"""Tests pour le module Courtroom Mode — rapport_tribunal + generate_courtroom_report."""
from __future__ import annotations

import unittest.mock as mock
import uuid as _uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from models.analysis import Analysis, AnalysisStatus, Verdict
from models.case import Case, CaseStatus
from models.media_file import MediaFile, MediaType, MediaStatus
from models.report import Report
from models.user import User


# ── Helpers (miroir de test_reports.py) ──────────────────────────────────────

def _make_case(db: Session, user: User) -> Case:
    from models.case import Jurisdiction
    case = Case(
        case_number=f"CTR-{_uuid.uuid4().hex[:6].upper()}",
        title="Tribunal courtroom test",
        description="desc",
        jurisdiction=Jurisdiction.federal,
        status=CaseStatus.open,
        created_by_id=user.id,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def _make_media(db: Session, case: Case) -> MediaFile:
    mf = MediaFile(
        uuid=str(_uuid.uuid4()),
        case_id=case.id,
        original_filename="exhibit_A.mp4",
        media_type=MediaType.video,
        mime_type="video/mp4",
        file_size_bytes=2048,
        status=MediaStatus.quarantine,
        hash_sha256="a" * 64,
        hash_blake3="b" * 64,
        hash_md5="c" * 32,
        ingested_by_id=case.created_by_id,
    )
    db.add(mf)
    db.commit()
    db.refresh(mf)
    return mf


def _make_analysis(
    db: Session,
    case: Case,
    user: User,
    verdict: Verdict = Verdict.deepfake,
    status: AnalysisStatus = AnalysisStatus.completed,
    final_score: float = 0.78,
) -> Analysis:
    mf = _make_media(db, case)
    analysis = Analysis(
        case_id=case.id,
        media_file_id=mf.id,
        status=status,
        final_score=final_score,
        confidence_low=final_score - 0.08,
        confidence_high=min(final_score + 0.08, 1.0),
        verdict=verdict,
        requested_by_id=user.id,
        model_far=0.018,
        model_frr=0.042,
        model_eer=0.031,
        model_auc=0.974,
        score_video_texture=0.80,
        score_video_temporal=0.75,
        score_rppg=0.60,
        score_biometrics=0.55,
        score_audio_model=0.70,
        score_audio_phase=0.45,
        score_metadata=0.30,
        xai_plain_explanation="Le fichier présente des anomalies de texture et temporelles significatives.",
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _make_report(db: Session, case: Case, analysis: Analysis, user: User) -> Report:
    report = Report(
        report_number=f"CTR-{_uuid.uuid4().hex[:6].upper()}",
        case_id=case.id,
        analysis_id=analysis.id,
        expert_id=user.id,
        expert_username=user.username,
        is_signed=True,
        report_hash_sha256="d" * 64,
        pdf_path="/tmp/fake_courtroom.pdf",
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


# ── Tests _build_courtroom_extras ─────────────────────────────────────────────

class TestBuildCourtroomExtras:
    def test_deepfake_verdict_language(self, db: Session, analyst_user: User):
        from reporting.pdf_generator import _build_courtroom_extras
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user, verdict=Verdict.deepfake)
        extras = _build_courtroom_extras(analysis)
        assert "manipulation numérique" in extras["judicial_verdict_fr"]
        assert "manipulation" in extras["judicial_verdict_en"].lower()

    def test_authentic_verdict_language(self, db: Session, analyst_user: User):
        from reporting.pdf_generator import _build_courtroom_extras
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user, verdict=Verdict.authentic, final_score=0.20)
        extras = _build_courtroom_extras(analysis)
        assert "n'a pas identifié" in extras["judicial_verdict_fr"]
        assert "did not identify" in extras["judicial_verdict_en"].lower()

    def test_undetermined_verdict_language(self, db: Session, analyst_user: User):
        from reporting.pdf_generator import _build_courtroom_extras
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user, verdict=Verdict.undetermined, final_score=0.45)
        extras = _build_courtroom_extras(analysis)
        assert "sans atteindre le seuil" in extras["judicial_verdict_fr"]

    def test_reliability_stats_structure(self, db: Session, analyst_user: User):
        from reporting.pdf_generator import _build_courtroom_extras
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        extras = _build_courtroom_extras(analysis)
        assert extras["reliability_detection_pct"] == pytest.approx(97.4, 0.1)
        assert extras["reliability_false_alarm"] == 18   # round(0.018 * 1000)
        assert extras["reliability_missed"] == 42        # round(0.042 * 1000)

    def test_court_indicators_all_seven(self, db: Session, analyst_user: User):
        from reporting.pdf_generator import _build_courtroom_extras
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        extras = _build_courtroom_extras(analysis)
        features = {ind["feature"] for ind in extras["court_indicators"]}
        assert features == {"texture", "temporal", "rppg", "biometrics", "audio", "phase", "metadata"}

    def test_court_indicators_sorted_descending(self, db: Session, analyst_user: User):
        from reporting.pdf_generator import _build_courtroom_extras
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        extras = _build_courtroom_extras(analysis)
        scores = [ind["score"] for ind in extras["court_indicators"]]
        assert scores == sorted(scores, reverse=True)

    def test_court_indicators_have_required_keys(self, db: Session, analyst_user: User):
        from reporting.pdf_generator import _build_courtroom_extras
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        extras = _build_courtroom_extras(analysis)
        for ind in extras["court_indicators"]:
            assert "label_fr" in ind
            assert "label_en" in ind
            assert "court_interpretation" in ind
            assert 0.0 <= ind["score"] <= 1.0

    def test_defaults_when_metrics_none(self, db: Session, analyst_user: User):
        from reporting.pdf_generator import _build_courtroom_extras
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        analysis.model_auc = None
        analysis.model_far = None
        analysis.model_frr = None
        analysis.model_eer = None
        extras = _build_courtroom_extras(analysis)
        # Métriques non mesurées → None (Brief v2 P0.2 interdit les valeurs inventées)
        assert extras["reliability_detection_pct"] is None
        assert extras["reliability_false_alarm"] is None
        assert extras["reliability_missed"] is None
        assert extras["eer_pct"] is None


# ── Tests endpoint POST /reports/{analysis_id}/courtroom ─────────────────────

class TestCourtroomEndpoint:
    def test_courtroom_not_found(self, client: TestClient, analyst_token: str):
        resp = client.post(
            "/reports/9999/courtroom",
            headers={"Authorization": f"Bearer {analyst_token}"},
            json={},
        )
        assert resp.status_code == 404

    def test_courtroom_analysis_not_completed(
        self, client: TestClient, analyst_token: str, analyst_user: User, db: Session
    ):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user, status=AnalysisStatus.pending)
        resp = client.post(
            f"/reports/{analysis.id}/courtroom",
            headers={"Authorization": f"Bearer {analyst_token}"},
            json={},
        )
        assert resp.status_code == 409

    def test_courtroom_requires_auth(self, client: TestClient):
        resp = client.post("/reports/1/courtroom", json={})
        assert resp.status_code == 401

    def test_courtroom_readonly_forbidden(
        self, client: TestClient, readonly_token: str, analyst_user: User, db: Session
    ):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        resp = client.post(
            f"/reports/{analysis.id}/courtroom",
            headers={"Authorization": f"Bearer {readonly_token}"},
            json={},
        )
        assert resp.status_code == 403

    def test_courtroom_success_returns_ctr_prefix(
        self, client: TestClient, analyst_token: str, analyst_user: User, db: Session
    ):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        fake_report = _make_report(db, case, analysis, analyst_user)
        fake_report.report_number = f"CTR-{case.case_number}-20260924-ABCDEF"
        db.commit()

        with mock.patch("reporting.pdf_generator.generate_courtroom_report", return_value=fake_report):
            resp = client.post(
                f"/reports/{analysis.id}/courtroom",
                headers={"Authorization": f"Bearer {analyst_token}"},
                json={
                    "expert_name": "Me. Sophie Tremblay",
                    "expert_title": "Expert en forensique numérique",
                    "court_file_number": "2026-QC-001",
                    "notes_for_court": "Pièce A-7",
                },
            )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["report_number"].startswith("CTR-")
        assert data["is_signed"] is True

    def test_courtroom_success_minimal_body(
        self, client: TestClient, analyst_token: str, analyst_user: User, db: Session
    ):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        fake_report = _make_report(db, case, analysis, analyst_user)

        with mock.patch("reporting.pdf_generator.generate_courtroom_report", return_value=fake_report):
            resp = client.post(
                f"/reports/{analysis.id}/courtroom",
                headers={"Authorization": f"Bearer {analyst_token}"},
                json={},
            )
        assert resp.status_code == 201


# ── Tests generate_courtroom_report (unit) ────────────────────────────────────

class TestGenerateCourtroomReport:
    def test_generates_ctr_prefixed_report_number(self, db: Session, analyst_user: User):
        from reporting.pdf_generator import _generate_courtroom_number
        case = _make_case(db, analyst_user)
        number = _generate_courtroom_number(case)
        assert number.startswith("CTR-")
        assert case.case_number in number

    def test_generate_creates_report_with_hash(self, db: Session, analyst_user: User, tmp_path):
        from reporting.pdf_generator import generate_courtroom_report
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)

        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 courtroom content")

        with (
            mock.patch("reporting.pdf_generator.settings") as mock_settings,
            mock.patch("reporting.pdf_generator.request_tsa_timestamp", return_value=None),
            mock.patch("reporting.pdf_generator.sign_audit_entry", return_value=("hash123", "sig456")),
        ):
            mock_settings.processed_dir = str(tmp_path)
            mock_settings.jwt_public_key = ""
            mock_settings.tsa_url = ""
            mock_settings.tsa_cert_path = ""

            # Patch WeasyPrint (imported lazily inside function)
            with mock.patch("weasyprint.HTML") as mock_html:
                mock_html.return_value.write_pdf = lambda path: Path(path).write_bytes(b"%PDF fake court")
                report = generate_courtroom_report(
                    analysis_id=analysis.id,
                    expert=analyst_user,
                    db=db,
                )

        assert report.report_number.startswith("CTR-")
        assert report.report_hash_sha256 is not None
        assert len(report.report_hash_sha256) == 64
