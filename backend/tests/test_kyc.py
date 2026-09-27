"""Tests — Router KYC /kyc/verify et endpoints associés."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from models.audit_log import AuditLog, AuditAction
from models.kyc_verification import KYCVerification, KYCVerdict
from models.user import User


# ── Fixtures KYC ──────────────────────────────────────────────────────────────

def _fake_file(name: str = "document.jpg", content: bytes = b"fakejpegdata") -> tuple[str, io.BytesIO, str]:
    return (name, io.BytesIO(content), "image/jpeg")


def _mock_ingest(case_id: int = 0, file_id_offset: int = 1):
    """Retourne un MediaFile factice."""
    mf = MagicMock()
    mf.id = file_id_offset
    mf.storage_key = None
    return MagicMock(media_file=mf, warnings=[])


def _patch_engines(face_verdict: str = "MATCH", face_score: float = 0.85):
    """Patch FaceMatchingEngine + OCREngine + ingest_file pour les tests."""
    from engines.biometric_engine import FaceMatchResult

    face_result = FaceMatchResult(
        similarity=face_score,
        is_match=(face_verdict == "MATCH"),
        face_found_doc=face_verdict not in ("NO_FACE", "UNAVAILABLE"),
        face_found_selfie=face_verdict not in ("NO_FACE", "UNAVAILABLE"),
        backend_used="insightface",
        error="no_face_backend" if face_verdict == "UNAVAILABLE" else None,
    )

    biometric_mock = MagicMock()
    biometric_mock.compare.return_value = face_result

    ocr_mock = MagicMock()
    ocr_mock.extract_mrz.return_value = None  # tesseract non dispo en test

    return biometric_mock, ocr_mock


# ── Tests POST /kyc/verify ────────────────────────────────────────────────────

class TestKYCVerify:

    def test_kyc_verify_pass(self, client: TestClient, analyst_token: str):
        bio_mock, ocr_mock = _patch_engines("MATCH", 0.90)

        with (
            patch("routers.kyc._get_biometric", return_value=bio_mock),
            patch("routers.kyc._get_ocr", return_value=ocr_mock),
            patch("routers.kyc.ingest_file", side_effect=[_mock_ingest(file_id_offset=10), _mock_ingest(file_id_offset=11)]),
        ):
            resp = client.post(
                "/kyc/verify",
                files={
                    "doc_file": _fake_file("passport.jpg"),
                    "selfie_file": _fake_file("selfie.jpg"),
                },
                headers={"Authorization": f"Bearer {analyst_token}"},
            )

        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["kyc_verdict"] == "PASS"
        assert data["face_match_verdict"] == "MATCH"
        assert data["face_match_score"] == pytest.approx(0.90, abs=1e-4)

    def test_kyc_verify_fail_no_match(self, client: TestClient, analyst_token: str):
        bio_mock, ocr_mock = _patch_engines("NO_MATCH", 0.10)

        with (
            patch("routers.kyc._get_biometric", return_value=bio_mock),
            patch("routers.kyc._get_ocr", return_value=ocr_mock),
            patch("routers.kyc.ingest_file", side_effect=[_mock_ingest(10), _mock_ingest(11)]),
        ):
            resp = client.post(
                "/kyc/verify",
                files={
                    "doc_file": _fake_file("id.jpg"),
                    "selfie_file": _fake_file("selfie.jpg"),
                },
                headers={"Authorization": f"Bearer {analyst_token}"},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["kyc_verdict"] == "FAIL"
        assert data["face_match_verdict"] == "NO_MATCH"
        assert len(data["kyc_verdict_reasons"]) > 0

    def test_kyc_verify_review_no_face(self, client: TestClient, analyst_token: str):
        bio_mock, ocr_mock = _patch_engines("NO_FACE", 0.0)

        with (
            patch("routers.kyc._get_biometric", return_value=bio_mock),
            patch("routers.kyc._get_ocr", return_value=ocr_mock),
            patch("routers.kyc.ingest_file", side_effect=[_mock_ingest(10), _mock_ingest(11)]),
        ):
            resp = client.post(
                "/kyc/verify",
                files={
                    "doc_file": _fake_file("id.jpg"),
                    "selfie_file": _fake_file("selfie.jpg"),
                },
                headers={"Authorization": f"Bearer {analyst_token}"},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["kyc_verdict"] == "REVIEW"

    def test_kyc_verify_no_selfie_gives_review(self, client: TestClient, analyst_token: str):
        """Sans selfie → face_match_verdict=UNAVAILABLE → verdict REVIEW."""
        bio_mock, ocr_mock = _patch_engines("MATCH", 0.90)

        with (
            patch("routers.kyc._get_biometric", return_value=bio_mock),
            patch("routers.kyc._get_ocr", return_value=ocr_mock),
            patch("routers.kyc.ingest_file", return_value=_mock_ingest(10)),
        ):
            resp = client.post(
                "/kyc/verify",
                files={"doc_file": _fake_file("id.jpg")},
                headers={"Authorization": f"Bearer {analyst_token}"},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["kyc_verdict"] == "REVIEW"
        assert data["face_match_verdict"] == "UNAVAILABLE"

    def test_kyc_verify_audit_logged(self, client: TestClient, analyst_token: str, db: Session):
        bio_mock, ocr_mock = _patch_engines("MATCH", 0.85)

        with (
            patch("routers.kyc._get_biometric", return_value=bio_mock),
            patch("routers.kyc._get_ocr", return_value=ocr_mock),
            patch("routers.kyc.ingest_file", side_effect=[_mock_ingest(20), _mock_ingest(21)]),
        ):
            resp = client.post(
                "/kyc/verify",
                files={
                    "doc_file": _fake_file("doc.jpg"),
                    "selfie_file": _fake_file("selfie.jpg"),
                },
                headers={"Authorization": f"Bearer {analyst_token}"},
            )

        assert resp.status_code == 201
        log_entry = (
            db.query(AuditLog)
            .filter(AuditLog.action == AuditAction.KYC_VERIFIED)
            .first()
        )
        assert log_entry is not None
        assert log_entry.details["verdict"] in ("PASS", "FAIL", "REVIEW")

    def test_kyc_verify_requires_auth(self, client: TestClient):
        resp = client.post(
            "/kyc/verify",
            files={"doc_file": _fake_file("doc.jpg")},
        )
        assert resp.status_code == 401

    def test_kyc_verify_readonly_forbidden(self, client: TestClient, readonly_token: str):
        bio_mock, ocr_mock = _patch_engines()
        with (
            patch("routers.kyc._get_biometric", return_value=bio_mock),
            patch("routers.kyc._get_ocr", return_value=ocr_mock),
            patch("routers.kyc.ingest_file", return_value=_mock_ingest()),
        ):
            resp = client.post(
                "/kyc/verify",
                files={"doc_file": _fake_file("doc.jpg")},
                headers={"Authorization": f"Bearer {readonly_token}"},
            )
        assert resp.status_code == 403


# ── Tests GET /kyc/{id} ───────────────────────────────────────────────────────

class TestKYCGet:

    def _create_verif(self, db: Session, user: User) -> KYCVerification:
        from models.case import Case, CaseStatus, Jurisdiction
        from models.media_file import MediaFile, MediaType, MediaStatus
        case = Case(
            case_number="KYC-GET-001",
            title="KYC Get Test",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=user.id,
        )
        db.add(case)
        db.flush()
        mf = MediaFile(
            uuid="aaaaaaaa-0000-0000-0000-000000000001",
            case_id=case.id,
            original_filename="doc.jpg",
            media_type=MediaType.document,
            mime_type="image/jpeg",
            file_size_bytes=1024,
            status=MediaStatus.verified,
            hash_sha256="a" * 64,
            hash_blake3="b" * 64,
            hash_md5="c" * 32,
            ingested_by_id=user.id,
        )
        db.add(mf)
        db.flush()

        verif = KYCVerification(
            doc_media_file_id=mf.id,
            requested_by_id=user.id,
            face_match_verdict="MATCH",
            face_match_score=0.88,
            kyc_verdict=KYCVerdict.PASS,
            kyc_verdict_reasons=[],
        )
        db.add(verif)
        db.commit()
        db.refresh(verif)
        return verif

    def test_get_verification(self, client: TestClient, analyst_token: str, analyst_user: User, db: Session):
        verif = self._create_verif(db, analyst_user)
        resp = client.get(
            f"/kyc/{verif.id}",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verification_id"] == verif.id
        assert data["kyc_verdict"] == "PASS"

    def test_get_verification_not_found(self, client: TestClient, analyst_token: str):
        resp = client.get(
            "/kyc/99999",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 404

    def test_list_case_verifications(self, client: TestClient, analyst_token: str, analyst_user: User, db: Session):
        from models.case import Case, CaseStatus, Jurisdiction
        case = Case(
            case_number="KYC-TEST-001",
            title="Test KYC",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=analyst_user.id,
        )
        db.add(case)
        db.flush()

        from models.media_file import MediaFile, MediaType, MediaStatus
        mf = MediaFile(
            uuid="aaaaaaaa-0000-0000-0000-000000000099",
            case_id=case.id,
            original_filename="doc.jpg",
            media_type=MediaType.document,
            mime_type="image/jpeg",
            file_size_bytes=1024,
            status=MediaStatus.verified,
            hash_sha256="d" * 64,
            hash_blake3="e" * 64,
            hash_md5="f" * 32,
            ingested_by_id=analyst_user.id,
        )
        db.add(mf)
        db.flush()

        verif = KYCVerification(
            case_id=case.id,
            doc_media_file_id=mf.id,
            requested_by_id=analyst_user.id,
            face_match_verdict="MATCH",
            kyc_verdict=KYCVerdict.PASS,
            kyc_verdict_reasons=[],
        )
        db.add(verif)
        db.commit()

        resp = client.get(
            f"/kyc/case/{case.id}",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 200
        assert len(resp.json()) >= 1


# ── Tests KYCVerification.compute_verdict ────────────────────────────────────

class TestKYCVerdict:

    def _make_verif(self, **kwargs) -> KYCVerification:
        return KYCVerification(
            doc_media_file_id=1,
            requested_by_id=1,
            face_match_verdict=kwargs.get("face_match_verdict", "MATCH"),
            face_match_score=kwargs.get("face_match_score", 0.9),
            document_deepfake_score=kwargs.get("document_deepfake_score"),
        )

    def test_verdict_pass(self):
        v = self._make_verif(face_match_verdict="MATCH")
        v.compute_verdict()
        assert v.kyc_verdict == KYCVerdict.PASS

    def test_verdict_fail_no_match(self):
        v = self._make_verif(face_match_verdict="NO_MATCH")
        v.compute_verdict()
        assert v.kyc_verdict == KYCVerdict.FAIL
        assert len(v.kyc_verdict_reasons) > 0

    def test_verdict_fail_high_deepfake(self):
        v = self._make_verif(face_match_verdict="MATCH", document_deepfake_score=0.85)
        v.compute_verdict()
        assert v.kyc_verdict == KYCVerdict.FAIL

    def test_verdict_review_no_face(self):
        v = self._make_verif(face_match_verdict="NO_FACE")
        v.compute_verdict()
        assert v.kyc_verdict == KYCVerdict.REVIEW

    def test_verdict_review_unavailable(self):
        v = self._make_verif(face_match_verdict="UNAVAILABLE")
        v.compute_verdict()
        assert v.kyc_verdict == KYCVerdict.REVIEW

    def test_verdict_review_medium_deepfake(self):
        v = self._make_verif(face_match_verdict="MATCH", document_deepfake_score=0.60)
        v.compute_verdict()
        assert v.kyc_verdict == KYCVerdict.REVIEW

    def test_fail_overrides_review(self):
        """FAIL ne peut pas être rétrogradé en REVIEW."""
        v = self._make_verif(face_match_verdict="NO_MATCH", document_deepfake_score=0.60)
        v.compute_verdict()
        assert v.kyc_verdict == KYCVerdict.FAIL
