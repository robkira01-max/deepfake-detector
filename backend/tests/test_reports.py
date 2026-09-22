"""Tests pour routers/reports.py et core/encryption.py."""
from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

import uuid as _uuid

from models.analysis import Analysis, AnalysisStatus, Verdict
from models.case import Case, CaseStatus
from models.media_file import MediaFile, MediaType, MediaStatus
from models.report import Report
from models.user import User


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_case(db: Session, user: User) -> Case:
    from models.case import Jurisdiction
    case = Case(
        case_number=f"RPT-{_uuid.uuid4().hex[:6].upper()}",
        title="Test report case",
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
    # Need a user to pass ingested_by_id — reuse case creator via case.created_by_id
    mf = MediaFile(
        uuid=str(_uuid.uuid4()),
        case_id=case.id,
        original_filename="test_video.mp4",
        media_type=MediaType.video,
        mime_type="video/mp4",
        file_size_bytes=1024,
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


def _make_analysis(db: Session, case: Case, user: User, status=AnalysisStatus.completed) -> Analysis:
    mf = _make_media(db, case)
    analysis = Analysis(
        case_id=case.id,
        media_file_id=mf.id,
        status=status,
        final_score=0.87,
        verdict=Verdict.deepfake,
        requested_by_id=user.id,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _make_report(db: Session, case: Case, analysis: Analysis, user: User, pdf_exists: bool = False) -> Report:
    pdf_path = "/tmp/test_report.pdf" if pdf_exists else "/tmp/nonexistent.pdf"
    if pdf_exists:
        Path(pdf_path).write_bytes(b"%PDF-1.4 fake content")
    report = Report(
        report_number="RPT-2026-001",
        case_id=case.id,
        analysis_id=analysis.id,
        expert_id=user.id,
        expert_username=user.username,
        is_signed=True,
        report_hash_sha256="abc123",
        pdf_path=pdf_path,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


# ── Tests encryption ───────────────────────────────────────────────────────────

class TestEncryption:
    def test_encrypt_decrypt_roundtrip_no_fernet(self):
        """Sans clé Fernet configurée, encrypt/decrypt sont des no-ops."""
        from core.encryption import encrypt_secret, decrypt_secret
        plaintext = "my_secret_totp_key"
        encrypted = encrypt_secret(plaintext)
        assert encrypted == plaintext
        decrypted = decrypt_secret(encrypted)
        assert decrypted == plaintext

    def test_encrypt_decrypt_with_fernet(self):
        """Avec clé Fernet, les données sont chiffrées puis déchiffrées correctement."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        with mock.patch("core.encryption.settings") as m:
            m.fernet = Fernet(key.encode())
            from core.encryption import encrypt_secret, decrypt_secret
            plaintext = "super_secret_mfa_key_AAABBBCCC"
            encrypted = encrypt_secret(plaintext)
            assert encrypted != plaintext
            decrypted = decrypt_secret(encrypted)
            assert decrypted == plaintext

    def test_decrypt_invalid_ciphertext_returns_ciphertext(self):
        """Une valeur non chiffrée est retournée telle quelle si Fernet ne peut pas la déchiffrer."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        with mock.patch("core.encryption.settings") as m:
            m.fernet = Fernet(key.encode())
            from core.encryption import decrypt_secret
            result = decrypt_secret("not_encrypted_at_all")
            assert result == "not_encrypted_at_all"

    def test_encrypt_empty_string_no_fernet(self):
        from core.encryption import encrypt_secret
        assert encrypt_secret("") == ""


# ── Tests token_blocklist ──────────────────────────────────────────────────────

class TestTokenBlocklist:
    def test_is_revoked_empty_jti_returns_false(self):
        """jti vide → False sans contacter Redis."""
        from core.token_blocklist import is_revoked
        assert is_revoked("") is False

    def test_revoke_token_no_redis(self):
        """Sans Redis, revoke_token loggue un warning et ne lève pas d'exception."""
        import core.token_blocklist as bl
        with mock.patch.object(bl, "_get_redis", return_value=None):
            bl._redis = None
            bl.revoke_token("test-jti-123", 1800)  # Should not raise

    def test_is_revoked_redis_unavailable_raises_503(self):
        """Sans Redis, is_revoked lève 503 (fail-secure)."""
        import core.token_blocklist as bl
        from fastapi import HTTPException
        with mock.patch.object(bl, "_get_redis", return_value=None):
            bl._redis = None
            with pytest.raises(HTTPException) as exc_info:
                bl.is_revoked("some-jti")
            assert exc_info.value.status_code == 503

    def test_revoke_and_check_with_mock_redis(self):
        """Avec Redis mocké, revoke + is_revoked fonctionne correctement."""
        import core.token_blocklist as bl
        r_mock = mock.MagicMock()
        r_mock.exists.return_value = 1
        with mock.patch.object(bl, "_get_redis", return_value=r_mock):
            bl.revoke_token("jti-abc", 1800)
            r_mock.setex.assert_called_once_with("blocklist:jti-abc", 1800, "1")
            assert bl.is_revoked("jti-abc") is True

    def test_is_not_revoked_with_mock_redis(self):
        """Token non révoqué → False."""
        import core.token_blocklist as bl
        r_mock = mock.MagicMock()
        r_mock.exists.return_value = 0
        with mock.patch.object(bl, "_get_redis", return_value=r_mock):
            assert bl.is_revoked("clean-jti") is False


# ── Tests reports router ───────────────────────────────────────────────────────

class TestReportsList:
    def test_list_reports_empty(self, client, analyst_token, analyst_user, db):
        case = _make_case(db, analyst_user)
        resp = client.get(
            f"/reports/case/{case.id}",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_reports_returns_reports(self, client, analyst_token, analyst_user, db):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        _make_report(db, case, analysis, analyst_user)
        resp = client.get(
            f"/reports/case/{case.id}",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["report_number"] == "RPT-2026-001"

    def test_list_reports_requires_auth(self, client, db, analyst_user):
        case = _make_case(db, analyst_user)
        resp = client.get(f"/reports/case/{case.id}")
        assert resp.status_code == 401

    def test_readonly_can_list_reports(self, client, readonly_token, analyst_user, db):
        case = _make_case(db, analyst_user)
        resp = client.get(
            f"/reports/case/{case.id}",
            headers={"Authorization": f"Bearer {readonly_token}"},
        )
        assert resp.status_code == 200


class TestReportDownload:
    def test_download_report_not_found(self, client, analyst_token):
        resp = client.get(
            "/reports/9999/download",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 404

    def test_download_no_pdf_path(self, client, analyst_token, analyst_user, db):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        report = Report(
            report_number="RPT-NO-PDF",
            case_id=case.id,
            analysis_id=analysis.id,
            expert_id=analyst_user.id,
            expert_username=analyst_user.username,
            is_signed=False,
            pdf_path=None,
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        resp = client.get(
            f"/reports/{report.id}/download",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 404

    def test_download_pdf_file_missing_on_disk(self, client, analyst_token, analyst_user, db):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        report = _make_report(db, case, analysis, analyst_user, pdf_exists=False)
        resp = client.get(
            f"/reports/{report.id}/download",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 404

    def test_download_requires_auth(self, client):
        resp = client.get("/reports/1/download")
        assert resp.status_code == 401


class TestGenerateReport:
    def test_generate_analysis_not_found(self, client, analyst_token):
        resp = client.post(
            "/reports/generate/9999",
            headers={"Authorization": f"Bearer {analyst_token}"},
            json={},
        )
        assert resp.status_code == 404

    def test_generate_analysis_not_completed(self, client, analyst_token, analyst_user, db):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user, status=AnalysisStatus.pending)
        resp = client.post(
            f"/reports/generate/{analysis.id}",
            headers={"Authorization": f"Bearer {analyst_token}"},
            json={},
        )
        assert resp.status_code == 409

    def test_generate_readonly_forbidden(self, client, readonly_token, analyst_user, db):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        resp = client.post(
            f"/reports/generate/{analysis.id}",
            headers={"Authorization": f"Bearer {readonly_token}"},
            json={},
        )
        assert resp.status_code == 403

    def test_generate_report_success(self, client, analyst_token, analyst_user, db):
        case = _make_case(db, analyst_user)
        analysis = _make_analysis(db, case, analyst_user)
        fake_report = _make_report(db, case, analysis, analyst_user)

        with mock.patch("reporting.pdf_generator.generate_pdf_report", return_value=fake_report):
            resp = client.post(
                f"/reports/generate/{analysis.id}",
                headers={"Authorization": f"Bearer {analyst_token}"},
                json={"expert_name": "Dr. Test", "notes_for_court": "Pièce A-1"},
            )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["report_number"] == "RPT-2026-001"
        assert data["is_signed"] is True

    def test_generate_requires_auth(self, client):
        resp = client.post("/reports/generate/1", json={})
        assert resp.status_code == 401
