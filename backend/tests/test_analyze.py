"""Tests des routes d'upload et d'analyse — /analyze/."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from models.user import User, UserRole


def _make_case(db: Session, owner: User, case_number: str = "CASE-001"):
    from models.case import Case, CaseStatus, Jurisdiction
    case = Case(
        case_number=case_number,
        title="Test Case",
        jurisdiction=Jurisdiction.federal,
        status=CaseStatus.open,
        created_by_id=owner.id,
        retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


class TestUploadMedia:
    """Tests de POST /analyze/upload/{case_id}."""

    def test_upload_to_nonexistent_case(
        self, client: TestClient, auth_analyst: dict, tiny_wav: Path
    ):
        """Upload vers un case inexistant → 404."""
        with open(tiny_wav, "rb") as f:
            resp = client.post(
                "/analyze/upload/99999",
                files={"file": ("test.wav", f, "audio/wav")},
                headers=auth_analyst,
            )
        assert resp.status_code == 404

    def test_upload_invalid_extension(
        self, client: TestClient, db: Session, auth_analyst: dict,
        analyst_user: User, tmp_path: Path
    ):
        """Fichier .exe rejeté (extension non autorisée) → 422."""
        case = _make_case(db, analyst_user, "UPLOAD-EXT-001")
        exe_file = tmp_path / "mal.exe"
        exe_file.write_bytes(b"MZ\x00\x00")

        with open(exe_file, "rb") as f:
            resp = client.post(
                f"/analyze/upload/{case.id}",
                files={"file": ("mal.exe", f, "application/octet-stream")},
                headers=auth_analyst,
            )
        assert resp.status_code == 422

    def test_upload_idor_other_case(
        self, client: TestClient, db: Session,
        auth_analyst: dict, admin_user: User, tiny_wav: Path
    ):
        """Upload vers un case d'un autre utilisateur → 403."""
        admin_case = _make_case(db, admin_user, "ADMIN-UPLOAD-001")

        with open(tiny_wav, "rb") as f:
            resp = client.post(
                f"/analyze/upload/{admin_case.id}",
                files={"file": ("test.wav", f, "audio/wav")},
                headers=auth_analyst,
            )
        assert resp.status_code == 403

    def test_upload_requires_auth(self, client: TestClient, tiny_wav: Path):
        """Sans token → 401."""
        with open(tiny_wav, "rb") as f:
            resp = client.post(
                "/analyze/upload/1",
                files={"file": ("test.wav", f, "audio/wav")},
            )
        assert resp.status_code == 401

    def test_upload_readonly_cannot_upload(
        self, client: TestClient, db: Session,
        auth_readonly: dict, analyst_user: User, tiny_wav: Path
    ):
        """Un utilisateur readonly ne peut pas uploader → 403."""
        case = _make_case(db, analyst_user, "READONLY-UPLOAD-001")
        with open(tiny_wav, "rb") as f:
            resp = client.post(
                f"/analyze/upload/{case.id}",
                files={"file": ("test.wav", f, "audio/wav")},
                headers=auth_readonly,
            )
        assert resp.status_code == 403


class TestGetAnalysis:
    """Tests de GET /analyze/{analysis_id}."""

    def _make_analysis(self, db: Session, case_id: int, owner: User):
        from models.analysis import Analysis, AnalysisStatus
        from models.media_file import MediaFile, MediaType
        # Créer un MediaFile factice
        mf = MediaFile(
            uuid="test-uuid-001",
            case_id=case_id,
            original_filename="test.wav",
            media_type=MediaType.audio,
            mime_type="audio/wav",
            file_size_bytes=3244,
            hash_sha256="a" * 64,
            hash_blake3="b" * 64,
            hash_md5="c" * 32,
            storage_key="/tmp/test.wav",
            ingested_by_id=owner.id,
        )
        db.add(mf)
        db.commit()
        db.refresh(mf)

        analysis = Analysis(
            case_id=case_id,
            media_file_id=mf.id,
            status=AnalysisStatus.pending,
            requested_by_id=owner.id,
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis

    def test_get_own_analysis(
        self, client: TestClient, db: Session,
        auth_analyst: dict, analyst_user: User
    ):
        """Un analyst peut lire ses propres analyses."""
        case = _make_case(db, analyst_user, "ANAL-OWN-001")
        analysis = self._make_analysis(db, case.id, analyst_user)

        resp = client.get(f"/analyze/{analysis.id}", headers=auth_analyst)
        assert resp.status_code == 200
        body = resp.json()
        # AnalysisResponse sérialise avec l'alias "id" (Field(alias="id"))
        assert body.get("id") == analysis.id or body.get("analysis_id") == analysis.id

    def test_get_analysis_idor(
        self, client: TestClient, db: Session,
        auth_analyst: dict, admin_user: User
    ):
        """Un analyst ne peut pas lire les analyses d'un autre → 403."""
        case = _make_case(db, admin_user, "ANAL-IDOR-001")
        analysis = self._make_analysis(db, case.id, admin_user)

        resp = client.get(f"/analyze/{analysis.id}", headers=auth_analyst)
        assert resp.status_code == 403

    def test_get_nonexistent_analysis(
        self, client: TestClient, auth_analyst: dict
    ):
        """Analyse inexistante → 404."""
        resp = client.get("/analyze/99999", headers=auth_analyst)
        assert resp.status_code == 404

    def test_admin_can_read_any_analysis(
        self, client: TestClient, db: Session,
        auth_admin: dict, analyst_user: User
    ):
        """Un admin peut lire toutes les analyses (pas de filtre IDOR pour admin)."""
        case = _make_case(db, analyst_user, "ANAL-ADMIN-001")
        analysis = self._make_analysis(db, case.id, analyst_user)

        resp = client.get(f"/analyze/{analysis.id}", headers=auth_admin)
        assert resp.status_code == 200

    def test_list_case_analyses_requires_ownership(
        self, client: TestClient, db: Session,
        auth_analyst: dict, admin_user: User
    ):
        """GET /analyze/case/{id} → 403 si l'analyst n'est pas propriétaire."""
        case = _make_case(db, admin_user, "ANAL-LIST-001")

        resp = client.get(f"/analyze/case/{case.id}", headers=auth_analyst)
        assert resp.status_code == 403

    def test_list_case_analyses_own(
        self, client: TestClient, db: Session,
        auth_analyst: dict, analyst_user: User
    ):
        """GET /analyze/case/{id} → liste vide si aucune analyse."""
        case = _make_case(db, analyst_user, "ANAL-LIST-OWN-001")

        resp = client.get(f"/analyze/case/{case.id}", headers=auth_analyst)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
