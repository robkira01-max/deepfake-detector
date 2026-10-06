"""Tests pour l'export batch PDF (ZIP) et CSV."""
from __future__ import annotations

import csv
import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from models.analysis import AnalysisStatus, Verdict


class TestBatchPdfExport:
    """Tests POST /export/batch-pdf."""

    def test_batch_pdf_empty_list_returns_422(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.post("/export/batch-pdf", json={"analysis_ids": []}, headers=auth_analyst)
        assert resp.status_code == 422

    def test_batch_pdf_too_many_returns_422(self, client: TestClient, auth_analyst: dict) -> None:
        ids = list(range(1, 52))  # 51 items
        resp = client.post("/export/batch-pdf", json={"analysis_ids": ids}, headers=auth_analyst)
        assert resp.status_code == 422

    def test_batch_pdf_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/export/batch-pdf", json={"analysis_ids": [1]})
        assert resp.status_code == 401

    def test_batch_pdf_skips_nonexistent_analysis(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        resp = client.post(
            "/export/batch-pdf", json={"analysis_ids": [99999]}, headers=auth_analyst
        )
        # Should return a ZIP with 0 files (skipped)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        buf = io.BytesIO(resp.content)
        with zipfile.ZipFile(buf) as zf:
            assert len(zf.namelist()) == 0

    def test_batch_pdf_skips_incomplete_analysis(
        self, client: TestClient, auth_analyst: dict, db
    ) -> None:
        from models.analysis import Analysis
        from models.case import Case
        from models.media_file import MediaFile, MediaType

        # Create a pending analysis
        case = Case(
            case_number="BATCH-TEST-001",
            title="Batch Test Case",
            jurisdiction="federal",
            status="open",
            created_by_id=1,
        )
        db.add(case)
        db.flush()

        import uuid as _uuid
        from models.media_file import MediaStatus
        media = MediaFile(
            uuid=str(_uuid.uuid4()),
            case_id=case.id,
            original_filename="test.mp4",
            media_type=MediaType.video,
            mime_type="video/mp4",
            file_size_bytes=1024,
            status=MediaStatus.verified,
            hash_sha256="a" * 64,
            hash_blake3="b" * 64,
            
            ingested_by_id=1,
        )
        db.add(media)
        db.flush()

        analysis = Analysis(
            case_id=case.id,
            media_file_id=media.id,
            status=AnalysisStatus.pending,
            requested_by_id=1,
        )
        db.add(analysis)
        db.commit()

        resp = client.post(
            "/export/batch-pdf",
            json={"analysis_ids": [analysis.id]},
            headers=auth_analyst,
        )
        assert resp.status_code == 200
        buf = io.BytesIO(resp.content)
        with zipfile.ZipFile(buf) as zf:
            assert len(zf.namelist()) == 0

    def test_batch_pdf_response_headers(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        resp = client.post(
            "/export/batch-pdf", json={"analysis_ids": [99999]}, headers=auth_analyst
        )
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert ".zip" in cd


class TestCsvExport:
    """Tests GET /export/cases/{case_id}/analyses.csv."""

    def test_csv_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/export/cases/1/analyses.csv")
        assert resp.status_code == 401

    def test_csv_not_found_case(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/export/cases/99999/analyses.csv", headers=auth_analyst)
        assert resp.status_code == 404

    def test_csv_empty_case(self, client: TestClient, auth_analyst: dict, db) -> None:
        from models.case import Case

        case = Case(
            case_number="CSV-EMPTY-001",
            title="Empty Case",
            jurisdiction="federal",
            status="open",
            created_by_id=1,
        )
        db.add(case)
        db.commit()

        resp = client.get(f"/export/cases/{case.id}/analyses.csv", headers=auth_analyst)
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        # Should have only header row
        lines = resp.content.decode("utf-8-sig").strip().splitlines()
        assert len(lines) == 1  # header only
        assert "analysis_id" in lines[0]

    def test_csv_with_completed_analysis(
        self, client: TestClient, auth_analyst: dict, db
    ) -> None:
        from models.analysis import Analysis
        from models.case import Case
        from models.media_file import MediaFile, MediaType
        from datetime import datetime, timezone

        case = Case(
            case_number="CSV-FULL-001",
            title="Full CSV Test",
            jurisdiction="ontario",
            status="completed",
            created_by_id=1,
        )
        db.add(case)
        db.flush()

        import uuid as _uuid2
        from models.media_file import MediaStatus
        media = MediaFile(
            uuid=str(_uuid2.uuid4()),
            case_id=case.id,
            original_filename="evidence.mp4",
            media_type=MediaType.video,
            mime_type="video/mp4",
            file_size_bytes=2048,
            status=MediaStatus.verified,
            hash_sha256="d" * 64,
            hash_blake3="e" * 64,
            
            ingested_by_id=1,
        )
        db.add(media)
        db.flush()

        analysis = Analysis(
            case_id=case.id,
            media_file_id=media.id,
            status=AnalysisStatus.completed,
            verdict=Verdict.deepfake,
            final_score=0.87,
            confidence_low=0.82,
            confidence_high=0.92,
            score_video_texture=0.91,
            score_audio_model=0.85,
            duration_seconds=45,
            requested_by_id=1,
            completed_at=datetime.now(timezone.utc),
        )
        db.add(analysis)
        db.commit()

        resp = client.get(f"/export/cases/{case.id}/analyses.csv", headers=auth_analyst)
        assert resp.status_code == 200

        content = resp.content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["case_number"] == "CSV-FULL-001"
        assert row["verdict"] == "DEEPFAKE DÉTECTÉ"
        assert float(row["final_score"]) == pytest.approx(0.87, abs=0.001)
        assert row["duration_seconds"] == "45"

    def test_csv_filename_contains_case_number(
        self, client: TestClient, auth_analyst: dict, db
    ) -> None:
        from models.case import Case

        case = Case(
            case_number="CSV-FN-2026",
            title="Filename Test",
            jurisdiction="federal",
            status="open",
            created_by_id=1,
        )
        db.add(case)
        db.commit()

        resp = client.get(f"/export/cases/{case.id}/analyses.csv", headers=auth_analyst)
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        assert "CSV-FN-2026" in cd

    def test_csv_headers_complete(
        self, client: TestClient, auth_analyst: dict, db
    ) -> None:
        from models.case import Case

        case = Case(
            case_number="CSV-HDR-001",
            title="Header Test",
            jurisdiction="federal",
            status="open",
            created_by_id=1,
        )
        db.add(case)
        db.commit()

        resp = client.get(f"/export/cases/{case.id}/analyses.csv", headers=auth_analyst)
        assert resp.status_code == 200

        content = resp.content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        expected_cols = [
            "analysis_id", "case_number", "verdict", "final_score",
            "score_video_texture", "score_audio_model", "duration_seconds",
        ]
        for col in expected_cols:
            assert col in (reader.fieldnames or []), f"Missing column: {col}"
