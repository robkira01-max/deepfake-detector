"""Tests pour les endpoints analytiques et métriques Prometheus."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from models.analysis import AnalysisStatus, Verdict


class TestVerdictDistribution:
    """Tests GET /analytics/verdicts."""

    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/analytics/verdicts")
        assert resp.status_code == 401

    def test_returns_distribution_structure(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        resp = client.get("/analytics/verdicts", headers=auth_analyst)
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "distribution" in data
        assert isinstance(data["total"], int)
        assert isinstance(data["distribution"], dict)

    def test_distribution_with_data(
        self, client: TestClient, auth_analyst: dict, db
    ) -> None:
        from models.analysis import Analysis
        from models.case import Case
        from models.media_file import MediaFile, MediaType

        case = Case(
            case_number="ANAL-DIST-001",
            title="Analytics Test",
            jurisdiction="federal",
            status="completed",
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

        for verdict in [Verdict.deepfake, Verdict.deepfake, Verdict.authentic]:
            a = Analysis(
                case_id=case.id,
                media_file_id=media.id,
                status=AnalysisStatus.completed,
                verdict=verdict,
                final_score=0.85 if verdict == Verdict.deepfake else 0.2,
                requested_by_id=1,
                completed_at=datetime.now(timezone.utc),
            )
            db.add(a)
        db.commit()

        resp = client.get("/analytics/verdicts", headers=auth_analyst)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 3


class TestTrends:
    """Tests GET /analytics/trends."""

    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/analytics/trends")
        assert resp.status_code == 401

    def test_default_monthly_period(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        resp = client.get("/analytics/trends", headers=auth_analyst)
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "monthly"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_daily_period(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/analytics/trends?period=daily", headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json()["period"] == "daily"

    def test_weekly_period(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/analytics/trends?period=weekly", headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json()["period"] == "weekly"

    def test_invalid_period_returns_422(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/analytics/trends?period=yearly", headers=auth_analyst)
        assert resp.status_code == 422

    def test_limit_parameter(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/analytics/trends?limit=6", headers=auth_analyst)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) <= 6

    def test_trends_data_structure(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/analytics/trends", headers=auth_analyst)
        data = resp.json()["data"]
        for bucket in data:
            assert "period" in bucket
            assert "total" in bucket
            assert "deepfake" in bucket
            assert "authentic" in bucket
            assert "undetermined" in bucket


class TestEngineScores:
    """Tests GET /analytics/engines."""

    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/analytics/engines")
        assert resp.status_code == 401

    def test_returns_all_engines(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/analytics/engines", headers=auth_analyst)
        assert resp.status_code == 200
        data = resp.json()
        assert "engines" in data
        expected = [
            "video_texture", "video_temporal", "rppg", "biometrics",
            "audio_model", "audio_phase", "metadata",
        ]
        for engine in expected:
            assert engine in data["engines"]

    def test_engine_stats_structure(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/analytics/engines", headers=auth_analyst)
        data = resp.json()
        for name, stats in data["engines"].items():
            assert "count" in stats
            # mean/stdev/min/max may be None if no data
            assert "mean" in stats


class TestCasesSummary:
    """Tests GET /analytics/cases-summary."""

    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/analytics/cases-summary")
        assert resp.status_code == 401

    def test_returns_correct_structure(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        resp = client.get("/analytics/cases-summary", headers=auth_analyst)
        assert resp.status_code == 200
        data = resp.json()
        assert "by_status" in data
        assert "by_jurisdiction" in data


class TestPrometheusMetrics:
    """Tests GET /analytics/metrics (Prometheus scrape endpoint)."""

    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/analytics/metrics")
        assert resp.status_code == 401

    def test_returns_prometheus_format(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        resp = client.get("/analytics/metrics", headers=auth_analyst)
        assert resp.status_code == 200
        content = resp.text
        # Prometheus text format starts with # HELP or # TYPE lines
        assert "deepfake_" in content

    def test_contains_expected_metrics(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        resp = client.get("/analytics/metrics", headers=auth_analyst)
        content = resp.text
        assert "deepfake_analyses_total" in content
        assert "deepfake_cases_open_total" in content
        assert "deepfake_analysis_duration_seconds" in content
        assert "deepfake_final_score" in content
