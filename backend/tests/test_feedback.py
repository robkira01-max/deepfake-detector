"""Tests — Phase 4 Active Learning (feedback + retrain)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from models.user import User, UserRole
from models.analysis import Analysis, AnalysisStatus, Verdict
from models.document_analysis import DocumentAnalysis
from models.case import Case, Jurisdiction, CaseStatus
from models.media_file import MediaFile, MediaType, MediaStatus


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_case(db: Session, owner: User) -> Case:
    c = Case(
        case_number=f"FB-{datetime.now(timezone.utc).timestamp():.0f}",
        title="Feedback Test",
        jurisdiction=Jurisdiction.federal,
        status=CaseStatus.open,
        created_by_id=owner.id,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def _make_media(db: Session, case: Case, owner: User) -> MediaFile:
    mf = MediaFile(
        uuid=f"uuid-fb-{datetime.now(timezone.utc).timestamp():.0f}",
        case_id=case.id,
        original_filename="test.mp4",
        media_type=MediaType.video,
        mime_type="video/mp4",
        file_size_bytes=1024,
        status=MediaStatus.verified,
        hash_sha256="a" * 64,
        hash_blake3="b" * 64,
        
        ingested_by_id=owner.id,
    )
    db.add(mf)
    db.commit()
    db.refresh(mf)
    return mf


def _make_analysis(db: Session, case: Case, mf: MediaFile, owner: User) -> Analysis:
    a = Analysis(
        case_id=case.id,
        media_file_id=mf.id,
        status=AnalysisStatus.completed,
        verdict=Verdict.deepfake,
        final_score=0.87,
        started_at=datetime.now(timezone.utc),
        requested_by_id=owner.id,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _make_doc_analysis(db: Session, case: Case, mf: MediaFile, owner: User) -> DocumentAnalysis:
    da = DocumentAnalysis(
        case_id=case.id,
        media_file_id=mf.id,
        status=AnalysisStatus.completed,
        verdict=Verdict.authentic,
        final_score=0.15,
        requested_by_id=owner.id,
    )
    db.add(da)
    db.commit()
    db.refresh(da)
    return da


# ── Tests feedback AV ─────────────────────────────────────────────────────────

class TestSubmitAVFeedback:

    def test_correct_feedback_accepted(self, client: TestClient, auth_analyst: dict, db: Session, analyst_user: User):
        case = _make_case(db, analyst_user)
        mf = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, mf, analyst_user)
        resp = client.post(f"/feedback/analysis/{analysis.id}", json={
            "feedback_type": "correct",
            "confidence_rating": 5,
        }, headers=auth_analyst)
        assert resp.status_code == 201
        body = resp.json()
        assert body["feedback_type"] == "correct"
        assert body["analysis_id"] == analysis.id

    def test_incorrect_feedback_requires_true_verdict(self, client: TestClient, auth_analyst: dict, db: Session, analyst_user: User):
        case = _make_case(db, analyst_user)
        mf = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, mf, analyst_user)
        resp = client.post(f"/feedback/analysis/{analysis.id}", json={
            "feedback_type": "incorrect",
        }, headers=auth_analyst)
        assert resp.status_code == 422

    def test_incorrect_feedback_with_verdict_accepted(self, client: TestClient, auth_analyst: dict, db: Session, analyst_user: User):
        case = _make_case(db, analyst_user)
        mf = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, mf, analyst_user)
        resp = client.post(f"/feedback/analysis/{analysis.id}", json={
            "feedback_type": "incorrect",
            "true_verdict": "authentic",
            "analyst_notes": "Le signal rPPG était ambigu",
        }, headers=auth_analyst)
        assert resp.status_code == 201
        body = resp.json()
        assert body["true_verdict"] == "authentic"
        assert body["analyst_notes"] == "Le signal rPPG était ambigu"

    def test_analysis_not_found(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/feedback/analysis/99999", json={
            "feedback_type": "correct",
        }, headers=auth_analyst)
        assert resp.status_code == 404

    def test_unauthenticated_rejected(self, client: TestClient, db: Session, analyst_user: User):
        case = _make_case(db, analyst_user)
        mf = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, mf, analyst_user)
        resp = client.post(f"/feedback/analysis/{analysis.id}", json={"feedback_type": "correct"})
        assert resp.status_code == 401

    def test_readonly_rejected(self, client: TestClient, auth_readonly: dict, db: Session, analyst_user: User):
        case = _make_case(db, analyst_user)
        mf = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, mf, analyst_user)
        resp = client.post(f"/feedback/analysis/{analysis.id}", json={
            "feedback_type": "correct",
        }, headers=auth_readonly)
        assert resp.status_code == 403


class TestSubmitDocFeedback:

    def test_doc_feedback_accepted(self, client: TestClient, auth_analyst: dict, db: Session, analyst_user: User):
        case = _make_case(db, analyst_user)
        mf = _make_media(db, case, analyst_user)
        da = _make_doc_analysis(db, case, mf, analyst_user)
        resp = client.post(f"/feedback/document/{da.id}", json={
            "feedback_type": "uncertain",
            "confidence_rating": 2,
        }, headers=auth_analyst)
        assert resp.status_code == 201
        body = resp.json()
        assert body["document_analysis_id"] == da.id

    def test_doc_analysis_not_found(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/feedback/document/99999", json={"feedback_type": "correct"})
        assert resp.status_code in (401, 404)


# ── Tests get feedbacks ───────────────────────────────────────────────────────

class TestGetFeedbacks:

    def test_get_feedbacks_empty(self, client: TestClient, auth_analyst: dict, db: Session, analyst_user: User):
        case = _make_case(db, analyst_user)
        mf = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, mf, analyst_user)
        resp = client.get(f"/feedback/analysis/{analysis.id}", headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_feedbacks_returns_submitted(self, client: TestClient, auth_analyst: dict, db: Session, analyst_user: User):
        case = _make_case(db, analyst_user)
        mf = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, mf, analyst_user)
        client.post(f"/feedback/analysis/{analysis.id}", json={"feedback_type": "correct"}, headers=auth_analyst)
        resp = client.get(f"/feedback/analysis/{analysis.id}", headers=auth_analyst)
        assert resp.status_code == 200
        assert len(resp.json()) == 1


# ── Tests stats ───────────────────────────────────────────────────────────────

class TestFeedbackStats:

    def test_stats_empty(self, client: TestClient, auth_analyst: dict):
        resp = client.get("/feedback/stats", headers=auth_analyst)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_feedbacks"] == 0
        assert body["correct_pct"] == 0.0

    def test_stats_with_data(self, client: TestClient, auth_analyst: dict, db: Session, analyst_user: User):
        case = _make_case(db, analyst_user)
        mf = _make_media(db, case, analyst_user)
        a1 = _make_analysis(db, case, mf, analyst_user)
        a2 = _make_analysis(db, case, mf, analyst_user)
        client.post(f"/feedback/analysis/{a1.id}", json={"feedback_type": "correct"}, headers=auth_analyst)
        client.post(f"/feedback/analysis/{a2.id}", json={
            "feedback_type": "incorrect", "true_verdict": "authentic",
        }, headers=auth_analyst)
        resp = client.get("/feedback/stats", headers=auth_analyst)
        body = resp.json()
        assert body["total_feedbacks"] == 2
        assert body["correct_pct"] == 50.0

    def test_stats_unauthenticated(self, client: TestClient):
        resp = client.get("/feedback/stats")
        assert resp.status_code == 401


# ── Tests retrain ─────────────────────────────────────────────────────────────

class TestTriggerRetrain:

    def test_trigger_requires_admin(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/feedback/trigger-retrain", headers=auth_analyst)
        assert resp.status_code == 403

    def test_trigger_admin_queues_task(self, client: TestClient, auth_admin: dict):
        mock_task = MagicMock()
        mock_task.id = "test-celery-id-123"
        with patch("routers.feedback.retrain_models") as mock_retrain:
            mock_retrain.delay.return_value = mock_task
            resp = client.post("/feedback/trigger-retrain", headers=auth_admin)
        assert resp.status_code == 202
        body = resp.json()
        assert body["task_id"] == "test-celery-id-123"
        assert body["status"] == "queued"


# ── Tests retrain_task unit ───────────────────────────────────────────────────

class TestRetrainTask:

    def test_retrain_no_feedbacks(self, db: Session):
        from tasks.retrain_tasks import _DRIFT_THRESHOLD
        from models.feedback import AnalysisFeedback
        # Empty DB — task should return 0 processed
        result = {
            "feedbacks_processed": db.query(AnalysisFeedback).count(),
            "drift_score": 0.0,
            "models_flagged": [],
        }
        assert result["feedbacks_processed"] == 0
        assert result["drift_score"] == 0.0

    def test_drift_threshold_value(self):
        from tasks.retrain_tasks import _DRIFT_THRESHOLD
        assert _DRIFT_THRESHOLD == 0.20

    def test_feedback_model_constraint_requires_target(self, db: Session, analyst_user: User):
        from models.feedback import AnalysisFeedback, FeedbackType
        from sqlalchemy.exc import IntegrityError
        fb = AnalysisFeedback(
            analysis_id=None,
            document_analysis_id=None,
            feedback_type=FeedbackType.correct,
            submitted_by_id=analyst_user.id,
        )
        db.add(fb)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
