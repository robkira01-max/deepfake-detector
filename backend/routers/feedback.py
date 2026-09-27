"""Active Learning — feedback des analystes sur les verdicts."""
from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.security import require_admin, require_analyst
from database import get_db
from tasks.retrain_tasks import retrain_models
from models.analysis import Analysis
from models.audit_log import AuditAction, AuditLog
from models.document_analysis import DocumentAnalysis
from models.feedback import AnalysisFeedback, FeedbackType, TrueVerdict
from models.user import User

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["Active Learning"])


# ── Schémas ───────────────────────────────────────────────────────────────────

class FeedbackIn(BaseModel):
    feedback_type: FeedbackType
    true_verdict: TrueVerdict | None = None
    analyst_notes: str | None = Field(None, max_length=2000)
    confidence_rating: int | None = Field(None, ge=1, le=5)

    @model_validator(mode="after")
    def require_verdict_when_incorrect(self) -> "FeedbackIn":
        if self.feedback_type == FeedbackType.incorrect and self.true_verdict is None:
            raise ValueError("true_verdict requis quand feedback_type est 'incorrect'")
        return self


class FeedbackOut(BaseModel):
    id: int
    analysis_id: int | None
    document_analysis_id: int | None
    feedback_type: FeedbackType
    true_verdict: TrueVerdict | None
    analyst_notes: str | None
    confidence_rating: int | None
    submitted_at: str
    submitted_by_id: int

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm(cls, obj: AnalysisFeedback) -> "FeedbackOut":
        return cls(
            id=obj.id,
            analysis_id=obj.analysis_id,
            document_analysis_id=obj.document_analysis_id,
            feedback_type=obj.feedback_type,
            true_verdict=obj.true_verdict,
            analyst_notes=obj.analyst_notes,
            confidence_rating=obj.confidence_rating,
            submitted_at=obj.submitted_at.isoformat(),
            submitted_by_id=obj.submitted_by_id,
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/analysis/{analysis_id}",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
)
def submit_av_feedback(
    analysis_id: int,
    payload: FeedbackIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_analyst)],
) -> FeedbackOut:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analyse introuvable")

    fb = AnalysisFeedback(
        analysis_id=analysis_id,
        feedback_type=payload.feedback_type,
        true_verdict=payload.true_verdict,
        analyst_notes=payload.analyst_notes,
        confidence_rating=payload.confidence_rating,
        submitted_by_id=user.id,
    )
    db.add(fb)
    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.FEEDBACK_SUBMITTED,
        resource_type="Analysis",
        resource_id=str(analysis_id),
        details={
            "feedback_type": payload.feedback_type.value,
            "true_verdict": payload.true_verdict.value if payload.true_verdict else None,
        },
    ))
    db.commit()
    db.refresh(fb)
    log.info("feedback_submitted", analysis_id=analysis_id, type=payload.feedback_type, user=user.username)
    return FeedbackOut.from_orm(fb)


@router.post(
    "/document/{document_analysis_id}",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
)
def submit_doc_feedback(
    document_analysis_id: int,
    payload: FeedbackIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_analyst)],
) -> FeedbackOut:
    doc = db.query(DocumentAnalysis).filter(DocumentAnalysis.id == document_analysis_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Analyse document introuvable")

    fb = AnalysisFeedback(
        document_analysis_id=document_analysis_id,
        feedback_type=payload.feedback_type,
        true_verdict=payload.true_verdict,
        analyst_notes=payload.analyst_notes,
        confidence_rating=payload.confidence_rating,
        submitted_by_id=user.id,
    )
    db.add(fb)
    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.FEEDBACK_SUBMITTED,
        resource_type="DocumentAnalysis",
        resource_id=str(document_analysis_id),
        details={
            "feedback_type": payload.feedback_type.value,
            "true_verdict": payload.true_verdict.value if payload.true_verdict else None,
        },
    ))
    db.commit()
    db.refresh(fb)
    return FeedbackOut.from_orm(fb)


@router.get("/analysis/{analysis_id}", response_model=list[FeedbackOut])
def get_av_feedbacks(
    analysis_id: int,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_analyst)],
) -> list[FeedbackOut]:
    fbs = (
        db.query(AnalysisFeedback)
        .filter(AnalysisFeedback.analysis_id == analysis_id)
        .order_by(AnalysisFeedback.submitted_at.desc())
        .all()
    )
    return [FeedbackOut.from_orm(f) for f in fbs]


@router.get("/stats")
def feedback_stats(
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_analyst)],
) -> dict:
    total = db.query(func.count(AnalysisFeedback.id)).scalar() or 0
    by_type = (
        db.query(AnalysisFeedback.feedback_type, func.count(AnalysisFeedback.id))
        .group_by(AnalysisFeedback.feedback_type)
        .all()
    )
    type_counts = {t.value: n for t, n in by_type}
    correct_pct = round(type_counts.get("correct", 0) / total * 100, 1) if total else 0.0
    return {
        "total_feedbacks": total,
        "by_type": type_counts,
        "correct_pct": correct_pct,
        "incorrect_pct": round(type_counts.get("incorrect", 0) / total * 100, 1) if total else 0.0,
    }


@router.post("/trigger-retrain", status_code=status.HTTP_202_ACCEPTED)
def trigger_retrain(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_admin)],
) -> dict:
    task = retrain_models.delay()
    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.RETRAIN_TRIGGERED,
        details={"celery_task_id": task.id},
    ))
    db.commit()
    log.info("retrain_triggered", task_id=task.id, user=user.username)
    return {"task_id": task.id, "status": "queued"}
