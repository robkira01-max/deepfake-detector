"""Feedback analyste sur un verdict — Phase 4 Active Learning."""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from sqlalchemy import String, Text, Integer, DateTime, Enum as SAEnum, ForeignKey, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class FeedbackType(str, enum.Enum):
    correct = "correct"
    incorrect = "incorrect"
    uncertain = "uncertain"


class TrueVerdict(str, enum.Enum):
    authentic = "authentic"
    deepfake = "deepfake"
    indeterminate = "indeterminate"


class AnalysisFeedback(Base):
    __tablename__ = "analysis_feedbacks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Cible (l'un ou l'autre — contrainte CHECK garantit qu'au moins un est non-null)
    analysis_id: Mapped[int | None] = mapped_column(
        ForeignKey("analyses.id"), nullable=True, index=True
    )
    document_analysis_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_analyses.id"), nullable=True, index=True
    )

    feedback_type: Mapped[FeedbackType] = mapped_column(
        SAEnum(FeedbackType), nullable=False
    )
    true_verdict: Mapped[TrueVerdict | None] = mapped_column(
        SAEnum(TrueVerdict), nullable=True
    )
    analyst_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5

    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    submitted_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    __table_args__ = (
        CheckConstraint(
            "analysis_id IS NOT NULL OR document_analysis_id IS NOT NULL",
            name="ck_feedback_target_not_null",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisFeedback id={self.id} type={self.feedback_type} "
            f"analysis={self.analysis_id} doc={self.document_analysis_id}>"
        )
