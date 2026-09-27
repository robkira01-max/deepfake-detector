"""Résultat d'analyse forensique de document (v2.0)."""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import String, Float, Text, DateTime, Enum as SAEnum, ForeignKey, JSON, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base
from models.analysis import AnalysisStatus, Verdict


class DocumentAnalysis(Base):
    __tablename__ = "document_analyses"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Liaisons
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), nullable=False, index=True)
    media_file_id: Mapped[int] = mapped_column(ForeignKey("media_files.id"), nullable=False, index=True)

    # Statut Celery
    status: Mapped[AnalysisStatus] = mapped_column(
        SAEnum(AnalysisStatus), default=AnalysisStatus.pending, nullable=False
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # Verdict
    verdict: Mapped[Verdict | None] = mapped_column(SAEnum(Verdict), nullable=True)
    verdict_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Scores composantes
    score_ela: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_clone: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_metadata_doc: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_font: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_text_ai: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Détails XAI
    anomalies: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    xai_plain_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    models_used: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Audit
    requested_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<DocumentAnalysis id={self.id} verdict={self.verdict_label} score={self.final_score}>"
