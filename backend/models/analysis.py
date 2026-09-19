"""Résultat d'analyse deepfake avec scores détaillés et métriques XAI."""
import enum
from datetime import datetime, timezone
from sqlalchemy import String, Float, DateTime, Enum as SAEnum, ForeignKey, JSON, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class AnalysisStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Verdict(str, enum.Enum):
    authentic = "AUTHENTIQUE"
    undetermined = "INDÉTERMINÉ"
    deepfake = "DEEPFAKE DÉTECTÉ"


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Liaisons
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), nullable=False, index=True)
    case: Mapped["Case"] = relationship("Case", back_populates="analyses")  # noqa: F821
    media_file_id: Mapped[int] = mapped_column(ForeignKey("media_files.id"), nullable=False, index=True)
    media_file: Mapped["MediaFile"] = relationship("MediaFile", back_populates="analyses")  # noqa: F821

    # Statut et tâche Celery
    status: Mapped[AnalysisStatus] = mapped_column(
        SAEnum(AnalysisStatus), default=AnalysisStatus.pending, nullable=False
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # Verdict final
    verdict: Mapped[Verdict | None] = mapped_column(SAEnum(Verdict), nullable=True)
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_high: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Scores par composante
    score_video_texture: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_video_temporal: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_rppg: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_biometrics: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_audio_model: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_audio_phase: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_metadata: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Métriques du modèle (pour le rapport Mohan)
    model_far: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_frr: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_eer: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_auc: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Versions des modèles utilisés (empreintes pour reproductibilité)
    models_used: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # XAI — données d'explicabilité
    xai_shap_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    xai_heatmap_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    xai_spectrogram_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    xai_suspicious_timecodes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    xai_plain_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Erreur éventuelle
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Durée d'analyse
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Dates
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    requested_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    def __repr__(self) -> str:
        return f"<Analysis id={self.id} verdict={self.verdict} score={self.final_score}>"
