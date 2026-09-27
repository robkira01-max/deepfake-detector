"""Version trackée d'un modèle ML — Phase 3 Model Registry + Brief v3 §0/§6."""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from sqlalchemy import String, Boolean, DateTime, ForeignKey, JSON, Index, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class ValidationStatus(str, enum.Enum):
    """Statut de validation d'une composante ML (Brief v3 §6).

    experimental : poids non validés deepfake (ImageNet, etc.) — contribue si ALLOW_EXPERIMENTAL_ENGINES=True
    validated    : critères §6 satisfaits + approval humain explicite
    disabled     : exclu définitivement de toute fusion (ex : tête Wav2Vec2 aléatoire)
    """
    experimental = "experimental"
    validated    = "validated"
    disabled     = "disabled"


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)

    # HuggingFace Hub
    hf_repo_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    hf_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Stockage local
    local_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Métriques (FAR, FRR, AUC, EER) — doit correspondre à un metrics.json haché
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Brief v3 §0.5 — SHA-256 du fichier metrics.json associé à ces métriques
    # None signifie que les métriques n'ont pas encore été mesurées sur un jeu de test indépendant
    metrics_source_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metrics_source_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Brief v3 §0.6 / §6 — statut de validation (experimental par défaut)
    validation_status: Mapped[ValidationStatus] = mapped_column(
        SAEnum(ValidationStatus, name="validationstatus"),
        nullable=False,
        default=ValidationStatus.experimental,
        index=True,
    )

    # FK vers le protocole d'entraînement utilisé (Brief v3 §0.1)
    training_protocol_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_protocols.id"), nullable=True
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    registered_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    __table_args__ = (
        Index("ix_model_versions_name_active", "model_name", "is_active"),
        Index("ix_model_versions_name_status", "model_name", "validation_status"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelVersion id={self.id} name={self.model_name} v={self.version} "
            f"status={self.validation_status.value} active={self.is_active}>"
        )
