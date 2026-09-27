"""Protocole d'entraînement — Brief v3 §0 (règles non négociables).

Cycle de vie d'un protocole :
  draft    → protocole en cours de rédaction, jeu de test non figé
  frozen   → hash SHA-256 calculé, jeu de test verrouillé, entraînement autorisé
  consumed → utilisé pour une version candidate (usage unique — Brief v3 §0.2)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Enum as SAEnum, ForeignKey, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column
import enum

from database import Base


class ProtocolStatus(str, enum.Enum):
    draft    = "draft"
    frozen   = "frozen"
    consumed = "consumed"


class TrainingProtocol(Base):
    __tablename__ = "training_protocols"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    engine_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Définition figée du jeu de test (Brief v3 §0.1)
    # Doit contenir : {"dataset": str, "version": str, "split_hash": str, "n_samples": int, "n_identities": int}
    test_set_definition: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Critères de validation (Brief v3 §6)
    # Ex. : {"far_max": 0.05, "frr_max": 0.10, "auc_min": 0.90, "eer_max": 0.08}
    metrics_targets: Mapped[dict] = mapped_column(JSON, nullable=False)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # SHA-256 de (engine_name + test_set_definition + metrics_targets) trié — calculé au freeze
    protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    status: Mapped[ProtocolStatus] = mapped_column(
        SAEnum(ProtocolStatus, name="protocolstatus"),
        nullable=False,
        default=ProtocolStatus.draft,
        index=True,
    )

    # Traçabilité
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    # FK vers model_versions (rempli au consume — Brief v3 §0.2 : usage unique)
    consumed_by_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    frozen_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_training_protocols_engine_status", "engine_name", "status"),
    )

    # ── Méthodes utilitaires ──────────────────────────────────────────────────

    def compute_hash(self) -> str:
        """Calcule le SHA-256 déterministe du protocole figé.

        La fonction est déterministe (sort_keys=True, séparateurs canoniques).
        Utilise SHA-256 comme référence probatoire (même convention que chain_of_custody).
        """
        payload = json.dumps(
            {
                "engine_name":        self.engine_name,
                "test_set_definition": self.test_set_definition,
                "metrics_targets":    self.metrics_targets,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """Vérifie que le hash stocké correspond à la définition actuelle."""
        if self.protocol_hash is None:
            return False
        return self.protocol_hash == self.compute_hash()

    def __repr__(self) -> str:
        return (
            f"<TrainingProtocol id={self.id} engine={self.engine_name} "
            f"status={self.status.value} hash={self.protocol_hash[:8] if self.protocol_hash else 'None'}>"
        )
