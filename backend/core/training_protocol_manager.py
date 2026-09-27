"""Gestionnaire de protocoles d'entraînement — Brief v3 §0 (règles non négociables).

Responsabilités :
  - Créer un protocole (statut draft)
  - Figer un protocole (draft → frozen, calcul SHA-256)
  - Consommer un protocole (frozen → consumed, usage unique §0.2)
  - Vérifier l'intégrité du hash d'un protocole figé
"""
from __future__ import annotations

from datetime import datetime, timezone

import structlog
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from models.audit_log import AuditAction, AuditLog
from models.training_protocol import TrainingProtocol, ProtocolStatus

log = structlog.get_logger(__name__)


class ProtocolError(Exception):
    """Levée sur violation des règles de protocole (Brief v3 §0)."""


# ── Création ──────────────────────────────────────────────────────────────────

def create_protocol(
    *,
    db: Session,
    name: str,
    engine_name: str,
    test_set_definition: dict,
    metrics_targets: dict,
    created_by_id: int,
    notes: str | None = None,
) -> TrainingProtocol:
    """Crée un protocole en statut draft."""
    proto = TrainingProtocol(
        name=name,
        engine_name=engine_name,
        test_set_definition=test_set_definition,
        metrics_targets=metrics_targets,
        notes=notes,
        created_by_id=created_by_id,
        status=ProtocolStatus.draft,
    )
    db.add(proto)
    db.flush()

    db.add(AuditLog(
        user_id=created_by_id,
        action=AuditAction.PROTOCOL_CREATED,
        resource_type="training_protocol",
        resource_id=str(proto.id),
        details={"engine_name": engine_name, "name": name},
    ))

    log.info("protocol_created", protocol_id=proto.id, engine=engine_name)
    return proto


# ── Gel ───────────────────────────────────────────────────────────────────────

def freeze_protocol(
    *,
    db: Session,
    protocol_id: int,
    frozen_by_id: int,
) -> TrainingProtocol:
    """Fige le protocole : calcule le SHA-256 et passe draft → frozen.

    Une fois figé, test_set_definition et metrics_targets ne doivent plus être modifiés.
    Le hash est calculé à partir de la définition au moment du freeze.

    Raises:
        ProtocolError: Si le protocole n'est pas en statut draft.
    """
    proto = _get_or_404(db, protocol_id)

    if proto.status != ProtocolStatus.draft:
        raise ProtocolError(
            f"Le protocole {protocol_id} est en statut '{proto.status.value}', "
            f"seul un protocole 'draft' peut être figé."
        )

    proto.protocol_hash = proto.compute_hash()
    proto.status = ProtocolStatus.frozen
    proto.frozen_at = datetime.now(timezone.utc)
    db.flush()

    db.add(AuditLog(
        user_id=frozen_by_id,
        action=AuditAction.PROTOCOL_FROZEN,
        resource_type="training_protocol",
        resource_id=str(proto.id),
        details={
            "engine_name":    proto.engine_name,
            "protocol_hash":  proto.protocol_hash,
            "frozen_at":      proto.frozen_at.isoformat(),
        },
    ))

    log.info("protocol_frozen", protocol_id=proto.id,
             engine=proto.engine_name, hash=proto.protocol_hash[:12])
    return proto


# ── Consommation (usage unique) ───────────────────────────────────────────────

def consume_protocol(
    *,
    db: Session,
    protocol_id: int,
    model_version_id: int,
    consumed_by_id: int,
) -> TrainingProtocol:
    """Marque le protocole comme consommé (frozen → consumed).

    Brief v3 §0.2 : le jeu de test n'est utilisé qu'une seule fois par version candidate.
    Un protocole consumed ne peut pas être réutilisé.

    Raises:
        ProtocolError: Si le protocole n'est pas frozen, ou déjà consumed.
    """
    proto = _get_or_404(db, protocol_id)

    if proto.status == ProtocolStatus.consumed:
        raise ProtocolError(
            f"Le protocole {protocol_id} est déjà consumed "
            f"(utilisé par la version modèle {proto.consumed_by_model_version_id}). "
            f"Brief v3 §0.2 : usage unique — créer un nouveau protocole."
        )

    if proto.status != ProtocolStatus.frozen:
        raise ProtocolError(
            f"Le protocole {protocol_id} est en statut '{proto.status.value}'. "
            f"Seul un protocole 'frozen' peut être consommé."
        )

    proto.status = ProtocolStatus.consumed
    proto.consumed_at = datetime.now(timezone.utc)
    proto.consumed_by_model_version_id = model_version_id
    db.flush()

    db.add(AuditLog(
        user_id=consumed_by_id,
        action=AuditAction.PROTOCOL_CONSUMED,
        resource_type="training_protocol",
        resource_id=str(proto.id),
        details={
            "engine_name":          proto.engine_name,
            "protocol_hash":        proto.protocol_hash,
            "model_version_id":     model_version_id,
            "consumed_at":          proto.consumed_at.isoformat(),
        },
    ))

    log.info("protocol_consumed", protocol_id=proto.id,
             engine=proto.engine_name, model_version_id=model_version_id)
    return proto


# ── Vérification d'intégrité ──────────────────────────────────────────────────

def verify_protocol_integrity(
    *,
    db: Session,
    protocol_id: int,
) -> tuple[bool, str]:
    """Recompute le hash et vérifie qu'il correspond au hash stocké.

    Returns:
        (ok: bool, detail: str)
        ok=True si le hash est intact, False si le protocole a été altéré.

    Un protocole dont le hash ne correspond pas doit être considéré comme compromis.
    """
    proto = _get_or_404(db, protocol_id)

    if proto.protocol_hash is None:
        return False, f"Protocole {protocol_id} non figé (hash absent)"

    computed = proto.compute_hash()
    if computed == proto.protocol_hash:
        return True, f"Intégrité vérifiée : {proto.protocol_hash[:16]}…"

    log.warning(
        "protocol_hash_mismatch",
        protocol_id=protocol_id,
        stored=proto.protocol_hash[:16],
        computed=computed[:16],
    )
    return (
        False,
        f"ALTÉRATION DÉTECTÉE — hash stocké={proto.protocol_hash[:16]}… "
        f"vs calculé={computed[:16]}…",
    )


# ── Guard PDF (Brief v3 §0.5) ─────────────────────────────────────────────────

def assert_metrics_have_source(model_version) -> None:  # type: ignore[type-arg]
    """Lève une erreur si des métriques sont présentes sans metrics_source_hash.

    À appeler dans pdf_generator.py avant d'afficher FAR/FRR/AUC dans un rapport.

    Brief v3 §0.5 : aucun chiffre dans un rapport sans metrics.json correspondant.
    """
    if model_version is None:
        return

    has_metrics = any(
        getattr(model_version, f, None) is not None
        for f in ("model_far", "model_frr", "model_eer", "model_auc")
    )
    if has_metrics and not getattr(model_version, "metrics_source_hash", None):
        raise ValueError(
            "Brief v3 §0.5 — Impossible d'afficher des métriques dans le rapport : "
            "aucun metrics_source_hash n'est associé à cette version de modèle. "
            "Fournissez un metrics.json validé et stockez son SHA-256 dans "
            "ModelVersion.metrics_source_hash avant de générer ce rapport."
        )


# ── Utilitaire interne ────────────────────────────────────────────────────────

def _get_or_404(db: Session, protocol_id: int) -> TrainingProtocol:
    proto = db.get(TrainingProtocol, protocol_id)
    if proto is None:
        raise ProtocolError(f"Protocole introuvable : id={protocol_id}")
    return proto
