"""Registre de modèles ML — Phase 3."""
from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.model_registry import ModelRegistry, ModelRegistryError
from core.security import require_admin, require_analyst
from database import get_db
from models.audit_log import AuditAction, AuditLog
from models.model_version import ModelVersion, ValidationStatus
from models.user import User

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Model Registry"])


# ── Schémas ───────────────────────────────────────────────────────────────────

class ModelVersionOut(BaseModel):
    id: int
    model_name: str
    version: str
    hf_repo_id: str | None
    hf_revision: str | None
    local_path: str | None
    sha256: str | None
    metrics: dict | None
    is_active: bool
    downloaded_at: str | None
    created_at: str
    # Brief v3 §6 — cycle de vie de validation
    validation_status: str
    training_protocol_id: int | None
    metrics_source_hash: str | None

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm(cls, obj: ModelVersion) -> "ModelVersionOut":
        vs = obj.validation_status
        return cls(
            id=obj.id,
            model_name=obj.model_name,
            version=obj.version,
            hf_repo_id=obj.hf_repo_id,
            hf_revision=obj.hf_revision,
            local_path=obj.local_path,
            sha256=obj.sha256,
            metrics=obj.metrics,
            is_active=obj.is_active,
            downloaded_at=obj.downloaded_at.isoformat() if obj.downloaded_at else None,
            created_at=obj.created_at.isoformat(),
            validation_status=vs.value if hasattr(vs, "value") else str(vs),
            training_protocol_id=obj.training_protocol_id,
            metrics_source_hash=obj.metrics_source_hash,
        )


class RegisterRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=128)
    version: str = Field(..., min_length=1, max_length=64)
    hf_repo_id: str | None = Field(None, max_length=256)
    hf_revision: str | None = Field(None, max_length=128)
    metrics: dict | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ModelVersionOut])
def list_all_versions(
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_admin)],
) -> list[ModelVersionOut]:
    versions = ModelRegistry.list_versions(db)
    return [ModelVersionOut.from_orm(v) for v in versions]


@router.get("/stats")
def registry_stats(
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_analyst)],
) -> dict:
    return ModelRegistry.get_registry_stats(db)


@router.get("/{model_name}", response_model=list[ModelVersionOut])
def list_model_versions(
    model_name: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_analyst)],
) -> list[ModelVersionOut]:
    versions = ModelRegistry.list_versions(db, model_name=model_name)
    return [ModelVersionOut.from_orm(v) for v in versions]


@router.post("/register", response_model=ModelVersionOut, status_code=status.HTTP_201_CREATED)
def register_version(
    payload: RegisterRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_admin)],
) -> ModelVersionOut:
    mv = ModelRegistry.register(
        db,
        model_name=payload.model_name,
        version=payload.version,
        user_id=user.id,
        hf_repo_id=payload.hf_repo_id,
        hf_revision=payload.hf_revision,
        metrics=payload.metrics,
    )
    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.MODEL_REGISTERED,
        resource_type="ModelVersion",
        resource_id=str(mv.id),
        details={"model_name": mv.model_name, "version": mv.version},
    ))
    db.commit()
    return ModelVersionOut.from_orm(mv)


@router.post("/{version_id}/pull", response_model=ModelVersionOut)
def pull_version(
    version_id: int,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_admin)],
) -> ModelVersionOut:
    try:
        mv = ModelRegistry.pull(db, version_id=version_id, user_id=user.id)
    except ModelRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.MODEL_PULLED,
        resource_type="ModelVersion",
        resource_id=str(mv.id),
        details={"model_name": mv.model_name, "version": mv.version, "local_path": mv.local_path},
    ))
    db.commit()
    return ModelVersionOut.from_orm(mv)


@router.post("/{version_id}/activate", response_model=ModelVersionOut)
def activate_version(
    version_id: int,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_admin)],
) -> ModelVersionOut:
    try:
        mv = ModelRegistry.activate(db, version_id=version_id)
    except ModelRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.MODEL_ACTIVATED,
        resource_type="ModelVersion",
        resource_id=str(mv.id),
        details={"model_name": mv.model_name, "version": mv.version},
    ))
    db.commit()
    return ModelVersionOut.from_orm(mv)


@router.patch("/{version_id}/validate", response_model=ModelVersionOut)
def validate_version(
    version_id: int,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_admin)],
) -> ModelVersionOut:
    """Marque une ModelVersion comme 'validated'.

    Pré-condition : metrics_source_hash doit être présent (Brief v3 §0.5).
    Retourne HTTP 422 si la pré-condition n'est pas satisfaite.
    """
    try:
        mv = ModelRegistry.set_validation_status(
            db, version_id=version_id, new_status=ValidationStatus.validated
        )
    except ModelRegistryError as exc:
        code = 404 if "introuvable" in str(exc) else status.HTTP_422_UNPROCESSABLE_ENTITY
        raise HTTPException(status_code=code, detail=str(exc)) from exc

    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.ENGINE_STATUS_CHANGED,
        resource_type="ModelVersion",
        resource_id=str(mv.id),
        details={"model_name": mv.model_name, "version": mv.version, "new_status": "validated"},
    ))
    db.commit()
    return ModelVersionOut.from_orm(mv)


# ── Validation des moteurs (Règle 10) ─────────────────────────────────────────

class EngineApprovalRequest(BaseModel):
    review_notes: str = Field(..., min_length=10, max_length=2000)


@router.get(
    "/engines/{engine_name}/validation-status",
    summary="État des 4 conditions de validation d'un moteur (Règle 10)",
)
def engine_validation_status(
    engine_name: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_analyst)],
) -> dict:
    from engines.validation_gate import EngineValidationGate, VALID_ENGINES  # noqa: PLC0415
    if engine_name not in VALID_ENGINES:
        raise HTTPException(
            status_code=422,
            detail=f"Moteur inconnu '{engine_name}'. Valeurs : {sorted(VALID_ENGINES)}",
        )
    gate = EngineValidationGate.check(engine_name, db)
    return gate.to_dict()


@router.post(
    "/engines/{engine_name}/approve",
    status_code=status.HTTP_201_CREATED,
    summary="Journalise l'approbation humaine (condition 4 Règle 10) — ne promeut pas encore",
)
def approve_engine(
    engine_name: str,
    payload: EngineApprovalRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_admin)],
) -> dict:
    from engines.validation_gate import VALID_ENGINES  # noqa: PLC0415
    if engine_name not in VALID_ENGINES:
        raise HTTPException(
            status_code=422,
            detail=f"Moteur inconnu '{engine_name}'. Valeurs : {sorted(VALID_ENGINES)}",
        )
    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.ENGINE_STATUS_CHANGED,
        resource_type="Engine",
        resource_id=engine_name,
        details={
            "engine_name": engine_name,
            "action": "approved",
            "approved_by": user.username,
            "review_notes": payload.review_notes,
        },
    ))
    db.commit()
    log.info("engine_approved", engine=engine_name, by=user.username)
    return {
        "engine_name": engine_name,
        "action": "approved",
        "by": user.username,
        "message": (
            "Approbation journalisée. Appeler POST /models/engines/{name}/promote "
            "quand les 3 autres conditions sont satisfaites."
        ),
    }


@router.post(
    "/engines/{engine_name}/promote",
    summary="Promeut un moteur de 'experimental' à 'validated' (Règle 10 — 4 conditions requises)",
)
def promote_engine(
    engine_name: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_admin)],
) -> dict:
    from engines.validation_gate import EngineValidationGate, VALID_ENGINES  # noqa: PLC0415
    if engine_name not in VALID_ENGINES:
        raise HTTPException(
            status_code=422,
            detail=f"Moteur inconnu '{engine_name}'. Valeurs : {sorted(VALID_ENGINES)}",
        )
    gate = EngineValidationGate.check(engine_name, db)
    if not gate.can_promote:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"Le moteur '{engine_name}' ne satisfait pas les 4 conditions (Règle 10).",
                "blocking_reasons": gate.blocking_reasons,
                "conditions": {
                    "metrics_ok": gate.metrics_ok,
                    "model_card_ok": gate.model_card_ok,
                    "protocol_ok": gate.protocol_ok,
                    "human_approval_ok": gate.human_approval_ok,
                },
            },
        )

    _write_engine_status(engine_name, "validated")

    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.ENGINE_STATUS_CHANGED,
        resource_type="Engine",
        resource_id=engine_name,
        details={
            "engine_name": engine_name,
            "new_status": "validated",
            "action": "promoted",
            "promoted_by": user.username,
        },
    ))
    db.commit()
    log.info("engine_promoted", engine=engine_name, by=user.username)
    return {
        "engine_name": engine_name,
        "new_status": "validated",
        "promoted_by": user.username,
        "message": (
            f"Le moteur '{engine_name}' est maintenant 'validated'. "
            "Les prochaines analyses l'incluront dans le score de fusion."
        ),
    }


def _write_engine_status(engine_name: str, new_status: str) -> None:
    """Met à jour evaluation/engine_statuses.json avec le nouveau statut."""
    import json as _json  # noqa: PLC0415
    from engines.validation_gate import _EVAL_ROOT  # noqa: PLC0415
    statuses_path = _EVAL_ROOT / "engine_statuses.json"
    try:
        data = _json.loads(statuses_path.read_text(encoding="utf-8"))
    except Exception:
        data = {"schema_version": "1.0", "statuses": {}}
    data.setdefault("statuses", {})[engine_name] = new_status
    statuses_path.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@router.patch("/{version_id}/disable", response_model=ModelVersionOut)
def disable_version(
    version_id: int,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_admin)],
) -> ModelVersionOut:
    """Désactive une ModelVersion — exclue de la fusion jusqu'à nouvelle validation."""
    try:
        mv = ModelRegistry.set_validation_status(
            db, version_id=version_id, new_status=ValidationStatus.disabled
        )
    except ModelRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.ENGINE_STATUS_CHANGED,
        resource_type="ModelVersion",
        resource_id=str(mv.id),
        details={"model_name": mv.model_name, "version": mv.version, "new_status": "disabled"},
    ))
    db.commit()
    return ModelVersionOut.from_orm(mv)
