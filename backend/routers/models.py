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
