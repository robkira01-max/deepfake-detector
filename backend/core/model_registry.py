"""Registre de modèles ML avec intégration HuggingFace Hub."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import structlog
from sqlalchemy.orm import Session

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]

from config import settings
from models.model_version import ModelVersion, ValidationStatus

log = structlog.get_logger(__name__)


class ModelRegistryError(Exception):
    """Levée sur erreur de registre de modèles."""


class ModelRegistry:
    """Interface CRUD pour les versions de modèles ML."""

    # ── Lecture ───────────────────────────────────────────────────────────────

    @staticmethod
    def list_versions(db: Session, model_name: str | None = None) -> list[ModelVersion]:
        q = db.query(ModelVersion)
        if model_name:
            q = q.filter(ModelVersion.model_name == model_name)
        return q.order_by(ModelVersion.created_at.desc()).all()

    @staticmethod
    def get_active(db: Session, model_name: str) -> ModelVersion | None:
        return (
            db.query(ModelVersion)
            .filter(ModelVersion.model_name == model_name, ModelVersion.is_active.is_(True))
            .first()
        )

    @staticmethod
    def get_by_id(db: Session, version_id: int) -> ModelVersion | None:
        return db.query(ModelVersion).filter(ModelVersion.id == version_id).first()

    @staticmethod
    def get_registry_stats(db: Session) -> dict:
        all_versions = db.query(ModelVersion).all()
        names = {v.model_name for v in all_versions}
        active = [v for v in all_versions if v.is_active]
        last_update = max((v.created_at for v in all_versions), default=None)
        status_counts: dict[str, int] = {"validated": 0, "experimental": 0, "disabled": 0}
        for v in all_versions:
            s = v.validation_status.value if hasattr(v.validation_status, "value") else str(v.validation_status)
            if s in status_counts:
                status_counts[s] += 1
        return {
            "total_versions": len(all_versions),
            "unique_models": len(names),
            "active_versions": len(active),
            "model_names": sorted(names),
            "last_update": last_update.isoformat() if last_update else None,
            "status_breakdown": status_counts,
        }

    # ── Écriture ──────────────────────────────────────────────────────────────

    @staticmethod
    def register(
        db: Session,
        model_name: str,
        version: str,
        user_id: int,
        hf_repo_id: str | None = None,
        hf_revision: str | None = None,
        metrics: dict | None = None,
    ) -> ModelVersion:
        mv = ModelVersion(
            model_name=model_name,
            version=version,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
            metrics=metrics or {},
            is_active=False,
            registered_by_id=user_id,
        )
        db.add(mv)
        db.commit()
        db.refresh(mv)
        log.info("model_registered", model=model_name, version=version, id=mv.id)
        return mv

    @staticmethod
    def pull(db: Session, version_id: int, user_id: int) -> ModelVersion:
        mv = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not mv:
            raise ModelRegistryError(f"ModelVersion {version_id} introuvable")

        if mv.hf_repo_id:
            try:
                cache_dir = Path(settings.hf_cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                if hf_hub_download is None:
                    raise ModelRegistryError("huggingface_hub non installé")

                local_path = hf_hub_download(
                    repo_id=mv.hf_repo_id,
                    filename=f"{mv.model_name}.bin",
                    revision=mv.hf_revision or "main",
                    cache_dir=str(cache_dir),
                    token=settings.hf_token or None,
                )
                mv.local_path = local_path
                mv.sha256 = _sha256(local_path)
                log.info("model_downloaded", model=mv.model_name, path=local_path)
            except Exception as exc:
                log.error("model_download_failed", model=mv.model_name, error=str(exc))
                raise ModelRegistryError(f"Téléchargement HF échoué : {exc}") from exc
        else:
            log.info("model_no_hf_repo", model=mv.model_name, version=mv.version)

        mv.downloaded_at = datetime.now(timezone.utc)
        _set_active(db, mv)
        db.commit()
        db.refresh(mv)
        return mv

    @staticmethod
    def set_validation_status(
        db: Session,
        version_id: int,
        new_status: ValidationStatus,
    ) -> ModelVersion:
        """Change le validation_status d'une ModelVersion.

        Règle Brief v3 §0.5 : le passage à 'validated' exige metrics_source_hash.
        Le passage à 'disabled' est toujours autorisé (y compris depuis 'validated').
        """
        mv = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not mv:
            raise ModelRegistryError(f"ModelVersion {version_id} introuvable")

        if new_status == ValidationStatus.validated and not mv.metrics_source_hash:
            raise ModelRegistryError(
                "Impossible de marquer 'validated' : metrics_source_hash absent. "
                "Fournissez un metrics.json validé (Brief v3 §0.5)."
            )

        old_status = mv.validation_status
        mv.validation_status = new_status
        db.commit()
        db.refresh(mv)
        log.info(
            "model_validation_status_changed",
            model=mv.model_name,
            version=mv.version,
            old=old_status,
            new=new_status,
            id=mv.id,
        )
        return mv

    @staticmethod
    def activate(db: Session, version_id: int) -> ModelVersion:
        mv = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not mv:
            raise ModelRegistryError(f"ModelVersion {version_id} introuvable")
        _set_active(db, mv)
        db.commit()
        db.refresh(mv)
        log.info("model_activated", model=mv.model_name, version=mv.version, id=mv.id)
        return mv


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_active(db: Session, target: ModelVersion) -> None:
    """Désactive toutes les versions du même model_name, active target."""
    db.query(ModelVersion).filter(
        ModelVersion.model_name == target.model_name,
        ModelVersion.id != target.id,
    ).update({"is_active": False})
    target.is_active = True


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
