"""Tests — Phase 3 Model Registry."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from models.user import User
from models.model_version import ModelVersion


# ── Helpers ───────────────────────────────────────────────────────────────────

def _register_version(
    db: Session,
    user: User,
    model_name: str = "video_engine",
    version: str = "1.0.0",
    hf_repo_id: str | None = None,
    metrics: dict | None = None,
) -> ModelVersion:
    from core.model_registry import ModelRegistry
    return ModelRegistry.register(
        db,
        model_name=model_name,
        version=version,
        user_id=user.id,
        hf_repo_id=hf_repo_id,
        metrics=metrics or {"auc": 0.95, "eer": 0.05},
    )


# ── ModelRegistry unit tests ──────────────────────────────────────────────────

class TestModelRegistryUnit:

    def test_register_creates_version(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        mv = ModelRegistry.register(db, "audio_engine", "2.0.0", admin_user.id)
        assert mv.id is not None
        assert mv.model_name == "audio_engine"
        assert mv.version == "2.0.0"
        assert mv.is_active is False

    def test_list_versions_all(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        ModelRegistry.register(db, "model_a", "1.0", admin_user.id)
        ModelRegistry.register(db, "model_b", "1.0", admin_user.id)
        versions = ModelRegistry.list_versions(db)
        names = {v.model_name for v in versions}
        assert "model_a" in names
        assert "model_b" in names

    def test_list_versions_filtered(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        ModelRegistry.register(db, "model_x", "1.0", admin_user.id)
        ModelRegistry.register(db, "model_x", "2.0", admin_user.id)
        ModelRegistry.register(db, "model_y", "1.0", admin_user.id)
        versions = ModelRegistry.list_versions(db, model_name="model_x")
        assert len(versions) == 2
        assert all(v.model_name == "model_x" for v in versions)

    def test_get_active_returns_none_when_none_active(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        ModelRegistry.register(db, "inactive_model", "1.0", admin_user.id)
        result = ModelRegistry.get_active(db, "inactive_model")
        assert result is None

    def test_activate_sets_active_flag(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        mv = ModelRegistry.register(db, "text_engine", "1.0", admin_user.id)
        assert mv.is_active is False
        activated = ModelRegistry.activate(db, mv.id)
        assert activated.is_active is True

    def test_activate_deactivates_others_same_name(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        mv1 = ModelRegistry.register(db, "doc_engine", "1.0", admin_user.id)
        mv2 = ModelRegistry.register(db, "doc_engine", "2.0", admin_user.id)
        ModelRegistry.activate(db, mv1.id)
        ModelRegistry.activate(db, mv2.id)

        db.expire_all()
        v1 = db.query(ModelVersion).filter(ModelVersion.id == mv1.id).first()
        v2 = db.query(ModelVersion).filter(ModelVersion.id == mv2.id).first()
        assert v1.is_active is False
        assert v2.is_active is True

    def test_activate_does_not_affect_other_model_names(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        mv_a = ModelRegistry.register(db, "engine_a", "1.0", admin_user.id)
        mv_b = ModelRegistry.register(db, "engine_b", "1.0", admin_user.id)
        ModelRegistry.activate(db, mv_a.id)
        ModelRegistry.activate(db, mv_b.id)

        db.expire_all()
        a = db.query(ModelVersion).filter(ModelVersion.id == mv_a.id).first()
        b = db.query(ModelVersion).filter(ModelVersion.id == mv_b.id).first()
        assert a.is_active is True
        assert b.is_active is True

    def test_activate_not_found_raises(self, db: Session):
        from core.model_registry import ModelRegistry, ModelRegistryError
        with pytest.raises(ModelRegistryError):
            ModelRegistry.activate(db, 99999)

    def test_pull_no_hf_repo_sets_downloaded_at(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        mv = ModelRegistry.register(db, "local_model", "1.0", admin_user.id, hf_repo_id=None)
        pulled = ModelRegistry.pull(db, mv.id, admin_user.id)
        assert pulled.downloaded_at is not None
        assert pulled.is_active is True

    def test_pull_with_hf_repo_calls_download(self, db: Session, admin_user: User, tmp_path):
        from core.model_registry import ModelRegistry
        fake_path = str(tmp_path / "model.bin")
        (tmp_path / "model.bin").write_bytes(b"fake model weights")
        mv = ModelRegistry.register(
            db, "hf_model", "1.0", admin_user.id,
            hf_repo_id="org/hf_model",
        )
        with patch("core.model_registry.hf_hub_download", return_value=fake_path), \
             patch("core.model_registry.settings") as mock_settings:
            mock_settings.hf_cache_dir = str(tmp_path)
            mock_settings.hf_token = None
            pulled = ModelRegistry.pull(db, mv.id, admin_user.id)
        assert pulled.local_path == fake_path
        assert pulled.sha256 is not None
        assert pulled.is_active is True

    def test_pull_hf_failure_raises(self, db: Session, admin_user: User, tmp_path):
        from core.model_registry import ModelRegistry, ModelRegistryError
        mv = ModelRegistry.register(db, "fail_model", "1.0", admin_user.id, hf_repo_id="org/fail")
        with patch("core.model_registry.hf_hub_download", side_effect=Exception("404")), \
             patch("core.model_registry.settings") as mock_settings:
            mock_settings.hf_cache_dir = str(tmp_path)
            mock_settings.hf_token = None
            with pytest.raises(ModelRegistryError, match="Téléchargement HF échoué"):
                ModelRegistry.pull(db, mv.id, admin_user.id)

    def test_pull_not_found_raises(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry, ModelRegistryError
        with pytest.raises(ModelRegistryError):
            ModelRegistry.pull(db, 99999, admin_user.id)

    def test_get_registry_stats_empty(self, db: Session):
        from core.model_registry import ModelRegistry
        stats = ModelRegistry.get_registry_stats(db)
        assert stats["total_versions"] == 0
        assert stats["unique_models"] == 0
        assert stats["active_versions"] == 0
        assert stats["last_update"] is None

    def test_get_registry_stats_populated(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        ModelRegistry.register(db, "m1", "1.0", admin_user.id)
        mv = ModelRegistry.register(db, "m1", "2.0", admin_user.id)
        ModelRegistry.activate(db, mv.id)
        ModelRegistry.register(db, "m2", "1.0", admin_user.id)

        stats = ModelRegistry.get_registry_stats(db)
        assert stats["total_versions"] == 3
        assert stats["unique_models"] == 2
        assert stats["active_versions"] == 1
        assert "m1" in stats["model_names"]
        assert stats["last_update"] is not None


# ── HTTP API tests ─────────────────────────────────────────────────────────────

class TestModelsAPI:

    def test_list_all_requires_admin(self, client: TestClient, auth_analyst: dict):
        resp = client.get("/models", headers=auth_analyst)
        assert resp.status_code == 403

    def test_list_all_admin_ok(self, client: TestClient, auth_admin: dict):
        resp = client.get("/models", headers=auth_admin)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_stats_requires_analyst(self, client: TestClient):
        resp = client.get("/models/stats")
        assert resp.status_code == 401

    def test_stats_analyst_ok(self, client: TestClient, auth_analyst: dict):
        resp = client.get("/models/stats", headers=auth_analyst)
        assert resp.status_code == 200
        body = resp.json()
        assert "total_versions" in body
        assert "unique_models" in body

    def test_register_requires_admin(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/models/register", json={
            "model_name": "test", "version": "1.0",
        }, headers=auth_analyst)
        assert resp.status_code == 403

    def test_register_admin_creates_version(self, client: TestClient, auth_admin: dict):
        resp = client.post("/models/register", json={
            "model_name": "video_engine",
            "version": "3.0.0",
            "hf_repo_id": "org/video-engine",
            "metrics": {"auc": 0.97, "eer": 0.03},
        }, headers=auth_admin)
        assert resp.status_code == 201
        body = resp.json()
        assert body["model_name"] == "video_engine"
        assert body["version"] == "3.0.0"
        assert body["is_active"] is False

    def test_list_by_model_name(self, client: TestClient, auth_analyst: dict, auth_admin: dict):
        client.post("/models/register", json={
            "model_name": "audio_engine", "version": "1.0",
        }, headers=auth_admin)
        resp = client.get("/models/audio_engine", headers=auth_analyst)
        assert resp.status_code == 200
        body = resp.json()
        assert all(v["model_name"] == "audio_engine" for v in body)

    def test_pull_not_found(self, client: TestClient, auth_admin: dict):
        resp = client.post("/models/99999/pull", headers=auth_admin)
        assert resp.status_code == 404

    def test_pull_no_hf_repo(self, client: TestClient, auth_admin: dict):
        reg = client.post("/models/register", json={
            "model_name": "local_only", "version": "1.0",
        }, headers=auth_admin)
        vid = reg.json()["id"]
        resp = client.post(f"/models/{vid}/pull", headers=auth_admin)
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

    def test_activate_requires_admin(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/models/1/activate", headers=auth_analyst)
        assert resp.status_code == 403

    def test_activate_not_found(self, client: TestClient, auth_admin: dict):
        resp = client.post("/models/99999/activate", headers=auth_admin)
        assert resp.status_code == 404

    def test_activate_sets_active(self, client: TestClient, auth_admin: dict):
        reg = client.post("/models/register", json={
            "model_name": "activate_test", "version": "1.0",
        }, headers=auth_admin)
        vid = reg.json()["id"]
        resp = client.post(f"/models/{vid}/activate", headers=auth_admin)
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

    def test_unauthenticated_list_rejected(self, client: TestClient):
        resp = client.get("/models")
        assert resp.status_code == 401

    def test_register_exposes_validation_status(self, client: TestClient, auth_admin: dict):
        resp = client.post("/models/register", json={
            "model_name": "status_test", "version": "1.0",
        }, headers=auth_admin)
        assert resp.status_code == 201
        body = resp.json()
        assert body["validation_status"] == "experimental"
        assert body["training_protocol_id"] is None
        assert body["metrics_source_hash"] is None

    def test_validate_requires_admin(self, client: TestClient, auth_analyst: dict):
        resp = client.patch("/models/1/validate", headers=auth_analyst)
        assert resp.status_code == 403

    def test_validate_not_found(self, client: TestClient, auth_admin: dict):
        resp = client.patch("/models/99999/validate", headers=auth_admin)
        assert resp.status_code == 404

    def test_validate_without_metrics_hash_returns_422(self, client: TestClient, auth_admin: dict):
        reg = client.post("/models/register", json={
            "model_name": "no_metrics_model", "version": "1.0",
        }, headers=auth_admin)
        vid = reg.json()["id"]
        resp = client.patch(f"/models/{vid}/validate", headers=auth_admin)
        assert resp.status_code == 422
        assert "metrics_source_hash" in resp.json()["detail"]

    def test_disable_requires_admin(self, client: TestClient, auth_analyst: dict):
        resp = client.patch("/models/1/disable", headers=auth_analyst)
        assert resp.status_code == 403

    def test_disable_not_found(self, client: TestClient, auth_admin: dict):
        resp = client.patch("/models/99999/disable", headers=auth_admin)
        assert resp.status_code == 404

    def test_disable_sets_disabled_status(self, client: TestClient, auth_admin: dict):
        reg = client.post("/models/register", json={
            "model_name": "to_disable", "version": "1.0",
        }, headers=auth_admin)
        vid = reg.json()["id"]
        resp = client.patch(f"/models/{vid}/disable", headers=auth_admin)
        assert resp.status_code == 200
        assert resp.json()["validation_status"] == "disabled"

    def test_stats_includes_status_breakdown(self, client: TestClient, auth_analyst: dict):
        resp = client.get("/models/stats", headers=auth_analyst)
        assert resp.status_code == 200
        body = resp.json()
        assert "status_breakdown" in body
        assert "experimental" in body["status_breakdown"]
        assert "validated" in body["status_breakdown"]
        assert "disabled" in body["status_breakdown"]


# ── Unit tests — set_validation_status ────────────────────────────────────────

class TestSetValidationStatus:

    def test_default_status_is_experimental(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        mv = ModelRegistry.register(db, "engine_default", "1.0", admin_user.id)
        assert mv.validation_status.value == "experimental"

    def test_disable_from_experimental(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        from models.model_version import ValidationStatus
        mv = ModelRegistry.register(db, "engine_d1", "1.0", admin_user.id)
        result = ModelRegistry.set_validation_status(db, mv.id, ValidationStatus.disabled)
        assert result.validation_status.value == "disabled"

    def test_validate_without_hash_raises(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry, ModelRegistryError
        from models.model_version import ValidationStatus
        mv = ModelRegistry.register(db, "engine_v1", "1.0", admin_user.id)
        with pytest.raises(ModelRegistryError, match="metrics_source_hash"):
            ModelRegistry.set_validation_status(db, mv.id, ValidationStatus.validated)

    def test_validate_with_hash_succeeds(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        from models.model_version import ValidationStatus
        mv = ModelRegistry.register(db, "engine_v2", "1.0", admin_user.id)
        mv.metrics_source_hash = "a" * 64
        db.commit()
        result = ModelRegistry.set_validation_status(db, mv.id, ValidationStatus.validated)
        assert result.validation_status.value == "validated"

    def test_validate_then_disable(self, db: Session, admin_user: User):
        from core.model_registry import ModelRegistry
        from models.model_version import ValidationStatus
        mv = ModelRegistry.register(db, "engine_v3", "1.0", admin_user.id)
        mv.metrics_source_hash = "b" * 64
        db.commit()
        ModelRegistry.set_validation_status(db, mv.id, ValidationStatus.validated)
        result = ModelRegistry.set_validation_status(db, mv.id, ValidationStatus.disabled)
        assert result.validation_status.value == "disabled"

    def test_not_found_raises(self, db: Session):
        from core.model_registry import ModelRegistry, ModelRegistryError
        from models.model_version import ValidationStatus
        with pytest.raises(ModelRegistryError, match="introuvable"):
            ModelRegistry.set_validation_status(db, 99999, ValidationStatus.disabled)
