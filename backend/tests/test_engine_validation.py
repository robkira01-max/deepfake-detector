"""Tests — EngineValidationGate + promote/approve endpoints (P3 Règle 10)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_valid_metrics(engine_name: str = "metadata") -> dict:
    return {
        "schema_version": "1.0",
        "engine_name": engine_name,
        "model_version": "v0.1.0",
        "protocol_hash": "abc" * 20,
        "train_dataset": {"name": "FaceForensics++", "version": "3", "split": "train", "n_samples": 2000, "n_identities": 100},
        "test_dataset": {"name": "DFDC", "version": "1", "split": "test", "n_samples": 1000, "n_identities": 50},
        "far": 0.05,
        "frr": 0.08,
        "eer": 0.065,
        "auc": 0.94,
        "threshold_used": 0.50,
        "evaluated_at": "2026-10-04T00:00:00Z",
        "evaluator": "analyst@test.ca",
    }


def _make_completed_protocol() -> str:
    return textwrap.dedent("""\
        schema_version: "1.0"
        protocol_id: "baseline_protocol"
        status: "completed"
        dataset_requirements:
          - jeu_test_independent: true
        metrics_required:
          - auc
        approval_required:
          - human_review: true
    """)


class TestValidationGate:
    """Tests unitaires de EngineValidationGate — toutes conditions mockées."""

    def _gate_with_all_ok(self, tmp_path):
        """Crée l'environnement complet pour can_promote=True."""
        from engines.validation_gate import EngineValidationGate, _EVAL_ROOT

        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(_make_valid_metrics("metadata")))
        model_card = tmp_path / "model_cards" / "metadata.md"
        model_card.parent.mkdir()
        model_card.write_text("# Metadata Engine\n\n" + "x" * 300)
        protocol = tmp_path / "baseline_protocol.yaml"
        protocol.write_text(_make_completed_protocol())

        mock_db = MagicMock()
        from models.audit_log import AuditAction, AuditLog
        mock_entry = MagicMock(spec=AuditLog)
        mock_entry.details = {"engine_name": "metadata", "action": "approved"}
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_entry]

        with patch("engines.validation_gate._METRICS_PATH", metrics_path), \
             patch("engines.validation_gate._MODEL_CARDS_DIR", tmp_path / "model_cards"), \
             patch("engines.validation_gate._PROTOCOL_PATH", protocol):
            return EngineValidationGate.check("metadata", mock_db)

    def test_all_conditions_met_can_promote(self, tmp_path):
        result = self._gate_with_all_ok(tmp_path)
        assert result.can_promote is True
        assert result.blocking_reasons == []

    def test_missing_metrics_blocks(self, tmp_path):
        from engines.validation_gate import EngineValidationGate
        with patch("engines.validation_gate._METRICS_PATH", tmp_path / "nonexistent.json"):
            result = EngineValidationGate.check("metadata", MagicMock())
        assert result.metrics_ok is False
        assert result.can_promote is False
        assert any("absent" in r and "metrics" in r for r in result.blocking_reasons)

    def test_wrong_engine_in_metrics_blocks(self, tmp_path):
        from engines.validation_gate import EngineValidationGate
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(_make_valid_metrics("texture")))
        with patch("engines.validation_gate._METRICS_PATH", metrics_path):
            result = EngineValidationGate.check("metadata", MagicMock())
        assert result.metrics_ok is False
        assert any("texture" in r for r in result.blocking_reasons)

    def test_missing_required_field_in_metrics_blocks(self, tmp_path):
        from engines.validation_gate import EngineValidationGate
        data = _make_valid_metrics("metadata")
        del data["auc"]
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(data))
        with patch("engines.validation_gate._METRICS_PATH", metrics_path):
            result = EngineValidationGate.check("metadata", MagicMock())
        assert result.metrics_ok is False

    def test_missing_model_card_blocks(self, tmp_path):
        from engines.validation_gate import EngineValidationGate
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(_make_valid_metrics("metadata")))
        empty_dir = tmp_path / "no_cards"
        empty_dir.mkdir()
        with patch("engines.validation_gate._METRICS_PATH", metrics_path), \
             patch("engines.validation_gate._MODEL_CARDS_DIR", empty_dir):
            result = EngineValidationGate.check("metadata", MagicMock())
        assert result.model_card_ok is False
        assert any("Model card absente" in r for r in result.blocking_reasons)

    def test_too_short_model_card_blocks(self, tmp_path):
        from engines.validation_gate import EngineValidationGate
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(_make_valid_metrics("metadata")))
        cards_dir = tmp_path / "model_cards"
        cards_dir.mkdir()
        (cards_dir / "metadata.md").write_text("# Short")
        with patch("engines.validation_gate._METRICS_PATH", metrics_path), \
             patch("engines.validation_gate._MODEL_CARDS_DIR", cards_dir):
            result = EngineValidationGate.check("metadata", MagicMock())
        assert result.model_card_ok is False
        assert any("trop courte" in r for r in result.blocking_reasons)

    def test_protocol_draft_blocks(self, tmp_path):
        from engines.validation_gate import EngineValidationGate
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(_make_valid_metrics("metadata")))
        cards_dir = tmp_path / "model_cards"
        cards_dir.mkdir()
        (cards_dir / "metadata.md").write_text("# Title\n\n" + "x" * 300)
        proto = tmp_path / "protocol.yaml"
        proto.write_text("status: 'draft'\n")
        with patch("engines.validation_gate._METRICS_PATH", metrics_path), \
             patch("engines.validation_gate._MODEL_CARDS_DIR", cards_dir), \
             patch("engines.validation_gate._PROTOCOL_PATH", proto):
            result = EngineValidationGate.check("metadata", MagicMock())
        assert result.protocol_ok is False
        assert any("draft" in r for r in result.blocking_reasons)

    def test_no_audit_approval_blocks(self, tmp_path):
        from engines.validation_gate import EngineValidationGate
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(_make_valid_metrics("metadata")))
        cards_dir = tmp_path / "model_cards"
        cards_dir.mkdir()
        (cards_dir / "metadata.md").write_text("# Title\n\n" + "x" * 300)
        proto = tmp_path / "protocol.yaml"
        proto.write_text(_make_completed_protocol())
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        with patch("engines.validation_gate._METRICS_PATH", metrics_path), \
             patch("engines.validation_gate._MODEL_CARDS_DIR", cards_dir), \
             patch("engines.validation_gate._PROTOCOL_PATH", proto):
            result = EngineValidationGate.check("metadata", mock_db)
        assert result.human_approval_ok is False
        assert any("approbation" in r for r in result.blocking_reasons)

    def test_no_db_blocks_human_approval(self, tmp_path):
        from engines.validation_gate import EngineValidationGate
        result = EngineValidationGate.check("metadata", db=None)
        assert result.human_approval_ok is False

    def test_to_dict_structure(self, tmp_path):
        from engines.validation_gate import EngineValidationGate
        with patch("engines.validation_gate._METRICS_PATH", tmp_path / "x"):
            result = EngineValidationGate.check("metadata", None)
        d = result.to_dict()
        assert "engine_name" in d
        assert "can_promote" in d
        assert "conditions" in d
        assert "blocking_reasons" in d

    def test_valid_engines_constant(self):
        from engines.validation_gate import VALID_ENGINES
        assert "metadata" in VALID_ENGINES
        assert "audio" in VALID_ENGINES
        assert "fake_engine" not in VALID_ENGINES


class TestFusionLoadEngineStatuses:
    """Tests de load_engine_statuses() — persistence dans engine_statuses.json."""

    def test_defaults_when_file_absent(self, tmp_path):
        from engines.fusion import load_engine_statuses, ENGINE_DEFAULT_STATUS
        with patch("engines.fusion._ENGINE_STATUSES_PATH", tmp_path / "nonexistent.json"):
            result = load_engine_statuses()
        assert result == ENGINE_DEFAULT_STATUS

    def test_loads_override_from_file(self, tmp_path):
        from engines.fusion import load_engine_statuses
        statuses_file = tmp_path / "engine_statuses.json"
        statuses_file.write_text(json.dumps({
            "schema_version": "1.0",
            "statuses": {"metadata": "validated"}
        }))
        with patch("engines.fusion._ENGINE_STATUSES_PATH", statuses_file):
            result = load_engine_statuses()
        assert result["metadata"] == "validated"
        assert result["audio"] == "disabled"  # preserved from defaults

    def test_ignores_unknown_engines(self, tmp_path):
        from engines.fusion import load_engine_statuses, ENGINE_DEFAULT_STATUS
        statuses_file = tmp_path / "engine_statuses.json"
        statuses_file.write_text(json.dumps({
            "statuses": {"unknown_engine": "validated", "metadata": "experimental"}
        }))
        with patch("engines.fusion._ENGINE_STATUSES_PATH", statuses_file):
            result = load_engine_statuses()
        assert "unknown_engine" not in result
        assert set(result.keys()) == set(ENGINE_DEFAULT_STATUS.keys())

    def test_ignores_invalid_status_values(self, tmp_path):
        from engines.fusion import load_engine_statuses, ENGINE_DEFAULT_STATUS
        statuses_file = tmp_path / "engine_statuses.json"
        statuses_file.write_text(json.dumps({
            "statuses": {"metadata": "invalid_status"}
        }))
        with patch("engines.fusion._ENGINE_STATUSES_PATH", statuses_file):
            result = load_engine_statuses()
        assert result["metadata"] == ENGINE_DEFAULT_STATUS["metadata"]

    def test_fallback_on_corrupt_file(self, tmp_path):
        from engines.fusion import load_engine_statuses, ENGINE_DEFAULT_STATUS
        statuses_file = tmp_path / "engine_statuses.json"
        statuses_file.write_text("NOT VALID JSON {{{")
        with patch("engines.fusion._ENGINE_STATUSES_PATH", statuses_file):
            result = load_engine_statuses()
        assert result == ENGINE_DEFAULT_STATUS


class TestEngineEndpoints:
    """Tests des routes /models/engines/{name}/."""

    def test_validation_status_unknown_engine(self, client, auth_analyst):
        r = client.get("/models/engines/fake_engine/validation-status", headers=auth_analyst)
        assert r.status_code == 422

    def test_validation_status_requires_auth(self, client):
        r = client.get("/models/engines/metadata/validation-status")
        assert r.status_code == 401

    def test_validation_status_returns_structure(self, client, auth_analyst):
        with patch("engines.validation_gate._METRICS_PATH", Path("/tmp/nonexistent_metrics.json")):
            r = client.get("/models/engines/metadata/validation-status", headers=auth_analyst)
        assert r.status_code == 200
        data = r.json()
        assert "can_promote" in data
        assert "conditions" in data
        assert "blocking_reasons" in data
        assert data["can_promote"] is False

    def test_approve_requires_admin(self, client, auth_analyst):
        r = client.post(
            "/models/engines/metadata/approve",
            json={"review_notes": "Reviewed and approved by expert."},
            headers=auth_analyst,
        )
        assert r.status_code == 403

    def test_approve_unknown_engine(self, client, auth_admin):
        r = client.post(
            "/models/engines/bad_engine/approve",
            json={"review_notes": "Test review notes here"},
            headers=auth_admin,
        )
        assert r.status_code == 422

    def test_approve_logs_audit(self, client, auth_admin):
        r = client.post(
            "/models/engines/metadata/approve",
            json={"review_notes": "Reviewed metadata heuristics, no ML model, low risk."},
            headers=auth_admin,
        )
        assert r.status_code == 201
        data = r.json()
        assert data["action"] == "approved"
        assert data["engine_name"] == "metadata"

    def test_promote_requires_admin(self, client, auth_analyst):
        r = client.post("/models/engines/metadata/promote", headers=auth_analyst)
        assert r.status_code == 403

    def test_promote_unknown_engine(self, client, auth_admin):
        r = client.post("/models/engines/bad_engine/promote", headers=auth_admin)
        assert r.status_code == 422

    def test_promote_blocked_when_gate_fails(self, client, auth_admin):
        with patch("engines.validation_gate._METRICS_PATH", Path("/tmp/nonexistent_metrics.json")):
            r = client.post("/models/engines/metadata/promote", headers=auth_admin)
        assert r.status_code == 409
        data = r.json()
        assert "blocking_reasons" in data["detail"]
        assert data["detail"]["conditions"]["metrics_ok"] is False
