"""Tests — TrainingProtocol + ProtocolManager (Brief v3 §0)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from models.training_protocol import TrainingProtocol, ProtocolStatus
from core.training_protocol_manager import (
    ProtocolError,
    create_protocol,
    freeze_protocol,
    consume_protocol,
    verify_protocol_integrity,
    assert_metrics_have_source,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

_TEST_SET_DEF = {
    "dataset": "FaceForensics++",
    "version": "1.0",
    "split": "test",
    "split_hash": "abc123",
    "n_samples": 1000,
    "n_identities": 200,
}

_METRICS_TARGETS = {
    "far_max": 0.05,
    "frr_max": 0.10,
    "auc_min": 0.90,
    "eer_max": 0.08,
}


def _make_db():
    """Mini-mock de session SQLAlchemy pour tests sans base réelle."""
    db = MagicMock()
    db.get.return_value = None  # 404 par défaut

    protocols: list[TrainingProtocol] = []

    def _add(obj):
        if isinstance(obj, TrainingProtocol):
            obj.id = len(protocols) + 1
            protocols.append(obj)

    db.add.side_effect = _add
    db.flush.return_value = None
    return db, protocols


# ── Tests TrainingProtocol (modèle) ───────────────────────────────────────────

class TestTrainingProtocolModel:

    def test_compute_hash_deterministic(self):
        proto = TrainingProtocol(
            engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS,
        )
        h1 = proto.compute_hash()
        h2 = proto.compute_hash()
        assert h1 == h2

    def test_compute_hash_is_sha256(self):
        proto = TrainingProtocol(
            engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS,
        )
        h = proto.compute_hash()
        assert len(h) == 64  # SHA-256 en hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_hash_different_engines(self):
        """Des engines différents produisent des hashes différents."""
        p1 = TrainingProtocol(engine_name="texture",
                              test_set_definition=_TEST_SET_DEF,
                              metrics_targets=_METRICS_TARGETS)
        p2 = TrainingProtocol(engine_name="temporal",
                              test_set_definition=_TEST_SET_DEF,
                              metrics_targets=_METRICS_TARGETS)
        assert p1.compute_hash() != p2.compute_hash()

    def test_compute_hash_sensitive_to_test_set(self):
        """Modifier le test_set_definition change le hash."""
        p1 = TrainingProtocol(engine_name="texture",
                              test_set_definition=_TEST_SET_DEF,
                              metrics_targets=_METRICS_TARGETS)
        modified = {**_TEST_SET_DEF, "n_samples": 999}
        p2 = TrainingProtocol(engine_name="texture",
                              test_set_definition=modified,
                              metrics_targets=_METRICS_TARGETS)
        assert p1.compute_hash() != p2.compute_hash()

    def test_verify_hash_true_when_intact(self):
        proto = TrainingProtocol(
            engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS,
        )
        proto.protocol_hash = proto.compute_hash()
        assert proto.verify_hash() is True

    def test_verify_hash_false_when_none(self):
        proto = TrainingProtocol(
            engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS,
        )
        proto.protocol_hash = None
        assert proto.verify_hash() is False

    def test_verify_hash_false_when_tampered(self):
        proto = TrainingProtocol(
            engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS,
        )
        proto.protocol_hash = "0" * 64  # hash invalide
        assert proto.verify_hash() is False


# ── Tests freeze_protocol ──────────────────────────────────────────────────────

class TestFreezeProtocol:

    def _make_draft(self) -> TrainingProtocol:
        proto = TrainingProtocol(
            id=1,
            engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS,
            created_by_id=1,
            status=ProtocolStatus.draft,
        )
        return proto

    def test_freeze_sets_hash_and_status(self):
        proto = self._make_draft()
        db = MagicMock()
        db.get.return_value = proto

        result = freeze_protocol(db=db, protocol_id=1, frozen_by_id=1)

        assert result.status == ProtocolStatus.frozen
        assert result.protocol_hash is not None
        assert len(result.protocol_hash) == 64
        assert result.frozen_at is not None

    def test_freeze_raises_if_not_draft(self):
        proto = self._make_draft()
        proto.status = ProtocolStatus.frozen
        db = MagicMock()
        db.get.return_value = proto

        with pytest.raises(ProtocolError, match="frozen"):
            freeze_protocol(db=db, protocol_id=1, frozen_by_id=1)

    def test_freeze_raises_if_consumed(self):
        proto = self._make_draft()
        proto.status = ProtocolStatus.consumed
        db = MagicMock()
        db.get.return_value = proto

        with pytest.raises(ProtocolError):
            freeze_protocol(db=db, protocol_id=1, frozen_by_id=1)


# ── Tests consume_protocol ────────────────────────────────────────────────────

class TestConsumeProtocol:

    def _make_frozen(self) -> TrainingProtocol:
        proto = TrainingProtocol(
            id=1,
            engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS,
            created_by_id=1,
            status=ProtocolStatus.frozen,
        )
        proto.protocol_hash = proto.compute_hash()
        return proto

    def test_consume_sets_status_and_version(self):
        proto = self._make_frozen()
        db = MagicMock()
        db.get.return_value = proto

        result = consume_protocol(db=db, protocol_id=1,
                                  model_version_id=42, consumed_by_id=1)

        assert result.status == ProtocolStatus.consumed
        assert result.consumed_by_model_version_id == 42
        assert result.consumed_at is not None

    def test_consume_raises_if_already_consumed(self):
        """Brief v3 §0.2 — usage unique."""
        proto = self._make_frozen()
        proto.status = ProtocolStatus.consumed
        proto.consumed_by_model_version_id = 7
        db = MagicMock()
        db.get.return_value = proto

        with pytest.raises(ProtocolError, match="consumed"):
            consume_protocol(db=db, protocol_id=1,
                             model_version_id=99, consumed_by_id=1)

    def test_consume_raises_if_not_frozen(self):
        proto = TrainingProtocol(
            id=1, status=ProtocolStatus.draft,
            engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS, created_by_id=1,
        )
        db = MagicMock()
        db.get.return_value = proto

        with pytest.raises(ProtocolError, match="draft"):
            consume_protocol(db=db, protocol_id=1,
                             model_version_id=1, consumed_by_id=1)


# ── Tests verify_protocol_integrity ──────────────────────────────────────────

class TestVerifyIntegrity:

    def test_ok_when_hash_intact(self):
        proto = TrainingProtocol(
            id=1, engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS, created_by_id=1,
            status=ProtocolStatus.frozen,
        )
        proto.protocol_hash = proto.compute_hash()
        db = MagicMock()
        db.get.return_value = proto

        ok, detail = verify_protocol_integrity(db=db, protocol_id=1)
        assert ok is True
        assert "Intégrité vérifiée" in detail

    def test_fail_when_hash_none(self):
        proto = TrainingProtocol(
            id=1, engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS, created_by_id=1,
            status=ProtocolStatus.draft,
        )
        proto.protocol_hash = None
        db = MagicMock()
        db.get.return_value = proto

        ok, detail = verify_protocol_integrity(db=db, protocol_id=1)
        assert ok is False

    def test_fail_when_tampered(self):
        proto = TrainingProtocol(
            id=1, engine_name="texture",
            test_set_definition=_TEST_SET_DEF,
            metrics_targets=_METRICS_TARGETS, created_by_id=1,
            status=ProtocolStatus.frozen,
        )
        proto.protocol_hash = proto.compute_hash()
        # Simuler une altération post-freeze
        proto.test_set_definition = {**_TEST_SET_DEF, "n_samples": 9999}
        db = MagicMock()
        db.get.return_value = proto

        ok, detail = verify_protocol_integrity(db=db, protocol_id=1)
        assert ok is False
        assert "ALTÉRATION" in detail


# ── Tests assert_metrics_have_source (Brief v3 §0.5) ─────────────────────────

class TestAssertMetricsHaveSource:

    def test_ok_when_no_metrics(self):
        mv = MagicMock()
        mv.model_far = None
        mv.model_frr = None
        mv.model_eer = None
        mv.model_auc = None
        mv.metrics_source_hash = None
        # Ne doit pas lever
        assert_metrics_have_source(mv)

    def test_ok_when_metrics_with_hash(self):
        mv = MagicMock()
        mv.model_far = 0.02
        mv.model_frr = 0.04
        mv.model_eer = 0.03
        mv.model_auc = 0.97
        mv.metrics_source_hash = "a" * 64
        # Ne doit pas lever
        assert_metrics_have_source(mv)

    def test_raises_when_metrics_without_hash(self):
        """Brief v3 §0.5 — métriques présentes mais sans metrics.json haché."""
        mv = MagicMock()
        mv.model_far = 0.02
        mv.model_frr = None
        mv.model_eer = None
        mv.model_auc = None
        mv.metrics_source_hash = None
        with pytest.raises(ValueError, match="metrics_source_hash"):
            assert_metrics_have_source(mv)

    def test_ok_when_model_version_is_none(self):
        # Cas où l'analyse n'a pas de modèle associé
        assert_metrics_have_source(None)


# ── Tests ENGINE_DEFAULT_STATUS (Brief v3 §0.6) ───────────────────────────────

class TestEngineDefaultStatus:

    def test_audio_is_disabled(self):
        from engines.fusion import ENGINE_DEFAULT_STATUS
        assert ENGINE_DEFAULT_STATUS["audio"] == "disabled"

    def test_all_others_are_experimental(self):
        from engines.fusion import ENGINE_DEFAULT_STATUS, WEIGHTS
        for engine in WEIGHTS:
            if engine != "audio":
                assert ENGINE_DEFAULT_STATUS[engine] == "experimental", (
                    f"Engine '{engine}' devrait être 'experimental' jusqu'à la validation §6"
                )

    def test_all_engines_in_weights_have_status(self):
        from engines.fusion import ENGINE_DEFAULT_STATUS, WEIGHTS
        for engine in WEIGHTS:
            assert engine in ENGINE_DEFAULT_STATUS, (
                f"Engine '{engine}' absent de ENGINE_DEFAULT_STATUS"
            )

    def test_fuse_scores_excludes_disabled(self):
        from engines.fusion import fuse_scores
        result = fuse_scores(score_audio=1.0)  # audio = disabled
        assert "audio" in result.unvalidated_components
        # Score ne doit pas être 1.0 (audio exclu)
        assert result.final_score < 0.5  # toutes autres scores = 0.0

    def test_fuse_scores_with_allow_experimental_false(self):
        """Avec allow_experimental_engines=False, seuls les engines 'validated' contribuent."""
        from engines.fusion import fuse_scores, ENGINE_DEFAULT_STATUS
        with patch("engines.fusion.settings") as mock_settings:
            mock_settings.allow_experimental_engines = False
            result = fuse_scores(score_texture=0.9, score_temporal=0.8)
        # Tous les engines sont experimental ou disabled → aucun ne contribue
        # Le score final doit être 0.0 (ou très proche si validated_weight_sum = 0)
        assert result.final_score == 0.0 or result.final_score < 0.01
        assert len(result.experimental_components) == 0
