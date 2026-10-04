"""Tests — C2PAVerifier et C2PAEngine (P2 provenance cryptographique)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.c2pa_verifier import C2PAVerifier, C2PAVerificationResult


# ── Manifestes de test ────────────────────────────────────────────────────────

_MANIFEST_AI_GENERATED = {
    "active_manifest": "urn:c2pa:test",
    "manifests": {
        "urn:c2pa:test": {
            "claim_generator": "GenAI-Tool/1.0",
            "claim_generator_info": [{"name": "GenAI-Tool", "version": "1.0"}],
            "assertions": [
                {"label": "c2pa.ai.generatedContent", "data": {}},
                {"label": "c2pa.hash.data", "data": {"algorithm": "sha256", "hash": "abc"}},
                {"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.created"}]}},
            ],
        }
    },
    "validation_status": "valid",
    "producer": "GenAI-Tool",
    "created_at": "2026-09-01T12:00:00Z",
}

_MANIFEST_CAMERA = {
    "active_manifest": "urn:c2pa:cam",
    "manifests": {
        "urn:c2pa:cam": {
            "claim_generator": "Canon-EOS/1.0",
            "claim_generator_info": [{"name": "Canon-EOS", "version": "1.0"}],
            "assertions": [
                {"label": "c2pa.hash.data", "data": {"algorithm": "sha256", "hash": "def"}},
                {"label": "c2pa.hash.boxes", "data": {}},
                {
                    "label": "c2pa.training-mining",
                    "data": {
                        "entries": [
                            {"use": "notAllowed", "constraint_info": "all uses prohibited"}
                        ]
                    },
                },
            ],
        }
    },
    "validation_status": "valid",
    "producer": "Canon-EOS",
    "created_at": "2026-09-01T10:00:00Z",
}

_MANIFEST_INVALID_HASH = {
    "active_manifest": "urn:c2pa:tampered",
    "manifests": {
        "urn:c2pa:tampered": {
            "claim_generator": "SomeTool/1.0",
            "assertions": [
                {"label": "c2pa.hash.data", "data": {}},
            ],
        }
    },
    "validation_status": "INVALID_HASH",
    "producer": "SomeTool",
    "created_at": None,
}


# ── Tests C2PAVerifier ────────────────────────────────────────────────────────

class TestC2PAVerifierNoManifest:

    def test_no_manifest_returns_no_manifest_result(self, tmp_path):
        p = tmp_path / "plain.mp4"
        p.write_bytes(b"fake")
        with patch("core.c2pa_verifier.C2PAHandler.read_manifest", return_value=None):
            result = C2PAVerifier.verify(p)
        assert result.has_manifest is False
        assert result.is_cryptographically_valid is None
        assert result.is_ai_generated is None
        assert result.hard_bindings_count == 0

    def test_no_manifest_to_dict_safe(self, tmp_path):
        p = tmp_path / "plain.jpg"
        p.write_bytes(b"fake")
        with patch("core.c2pa_verifier.C2PAHandler.read_manifest", return_value=None):
            result = C2PAVerifier.verify(p)
        d = result.to_dict()
        assert d["has_manifest"] is False
        assert d["is_ai_generated"] is None

    def test_exception_in_handler_returns_no_manifest(self, tmp_path):
        p = tmp_path / "bad.mp4"
        p.write_bytes(b"fake")
        with patch("core.c2pa_verifier.C2PAHandler.read_manifest", side_effect=RuntimeError("crash")):
            # C2PAVerifier.verify does NOT catch exceptions — caller should handle
            with pytest.raises(RuntimeError):
                C2PAVerifier.verify(p)


class TestC2PAVerifierAIGenerated:

    def _mock_handler(self, manifest: dict):
        """Patch C2PAHandler methods to return fixture data."""
        from core.c2pa_handler import C2PAHandler

        def _read(file_path):
            return manifest

        def _extract(m):
            active_label = m.get("active_manifest")
            active = m.get("manifests", {}).get(active_label) if active_label else None
            return active.get("assertions", []) if active else []

        def _summary(m):
            from core.c2pa_handler import C2PAHandler as H
            assertions = _extract(m)
            tool = None
            active_label = m.get("active_manifest")
            active = m.get("manifests", {}).get(active_label) if active_label else None
            if active:
                sw = active.get("claim_generator_info", [])
                tool = sw[0].get("name") if (isinstance(sw, list) and sw) else None
            validation = m.get("validation_status", "unknown")
            is_valid = validation in ("valid", "Valid")
            return {
                "has_c2pa": True,
                "producer": m.get("producer"),
                "tool": tool,
                "created_at": m.get("created_at"),
                "actions_count": 0,
                "is_valid": is_valid,
                "validation_errors": [] if is_valid else [validation],
            }

        return patch.multiple(
            "core.c2pa_verifier.C2PAHandler",
            read_manifest=staticmethod(_read),
            extract_assertions=staticmethod(_extract),
            get_provenance_summary=staticmethod(_summary),
        )

    def test_ai_generated_assertion_detected(self, tmp_path):
        p = tmp_path / "ai.jpg"
        p.write_bytes(b"fake")
        with self._mock_handler(_MANIFEST_AI_GENERATED):
            result = C2PAVerifier.verify(p)
        assert result.has_manifest is True
        assert result.is_ai_generated is True
        assert result.is_cryptographically_valid is True
        assert result.hard_bindings_count == 1
        assert "c2pa.created" in result.actions

    def test_camera_manifest_not_ai_generated(self, tmp_path):
        p = tmp_path / "photo.jpg"
        p.write_bytes(b"fake")
        with self._mock_handler(_MANIFEST_CAMERA):
            result = C2PAVerifier.verify(p)
        assert result.has_manifest is True
        assert result.is_ai_generated is False
        assert result.is_cryptographically_valid is True
        assert result.hard_bindings_count == 2

    def test_training_opt_out_detected(self, tmp_path):
        p = tmp_path / "photo.jpg"
        p.write_bytes(b"fake")
        with self._mock_handler(_MANIFEST_CAMERA):
            result = C2PAVerifier.verify(p)
        assert result.ai_training_opted_out is True

    def test_invalid_manifest_detected(self, tmp_path):
        p = tmp_path / "tampered.mp4"
        p.write_bytes(b"fake")
        with self._mock_handler(_MANIFEST_INVALID_HASH):
            result = C2PAVerifier.verify(p)
        assert result.has_manifest is True
        assert result.is_cryptographically_valid is False
        assert len(result.validation_errors) > 0

    def test_hard_bindings_counted(self, tmp_path):
        p = tmp_path / "photo.jpg"
        p.write_bytes(b"fake")
        with self._mock_handler(_MANIFEST_CAMERA):
            result = C2PAVerifier.verify(p)
        assert result.hard_bindings_count == 2  # c2pa.hash.data + c2pa.hash.boxes

    def test_producer_extracted(self, tmp_path):
        p = tmp_path / "ai.jpg"
        p.write_bytes(b"fake")
        with self._mock_handler(_MANIFEST_AI_GENERATED):
            result = C2PAVerifier.verify(p)
        assert result.producer == "GenAI-Tool"

    def test_to_dict_complete(self, tmp_path):
        p = tmp_path / "ai.jpg"
        p.write_bytes(b"fake")
        with self._mock_handler(_MANIFEST_AI_GENERATED):
            result = C2PAVerifier.verify(p)
        d = result.to_dict()
        required_keys = [
            "has_manifest", "is_cryptographically_valid", "producer", "tool",
            "created_at", "is_ai_generated", "ai_training_opted_out",
            "hard_bindings_count", "soft_bindings_count", "actions",
            "validation_errors", "raw_assertions_count",
        ]
        for key in required_keys:
            assert key in d, f"Clé manquante : {key}"


# ── Tests C2PAEngine ──────────────────────────────────────────────────────────

class TestC2PAEngine:

    def _make_verifier_patch(self, result):
        return patch("engines.c2pa_engine.C2PAVerifier.verify", return_value=result)

    def _no_manifest_result(self):
        return C2PAVerificationResult(
            has_manifest=False,
            is_cryptographically_valid=None,
            producer=None,
            tool=None,
            created_at=None,
            is_ai_generated=None,
            ai_training_opted_out=None,
            hard_bindings_count=0,
            soft_bindings_count=0,
        )

    def _ai_generated_result(self):
        return C2PAVerificationResult(
            has_manifest=True,
            is_cryptographically_valid=True,
            producer="GenAI-Tool",
            tool="GenAI-Tool",
            created_at="2026-09-01T12:00:00Z",
            is_ai_generated=True,
            ai_training_opted_out=None,
            hard_bindings_count=1,
            soft_bindings_count=0,
        )

    def _camera_result(self):
        return C2PAVerificationResult(
            has_manifest=True,
            is_cryptographically_valid=True,
            producer="Canon-EOS",
            tool="Canon-EOS",
            created_at="2026-09-01T10:00:00Z",
            is_ai_generated=False,
            ai_training_opted_out=True,
            hard_bindings_count=2,
            soft_bindings_count=0,
        )

    def _invalid_manifest_result(self):
        return C2PAVerificationResult(
            has_manifest=True,
            is_cryptographically_valid=False,
            producer="SomeTool",
            tool=None,
            created_at=None,
            is_ai_generated=False,
            ai_training_opted_out=None,
            hard_bindings_count=1,
            soft_bindings_count=0,
            validation_errors=["INVALID_HASH"],
        )

    def test_engine_name(self):
        from engines.c2pa_engine import C2PAEngine
        assert C2PAEngine.name == "c2pa"

    def test_engine_version(self):
        from engines.c2pa_engine import C2PAEngine
        assert C2PAEngine.version == "1.0.0"

    def test_supports_image_video_audio_document(self):
        from engines.c2pa_engine import C2PAEngine
        engine = C2PAEngine()
        for media_type in ("image", "video", "audio", "document"):
            assert engine.supports(media_type) is True

    def test_no_manifest_score_zero_confidence_zero(self, tmp_path):
        from engines.c2pa_engine import C2PAEngine
        engine = C2PAEngine()
        p = tmp_path / "x.mp4"
        p.touch()
        with self._make_verifier_patch(self._no_manifest_result()):
            result = engine.analyze(p)
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.error is None

    def test_ai_generated_score_1_confidence_1(self, tmp_path):
        from engines.c2pa_engine import C2PAEngine
        engine = C2PAEngine()
        p = tmp_path / "x.jpg"
        p.touch()
        with self._make_verifier_patch(self._ai_generated_result()):
            result = engine.analyze(p)
        assert result.score == 1.0
        assert result.confidence == 1.0

    def test_camera_manifest_score_zero_confidence_high(self, tmp_path):
        from engines.c2pa_engine import C2PAEngine
        engine = C2PAEngine()
        p = tmp_path / "x.jpg"
        p.touch()
        with self._make_verifier_patch(self._camera_result()):
            result = engine.analyze(p)
        assert result.score == 0.0
        assert result.confidence >= 0.8

    def test_invalid_manifest_score_07(self, tmp_path):
        from engines.c2pa_engine import C2PAEngine
        engine = C2PAEngine()
        p = tmp_path / "x.jpg"
        p.touch()
        with self._make_verifier_patch(self._invalid_manifest_result()):
            result = engine.analyze(p)
        assert result.score == pytest.approx(0.7)
        assert result.confidence >= 0.7

    def test_exception_returns_error_result(self, tmp_path):
        from engines.c2pa_engine import C2PAEngine
        engine = C2PAEngine()
        p = tmp_path / "x.mp4"
        p.touch()
        with patch("engines.c2pa_engine.C2PAVerifier.verify", side_effect=RuntimeError("boom")):
            result = engine.analyze(p)
        assert result.error == "boom"
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_is_engine_plugin(self):
        from engines.c2pa_engine import C2PAEngine
        from engines.plugin import EnginePlugin
        assert issubclass(C2PAEngine, EnginePlugin)

    def test_metadata_contains_c2pa_fields(self, tmp_path):
        from engines.c2pa_engine import C2PAEngine
        engine = C2PAEngine()
        p = tmp_path / "x.jpg"
        p.touch()
        with self._make_verifier_patch(self._ai_generated_result()):
            result = engine.analyze(p)
        assert "has_manifest" in result.metadata
        assert result.metadata["is_ai_generated"] is True

    def test_can_register_in_plugin_registry(self):
        """C2PAEngine s'enregistre sans erreur dans PluginRegistry."""
        from engines.c2pa_engine import C2PAEngine
        from engines.registry import PluginRegistry
        registry = PluginRegistry()
        registry.register(C2PAEngine, status="experimental", weight=0.1)
        assert "c2pa" in registry


# ── Tests endpoint C2PA ───────────────────────────────────────────────────────

class TestC2PAEndpoint:

    def test_verify_requires_auth(self, client):
        r = client.get("/c2pa/verify/1")
        assert r.status_code == 401

    def test_verify_file_not_found(self, client, auth_analyst):
        r = client.get("/c2pa/verify/9999", headers=auth_analyst)
        assert r.status_code == 404

    def _make_media_file(self, db, user, storage_path):
        """Crée un Case + MediaFile dans la DB de test."""
        import uuid as _uuid
        from models.case import Case, CaseStatus, Jurisdiction
        from models.media_file import MediaFile, MediaType, MediaStatus

        case = Case(
            case_number=f"C2PA-{_uuid.uuid4().hex[:6].upper()}",
            title="C2PA test case",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=user.id,
        )
        db.add(case)
        db.commit()
        db.refresh(case)

        mf = MediaFile(
            uuid=str(_uuid.uuid4()),
            case_id=case.id,
            original_filename=storage_path.name,
            media_type=MediaType.video,
            mime_type="image/jpeg",
            file_size_bytes=len(storage_path.read_bytes()),
            status=MediaStatus.quarantine,
            hash_sha256="a" * 64,
            hash_blake3="b" * 64,
            ingested_by_id=user.id,
            storage_key=str(storage_path),
        )
        db.add(mf)
        db.commit()
        db.refresh(mf)
        return mf

    def test_verify_returns_c2pa_result(self, client, db, auth_analyst, analyst_user, tmp_path):
        from core.c2pa_verifier import C2PAVerificationResult

        fake_file = tmp_path / "test.jpg"
        fake_file.write_bytes(b"fake jpeg content")
        mf = self._make_media_file(db, analyst_user, fake_file)

        no_manifest = C2PAVerificationResult(
            has_manifest=False,
            is_cryptographically_valid=None,
            producer=None, tool=None, created_at=None,
            is_ai_generated=None, ai_training_opted_out=None,
            hard_bindings_count=0, soft_bindings_count=0,
        )

        with patch("core.c2pa_verifier.C2PAVerifier.verify", return_value=no_manifest):
            r = client.get(f"/c2pa/verify/{mf.id}", headers=auth_analyst)

        assert r.status_code == 200
        data = r.json()
        assert data["has_manifest"] is False
        assert "disclaimer" in data
        assert data["file_id"] == mf.id

    def test_verify_ai_generated_manifest(self, client, db, auth_analyst, analyst_user, tmp_path):
        from core.c2pa_verifier import C2PAVerificationResult

        fake_file = tmp_path / "ai_gen.jpg"
        fake_file.write_bytes(b"fake ai content")
        mf = self._make_media_file(db, analyst_user, fake_file)

        ai_result = C2PAVerificationResult(
            has_manifest=True,
            is_cryptographically_valid=True,
            producer="DALL-E", tool="DALL-E",
            created_at="2026-10-04T00:00:00Z",
            is_ai_generated=True, ai_training_opted_out=None,
            hard_bindings_count=1, soft_bindings_count=0,
            actions=["c2pa.created"],
        )

        with patch("core.c2pa_verifier.C2PAVerifier.verify", return_value=ai_result):
            r = client.get(f"/c2pa/verify/{mf.id}", headers=auth_analyst)

        assert r.status_code == 200
        data = r.json()
        assert data["has_manifest"] is True
        assert data["is_ai_generated"] is True
        assert data["producer"] == "DALL-E"
