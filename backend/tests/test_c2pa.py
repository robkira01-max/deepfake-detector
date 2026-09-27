"""Tests — Phase 5 C2PA (provenance cryptographique)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from core.c2pa_handler import C2PAHandler, C2PAUnavailable


# ── Données de test ────────────────────────────────────────────────────────────

_FAKE_MANIFEST = {
    "active_manifest": "urn:c2pa:test:manifest",
    "manifests": {
        "urn:c2pa:test:manifest": {
            "claim_generator": "AdobePhotoshop/25.0",
            "claim_generator_info": [{"name": "AdobePhotoshop", "version": "25.0"}],
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {"actions": [{"action": "c2pa.edited"}]},
                },
                {
                    "label": "stds.schema-org.CreativeWork",
                    "data": {"dateCreated": "2026-09-01T12:00:00Z"},
                },
                {
                    "label": "c2pa.hash.data",
                    "data": {"algorithm": "sha256", "hash": "abc123"},
                },
            ],
        }
    },
    "validation_status": "valid",
    "producer": "AdobePhotoshop",
    "created_at": "2026-09-01T12:00:00Z",
}

_EMPTY_MANIFEST_JSON = json.dumps({
    "active_manifest": "urn:c2pa:empty",
    "manifests": {
        "urn:c2pa:empty": {
            "claim_generator": "TestTool/1.0",
            "assertions": [],
        }
    },
})

_REAL_MANIFEST_JSON = json.dumps({
    "active_manifest": "urn:c2pa:test:manifest",
    "manifests": {
        "urn:c2pa:test:manifest": {
            "claim_generator": "AdobePhotoshop/25.0",
            "claim_generator_info": [{"name": "AdobePhotoshop", "version": "25.0"}],
            "assertions": [
                {"label": "c2pa.actions", "data": {}},
                {
                    "label": "stds.schema-org.CreativeWork",
                    "data": {"dateCreated": "2026-09-01T12:00:00Z"},
                },
            ],
        }
    },
})


# ── Tests read_manifest ────────────────────────────────────────────────────────

class TestReadManifest:

    def test_c2pa_unavailable_returns_none(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"fake")
        with patch("core.c2pa_handler._get_c2pa", side_effect=C2PAUnavailable("not installed")):
            result = C2PAHandler.read_manifest(str(p))
        assert result is None

    def test_reader_returns_none_when_no_manifest(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.write_bytes(b"fake mp4")
        mock_c2pa = MagicMock()
        mock_c2pa.Reader.try_create.return_value = None
        with patch("core.c2pa_handler._get_c2pa", return_value=mock_c2pa):
            result = C2PAHandler.read_manifest(str(p))
        assert result is None

    def test_reader_exception_returns_none(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.write_bytes(b"fake")
        mock_c2pa = MagicMock()
        mock_c2pa.Reader.try_create.side_effect = Exception("unsupported format")
        with patch("core.c2pa_handler._get_c2pa", return_value=mock_c2pa):
            result = C2PAHandler.read_manifest(str(p))
        assert result is None

    def test_read_valid_manifest(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"fake jpeg")
        mock_reader = MagicMock()
        mock_reader.json.return_value = _REAL_MANIFEST_JSON
        mock_reader.get_validation_state.return_value = "valid"
        mock_c2pa = MagicMock()
        mock_c2pa.Reader.try_create.return_value = mock_reader
        with patch("core.c2pa_handler._get_c2pa", return_value=mock_c2pa):
            result = C2PAHandler.read_manifest(str(p))
        assert result is not None
        assert result["active_manifest"] == "urn:c2pa:test:manifest"
        assert result["producer"] == "AdobePhotoshop"
        assert result["created_at"] == "2026-09-01T12:00:00Z"
        assert result["validation_status"] == "valid"

    def test_read_manifest_empty_json_returns_none(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"fake")
        mock_reader = MagicMock()
        mock_reader.json.return_value = None
        mock_c2pa = MagicMock()
        mock_c2pa.Reader.try_create.return_value = mock_reader
        with patch("core.c2pa_handler._get_c2pa", return_value=mock_c2pa):
            result = C2PAHandler.read_manifest(str(p))
        assert result is None

    def test_accepts_path_object(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"fake")
        mock_c2pa = MagicMock()
        mock_c2pa.Reader.try_create.return_value = None
        with patch("core.c2pa_handler._get_c2pa", return_value=mock_c2pa):
            result = C2PAHandler.read_manifest(p)  # Path object, not str
        assert result is None


# ── Tests extract_assertions ──────────────────────────────────────────────────

class TestExtractAssertions:

    def test_empty_manifest_returns_empty(self):
        result = C2PAHandler.extract_assertions({})
        assert result == []

    def test_none_returns_empty(self):
        result = C2PAHandler.extract_assertions(None)
        assert result == []

    def test_no_active_manifest_returns_empty(self):
        result = C2PAHandler.extract_assertions({"active_manifest": None, "manifests": {}})
        assert result == []

    def test_extracts_assertions_from_active(self):
        result = C2PAHandler.extract_assertions(_FAKE_MANIFEST)
        assert len(result) == 3
        labels = [a["label"] for a in result]
        assert "c2pa.actions" in labels
        assert "stds.schema-org.CreativeWork" in labels
        assert "c2pa.hash.data" in labels


# ── Tests get_provenance_summary ──────────────────────────────────────────────

class TestGetProvenanceSummary:

    def test_none_manifest_returns_no_c2pa(self):
        result = C2PAHandler.get_provenance_summary(None)
        assert result["has_c2pa"] is False
        assert result["producer"] is None
        assert result["is_valid"] is None
        assert result["actions_count"] == 0

    def test_empty_dict_returns_no_c2pa(self):
        result = C2PAHandler.get_provenance_summary({})
        assert result["has_c2pa"] is False

    def test_valid_manifest_summary(self):
        result = C2PAHandler.get_provenance_summary(_FAKE_MANIFEST)
        assert result["has_c2pa"] is True
        assert result["producer"] == "AdobePhotoshop"
        assert result["created_at"] == "2026-09-01T12:00:00Z"
        assert result["actions_count"] == 1
        assert result["is_valid"] is True
        assert result["validation_errors"] == []

    def test_invalid_manifest_flags_errors(self):
        invalid = {**_FAKE_MANIFEST, "validation_status": "INVALID_HASH"}
        result = C2PAHandler.get_provenance_summary(invalid)
        assert result["is_valid"] is False
        assert len(result["validation_errors"]) == 1

    def test_tool_extracted_from_claim_generator_info(self):
        result = C2PAHandler.get_provenance_summary(_FAKE_MANIFEST)
        assert result["tool"] == "AdobePhotoshop"

    def test_no_actions_count_zero(self):
        manifest_no_actions = {
            "active_manifest": "urn:test",
            "manifests": {
                "urn:test": {
                    "assertions": [
                        {"label": "c2pa.hash.data", "data": {}},
                    ],
                }
            },
            "validation_status": "valid",
            "producer": "TestTool",
            "created_at": None,
        }
        result = C2PAHandler.get_provenance_summary(manifest_no_actions)
        assert result["actions_count"] == 0
