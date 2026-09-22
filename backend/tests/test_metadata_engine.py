"""Tests du MetadataEngine — couverture des vérifications forensiques.

Pas d'appel à ffprobe réel : on patch subprocess.run et on injecte
directement des métadonnées JSON synthétiques.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ffprobe_output(
    tags: dict | None = None,
    streams: list | None = None,
    duration: str = "10.0",
    bit_rate: str = "500000",
) -> str:
    """Génère un JSON ffprobe synthétique."""
    return json.dumps({
        "format": {
            "duration": duration,
            "bit_rate": bit_rate,
            "tags": tags or {},
        },
        "streams": streams or [],
    })


def _ffprobe_success(stdout: str) -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    return m


def _ffprobe_fail() -> MagicMock:
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    return m


# ── Tests : extraction métadonnées ────────────────────────────────────────────

class TestMetadataExtraction:

    def test_ffprobe_success_populates_metadata(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "video.mp4"
        dummy.write_bytes(b"\x00" * 100)

        output = _make_ffprobe_output(tags={"encoder": "HandBrake 1.6"})
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        assert result.error is None
        assert 0.0 <= result.score <= 1.0
        assert "ffprobe" in result.metadata_raw

    def test_ffprobe_unavailable_uses_fallback(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "audio.wav"
        dummy.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("subprocess.run", side_effect=FileNotFoundError("ffprobe not found")):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        # Fallback doit fonctionner sans ffprobe
        assert result.error is None or "métadonnées" in (result.error or "")
        assert result.score >= 0.0

    def test_missing_file_returns_error(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        engine = MetadataEngine()
        result = engine.analyze(Path("/nonexistent/totally_fake_path.mp4"))

        assert result.error is not None
        assert "introuvable" in result.error.lower()
        assert result.score == 0.0

    def test_ffprobe_invalid_json_uses_fallback(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "corrupt.mp4"
        dummy.write_bytes(b"\x00" * 50)

        bad_response = MagicMock()
        bad_response.returncode = 0
        bad_response.stdout = "NOT_VALID_JSON"

        with patch("subprocess.run", return_value=bad_response):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        # Ne doit pas lever une exception
        assert isinstance(result.score, float)

    def test_ffprobe_timeout_uses_fallback(self, tmp_path):
        import subprocess
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "video.mp4"
        dummy.write_bytes(b"\x00" * 50)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 30)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        assert isinstance(result.score, float)


# ── Tests : détection d'outils deepfake ──────────────────────────────────────

class TestDeepfakeToolDetection:

    @pytest.mark.parametrize("tool_name,expected_pattern", [
        ("DeepFaceLab 3.0", "deepfacelab"),
        ("FaceFusion 2.5", "facefusion"),
        ("InsightFace", "insightface"),
        ("Roop-Next", "roop"),
        ("SimSwap v2", "simswap"),
        ("Wav2Lip", "wav2lip"),
        ("SadTalker v0.0.2", "sadtalker"),
        ("stable-diffusion-video", "diffusion"),
        ("GFPGAN", "gfpgan"),
        ("CodeFormer", "codeformer"),
        ("Real-ESRGAN", "real-esrgan"),
    ])
    def test_known_deepfake_tool_detected(self, tmp_path, tool_name, expected_pattern):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "fake.mp4"
        dummy.write_bytes(b"\x00" * 100)

        output = _make_ffprobe_output(tags={"encoder": tool_name, "comment": tool_name})
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        # Score attendu > 0.3 (le poids de tool_score=0.9 dans la moyenne pondérée est 2/4.5)
        assert result.score >= 0.3, f"Score trop bas pour {tool_name} : {result.score}"
        assert result.tool_detected is not None
        assert any(f["type"] == "deepfake_tool_signature" for f in result.findings)

    def test_legitimate_tool_reduces_score(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "legit.mp4"
        dummy.write_bytes(b"\x00" * 100)

        output = _make_ffprobe_output(tags={"encoder": "Adobe Premiere Pro 2024"})
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        # Adobe Premiere = outil légitime → score faible
        assert result.score < 0.3, f"Score trop haut pour Adobe Premiere : {result.score}"

    def test_no_tool_info_neutral_score(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "neutral.mp4"
        dummy.write_bytes(b"\x00" * 100)

        output = _make_ffprobe_output(tags={})
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        # Pas d'info = score neutre (pas de flag deepfake)
        assert result.score < 0.5


# ── Tests : anomalies temporelles ────────────────────────────────────────────

class TestTemporalAnomalies:

    def test_future_creation_date_flagged(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "future.mp4"
        dummy.write_bytes(b"\x00" * 100)

        # Année dans le futur lointain = suspecte
        output = _make_ffprobe_output(tags={"creation_time": "2099-01-01T00:00:00Z"})
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        assert result.temporal_anomaly is True
        assert any(f["type"] == "temporal_anomaly" for f in result.findings)

    def test_old_creation_date_flagged(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "old.mp4"
        dummy.write_bytes(b"\x00" * 100)

        # Année avant 1990 = suspecte (deepfake tools n'existaient pas)
        output = _make_ffprobe_output(tags={"creation_time": "1985-06-15T00:00:00Z"})
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        assert result.temporal_anomaly is True

    def test_valid_date_no_anomaly(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "normal.mp4"
        dummy.write_bytes(b"\x00" * 100)

        output = _make_ffprobe_output(tags={"creation_time": "2024-03-15T10:30:00Z"})
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        assert result.temporal_anomaly is False


# ── Tests : paramètres codec ──────────────────────────────────────────────────

class TestCodecParams:

    def test_suspicious_generic_encoder_flagged(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "suspect.mp4"
        dummy.write_bytes(b"\x00" * 100)

        streams = [{"tags": {"encoder": "lavc 59.37.100 libx264"}, "codec_type": "video"}]
        output = _make_ffprobe_output(streams=streams)
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        # 'lavc' est dans _SUSPICIOUS_CODEC_PARAMS
        suspicious = [f for f in result.findings if f["type"] == "generic_encoder"]
        assert len(suspicious) > 0

    def test_fps_mismatch_flagged(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "fps_mismatch.mp4"
        dummy.write_bytes(b"\x00" * 100)

        # 25fps déclaré mais nb_frames/duration = ~10fps
        streams = [{
            "codec_type": "video",
            "tags": {},
            "r_frame_rate": "25/1",  # 25fps déclaré
            "nb_frames": "100",
            "duration": "20.0",     # mais seulement 5fps réels
        }]
        output = _make_ffprobe_output(streams=streams)
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        fps_findings = [f for f in result.findings if f["type"] == "fps_mismatch"]
        assert len(fps_findings) > 0

    def test_streams_without_encoder_no_crash(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "no_encoder.mp4"
        dummy.write_bytes(b"\x00" * 100)

        streams = [{"codec_type": "audio", "tags": {}}]
        output = _make_ffprobe_output(streams=streams)
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        assert isinstance(result.score, float)


# ── Tests : anomalies de données ──────────────────────────────────────────────

class TestDataAnomalies:

    def test_oversized_file_flagged(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "oversized.mp4"
        # 1 MB de données
        dummy.write_bytes(b"\x00" * 1_000_000)

        # Déclarer un débit très faible → ratio > 3.0
        # bit_rate=8000 * duration=1.0 → expected_size = 1000 bytes, actual = 1MB
        output = _make_ffprobe_output(
            tags={},
            duration="1.0",
            bit_rate="8000",  # 8 kbps → 1000 bytes attendus vs 1MB réel
        )
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        oversized = [f for f in result.findings if f["type"] == "oversized_for_bitrate"]
        assert len(oversized) > 0


# ── Tests : flatten_to_text ───────────────────────────────────────────────────

class TestFlattenToText:

    def test_flatten_string(self):
        from engines.metadata_engine import MetadataEngine
        engine = MetadataEngine()
        assert "hello" in engine._flatten_to_text("hello world")

    def test_flatten_dict(self):
        from engines.metadata_engine import MetadataEngine
        engine = MetadataEngine()
        result = engine._flatten_to_text({"key": "deepfacelab", "nested": {"x": "wav2lip"}})
        assert "deepfacelab" in result
        assert "wav2lip" in result

    def test_flatten_list(self):
        from engines.metadata_engine import MetadataEngine
        engine = MetadataEngine()
        result = engine._flatten_to_text(["item1", {"tool": "sadtalker"}])
        assert "item1" in result
        assert "sadtalker" in result

    def test_flatten_depth_limit(self):
        """La récursion s'arrête à depth=5 pour éviter les boucles infinies."""
        from engines.metadata_engine import MetadataEngine
        engine = MetadataEngine()

        # Créer une structure très profonde
        deep = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": {"l7": "deep_value"}}}}}}}
        result = engine._flatten_to_text(deep)
        # Ne doit pas lever RecursionError
        assert isinstance(result, str)

    def test_flatten_non_string_values(self):
        from engines.metadata_engine import MetadataEngine
        engine = MetadataEngine()

        # Nombres, None, bool — pas d'erreur
        result = engine._flatten_to_text({"count": 42, "flag": True, "nothing": None})
        assert isinstance(result, str)


# ── Tests : score global ──────────────────────────────────────────────────────

class TestScoreIntegrity:

    def test_score_always_in_0_1(self, tmp_path):
        from engines.metadata_engine import MetadataEngine

        dummy = tmp_path / "test.mp4"
        dummy.write_bytes(b"\x00" * 200)

        # Combinaison explosive : deepfake tool + date suspecte
        output = _make_ffprobe_output(
            tags={"encoder": "DeepFaceLab", "creation_time": "2099-12-31T00:00:00Z"},
        )
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        assert 0.0 <= result.score <= 1.0

    def test_result_fields_types(self, tmp_path):
        from engines.metadata_engine import MetadataEngine, MetadataResult

        dummy = tmp_path / "typed.mp4"
        dummy.write_bytes(b"\x00" * 100)

        output = _make_ffprobe_output()
        with patch("subprocess.run", return_value=_ffprobe_success(output)):
            engine = MetadataEngine()
            result = engine.analyze(dummy)

        assert isinstance(result, MetadataResult)
        assert isinstance(result.score, float)
        assert isinstance(result.findings, list)
        assert isinstance(result.metadata_raw, dict)
        assert isinstance(result.temporal_anomaly, bool)
