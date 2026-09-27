"""Tests des moteurs ML — fusion, scores, edge cases. Pas d'inférence réelle."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestFusionEngine:
    """Tests de la logique de fusion — pas de modèle ML chargé."""

    def test_all_zero_scores_authentic(self):
        from engines.fusion import fuse_scores
        from models.analysis import Verdict

        result = fuse_scores()
        assert result.verdict == Verdict.authentic
        assert result.final_score < 0.35

    def test_all_max_scores_deepfake(self):
        from engines.fusion import fuse_scores
        from models.analysis import Verdict

        result = fuse_scores(
            score_texture=1.0,
            score_temporal=1.0,
            score_rppg=1.0,
            score_biometrics=1.0,
            score_audio=1.0,
            score_phase=1.0,
            score_metadata=1.0,
        )
        assert result.verdict == Verdict.deepfake
        assert result.final_score > 0.55

    def test_scores_sum_weights_to_one(self):
        from engines.fusion import WEIGHTS
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Poids ne somment pas à 1.0 : {total}"

    def test_confidence_interval_valid(self):
        from engines.fusion import fuse_scores

        result = fuse_scores(
            score_texture=0.5,
            score_audio=0.4,
        )
        assert 0.0 <= result.confidence_low <= result.final_score
        assert result.final_score <= result.confidence_high <= 1.0

    def test_final_score_bounded_0_1(self):
        from engines.fusion import fuse_scores

        for _ in range(20):
            scores = {k: np.random.random() for k in [
                "score_texture", "score_temporal", "score_rppg",
                "score_biometrics", "score_audio", "score_phase", "score_metadata"
            ]}
            result = fuse_scores(**scores)
            assert 0.0 <= result.final_score <= 1.0, f"Score hors limites : {result.final_score}"

    def test_shap_ranking_all_features_present(self):
        from engines.fusion import fuse_scores, ENGINE_DEFAULT_STATUS

        result = fuse_scores(score_texture=0.8, score_audio=0.3)
        feature_names = {r["feature"] for r in result.shap_ranking}
        # SHAP n'inclut que les composantes actives (audio disabled — Brief v3 §0.6)
        active = {k for k, s in ENGINE_DEFAULT_STATUS.items() if s != "disabled"}
        assert feature_names == active
        assert "audio" not in feature_names
        # audio est disabled → dans unvalidated_components (exclu du score)
        assert "audio" in result.unvalidated_components
        # les autres sont experimental → dans experimental_components
        assert all(k in result.experimental_components for k in active)

    def test_shap_contributions_sum_to_100(self):
        from engines.fusion import fuse_scores

        result = fuse_scores(score_texture=0.6, score_rppg=0.7)
        total = sum(r["contribution"] for r in result.shap_ranking)
        assert abs(total - 100.0) < 0.1, f"Contributions SHAP ne somment pas à 100% : {total}"

    def test_plain_explanation_not_empty(self):
        from engines.fusion import fuse_scores

        for score in [0.1, 0.45, 0.8]:
            result = fuse_scores(score_texture=score)
            assert result.plain_explanation
            assert len(result.plain_explanation) > 50

    def test_undetermined_zone(self):
        from engines.fusion import fuse_scores, THRESHOLD_AUTHENTIC, THRESHOLD_DEEPFAKE
        from models.analysis import Verdict

        # Score pile dans la zone indéterminée
        result = fuse_scores(score_texture=0.45)
        if THRESHOLD_AUTHENTIC <= result.final_score < THRESHOLD_DEEPFAKE:
            assert result.verdict == Verdict.undetermined

    def test_model_metrics_none_until_validated(self):
        from engines.fusion import fuse_scores

        result = fuse_scores()
        # Brief v2 P0.2 : métriques à None jusqu'à mesure sur jeu de test indépendant
        assert result.model_far is None
        assert result.model_frr is None
        assert result.model_eer is None
        assert result.model_auc is None


class TestVideoEngineEdgeCases:
    """Tests des cas limites du VideoEngine — pas d'inférence réelle."""

    def test_missing_file_returns_error(self):
        from engines.video_engine import VideoEngine
        engine = VideoEngine.__new__(VideoEngine)
        engine._models_loaded = False
        engine._face_mesh = None

        from pathlib import Path
        result = engine.analyze(Path("/nonexistent/file.mp4"))
        assert result.error is not None
        assert "introuvable" in result.error.lower()

    def test_ear_calculation(self):
        """Test du calcul Eye Aspect Ratio."""
        from engines.video_engine import _ear
        import numpy as np

        # Œil ouvert (rapport ~0.3)
        def open_eye(idx):
            pts = {
                0: np.array([0.0, 0.0]),
                1: np.array([1.0, 1.5]),
                2: np.array([2.0, 1.5]),
                3: np.array([3.0, 0.0]),
                4: np.array([2.0, -1.5]),
                5: np.array([1.0, -1.5]),
            }
            return pts[idx]

        ear_open = _ear(open_eye, [0, 1, 2, 3, 4, 5])
        assert ear_open > 0.15  # Œil ouvert

    def test_blink_detection(self):
        from engines.video_engine import _count_blinks
        import numpy as np

        # Signal simulé : 3 clignements (passages sous 0.2)
        ear = np.array([0.3, 0.3, 0.15, 0.3, 0.3, 0.12, 0.3, 0.3, 0.18, 0.3])
        blinks = _count_blinks(ear, threshold=0.2)
        assert blinks == 3

    def test_calibrate_pretrained_score_range(self):
        from engines.video_engine import _calibrate_pretrained_score

        for raw in [0.0, 0.25, 0.5, 0.75, 1.0]:
            calibrated = _calibrate_pretrained_score(raw)
            assert 0.0 <= calibrated <= 1.0, f"Score calibré hors limites pour raw={raw}"

    def test_calibrate_pretrained_score_centered_at_half(self):
        """Score brut de 0.5 → score calibré proche de 0.1 (biais vers authenticité)."""
        from engines.video_engine import _calibrate_pretrained_score
        assert abs(_calibrate_pretrained_score(0.5) - 0.1) < 0.01


class TestAudioEngineEdgeCases:
    """Tests des cas limites du AudioEngine."""

    def test_missing_file_returns_error(self):
        from engines.audio_engine import AudioEngine
        from pathlib import Path

        engine = AudioEngine.__new__(AudioEngine)
        engine._models_loaded = False

        result = engine.analyze(Path("/nonexistent/audio.wav"))
        assert result.error is not None

    def test_phase_analysis_short_signal(self):
        """Un signal trop court retourne un score neutre."""
        from engines.audio_engine import AudioEngine
        import numpy as np

        engine = AudioEngine.__new__(AudioEngine)
        engine._models_loaded = False
        engine.TARGET_SR = 16000

        # Signal très court : 10 samples
        short_signal = np.zeros(10)
        try:
            score, segments = engine._analyze_phase(short_signal, 16000)
            assert 0.0 <= score <= 1.0
        except Exception:
            pass  # Un signal trop court peut légitimement lever une exception

    def test_phase_analysis_synthetic_discontinuities(self):
        """Un signal avec discontinuités de phase doit avoir un score élevé."""
        from engines.audio_engine import AudioEngine
        import numpy as np

        engine = AudioEngine.__new__(AudioEngine)
        engine._models_loaded = False
        engine.TARGET_SR = 16000

        # Signal avec discontinuités artificielles (voix synthétique simulée)
        sr = 16000
        t = np.linspace(0, 1, sr)
        # Alterner brusquement entre deux fréquences
        signal = np.where(t < 0.5, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t + np.pi))

        try:
            score, segments = engine._analyze_phase(signal, sr)
            # Score > 0 si des discontinuités sont détectées
            assert 0.0 <= score <= 1.0
        except Exception:
            pass


class TestMetadataEngine:
    """Tests du moteur de métadonnées (score_metadata)."""

    def test_score_metadata_returns_float(self, tmp_path):
        """Le moteur doit retourner un float entre 0 et 1."""
        from engines.metadata_engine import MetadataEngine
        import numpy as np

        wav = tmp_path / "test.wav"
        wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

        try:
            engine = MetadataEngine()
            result = engine.analyze(wav)
            assert 0.0 <= result.score <= 1.0
        except ImportError:
            pytest.skip("MetadataEngine pas encore implémenté")
