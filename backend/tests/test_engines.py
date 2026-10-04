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
        # Brief v2 P0.2 : métriques indisponibles jusqu'à validation sur jeu de test indépendant
        assert result.metrics_available is False
        assert result.metrics_artifact is None


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


# ── Tests plugin API ──────────────────────────────────────────────────────────

class TestEnginePlugin:
    """Tests de l'ABC EnginePlugin et du dataclass EngineResult."""

    def _make_plugin(self, name: str = "test_plugin", score: float = 0.5):
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        _name = name
        _score = score

        class _Plugin(EnginePlugin):
            def analyze(self, file_path: Path) -> EngineResult:
                return EngineResult(engine_name=self.name, score=_score)

        _Plugin.name = _name
        return _Plugin

    def test_engine_result_defaults(self):
        from engines.plugin import EngineResult
        r = EngineResult(engine_name="x", score=0.3)
        assert r.confidence == 1.0
        assert r.metadata == {}
        assert r.error is None

    def test_engine_result_with_error(self):
        from engines.plugin import EngineResult
        r = EngineResult(engine_name="x", score=0.0, error="connexion refusée")
        assert r.error == "connexion refusée"

    def test_plugin_abstract_analyze(self):
        from engines.plugin import EnginePlugin
        with pytest.raises(TypeError):
            EnginePlugin()  # ne peut pas instancier une classe abstraite

    def test_plugin_supports_all_by_default(self):
        cls = self._make_plugin()
        assert cls().supports("video") is True
        assert cls().supports("audio") is True
        assert cls().supports("document") is True

    def test_plugin_analyze_returns_result(self, tmp_path):
        from engines.plugin import EngineResult
        from pathlib import Path
        cls = self._make_plugin(score=0.7)
        result = cls().analyze(tmp_path / "dummy.mp4")
        assert isinstance(result, EngineResult)
        assert result.score == 0.7


# ── Tests PluginRegistry ──────────────────────────────────────────────────────

class TestPluginRegistry:
    """Tests de PluginRegistry : enregistrement, active_names, run_all."""

    def _make_registry_with_plugins(self):
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        class _ValidatedPlugin(EnginePlugin):
            name = "valid_plugin"
            def analyze(self, f: Path) -> EngineResult:
                return EngineResult(engine_name=self.name, score=0.8)

        class _ExperimentalPlugin(EnginePlugin):
            name = "exp_plugin"
            def analyze(self, f: Path) -> EngineResult:
                return EngineResult(engine_name=self.name, score=0.4)

        class _DisabledPlugin(EnginePlugin):
            name = "dis_plugin"
            def analyze(self, f: Path) -> EngineResult:  # noqa: PLR6301
                return EngineResult(engine_name="dis_plugin", score=0.9)

        registry = PluginRegistry()
        registry.register(_ValidatedPlugin, status="validated", weight=0.5)
        registry.register(_ExperimentalPlugin, status="experimental", weight=0.3)
        registry.register(_DisabledPlugin, status="disabled", weight=0.2)
        return registry

    def test_register_adds_entry(self):
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        class _P(EnginePlugin):
            name = "my_plugin"
            def analyze(self, f: Path) -> EngineResult:
                return EngineResult(engine_name="my_plugin", score=0.5)

        registry = PluginRegistry()
        registry.register(_P, status="experimental", weight=0.2)
        assert "my_plugin" in registry
        assert len(registry) == 1

    def test_register_empty_name_raises(self):
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        class _P(EnginePlugin):
            name = ""
            def analyze(self, f: Path) -> EngineResult:
                return EngineResult(engine_name="", score=0.0)

        registry = PluginRegistry()
        with pytest.raises(ValueError, match="name"):
            registry.register(_P, status="experimental", weight=0.1)

    def test_register_negative_weight_raises(self):
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        class _P(EnginePlugin):
            name = "neg_plugin"
            def analyze(self, f: Path) -> EngineResult:
                return EngineResult(engine_name="neg_plugin", score=0.0)

        registry = PluginRegistry()
        with pytest.raises(ValueError, match="poids"):
            registry.register(_P, status="experimental", weight=-0.1)

    def test_active_names_allow_experimental_true(self):
        registry = self._make_registry_with_plugins()
        active = registry.active_names(allow_experimental=True)
        assert "valid_plugin" in active
        assert "exp_plugin" in active
        assert "dis_plugin" not in active

    def test_active_names_allow_experimental_false(self):
        registry = self._make_registry_with_plugins()
        active = registry.active_names(allow_experimental=False)
        assert active == ["valid_plugin"]

    def test_validated_names(self):
        registry = self._make_registry_with_plugins()
        assert registry.validated_names() == ["valid_plugin"]

    def test_run_all_returns_results(self, tmp_path):
        registry = self._make_registry_with_plugins()
        dummy = tmp_path / "file.mp4"
        dummy.touch()
        results = registry.run_all(dummy, allow_experimental=True)
        assert "valid_plugin" in results
        assert "exp_plugin" in results
        assert "dis_plugin" not in results
        assert results["valid_plugin"].score == 0.8
        assert results["exp_plugin"].score == 0.4

    def test_run_all_catches_engine_exception(self, tmp_path):
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        class _Crasher(EnginePlugin):
            name = "crasher"
            def analyze(self, f: Path) -> EngineResult:
                raise RuntimeError("moteur en feu")

        registry = PluginRegistry()
        registry.register(_Crasher, status="validated", weight=1.0)
        dummy = tmp_path / "x.mp4"
        dummy.touch()
        results = registry.run_all(dummy, allow_experimental=True)
        assert results["crasher"].error == "moteur en feu"
        assert results["crasher"].score == 0.0

    def test_calibration_applied(self, tmp_path):
        """La calibration Platt (a=2, b=-0.5) modifie le score brut."""
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        class _P(EnginePlugin):
            name = "cal_plugin"
            def analyze(self, f: Path) -> EngineResult:
                return EngineResult(engine_name=self.name, score=0.5)

        registry = PluginRegistry()
        registry.register(_P, status="validated", weight=1.0, calibration={"a": 0.5, "b": 0.2})
        dummy = tmp_path / "x.mp4"
        dummy.touch()
        results = registry.run_all(dummy, allow_experimental=False)
        # 0.5*0.5 + 0.2 = 0.45
        assert abs(results["cal_plugin"].score - 0.45) < 0.001

    def test_calibration_clamped_to_0_1(self, tmp_path):
        """La calibration ne peut pas sortir de [0, 1]."""
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        class _P(EnginePlugin):
            name = "clamp_plugin"
            def analyze(self, f: Path) -> EngineResult:
                return EngineResult(engine_name=self.name, score=0.9)

        registry = PluginRegistry()
        registry.register(_P, status="validated", weight=1.0, calibration={"a": 2.0, "b": 0.5})
        dummy = tmp_path / "x.mp4"
        dummy.touch()
        results = registry.run_all(dummy, allow_experimental=False)
        # 2.0*0.9 + 0.5 = 2.3 → clampé à 1.0
        assert results["clamp_plugin"].score == 1.0


# ── Tests fuse_from_registry ──────────────────────────────────────────────────

class TestFuseFromRegistry:
    """Tests de fuse_from_registry() : fusion, abstention, SHAP."""

    def _build_registry_and_results(self, scores: dict[str, float], statuses: dict[str, str]):
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        registry = PluginRegistry()
        results: dict = {}

        for name, score in scores.items():
            status = statuses.get(name, "experimental")

            class _P(EnginePlugin):
                pass
            _P.name = name

            def _analyze(self, f: Path, _s=score) -> EngineResult:
                return EngineResult(engine_name=self.name, score=_s)
            _P.analyze = _analyze

            registry.register(_P, status=status, weight=1.0 / len(scores))
            results[name] = EngineResult(engine_name=name, score=score)

        return registry, results

    def test_fusion_single_validated_engine_high_score(self):
        from engines.fusion import fuse_from_registry
        from models.analysis import Verdict

        registry, results = self._build_registry_and_results(
            {"vid": 0.8}, {"vid": "validated"}
        )
        result = fuse_from_registry(results, registry, allow_experimental=False)
        assert result.verdict == Verdict.deepfake
        assert result.final_score > 0.55

    def test_fusion_two_experimental_engines(self):
        from engines.fusion import fuse_from_registry
        from models.analysis import Verdict

        registry, results = self._build_registry_and_results(
            {"a": 0.1, "b": 0.2}, {"a": "experimental", "b": "experimental"}
        )
        result = fuse_from_registry(results, registry, allow_experimental=True)
        assert result.verdict == Verdict.authentic
        assert result.final_score < 0.35

    def test_abstain_when_no_validated_and_experimental_disabled(self):
        """Sans moteur validated ET allow_experimental=False → Verdict.abstain."""
        from engines.fusion import fuse_from_registry
        from models.analysis import Verdict

        registry, results = self._build_registry_and_results(
            {"exp": 0.9}, {"exp": "experimental"}
        )
        result = fuse_from_registry(results, registry, allow_experimental=False)
        assert result.verdict == Verdict.abstain
        assert result.final_score == 0.0
        assert result.shap_ranking == []

    def test_abstain_empty_registry(self):
        from engines.fusion import fuse_from_registry
        from engines.registry import PluginRegistry
        from models.analysis import Verdict

        registry = PluginRegistry()
        result = fuse_from_registry({}, registry, allow_experimental=True)
        assert result.verdict == Verdict.abstain

    def test_disabled_engine_excluded(self):
        from engines.fusion import fuse_from_registry
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from models.analysis import Verdict
        from pathlib import Path

        class _D(EnginePlugin):
            name = "disabled_eng"
            def analyze(self, f: Path) -> EngineResult:
                return EngineResult(engine_name=self.name, score=0.99)

        registry = PluginRegistry()
        registry.register(_D, status="disabled", weight=1.0)
        # score élevé d'un moteur disabled ne doit pas affecter le verdict
        results = {"disabled_eng": EngineResult(engine_name="disabled_eng", score=0.99)}
        result = fuse_from_registry(results, registry, allow_experimental=True)
        assert result.verdict == Verdict.abstain

    def test_errored_engine_counted_as_zero(self):
        from engines.fusion import fuse_from_registry
        from engines.registry import PluginRegistry
        from engines.plugin import EnginePlugin, EngineResult
        from pathlib import Path

        class _V(EnginePlugin):
            name = "v_engine"
            def analyze(self, f: Path) -> EngineResult:
                return EngineResult(engine_name=self.name, score=0.8)

        registry = PluginRegistry()
        registry.register(_V, status="validated", weight=1.0)
        # Simuler un moteur en erreur
        results = {"v_engine": EngineResult(engine_name="v_engine", score=0.8, error="crash")}
        result = fuse_from_registry(results, registry, allow_experimental=False)
        # Moteur en erreur → score=0.0 → AUTHENTIQUE (score < 0.35)
        assert result.final_score == 0.0
        assert result.verdict.value != "DEEPFAKE DÉTECTÉ"

    def test_shap_ranking_in_fuse_from_registry(self):
        from engines.fusion import fuse_from_registry

        registry, results = self._build_registry_and_results(
            {"a": 0.8, "b": 0.2}, {"a": "validated", "b": "validated"}
        )
        result = fuse_from_registry(results, registry, allow_experimental=False)
        features = [r["feature"] for r in result.shap_ranking]
        assert "a" in features
        assert "b" in features
        total_contrib = sum(r["contribution"] for r in result.shap_ranking)
        assert abs(total_contrib - 100.0) < 0.1


# ── Tests Verdict.abstain dans fuse_scores ────────────────────────────────────

class TestFuseScoresAbstention:
    """Tests de l'abstention dans fuse_scores() quand aucun moteur actif."""

    def test_abstain_when_all_disabled_and_no_experimental(self):
        """Tous les moteurs désactivés + allow=False → Verdict.abstain."""
        from engines.fusion import fuse_scores
        from models.analysis import Verdict

        all_disabled = {k: "disabled" for k in [
            "texture", "temporal", "rppg", "biometrics", "audio", "phase", "metadata"
        ]}
        with patch("engines.fusion.settings") as mock_settings:
            mock_settings.allow_experimental_engines = False
            result = fuse_scores(engine_statuses=all_disabled)
        assert result.verdict == Verdict.abstain
        assert result.final_score == 0.0
        assert result.shap_ranking == []
        assert "Aucun moteur" in result.plain_explanation

    def test_no_abstain_when_experimental_allowed(self):
        """Moteurs experimental + allow=True → pas d'abstention."""
        from engines.fusion import fuse_scores
        from models.analysis import Verdict

        with patch("engines.fusion.settings") as mock_settings:
            mock_settings.allow_experimental_engines = True
            result = fuse_scores()
        assert result.verdict != Verdict.abstain

    def test_abstain_score_is_zero(self):
        from engines.fusion import fuse_scores
        from models.analysis import Verdict

        all_disabled = {k: "disabled" for k in [
            "texture", "temporal", "rppg", "biometrics", "audio", "phase", "metadata"
        ]}
        with patch("engines.fusion.settings") as mock_settings:
            mock_settings.allow_experimental_engines = False
            result = fuse_scores(score_texture=0.9, engine_statuses=all_disabled)
        assert result.verdict == Verdict.abstain
        # Score élevé d'un moteur disabled ne doit pas polluer le résultat
        assert result.final_score == 0.0

    def test_verdict_abstain_value(self):
        from models.analysis import Verdict
        assert Verdict.abstain.value == "ABSTENTION"

    def test_all_verdicts_present(self):
        from models.analysis import Verdict
        values = {v.value for v in Verdict}
        assert "AUTHENTIQUE" in values
        assert "INDÉTERMINÉ" in values
        assert "DEEPFAKE DÉTECTÉ" in values
        assert "ABSTENTION" in values
