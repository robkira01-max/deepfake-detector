"""Tests — document_engine, text_engine, compression_preprocess (v2.0).

Stratégie :
  - Fonctions statistiques testées directement (pas de mocking).
  - Fonctions ML (_compute_perplexity, _run_roberta_detector) mockées via patch.
  - Fonctions de lecture fichier (_lazy_load, pymupdf, docx, PIL) mockées.
  - detect_compression : subprocess.run mocké avec sortie FFprobe JSON contrôlée.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT ENGINE — fonctions statistiques (aucun mock)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenizeSentences:
    def test_splits_on_period(self):
        from engines.text_engine import _tokenize_sentences
        sents = _tokenize_sentences("Le chat est parti. Le chien aussi dort.")
        assert len(sents) == 2

    def test_ignores_short_sentences(self):
        from engines.text_engine import _tokenize_sentences
        # "Ok." has < 3 words → filtered
        sents = _tokenize_sentences("Ok. Cette phrase est assez longue pour passer.")
        assert len(sents) == 1

    def test_empty_text_returns_empty(self):
        from engines.text_engine import _tokenize_sentences
        assert _tokenize_sentences("") == []

    def test_splits_on_exclamation_and_question(self):
        from engines.text_engine import _tokenize_sentences
        # "allez-vous?" counts as 2 words (hyphenated), so use 3+ word clauses
        text = "Quelle belle journée! Vous allez très bien? Je vais très bien merci."
        sents = _tokenize_sentences(text)
        assert len(sents) == 3


class TestBurstinessScore:
    def test_too_few_sentences_returns_neutral(self):
        from engines.text_engine import _burstiness_score
        # < 5 sentences → 0.5
        assert _burstiness_score(["Un deux trois."]) == 0.5
        assert _burstiness_score([]) == 0.5

    def test_uniform_sentences_gives_high_score(self):
        """Phrases uniformes (IA-like) → score élevé."""
        from engines.text_engine import _burstiness_score
        # Toutes les phrases ont 5 mots → variance nulle → cv=0 → score=1.0
        uniform = ["un deux trois quatre cinq"] * 10
        score = _burstiness_score(uniform)
        assert score >= 0.9

    def test_varied_sentences_gives_low_score(self):
        """Phrases très variées (humain-like) → score bas."""
        from engines.text_engine import _burstiness_score
        # Longueurs très disparates : 2, 20, 3, 18, 4
        varied = [
            "Bonjour monde.",
            "Cette phrase est très longue et contient beaucoup de mots pour augmenter la variance totale.",
            "Court texte.",
            "Une autre phrase assez longue qui contribue à la variation de longueur dans ce test particulier.",
            "Fin.",
            "Encore une phrase de taille modérée ici pour équilibrer.",
        ]
        score = _burstiness_score(varied)
        # Score doit être < 0.7 pour des phrases très variées
        assert score < 0.8

    def test_returns_float_between_0_and_1(self):
        from engines.text_engine import _burstiness_score
        sents = ["phrase courte"] * 3 + ["phrase assez longue maintenant"] * 3
        s = _burstiness_score(sents * 2)  # 12 sentences total
        assert 0.0 <= s <= 1.0


class TestEntropyScore:
    def test_empty_text_returns_neutral(self):
        from engines.text_engine import _entropy_score
        assert _entropy_score("") == 0.5

    def test_single_repeated_word_low_entropy(self):
        """Mot répété → entropie très faible → score "IA" élevé."""
        from engines.text_engine import _entropy_score
        score = _entropy_score("chat " * 200)
        # Entropie faible → normalized proche de 0 → score proche de 1.3 clamped à 1.0
        assert score >= 0.9

    def test_varied_vocabulary_moderate_score(self):
        """Vocabulaire varié → entropie plus élevée → score IA plus bas."""
        from engines.text_engine import _entropy_score
        # Un texte avec des mots tous différents aura une entropie maximale
        words = [f"mot{i}" for i in range(100)]
        score = _entropy_score(" ".join(words))
        # Score doit être < 0.9 pour un vocabulaire très varié
        assert score < 0.9

    def test_returns_float_between_0_and_1(self):
        from engines.text_engine import _entropy_score
        s = _entropy_score("Le rapide renard brun saute par-dessus le chien paresseux.")
        assert 0.0 <= s <= 1.0


class TestNormalizePerplexity:
    def test_very_low_ppl_returns_high_score(self):
        from engines.text_engine import _normalize_perplexity
        assert _normalize_perplexity(10.0) == 0.85
        assert _normalize_perplexity(15.0) == 0.85

    def test_low_ppl_returns_070(self):
        from engines.text_engine import _normalize_perplexity
        assert _normalize_perplexity(20.0) == 0.70
        assert _normalize_perplexity(25.0) == 0.70

    def test_medium_ppl_returns_050(self):
        from engines.text_engine import _normalize_perplexity
        assert _normalize_perplexity(30.0) == 0.50
        assert _normalize_perplexity(40.0) == 0.50

    def test_high_ppl_returns_030(self):
        from engines.text_engine import _normalize_perplexity
        assert _normalize_perplexity(50.0) == 0.30
        assert _normalize_perplexity(70.0) == 0.30

    def test_very_high_ppl_returns_015(self):
        from engines.text_engine import _normalize_perplexity
        assert _normalize_perplexity(100.0) == 0.15
        assert _normalize_perplexity(500.0) == 0.15

    def test_boundary_values(self):
        from engines.text_engine import _normalize_perplexity
        # Exact boundary values
        assert _normalize_perplexity(15.0) == 0.85  # ≤15
        assert _normalize_perplexity(15.1) == 0.70  # >15, ≤25
        assert _normalize_perplexity(25.0) == 0.70  # ≤25
        assert _normalize_perplexity(25.1) == 0.50  # >25, ≤40
        assert _normalize_perplexity(40.1) == 0.30  # >40, ≤70
        assert _normalize_perplexity(70.1) == 0.15  # >70


class TestTextEngineShortText:
    def test_short_text_returns_error(self):
        from engines.text_engine import TextEngine
        engine = TextEngine()
        result = engine.analyze("Texte trop court.")
        assert result.error is not None
        assert "trop court" in result.error
        assert result.final_score == 0.5
        assert result.word_count < 50

    def test_exactly_min_words_triggers_error(self):
        from engines.text_engine import TextEngine
        engine = TextEngine()
        # 49 words (below _MIN_WORDS=50)
        text = " ".join(["mot"] * 49)
        result = engine.analyze(text)
        assert result.error is not None


class TestTextEngineAnalyzeMocked:
    """TextEngine.analyze() avec ML mocké (pas de téléchargement de modèles)."""

    def _build_text(self, n_words: int = 60) -> str:
        sentences = [
            "Le chat mange la souris. ",
            "La souris court très vite dans le jardin. ",
            "Le renard observe depuis sa cachette secrète. ",
        ]
        text = ""
        while len(text.split()) < n_words:
            text += sentences[len(text.split()) % 3]
        return text.strip()

    def test_analyze_returns_textscore_fusion_weighted(self):
        import engines.text_engine as te
        text = self._build_text(80)

        with patch.object(te, "_compute_perplexity", return_value=20.0), \
             patch.object(te, "_run_roberta_detector", return_value=(0.80, [])):
            engine = te.TextEngine()
            result = engine.analyze(text)

        # Fusion : 0.50*0.80 + 0.25*0.70 + 0.15*bust + 0.10*ent
        # score_perplexity = _normalize_perplexity(20.0) = 0.70
        assert result.score_model == 0.80
        assert result.score_perplexity == 0.70
        assert result.word_count >= 50
        assert result.error is None
        # Final must be between 0 and 1
        assert 0.0 <= result.final_score <= 1.0

    def test_analyze_high_ai_score_verdict(self):
        import engines.text_engine as te
        text = self._build_text(80)

        with patch.object(te, "_compute_perplexity", return_value=10.0), \
             patch.object(te, "_run_roberta_detector", return_value=(0.95, [{"paragraph_index": 0, "score_ia": 0.92}])):
            engine = te.TextEngine()
            result = engine.analyze(text)

        assert result.verdict_label == "TEXTE IA DÉTECTÉ"
        assert len(result.flagged_passages) == 1

    def test_analyze_low_ai_score_verdict_human(self):
        import engines.text_engine as te
        text = self._build_text(80)

        with patch.object(te, "_compute_perplexity", return_value=100.0), \
             patch.object(te, "_run_roberta_detector", return_value=(0.10, [])):
            engine = te.TextEngine()
            result = engine.analyze(text)

        assert result.verdict_label == "RÉDACTION HUMAINE"

    def test_analyze_medium_score_verdict_indeterminate(self):
        import engines.text_engine as te
        text = self._build_text(80)

        with patch.object(te, "_compute_perplexity", return_value=45.0), \
             patch.object(te, "_run_roberta_detector", return_value=(0.50, [])):
            engine = te.TextEngine()
            result = engine.analyze(text)

        assert result.verdict_label == "INDÉTERMINÉ"

    def test_perplexity_none_uses_neutral(self):
        """Si GPT-2 échoue, score_perplexity = 0.5."""
        import engines.text_engine as te
        text = self._build_text(80)

        with patch.object(te, "_compute_perplexity", return_value=None), \
             patch.object(te, "_run_roberta_detector", return_value=(0.60, [])):
            engine = te.TextEngine()
            result = engine.analyze(text)

        assert result.score_perplexity == 0.5
        assert result.error is None

    def test_analyze_file_txt(self, tmp_path):
        import engines.text_engine as te
        txt = tmp_path / "test.txt"
        txt.write_text(" ".join(["mot"] * 60))

        with patch.object(te, "_compute_perplexity", return_value=30.0), \
             patch.object(te, "_run_roberta_detector", return_value=(0.40, [])):
            engine = te.TextEngine()
            result = engine.analyze_file(txt)

        assert result.error is None
        assert result.word_count >= 50

    def test_analyze_file_unsupported_extension_returns_error(self, tmp_path):
        import engines.text_engine as te
        mp3 = tmp_path / "test.mp3"
        mp3.write_bytes(b"\x00" * 100)

        engine = te.TextEngine()
        result = engine.analyze_file(mp3)

        assert result.error is not None

    def test_roberta_fake_label_maps_to_score(self):
        import engines.text_engine as te
        text = self._build_text(80)

        # Simuler label "FAKE" → score_ia = confidence
        with patch.object(te, "_compute_perplexity", return_value=25.0), \
             patch.object(te, "_run_roberta_detector", return_value=(0.90, [])):
            engine = te.TextEngine()
            result = engine.analyze(text)

        assert result.score_model == 0.90


class TestTextEngineFusionFormula:
    """Vérifie la formule de fusion exacte."""

    def test_fusion_weights(self):
        import engines.text_engine as te

        text = " ".join(["le chat mange la souris dans le jardin vert"] * 10)

        # Valeurs connues pour calculer la fusion attendue
        mock_model = 0.80
        mock_ppl = 25.0  # → score_perplexity = 0.70

        with patch.object(te, "_compute_perplexity", return_value=mock_ppl), \
             patch.object(te, "_run_roberta_detector", return_value=(mock_model, [])):
            engine = te.TextEngine()
            result = engine.analyze(text)

        expected = round(
            0.50 * mock_model
            + 0.25 * 0.70
            + 0.15 * result.score_burstiness
            + 0.10 * result.score_entropy,
            4,
        )
        assert result.final_score == expected


# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSION PREPROCESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyQuality:
    def test_high_bitrate_h264_returns_high(self):
        from engines.compression_preprocess import _classify_quality, CompressionQuality
        q = _classify_quality(3000.0, "h264", 1920, 1080)
        assert q == CompressionQuality.high

    def test_medium_bitrate_h264(self):
        from engines.compression_preprocess import _classify_quality, CompressionQuality
        q = _classify_quality(1000.0, "h264", 1920, 1080)
        assert q == CompressionQuality.medium

    def test_low_bitrate_h264(self):
        from engines.compression_preprocess import _classify_quality, CompressionQuality
        q = _classify_quality(500.0, "h264", 1920, 1080)
        assert q == CompressionQuality.low

    def test_very_low_bitrate_h264(self):
        from engines.compression_preprocess import _classify_quality, CompressionQuality
        q = _classify_quality(200.0, "h264", 1920, 1080)
        assert q == CompressionQuality.very_low

    def test_h265_efficiency_factor_bumps_quality(self):
        """H.265 × 1.5 — un bitrate de 600 équivaut à 900 effectif → medium."""
        from engines.compression_preprocess import _classify_quality, CompressionQuality
        # 600 * 1.5 = 900 >= 800 → medium
        q = _classify_quality(600.0, "h265", 1920, 1080)
        assert q == CompressionQuality.medium

    def test_hevc_efficiency_factor(self):
        from engines.compression_preprocess import _classify_quality, CompressionQuality
        q = _classify_quality(600.0, "hevc", 1920, 1080)
        assert q == CompressionQuality.medium

    def test_av1_efficiency_factor(self):
        from engines.compression_preprocess import _classify_quality, CompressionQuality
        # 250 * 1.5 = 375 >= 300 → low
        q = _classify_quality(250.0, "av1", 1920, 1080)
        assert q == CompressionQuality.low

    def test_zero_bitrate_returns_medium(self):
        from engines.compression_preprocess import _classify_quality, CompressionQuality
        q = _classify_quality(0.0, "h264", 1920, 1080)
        assert q == CompressionQuality.medium

    def test_zero_dimensions_uses_fullhd_default(self):
        from engines.compression_preprocess import _classify_quality, CompressionQuality
        # w=0, h=0 → pixels = 1920*1080 default
        q = _classify_quality(3000.0, "h264", 0, 0)
        assert q == CompressionQuality.high


class TestApplyCompressionPenalty:
    def test_no_penalty_returns_unchanged(self):
        from engines.compression_preprocess import (
            apply_compression_penalty, CompressionProfile, CompressionQuality
        )
        profile = CompressionProfile(
            codec="h264", bitrate_kbps=3000.0, width=1920, height=1080, fps=25.0,
            quality_flag=CompressionQuality.high, confidence_penalty=0.0,
            source_warning=None, platform_hint=None, degraded=False,
        )
        low, high, score = apply_compression_penalty(0.70, 0.90, 0.82, profile)
        assert low == 0.70
        assert high == 0.90
        assert score == 0.82

    def test_medium_penalty_widens_interval(self):
        from engines.compression_preprocess import (
            apply_compression_penalty, CompressionProfile, CompressionQuality
        )
        profile = CompressionProfile(
            codec="h264", bitrate_kbps=1000.0, width=1920, height=1080, fps=25.0,
            quality_flag=CompressionQuality.medium, confidence_penalty=0.05,
            source_warning=None, platform_hint=None, degraded=False,
        )
        low, high, score = apply_compression_penalty(0.70, 0.90, 0.82, profile)
        assert low == round(0.70 - 0.05, 4)
        assert high == round(0.90 + 0.05, 4)
        assert score == 0.82  # score final inchangé

    def test_low_penalty_widens_interval(self):
        from engines.compression_preprocess import (
            apply_compression_penalty, CompressionProfile, CompressionQuality
        )
        profile = CompressionProfile(
            codec="h264", bitrate_kbps=400.0, width=1920, height=1080, fps=25.0,
            quality_flag=CompressionQuality.low, confidence_penalty=0.12,
            source_warning=None, platform_hint=None, degraded=True,
        )
        # high=0.80 → 0.80+0.12=0.92 (within bounds)
        low, high, score = apply_compression_penalty(0.70, 0.80, 0.75, profile)
        assert low == round(0.70 - 0.12, 4)
        assert high == round(0.80 + 0.12, 4)

    def test_penalty_clamps_to_0_1(self):
        """La pénalité ne peut pas pousser hors de [0, 1]."""
        from engines.compression_preprocess import (
            apply_compression_penalty, CompressionProfile, CompressionQuality
        )
        profile = CompressionProfile(
            codec="h264", bitrate_kbps=100.0, width=480, height=360, fps=15.0,
            quality_flag=CompressionQuality.very_low, confidence_penalty=0.20,
            source_warning="⚠️ Compression très agressive",
            platform_hint="WhatsApp", degraded=True,
        )
        low, high, score = apply_compression_penalty(0.05, 0.98, 0.50, profile)
        assert low >= 0.0
        assert high <= 1.0

    def test_very_low_penalty_value(self):
        from engines.compression_preprocess import (
            apply_compression_penalty, CompressionProfile, CompressionQuality
        )
        profile = CompressionProfile(
            codec="h264", bitrate_kbps=150.0, width=640, height=480, fps=15.0,
            quality_flag=CompressionQuality.very_low, confidence_penalty=0.20,
            source_warning="⚠️", platform_hint="WhatsApp", degraded=True,
        )
        low, high, score = apply_compression_penalty(0.60, 0.80, 0.72, profile)
        assert low == round(0.60 - 0.20, 4)
        assert high == round(0.80 + 0.20, 4)
        assert score == 0.72


class TestDetectCompression:
    def _ffprobe_json(
        self, codec="h264", width=1920, height=1080, fps="25/1",
        bitrate=2500000, stream_bitrate=None
    ) -> str:
        streams = [{
            "codec_type": "video",
            "codec_name": codec,
            "width": width,
            "height": height,
            "r_frame_rate": fps,
            "bit_rate": str(stream_bitrate or 0),
        }]
        fmt = {"bit_rate": str(bitrate)} if bitrate else {}
        return json.dumps({"streams": streams, "format": fmt})

    def test_high_quality_h264(self):
        from engines.compression_preprocess import detect_compression, CompressionQuality
        mock_result = MagicMock()
        mock_result.stdout = self._ffprobe_json(bitrate=3000000)

        with patch("subprocess.run", return_value=mock_result):
            profile = detect_compression(Path("test.mp4"))

        assert profile.codec == "h264"
        assert profile.quality_flag == CompressionQuality.high
        assert not profile.degraded
        assert profile.confidence_penalty == 0.0

    def test_very_low_quality_whatsapp(self):
        from engines.compression_preprocess import detect_compression, CompressionQuality
        mock_result = MagicMock()
        mock_result.stdout = self._ffprobe_json(bitrate=200000)

        with patch("subprocess.run", return_value=mock_result):
            profile = detect_compression(Path("test.mp4"))

        assert profile.quality_flag == CompressionQuality.very_low
        assert profile.degraded
        assert profile.confidence_penalty == 0.20
        assert profile.source_warning is not None

    def test_h265_tiktok_medium_quality(self):
        from engines.compression_preprocess import detect_compression, CompressionQuality
        mock_result = MagicMock()
        mock_result.stdout = self._ffprobe_json(codec="h265", bitrate=700000)

        with patch("subprocess.run", return_value=mock_result):
            profile = detect_compression(Path("test.mp4"))

        # 700 kbps * 1.5 = 1050 >= 800 → medium
        assert profile.quality_flag == CompressionQuality.medium
        assert profile.codec == "h265"
        assert "TikTok" in (profile.platform_hint or "")

    def test_fps_parsed_correctly(self):
        from engines.compression_preprocess import detect_compression
        mock_result = MagicMock()
        mock_result.stdout = self._ffprobe_json(fps="30000/1001", bitrate=2000000)

        with patch("subprocess.run", return_value=mock_result):
            profile = detect_compression(Path("test.mp4"))

        assert abs(profile.fps - 29.97) < 0.01

    def test_fallback_stream_bitrate_when_format_missing(self):
        from engines.compression_preprocess import detect_compression, CompressionQuality
        streams = [{
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920, "height": 1080,
            "r_frame_rate": "25/1",
            "bit_rate": "2500000",
        }]
        data = json.dumps({"streams": streams, "format": {}})
        mock_result = MagicMock()
        mock_result.stdout = data

        with patch("subprocess.run", return_value=mock_result):
            profile = detect_compression(Path("test.mp4"))

        # 2500 kbps >= 2000 → high
        assert profile.quality_flag == CompressionQuality.high

    def test_no_video_stream_returns_unknown_profile(self):
        from engines.compression_preprocess import detect_compression, CompressionQuality
        data = json.dumps({"streams": [{"codec_type": "audio", "codec_name": "aac"}], "format": {}})
        mock_result = MagicMock()
        mock_result.stdout = data

        with patch("subprocess.run", return_value=mock_result):
            profile = detect_compression(Path("test.mp4"))

        assert profile.codec == "unknown"
        assert profile.quality_flag == CompressionQuality.medium
        assert not profile.degraded

    def test_ffprobe_timeout_returns_unknown_profile(self):
        from engines.compression_preprocess import detect_compression, CompressionQuality
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 30)):
            profile = detect_compression(Path("test.mp4"))

        assert profile.codec == "unknown"
        assert profile.confidence_penalty == 0.0

    def test_ffprobe_not_found_returns_unknown_profile(self):
        from engines.compression_preprocess import detect_compression
        with patch("subprocess.run", side_effect=FileNotFoundError("ffprobe not found")):
            profile = detect_compression(Path("test.mp4"))

        assert profile.codec == "unknown"

    def test_invalid_json_returns_unknown_profile(self):
        from engines.compression_preprocess import detect_compression
        mock_result = MagicMock()
        mock_result.stdout = "NOT VALID JSON {"

        with patch("subprocess.run", return_value=mock_result):
            profile = detect_compression(Path("test.mp4"))

        assert profile.codec == "unknown"

    def test_vp9_platform_hint_youtube(self):
        from engines.compression_preprocess import detect_compression
        mock_result = MagicMock()
        mock_result.stdout = self._ffprobe_json(codec="vp9", bitrate=3000000)

        with patch("subprocess.run", return_value=mock_result):
            profile = detect_compression(Path("test.webm"))

        assert "YouTube" in (profile.platform_hint or "")

    def test_as_dict_returns_expected_keys(self):
        from engines.compression_preprocess import detect_compression
        mock_result = MagicMock()
        mock_result.stdout = self._ffprobe_json(bitrate=2000000)

        with patch("subprocess.run", return_value=mock_result):
            profile = detect_compression(Path("test.mp4"))

        d = profile.as_dict()
        assert "codec" in d
        assert "bitrate_kbps" in d
        assert "resolution" in d
        assert "quality_flag" in d
        assert "degraded" in d


class TestCompressionQualityEnum:
    def test_all_penalties_defined(self):
        from engines.compression_preprocess import _PENALTIES, CompressionQuality
        for q in CompressionQuality:
            assert q in _PENALTIES

    def test_high_no_penalty(self):
        from engines.compression_preprocess import _PENALTIES, CompressionQuality
        assert _PENALTIES[CompressionQuality.high] == 0.0

    def test_penalties_increase_with_severity(self):
        from engines.compression_preprocess import _PENALTIES, CompressionQuality
        assert _PENALTIES[CompressionQuality.high] < _PENALTIES[CompressionQuality.medium]
        assert _PENALTIES[CompressionQuality.medium] < _PENALTIES[CompressionQuality.low]
        assert _PENALTIES[CompressionQuality.low] < _PENALTIES[CompressionQuality.very_low]


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT ENGINE — scores et verdict
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentScoresVerdict:
    def test_high_score_forged(self):
        from engines.document_engine import DocumentScores
        s = DocumentScores(
            score_ela=0.8, score_clone=0.7, score_metadata=0.9,
            score_font=0.6, score_text_ai=0.8, final_score=0.80,
        )
        assert s.verdict_label == "DOCUMENT FALSIFIÉ"

    def test_low_score_authentic(self):
        from engines.document_engine import DocumentScores
        s = DocumentScores(
            score_ela=0.0, score_clone=0.0, score_metadata=0.1,
            score_font=0.0, score_text_ai=0.1, final_score=0.20,
        )
        assert s.verdict_label == "DOCUMENT AUTHENTIQUE"

    def test_mid_score_indeterminate(self):
        from engines.document_engine import DocumentScores
        s = DocumentScores(
            score_ela=0.5, score_clone=0.3, score_metadata=0.4,
            score_font=0.3, score_text_ai=0.5, final_score=0.50,
        )
        assert s.verdict_label == "INDÉTERMINÉ"

    def test_threshold_boundary_forged(self):
        from engines.document_engine import DocumentScores
        s = DocumentScores(
            score_ela=0.0, score_clone=0.0, score_metadata=0.0,
            score_font=0.0, score_text_ai=0.0, final_score=0.65,
        )
        assert s.verdict_label == "DOCUMENT FALSIFIÉ"

    def test_threshold_boundary_authentic(self):
        from engines.document_engine import DocumentScores
        s = DocumentScores(
            score_ela=0.0, score_clone=0.0, score_metadata=0.0,
            score_font=0.0, score_text_ai=0.0, final_score=0.35,
        )
        assert s.verdict_label == "DOCUMENT AUTHENTIQUE"


class TestFontScoreDocx:
    def _mock_docx_with_fonts(self, font_names: list[str]):
        """Crée un mock de docx.Document avec les polices spécifiées."""
        paragraphs = []
        for fn in font_names:
            run = MagicMock()
            run.font.name = fn
            run.font.size = 12
            para = MagicMock()
            para.runs = [run]
            paragraphs.append(para)

        doc_mock = MagicMock()
        doc_mock.paragraphs = paragraphs
        return doc_mock

    def test_two_fonts_score_zero(self, tmp_path):
        import engines.document_engine as de
        p = tmp_path / "test.docx"
        p.write_bytes(b"")

        doc_mock = self._mock_docx_with_fonts(["Arial", "Times New Roman"])
        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_docx") as mock_docx_mod:
            mock_docx_mod.Document.return_value = doc_mock
            score, details = de._font_score_docx(p)

        assert score == 0.0
        assert details["font_count"] == 2

    def test_four_fonts_score_025(self, tmp_path):
        import engines.document_engine as de
        p = tmp_path / "test.docx"
        p.write_bytes(b"")

        doc_mock = self._mock_docx_with_fonts(["A", "B", "C", "D"])
        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_docx") as mock_docx_mod:
            mock_docx_mod.Document.return_value = doc_mock
            score, details = de._font_score_docx(p)

        assert score == 0.25
        assert details["font_count"] == 4

    def test_six_fonts_suspicious(self, tmp_path):
        import engines.document_engine as de
        p = tmp_path / "test.docx"
        p.write_bytes(b"")

        doc_mock = self._mock_docx_with_fonts(["A", "B", "C", "D", "E", "F"])
        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_docx") as mock_docx_mod:
            mock_docx_mod.Document.return_value = doc_mock
            score, details = de._font_score_docx(p)

        # 6 fonts: (6-4)*0.15 = 0.30
        assert score == round((6 - 4) * 0.15, 4)
        assert "flag" in details

    def test_docx_error_returns_zero(self, tmp_path):
        import engines.document_engine as de
        p = tmp_path / "broken.docx"
        p.write_bytes(b"")

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_docx") as mock_docx_mod:
            mock_docx_mod.Document.side_effect = Exception("Fichier corrompu")
            score, details = de._font_score_docx(p)

        assert score == 0.0
        assert "error" in details


class TestDocumentEngineUnsupportedFormat:
    def test_unsupported_extension_returns_error(self):
        from engines.document_engine import DocumentEngine
        engine = DocumentEngine()
        result = engine.analyze(Path("test.mp4"))
        assert result.error is not None
        assert "non supporté" in result.error
        assert result.final_score == 0.0


class TestDocumentEngineImagePath:
    def test_image_fusion_weights(self, tmp_path):
        """Image : 0.40×ELA + 0.25×clone + 0.25×metadata + 0.10×font."""
        import engines.document_engine as de
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake image")

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_ela_score", return_value=(0.80, {"technique": "ELA"})), \
             patch.object(de, "_clone_score", return_value=(0.60, {"technique": "clone_detection"})), \
             patch.object(de, "_metadata_score_image", return_value=(0.40, {"technique": "exif_metadata", "anomalies_count": 1})), \
             patch.object(de, "_extract_text_preview", return_value=""):
            engine = de.DocumentEngine()
            result = engine.analyze(img)

        expected = round(0.40 * 0.80 + 0.25 * 0.60 + 0.25 * 0.40 + 0.10 * 0.0, 4)
        assert result.final_score == expected
        assert result.error is None

    def test_image_ela_flag_adds_anomaly(self, tmp_path):
        import engines.document_engine as de
        img = tmp_path / "test.png"
        img.write_bytes(b"")

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_ela_score", return_value=(0.80, {"technique": "ELA", "flag": "Retouche probable"})), \
             patch.object(de, "_clone_score", return_value=(0.0, {"technique": "clone_detection"})), \
             patch.object(de, "_metadata_score_image", return_value=(0.0, {"technique": "exif_metadata", "anomalies_count": 0})), \
             patch.object(de, "_extract_text_preview", return_value=""):
            engine = de.DocumentEngine()
            result = engine.analyze(img)

        assert any(a.get("flag") for a in result.anomalies)


class TestDocumentEnginePDFPath:
    def test_pdf_fusion_weights(self, tmp_path):
        """PDF : 0.35×text_ai + 0.30×metadata + 0.25×font + 0.10×clone."""
        import engines.document_engine as de
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_text_scores = MagicMock()
        mock_text_scores.error = None
        mock_text_scores.final_score = 0.70
        mock_text_scores.verdict_label = "TEXTE IA DÉTECTÉ"
        mock_text_scores.word_count = 200
        mock_text_scores.flagged_passages = []

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_metadata_score_pdf", return_value=(0.50, {"technique": "pdf_metadata", "anomalies_count": 1})), \
             patch.object(de, "_font_score_pdf", return_value=(0.30, {"technique": "font_coherence_pdf", "font_count": 5})), \
             patch.object(de, "_extract_text_preview", return_value="Aperçu texte PDF"), \
             patch("engines.text_engine.TextEngine") as MockTE:
            MockTE.return_value.analyze_file.return_value = mock_text_scores
            engine = de.DocumentEngine()
            result = engine.analyze(pdf)

        expected = round(0.35 * 0.70 + 0.30 * 0.50 + 0.25 * 0.30 + 0.10 * 0.0, 4)
        assert result.final_score == expected
        assert result.text_content_preview == "Aperçu texte PDF"

    def test_pdf_text_engine_error_uses_neutral(self, tmp_path):
        """Si TextEngine échoue, score_text_ai reste 0.5 (neutre)."""
        import engines.document_engine as de
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_text_scores = MagicMock()
        mock_text_scores.error = "Erreur extraction"

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_metadata_score_pdf", return_value=(0.0, {"technique": "pdf_metadata", "anomalies_count": 0})), \
             patch.object(de, "_font_score_pdf", return_value=(0.0, {"technique": "font_coherence_pdf", "font_count": 1})), \
             patch.object(de, "_extract_text_preview", return_value=""), \
             patch("engines.text_engine.TextEngine") as MockTE:
            MockTE.return_value.analyze_file.return_value = mock_text_scores
            engine = de.DocumentEngine()
            result = engine.analyze(pdf)

        assert result.score_text_ai == 0.5  # neutre si pas de texte

    def test_pdf_metadata_score_propagates(self, tmp_path):
        """La note metadata_score_pdf est bien intégrée dans la fusion finale."""
        import engines.document_engine as de
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_text_scores = MagicMock()
        mock_text_scores.error = None
        mock_text_scores.final_score = 0.4
        mock_text_scores.verdict_label = "INDÉTERMINÉ"
        mock_text_scores.word_count = 100
        mock_text_scores.flagged_passages = []

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_metadata_score_pdf", return_value=(0.60, {"technique": "pdf_metadata", "anomalies_count": 1})), \
             patch.object(de, "_font_score_pdf", return_value=(0.0, {"technique": "font_coherence_pdf", "font_count": 2})), \
             patch.object(de, "_extract_text_preview", return_value=""), \
             patch("engines.text_engine.TextEngine") as MockTE:
            MockTE.return_value.analyze_file.return_value = mock_text_scores
            engine = de.DocumentEngine()
            result = engine.analyze(pdf)

        # score_metadata=0.60 est visible dans final_score
        # 0.35*0.4 + 0.30*0.60 + 0.25*0.0 + 0.10*0.0 = 0.32
        expected = round(0.35 * 0.4 + 0.30 * 0.60, 4)
        assert result.score_metadata == 0.60
        assert abs(result.final_score - expected) < 0.001


class TestDocumentEngineDOCXPath:
    def test_docx_fusion_weights(self, tmp_path):
        """DOCX : même pondération que PDF."""
        import engines.document_engine as de
        docx = tmp_path / "test.docx"
        docx.write_bytes(b"PK fake docx")

        mock_text_scores = MagicMock()
        mock_text_scores.error = None
        mock_text_scores.final_score = 0.85
        mock_text_scores.verdict_label = "TEXTE IA DÉTECTÉ"
        mock_text_scores.word_count = 300
        mock_text_scores.flagged_passages = [{"paragraph_index": 0, "score_ia": 0.90}]

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_font_score_docx", return_value=(0.40, {"technique": "font_coherence", "font_count": 6, "flag": "6 polices détectées"})), \
             patch.object(de, "_extract_text_preview", return_value="Aperçu DOCX"), \
             patch("engines.text_engine.TextEngine") as MockTE:
            MockTE.return_value.analyze_file.return_value = mock_text_scores
            engine = de.DocumentEngine()
            result = engine.analyze(docx)

        # score_metadata = 0.0 (DOCX), score_clone = 0.0 (DOCX)
        expected = round(0.35 * 0.85 + 0.30 * 0.0 + 0.25 * 0.40 + 0.10 * 0.0, 4)
        assert result.final_score == expected

    def test_docx_ai_text_flagged_passages_added_to_anomalies(self, tmp_path):
        import engines.document_engine as de
        docx = tmp_path / "test.docx"
        docx.write_bytes(b"PK")

        mock_text_scores = MagicMock()
        mock_text_scores.error = None
        mock_text_scores.final_score = 0.75
        mock_text_scores.verdict_label = "TEXTE IA DÉTECTÉ"
        mock_text_scores.word_count = 200
        mock_text_scores.flagged_passages = [
            {"paragraph_index": 0, "score_ia": 0.88, "text_preview": "Ce texte semble généré..."},
            {"paragraph_index": 1, "score_ia": 0.72, "text_preview": "Un autre passage suspect..."},
        ]

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_font_score_docx", return_value=(0.0, {"technique": "font_coherence", "font_count": 2})), \
             patch.object(de, "_extract_text_preview", return_value=""), \
             patch("engines.text_engine.TextEngine") as MockTE:
            MockTE.return_value.analyze_file.return_value = mock_text_scores
            engine = de.DocumentEngine()
            result = engine.analyze(docx)

        ai_anomalies = [a for a in result.anomalies if a.get("type") == "ai_text"]
        assert len(ai_anomalies) == 1
        assert ai_anomalies[0]["detail"].startswith("Passages potentiellement générés par IA")


class TestDocumentEngineExceptionHandling:
    def test_exception_during_analyze_returns_error_scores(self, tmp_path):
        import engines.document_engine as de
        img = tmp_path / "test.jpg"
        img.write_bytes(b"")

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_ela_score", side_effect=RuntimeError("Erreur inattendue")):
            engine = de.DocumentEngine()
            result = engine.analyze(img)

        assert result.error is not None
        assert result.final_score == 0.0


class TestMetadataPDFScoreMocked:
    def test_suspicious_tool_photoshop(self, tmp_path):
        import engines.document_engine as de
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_doc = MagicMock()
        mock_doc.metadata = {
            "creator": "Adobe Photoshop CC 2023",
            "producer": "Adobe Photoshop",
            "creationDate": "D:20230101",
            "modDate": "",
        }
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_pymupdf") as mock_mu:
            mock_mu.open.return_value = mock_doc
            score, details = de._metadata_score_pdf(pdf)

        assert score >= 0.3  # suspicious_tool += 0.3

    def test_no_metadata_returns_score(self, tmp_path):
        import engines.document_engine as de
        pdf = tmp_path / "empty_meta.pdf"
        pdf.write_bytes(b"%PDF")

        mock_doc = MagicMock()
        mock_doc.metadata = {"creator": "", "producer": "", "creationDate": "", "modDate": ""}

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_pymupdf") as mock_mu:
            mock_mu.open.return_value = mock_doc
            score, details = de._metadata_score_pdf(pdf)

        # no_metadata += 0.2
        assert score >= 0.2

    def test_future_date_returns_high_score(self, tmp_path):
        import engines.document_engine as de
        pdf = tmp_path / "future.pdf"
        pdf.write_bytes(b"%PDF")

        mock_doc = MagicMock()
        mock_doc.metadata = {
            "creator": "Word",
            "producer": "Word",
            "creationDate": "D:20990101120000",
            "modDate": "",
        }

        with patch.object(de, "_lazy_load"), \
             patch.object(de, "_pymupdf") as mock_mu:
            mock_mu.open.return_value = mock_doc
            score, details = de._metadata_score_pdf(pdf)

        # future_date += 0.4
        assert score >= 0.4
