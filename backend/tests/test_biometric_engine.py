"""Tests — BiometricEngine face matching KYC."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vec(v: list[float]) -> np.ndarray:
    return np.array(v, dtype=float)


def _norm_vec(size: int = 10, value: float = 1.0) -> np.ndarray:
    v = np.ones(size) * value
    return v / np.linalg.norm(v)


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestCosineSimilarity:

    def test_identical_vectors(self):
        from engines.biometric_engine import _cosine_similarity
        v = _norm_vec()
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        from engines.biometric_engine import _cosine_similarity
        v = _norm_vec()
        assert _cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        from engines.biometric_engine import _cosine_similarity
        v = _norm_vec()
        z = np.zeros(10)
        assert _cosine_similarity(v, z) == 0.0

    def test_orthogonal_vectors(self):
        from engines.biometric_engine import _cosine_similarity
        a = _vec([1, 0, 0, 0])
        b = _vec([0, 1, 0, 0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


class TestFaceMatchingEngine:

    def _make_engine(self, backend: str = "insightface") -> object:
        from engines.biometric_engine import FaceMatchingEngine
        eng = FaceMatchingEngine.__new__(FaceMatchingEngine)
        eng._backend = backend
        eng._app = MagicMock() if backend == "insightface" else None
        return eng

    def test_is_available_insightface(self):
        eng = self._make_engine("insightface")
        assert eng.is_available() is True

    def test_is_available_deepface(self):
        eng = self._make_engine("deepface")
        assert eng.is_available() is True

    def test_is_available_unavailable(self):
        eng = self._make_engine("unavailable")
        assert eng.is_available() is False

    def test_compare_unavailable_returns_error(self):
        from engines.biometric_engine import FaceMatchingEngine
        eng = self._make_engine("unavailable")
        result = eng.compare("doc.jpg", "selfie.jpg")
        assert result.is_match is False
        assert result.error == "no_face_backend"
        assert result.backend_used == "unavailable"

    def test_compare_match(self):
        eng = self._make_engine("insightface")
        v = _norm_vec(512)
        with patch.object(eng, "extract_embedding", return_value=v):
            result = eng.compare("doc.jpg", "selfie.jpg")
        assert result.is_match is True
        assert result.similarity == pytest.approx(1.0, abs=1e-5)
        assert result.face_found_doc is True
        assert result.face_found_selfie is True

    def test_compare_no_match(self):
        eng = self._make_engine("insightface")
        v1 = _norm_vec(512, 1.0)
        v2 = -_norm_vec(512, 1.0)
        calls = [v1, v2]
        with patch.object(eng, "extract_embedding", side_effect=calls):
            result = eng.compare("doc.jpg", "selfie.jpg")
        assert result.is_match is False
        assert result.similarity < 0

    def test_no_face_doc(self):
        eng = self._make_engine("insightface")
        calls = [None, _norm_vec(512)]
        with patch.object(eng, "extract_embedding", side_effect=calls):
            result = eng.compare("doc.jpg", "selfie.jpg")
        assert result.face_found_doc is False
        assert result.is_match is False

    def test_no_face_selfie(self):
        eng = self._make_engine("insightface")
        calls = [_norm_vec(512), None]
        with patch.object(eng, "extract_embedding", side_effect=calls):
            result = eng.compare("doc.jpg", "selfie.jpg")
        assert result.face_found_selfie is False
        assert result.is_match is False

    def test_backend_unavailable_when_both_fail(self):
        """Si insightface ET deepface lèvent une exception → backend=unavailable."""
        with (
            patch("engines.biometric_engine._INSIGHTFACE_AVAILABLE", True),
            patch("engines.biometric_engine._DEEPFACE_AVAILABLE", False),
            patch("engines.biometric_engine._FaceAnalysis", side_effect=RuntimeError("no model")),
        ):
            from engines.biometric_engine import FaceMatchingEngine
            eng = FaceMatchingEngine()
            assert eng._backend == "unavailable"

    def test_deepface_fallback_when_insightface_fails(self):
        with (
            patch("engines.biometric_engine._INSIGHTFACE_AVAILABLE", True),
            patch("engines.biometric_engine._DEEPFACE_AVAILABLE", True),
            patch("engines.biometric_engine._FaceAnalysis", side_effect=RuntimeError("fail")),
        ):
            from engines.biometric_engine import FaceMatchingEngine
            eng = FaceMatchingEngine()
            assert eng._backend == "deepface"
