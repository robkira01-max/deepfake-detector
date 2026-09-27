"""FaceMatchingEngine — face matching KYC via insightface (ArcFace) + deepface fallback."""
from __future__ import annotations

import numpy as np
import structlog
from dataclasses import dataclass

log = structlog.get_logger(__name__)

# ── Lazy backend imports ──────────────────────────────────────────────────────
try:
    import insightface
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except Exception:  # noqa: BLE001
    _INSIGHTFACE_AVAILABLE = False

try:
    from deepface import DeepFace as _DeepFace
    _DEEPFACE_AVAILABLE = True
except Exception:  # noqa: BLE001
    _DEEPFACE_AVAILABLE = False

# Seuils de correspondance
_THRESHOLD_INSIGHTFACE = 0.40   # cosine similarity > seuil → match
_THRESHOLD_DEEPFACE    = 0.68   # distance cosine < seuil → match


@dataclass
class FaceMatchResult:
    similarity: float
    is_match: bool
    face_found_doc: bool
    face_found_selfie: bool
    backend_used: str   # "insightface" | "deepface" | "unavailable"
    error: str | None = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class FaceMatchingEngine:
    """Moteur de correspondance faciale KYC.

    Essaie insightface buffalo_s (ArcFace ONNX) en premier ; tombe sur deepface
    si insightface n'est pas disponible ou échoue à charger.
    """

    def __init__(self) -> None:
        self._backend = "unavailable"
        self._app: object | None = None

        if _INSIGHTFACE_AVAILABLE:
            try:
                app = _FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=0, det_size=(640, 640))
                self._app = app
                self._backend = "insightface"
                log.info("biometric_backend_loaded", backend="insightface")
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("insightface_load_failed", error=str(exc))

        if _DEEPFACE_AVAILABLE:
            self._backend = "deepface"
            log.info("biometric_backend_loaded", backend="deepface")
            return

        log.warning("biometric_backend_unavailable")

    def is_available(self) -> bool:
        return self._backend != "unavailable"

    def extract_embedding(self, image_path: str) -> np.ndarray | None:
        """Extrait l'embedding du premier visage détecté (vecteur 512-dim)."""
        if self._backend == "insightface" and self._app is not None:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return None
            faces = self._app.get(img)
            if not faces:
                return None
            return faces[0].embedding

        if self._backend == "deepface":
            try:
                result = _DeepFace.represent(
                    img_path=image_path,
                    model_name="ArcFace",
                    enforce_detection=True,
                    detector_backend="opencv",
                )
                if result:
                    return np.array(result[0]["embedding"])
            except Exception:  # noqa: BLE001
                return None

        return None

    def compare(self, doc_image_path: str, selfie_image_path: str) -> FaceMatchResult:
        """Compare le visage du document avec le selfie."""
        if self._backend == "unavailable":
            return FaceMatchResult(
                similarity=0.0,
                is_match=False,
                face_found_doc=False,
                face_found_selfie=False,
                backend_used="unavailable",
                error="no_face_backend",
            )

        emb_doc = self.extract_embedding(doc_image_path)
        emb_selfie = self.extract_embedding(selfie_image_path)

        face_doc     = emb_doc is not None
        face_selfie  = emb_selfie is not None

        if not face_doc or not face_selfie:
            return FaceMatchResult(
                similarity=0.0,
                is_match=False,
                face_found_doc=face_doc,
                face_found_selfie=face_selfie,
                backend_used=self._backend,
            )

        similarity = _cosine_similarity(emb_doc, emb_selfie)

        if self._backend == "insightface":
            is_match = similarity > _THRESHOLD_INSIGHTFACE
        else:
            # deepface returns cosine distance; similarity here is already 0-1 (we use cosine)
            is_match = similarity > (1.0 - _THRESHOLD_DEEPFACE)

        return FaceMatchResult(
            similarity=similarity,
            is_match=is_match,
            face_found_doc=True,
            face_found_selfie=True,
            backend_used=self._backend,
        )
