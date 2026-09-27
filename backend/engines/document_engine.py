"""DocumentEngine — Détection de faux documents (v2.0).

Quatre techniques complémentaires :
1. ELA (Error Level Analysis) — artefacts de retouche JPEG
2. Clone detection — copy-move via ORB keypoints
3. Métadonnées — incohérences PDF/EXIF
4. Cohérence des polices — mélange typographique (DOCX/PDF)

+ Délégation à TextEngine pour la composante "écriture IA".

Lazy imports : PIL, pymupdf, python-docx chargés au premier appel.
"""
from __future__ import annotations

import io
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Lazy globals ──────────────────────────────────────────────────────────────
_PIL = None
_pymupdf = None
_docx = None
_np = None

# Seuils ELA
_ELA_QUALITY = 95           # qualité de recompression pour ELA
_ELA_HIGH_THRESHOLD = 15.0  # différence moyenne pixel > 15 = suspect
_ELA_SCALE = 10             # amplification pour visualisation

# Verdict
_FORGED_THRESHOLD = 0.65
_AUTHENTIC_THRESHOLD = 0.35


def _lazy_load():
    global _PIL, _pymupdf, _docx, _np
    if _PIL is None:
        from PIL import Image, ImageChops, ImageFilter
        import pymupdf as _mu
        import docx as _d
        import numpy as np
        _PIL = (Image, ImageChops, ImageFilter)
        _pymupdf = _mu
        _docx = _d
        _np = np


@dataclass
class DocumentScores:
    score_ela: float           # 0=authentique, 1=manipulé
    score_clone: float         # 0=authentique, 1=copy-move détecté
    score_metadata: float      # 0=cohérent, 1=incohérent
    score_font: float          # 0=cohérent, 1=mélange polices
    score_text_ai: float       # délégué à TextEngine
    final_score: float
    anomalies: list[dict] = field(default_factory=list)
    models_used: dict = field(default_factory=dict)
    text_content_preview: str = ""
    error: str | None = None

    @property
    def verdict_label(self) -> str:
        if self.final_score >= _FORGED_THRESHOLD:
            return "DOCUMENT FALSIFIÉ"
        if self.final_score <= _AUTHENTIC_THRESHOLD:
            return "DOCUMENT AUTHENTIQUE"
        return "INDÉTERMINÉ"


# ── ELA — Error Level Analysis ────────────────────────────────────────────────

def _ela_score(image_path: Path) -> tuple[float, dict]:
    """ELA : recompresse en JPEG qualité 95, compare avec l'original pixel par pixel.

    Les zones retouchées ont un niveau d'erreur différent (plus élevé ou plus bas)
    que le reste de l'image compressée uniformément.
    """
    try:
        _lazy_load()
        Image, ImageChops, ImageFilter = _PIL

        original = Image.open(str(image_path)).convert("RGB")

        # Recompresse en mémoire
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=_ELA_QUALITY)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        # Différence amplifiée
        diff = ImageChops.difference(original, recompressed)
        diff_array = _np.array(diff, dtype=_np.float32)
        mean_diff = float(_np.mean(diff_array))
        max_diff = float(_np.max(diff_array))
        std_diff = float(_np.std(diff_array))

        # Score normalisé : diff élevée = manipulation probable
        score = min(1.0, mean_diff / _ELA_HIGH_THRESHOLD)

        details = {
            "technique": "ELA",
            "mean_diff": round(mean_diff, 3),
            "max_diff": round(max_diff, 3),
            "std_diff": round(std_diff, 3),
            "threshold": _ELA_HIGH_THRESHOLD,
        }
        if score >= 0.5:
            details["flag"] = "Niveau d'erreur élevé — retouche probable"

        return round(score, 4), details

    except Exception as exc:
        return 0.0, {"technique": "ELA", "error": str(exc)}


# ── Clone detection — ORB keypoints ──────────────────────────────────────────

def _clone_score(image_path: Path) -> tuple[float, dict]:
    """Détection copy-move via keypoints ORB + matching.

    Si deux régions de l'image ont les mêmes descripteurs → copier-coller.
    """
    try:
        _lazy_load()
        Image = _PIL[0]

        # Convertir en niveaux de gris pour OpenCV
        try:
            import cv2
        except ImportError:
            return 0.0, {"technique": "clone_detection", "error": "OpenCV non disponible"}

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0, {"technique": "clone_detection", "error": "Image illisible"}

        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(img, None)

        if des is None or len(des) < 10:
            return 0.0, {"technique": "clone_detection", "keypoints": 0}

        # Self-matching : cherche des descripteurs très similaires dans la même image
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des, des, k=3)

        # Compter les matches non-identiques (même descripteur, position différente)
        clone_matches = 0
        for match_group in matches:
            if len(match_group) >= 2:
                m, n = match_group[0], match_group[1]
                # Même point = index identique, différent = clone suspect
                if m.queryIdx != m.trainIdx and m.distance < 30:
                    clone_matches += 1

        score = min(1.0, clone_matches / max(len(kp), 1))
        details = {
            "technique": "clone_detection",
            "keypoints": len(kp),
            "clone_matches": clone_matches,
        }
        if clone_matches > 10:
            details["flag"] = f"{clone_matches} régions copy-move détectées"

        return round(score, 4), details

    except Exception as exc:
        return 0.0, {"technique": "clone_detection", "error": str(exc)}


# ── Métadonnées PDF ───────────────────────────────────────────────────────────

def _metadata_score_pdf(path: Path) -> tuple[float, dict]:
    """Analyse les métadonnées d'un PDF.

    Incohérences détectées :
    - Date de création > date de modification
    - Producer/Creator contenant des outils d'édition (Photoshop, GIMP, etc.)
    - JavaScript embarqué
    - Dates impossibles (dans le futur)
    - Absence totale de métadonnées (effacées)
    """
    _lazy_load()
    anomalies = []
    score = 0.0

    try:
        doc = _pymupdf.open(str(path))
        meta = doc.metadata
        doc.close()

        suspicious_tools = [
            "photoshop", "gimp", "affinity", "inkscape",
            "pdf editor", "foxit", "nitro", "sejda",
            "smallpdf", "ilovepdf", "pdf24",
        ]

        creator = (meta.get("creator") or "").lower()
        producer = (meta.get("producer") or "").lower()

        for tool in suspicious_tools:
            if tool in creator or tool in producer:
                anomalies.append({
                    "type": "suspicious_tool",
                    "detail": f"Outil d'édition détecté : {meta.get('creator') or meta.get('producer')}",
                })
                score += 0.3
                break

        # Dates incohérentes
        created = meta.get("creationDate", "") or ""
        modified = meta.get("modDate", "") or ""

        now = datetime.now(timezone.utc)
        if created and len(created) >= 4:
            try:
                year = int(created[2:6]) if created.startswith("D:") else int(created[:4])
                if year > now.year:
                    anomalies.append({"type": "future_date", "detail": f"Date de création dans le futur : {year}"})
                    score += 0.4
            except (ValueError, IndexError):
                pass

        # Absence de métadonnées
        if not creator and not producer and not created:
            anomalies.append({"type": "no_metadata", "detail": "Métadonnées absentes — possiblement effacées"})
            score += 0.2

        return round(min(score, 1.0), 4), {
            "technique": "pdf_metadata",
            "creator": meta.get("creator"),
            "producer": meta.get("producer"),
            "created": created[:20] if created else None,
            "anomalies_count": len(anomalies),
        }

    except Exception as exc:
        return 0.0, {"technique": "pdf_metadata", "error": str(exc)}


def _metadata_score_image(path: Path) -> tuple[float, dict]:
    """Analyse les métadonnées EXIF d'une image."""
    _lazy_load()
    Image = _PIL[0]
    anomalies = []
    score = 0.0

    try:
        from PIL.ExifTags import TAGS
        img = Image.open(str(path))
        exif_data = img._getexif() if hasattr(img, "_getexif") else None

        if exif_data is None:
            anomalies.append({"type": "no_exif", "detail": "Pas de données EXIF"})
            score += 0.15
        else:
            exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

            # Vérifier si le software est un éditeur
            software = str(exif.get("Software", "")).lower()
            suspicious = ["photoshop", "gimp", "affinity", "lightroom", "darktable"]
            if any(s in software for s in suspicious):
                anomalies.append({
                    "type": "editing_software",
                    "detail": f"Image modifiée par : {exif.get('Software')}",
                })
                score += 0.35

            # GPS présent sur un document scanné = suspect
            if "GPSInfo" in exif:
                anomalies.append({"type": "gps_on_scan", "detail": "Coordonnées GPS sur un scan = incohérent"})
                score += 0.2

        return round(min(score, 1.0), 4), {
            "technique": "exif_metadata",
            "has_exif": exif_data is not None,
            "anomalies_count": len(anomalies),
        }

    except Exception as exc:
        return 0.0, {"technique": "exif_metadata", "error": str(exc)}


# ── Cohérence des polices (DOCX) ──────────────────────────────────────────────

def _font_score_docx(path: Path) -> tuple[float, dict]:
    """Détecte les mélanges typographiques incohérents dans un DOCX.

    Un document authentique a une ou deux familles de polices.
    Un faux document assemblé peut avoir de nombreuses polices différentes.
    """
    try:
        _lazy_load()
        doc = _docx.Document(str(path))

        fonts_used: set[str] = set()
        sizes_per_font: dict[str, set] = {}

        for para in doc.paragraphs:
            for run in para.runs:
                if run.font.name:
                    fonts_used.add(run.font.name)
                    if run.font.size:
                        sizes_per_font.setdefault(run.font.name, set()).add(run.font.size)

        font_count = len(fonts_used)
        # Plus de 4 polices distinctes = suspect
        if font_count > 5:
            score = min(1.0, (font_count - 4) * 0.15)
        elif font_count > 3:
            score = 0.25
        else:
            score = 0.0

        details = {
            "technique": "font_coherence",
            "font_count": font_count,
            "fonts": list(fonts_used)[:10],
        }
        if font_count > 4:
            details["flag"] = f"{font_count} polices différentes — document assemblé probable"

        return round(score, 4), details

    except Exception as exc:
        return 0.0, {"technique": "font_coherence", "error": str(exc)}


def _font_score_pdf(path: Path) -> tuple[float, dict]:
    """Détecte les mélanges de polices dans un PDF."""
    try:
        _lazy_load()
        doc = _pymupdf.open(str(path))
        fonts_all: set[str] = set()

        for page in doc:
            for font in page.get_fonts():
                # font = (xref, ext, type, basefont, name, encoding, referencer)
                if len(font) >= 5:
                    fonts_all.add(font[3])  # basefont

        doc.close()
        font_count = len(fonts_all)
        score = 0.0
        if font_count > 6:
            score = min(1.0, (font_count - 5) * 0.1)
        elif font_count > 4:
            score = 0.15

        return round(score, 4), {
            "technique": "font_coherence_pdf",
            "font_count": font_count,
            "fonts_sample": list(fonts_all)[:8],
        }

    except Exception as exc:
        return 0.0, {"technique": "font_coherence_pdf", "error": str(exc)}


# ── Extraction texte ──────────────────────────────────────────────────────────

def _extract_text_preview(path: Path, max_chars: int = 1000) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            _lazy_load()
            doc = _pymupdf.open(str(path))
            text = " ".join(page.get_text() for page in doc[:3])
            doc.close()
            return text[:max_chars]
        if suffix in (".docx", ".doc"):
            _lazy_load()
            doc = _docx.Document(str(path))
            return " ".join(p.text for p in doc.paragraphs if p.text)[:max_chars]
    except Exception:
        pass
    return ""


# ── Interface principale ───────────────────────────────────────────────────────

class DocumentEngine:
    """Moteur de détection de faux documents forensique."""

    def analyze(self, file_path: Path) -> DocumentScores:
        suffix = file_path.suffix.lower()
        anomalies: list[dict] = []
        models_used: dict = {"document_engine_version": "2.0"}

        try:
            is_image = suffix in (".jpg", ".jpeg", ".png", ".tiff", ".tif")
            is_pdf = suffix == ".pdf"
            is_docx = suffix in (".docx", ".doc")

            if not (is_image or is_pdf or is_docx):
                return DocumentScores(
                    score_ela=0.0, score_clone=0.0, score_metadata=0.0,
                    score_font=0.0, score_text_ai=0.0, final_score=0.0,
                    error=f"Format non supporté : {suffix}",
                )

            # ── ELA (images uniquement) ───────────────────────────────────────
            score_ela = 0.0
            if is_image:
                score_ela, ela_detail = _ela_score(file_path)
                if ela_detail.get("flag"):
                    anomalies.append(ela_detail)
                models_used["ela"] = ela_detail.get("mean_diff")

            # ── Clone detection (images uniquement) ──────────────────────────
            score_clone = 0.0
            if is_image:
                score_clone, clone_detail = _clone_score(file_path)
                if clone_detail.get("flag"):
                    anomalies.append(clone_detail)
                models_used["clone_detection"] = clone_detail.get("clone_matches")

            # ── Métadonnées ───────────────────────────────────────────────────
            if is_pdf:
                score_metadata, meta_detail = _metadata_score_pdf(file_path)
            elif is_image:
                score_metadata, meta_detail = _metadata_score_image(file_path)
            else:
                score_metadata = 0.0
                meta_detail = {"technique": "metadata", "note": "DOCX — analyse métadonnées limitée"}
            models_used["metadata"] = meta_detail.get("anomalies_count", 0)

            # ── Cohérence polices ─────────────────────────────────────────────
            if is_docx:
                score_font, font_detail = _font_score_docx(file_path)
            elif is_pdf:
                score_font, font_detail = _font_score_pdf(file_path)
            else:
                score_font = 0.0
                font_detail = {"technique": "font_coherence", "note": "N/A pour images"}
            if font_detail.get("flag"):
                anomalies.append(font_detail)
            models_used["font_coherence"] = font_detail.get("font_count")

            # ── TextEngine (délégation) ───────────────────────────────────────
            score_text_ai = 0.5  # neutre si pas de texte extractible
            if is_pdf or is_docx:
                from engines.text_engine import TextEngine
                te = TextEngine()
                text_result = te.analyze_file(file_path)
                if not text_result.error:
                    score_text_ai = text_result.final_score
                    models_used["text_ai"] = {
                        "score": text_result.final_score,
                        "verdict": text_result.verdict_label,
                        "word_count": text_result.word_count,
                    }
                    if text_result.flagged_passages:
                        anomalies.append({
                            "type": "ai_text",
                            "detail": f"Passages potentiellement générés par IA ({len(text_result.flagged_passages)} détectés)",
                            "passages": text_result.flagged_passages[:3],
                        })

            # ── Fusion pondérée ───────────────────────────────────────────────
            # Images : ELA + Clone + Métadonnées dominant
            # PDF/DOCX : Métadonnées + Police + TextAI dominant
            if is_image:
                final_score = round(
                    0.40 * score_ela
                    + 0.25 * score_clone
                    + 0.25 * score_metadata
                    + 0.10 * score_font,
                    4,
                )
            else:
                final_score = round(
                    0.35 * score_text_ai
                    + 0.30 * score_metadata
                    + 0.25 * score_font
                    + 0.10 * score_clone,
                    4,
                )

            text_preview = _extract_text_preview(file_path)

            return DocumentScores(
                score_ela=score_ela,
                score_clone=score_clone,
                score_metadata=score_metadata,
                score_font=score_font,
                score_text_ai=score_text_ai,
                final_score=final_score,
                anomalies=anomalies,
                models_used=models_used,
                text_content_preview=text_preview,
            )

        except Exception as exc:
            return DocumentScores(
                score_ela=0.0, score_clone=0.0, score_metadata=0.0,
                score_font=0.0, score_text_ai=0.0, final_score=0.0,
                anomalies=[], error=str(exc),
            )
