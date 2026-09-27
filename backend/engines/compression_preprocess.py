"""CompressionPreprocess — Détection et compensation de la compression vidéo (v2.0).

Détecte si une vidéo a subi une compression agressive (WhatsApp, TikTok, Facebook)
et ajuste le score de confiance en conséquence avec un warning dans le rapport.
"""
from __future__ import annotations

import enum
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path


class CompressionQuality(str, enum.Enum):
    high = "HIGH"         # Bitrate élevé, peu de perte — analyse fiable
    medium = "MEDIUM"     # Compression modérée — analyse correcte
    low = "LOW"           # Compression forte (Facebook, YouTube) — précision réduite
    very_low = "VERY_LOW" # Très forte (WhatsApp, TikTok) — précision fortement réduite


# Seuils bitrate vidéo en kbps
_BITRATE_HIGH = 2000
_BITRATE_MEDIUM = 800
_BITRATE_LOW = 300

# Pénalités de confiance (soustraites de l'intervalle de confiance)
_PENALTIES = {
    CompressionQuality.high:     0.00,
    CompressionQuality.medium:   0.05,
    CompressionQuality.low:      0.12,
    CompressionQuality.very_low: 0.20,
}

# Messages warning pour le rapport
_WARNINGS = {
    CompressionQuality.high: None,
    CompressionQuality.medium: None,
    CompressionQuality.low: (
        "⚠️ Compression vidéo détectée (qualité faible) — "
        "précision des moteurs visuels réduite à ~65%. "
        "Recommandé : obtenir le fichier source original."
    ),
    CompressionQuality.very_low: (
        "⚠️ Compression vidéo très agressive détectée (WhatsApp/TikTok probable) — "
        "précision des moteurs visuels réduite à ~50%. "
        "Les moteurs audio restent fiables. "
        "RECOMMANDATION FORTE : demander le fichier source original à la source."
    ),
}

# Codecs associés aux plateformes connues
_PLATFORM_HINTS = {
    "h264": "Générique (WhatsApp, Facebook, Telegram)",
    "h265": "TikTok / Instagram / iPhone",
    "hevc": "TikTok / Instagram / iPhone",
    "vp9":  "YouTube / Google",
    "av1":  "YouTube (haute qualité) / Discord",
    "mpeg4": "Ancien format / MMS",
    "wmv3":  "Windows Media",
}


@dataclass
class CompressionProfile:
    codec: str
    bitrate_kbps: float
    width: int
    height: int
    fps: float
    quality_flag: CompressionQuality
    confidence_penalty: float
    source_warning: str | None
    platform_hint: str | None
    degraded: bool

    def as_dict(self) -> dict:
        return {
            "codec": self.codec,
            "bitrate_kbps": self.bitrate_kbps,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "quality_flag": self.quality_flag.value,
            "confidence_penalty": self.confidence_penalty,
            "degraded": self.degraded,
            "platform_hint": self.platform_hint,
        }


def detect_compression(file_path: Path) -> CompressionProfile:
    """Analyse la compression d'une vidéo via FFprobe.

    Retourne un CompressionProfile avec la qualité estimée et les pénalités.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return _unknown_profile()

    # Extraire le stream vidéo
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        return _unknown_profile()

    codec = video_stream.get("codec_name", "unknown").lower()
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    # FPS
    fps = 25.0
    fps_str = video_stream.get("r_frame_rate", "25/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        try:
            fps = float(num) / float(den) if float(den) != 0 else 25.0
        except (ValueError, ZeroDivisionError):
            fps = 25.0

    # Bitrate total en kbps
    bitrate_bps = float(data.get("format", {}).get("bit_rate", 0))
    bitrate_kbps = bitrate_bps / 1000.0

    # Si bitrate global absent, utiliser bitrate stream
    if bitrate_kbps == 0:
        bitrate_kbps = float(video_stream.get("bit_rate", 0)) / 1000.0

    # Qualité
    quality = _classify_quality(bitrate_kbps, codec, width, height)

    platform_hint = next(
        (hint for key, hint in _PLATFORM_HINTS.items() if key in codec),
        None,
    )

    return CompressionProfile(
        codec=codec,
        bitrate_kbps=round(bitrate_kbps, 1),
        width=width,
        height=height,
        fps=round(fps, 2),
        quality_flag=quality,
        confidence_penalty=_PENALTIES[quality],
        source_warning=_WARNINGS[quality],
        platform_hint=platform_hint,
        degraded=quality in (CompressionQuality.low, CompressionQuality.very_low),
    )


def _classify_quality(bitrate_kbps: float, codec: str, w: int, h: int) -> CompressionQuality:
    """Classifie la qualité selon le bitrate et la résolution."""
    # Normaliser bitrate par rapport à la résolution (pixels)
    pixels = w * h if w and h else 1920 * 1080
    bitrate_per_mpx = bitrate_kbps / max(pixels / 1_000_000, 0.1)

    # H.265 est plus efficace — ajustement
    efficiency_factor = 1.5 if codec in ("h265", "hevc", "av1") else 1.0

    effective_bitrate = bitrate_kbps * efficiency_factor

    if effective_bitrate == 0:
        return CompressionQuality.medium  # inconnu = on ne pénalise pas

    if effective_bitrate >= _BITRATE_HIGH:
        return CompressionQuality.high
    if effective_bitrate >= _BITRATE_MEDIUM:
        return CompressionQuality.medium
    if effective_bitrate >= _BITRATE_LOW:
        return CompressionQuality.low
    return CompressionQuality.very_low


def _unknown_profile() -> CompressionProfile:
    return CompressionProfile(
        codec="unknown",
        bitrate_kbps=0.0,
        width=0,
        height=0,
        fps=0.0,
        quality_flag=CompressionQuality.medium,
        confidence_penalty=0.0,
        source_warning=None,
        platform_hint=None,
        degraded=False,
    )


def apply_compression_penalty(
    confidence_low: float,
    confidence_high: float,
    final_score: float,
    profile: CompressionProfile,
) -> tuple[float, float, float]:
    """Ajuste l'intervalle de confiance selon la qualité de compression.

    Retourne (confidence_low_adjusted, confidence_high_adjusted, final_score_unchanged).
    Le score final n'est pas modifié — seul l'IC est élargi pour refléter l'incertitude.
    """
    penalty = profile.confidence_penalty
    if penalty == 0.0:
        return confidence_low, confidence_high, final_score

    # Élargir l'intervalle de confiance (refléter l'incertitude)
    new_low = max(0.0, confidence_low - penalty)
    new_high = min(1.0, confidence_high + penalty)
    return round(new_low, 4), round(new_high, 4), final_score
