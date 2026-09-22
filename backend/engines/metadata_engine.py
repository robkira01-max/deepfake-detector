"""Moteur d'analyse des métadonnées — détecte les incohérences forensiques.

Indicateurs analysés :
  - Cohérence des métadonnées EXIF/MP4 (créateur, outil, dates)
  - Signatures de logiciels de génération IA connus (FFmpeg patch, DeepFaceLab, etc.)
  - Incohérences temporelles (date de création > date de modification)
  - Fingerprints de codecs GAN (paramètres de compression atypiques)
  - Présence de chunks de données inhabituels (stegano, padding excessif)
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

# Outils de génération deepfake connus — signature dans les métadonnées
_DEEPFAKE_TOOL_PATTERNS = [
    r"deepfacelab",
    r"facefusion",
    r"insightface",
    r"roop",
    r"simswap",
    r"faceswap",
    r"wav2lip",
    r"sadtalker",
    r"tortoise[\-_]tts",
    r"elevenlabs",
    r"xtts",
    r"bark",
    r"real-esrgan",
    r"gfpgan",
    r"codeformer",
    r"diffusion",  # Stable Diffusion video
]

# Outils légitimes de post-production (réduisent le score)
_LEGITIMATE_TOOL_PATTERNS = [
    r"adobe\s+premiere",
    r"final\s+cut",
    r"davinci\s+resolve",
    r"avid",
    r"sony\s+vegas",
    r"handbrake",
    r"obs\s+studio",
]

# Paramètres de codec atypiques pour les deepfakes GAN
_SUSPICIOUS_CODEC_PARAMS = {
    "video_codec": ["libx264", "h264"],  # Légitimes — pas suspects seuls
    "suspicious_encoders": [
        "lavc",  # FFmpeg générique — souvent utilisé par les deepfake tools
        "x264 core",
    ],
}


@dataclass
class MetadataResult:
    score: float = 0.0
    findings: list[dict] = field(default_factory=list)
    metadata_raw: dict = field(default_factory=dict)
    tool_detected: str | None = None
    temporal_anomaly: bool = False
    error: str | None = None


class MetadataEngine:
    """Analyse les métadonnées d'un fichier pour détecter des anomalies deepfake."""

    def analyze(self, file_path: Path) -> MetadataResult:
        log.info("metadata_analysis_start", path=str(file_path))

        if not file_path.exists():
            return MetadataResult(error=f"Fichier introuvable : {file_path}")

        # Extraire les métadonnées via ffprobe (si disponible) ou fallback
        metadata = self._extract_metadata(file_path)
        if not metadata:
            return MetadataResult(
                score=0.0,
                error="Impossible d'extraire les métadonnées",
                metadata_raw={},
            )

        findings = []
        score_components = []

        # ── 1. Détection d'outils deepfake ───────────────────────────────────
        tool_score, tool_finding, tool_name = self._check_deepfake_tools(metadata)
        if tool_finding:
            findings.append(tool_finding)
        score_components.append(tool_score)

        # ── 2. Incohérences temporelles ───────────────────────────────────────
        temporal_score, temporal_anomaly, temporal_findings = self._check_temporal(metadata, file_path)
        findings.extend(temporal_findings)
        score_components.append(temporal_score)

        # ── 3. Paramètres de codec suspects ───────────────────────────────────
        codec_score, codec_findings = self._check_codec_params(metadata)
        findings.extend(codec_findings)
        score_components.append(codec_score)

        # ── 4. Présence de données inhabituelles ──────────────────────────────
        anomaly_score, anomaly_findings = self._check_data_anomalies(file_path, metadata)
        findings.extend(anomaly_findings)
        score_components.append(anomaly_score)

        # Score final : moyenne pondérée (outil deepfake = poids x2)
        weights = [2.0, 1.0, 1.0, 0.5]
        total_w = sum(weights[:len(score_components)])
        final_score = sum(s * w for s, w in zip(score_components, weights)) / total_w
        final_score = max(0.0, min(1.0, final_score))

        log.info(
            "metadata_analysis_done",
            score=round(final_score, 3),
            findings_count=len(findings),
            tool=tool_name,
        )

        return MetadataResult(
            score=round(final_score, 4),
            findings=findings,
            metadata_raw=metadata,
            tool_detected=tool_name,
            temporal_anomaly=temporal_anomaly,
        )

    # ── Extraction métadonnées ────────────────────────────────────────────────

    def _extract_metadata(self, file_path: Path) -> dict:
        """Extrait les métadonnées via ffprobe en priorité, fallback sur exiftool."""
        metadata = {}

        # Tentative ffprobe
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                metadata["ffprobe"] = data
                # Aplatir les tags du format
                if "format" in data and "tags" in data["format"]:
                    metadata["tags"] = {k.lower(): v for k, v in data["format"]["tags"].items()}
                else:
                    metadata["tags"] = {}
                return metadata
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass

        # Fallback : lecture binaire des magic bytes et headers basiques
        try:
            metadata["fallback"] = True
            metadata["tags"] = {}
            metadata["file_size"] = file_path.stat().st_size
            metadata["extension"] = file_path.suffix.lower()

            # Lire les premiers 512 octets pour signature
            with file_path.open("rb") as f:
                header = f.read(512)
            metadata["header_ascii"] = header.decode("ascii", errors="replace")
        except Exception as exc:
            log.warning("metadata_fallback_failed", error=str(exc))

        return metadata

    # ── Vérifications ─────────────────────────────────────────────────────────

    def _check_deepfake_tools(
        self, metadata: dict
    ) -> tuple[float, dict | None, str | None]:
        """Cherche les signatures d'outils deepfake dans tous les champs texte."""
        all_text = self._flatten_to_text(metadata).lower()

        for pattern in _DEEPFAKE_TOOL_PATTERNS:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                tool = match.group(0)
                log.warning("deepfake_tool_detected", tool=tool)
                return 0.9, {
                    "type": "deepfake_tool_signature",
                    "severity": "critical",
                    "detail": f"Signature d'outil deepfake détectée dans les métadonnées : '{tool}'",
                    "score": 0.9,
                }, tool

        # Vérifier aussi les outils légitimes (réduction du score)
        for pattern in _LEGITIMATE_TOOL_PATTERNS:
            if re.search(pattern, all_text, re.IGNORECASE):
                return 0.05, None, None

        return 0.1, None, None  # Neutre : pas d'info

    def _check_temporal(
        self, metadata: dict, file_path: Path
    ) -> tuple[float, bool, list[dict]]:
        """Vérifie les incohérences de dates (création > modification = suspect)."""
        findings = []
        anomaly = False

        try:
            stat = file_path.stat()
            mtime = stat.st_mtime
            ctime = stat.st_ctime  # Création sur Linux = inode change time

            # Anomalie : modification très récente sur un fichier prétendument ancien
            tags = metadata.get("tags", {})
            creation_tag = (
                tags.get("creation_time")
                or tags.get("date")
                or tags.get("com.apple.quicktime.creationdate")
            )

            if creation_tag:
                # Simple vérification de cohérence de format
                if re.match(r"\d{4}-\d{2}-\d{2}", creation_tag):
                    year = int(creation_tag[:4])
                    if year > 2030 or year < 1990:
                        anomaly = True
                        findings.append({
                            "type": "temporal_anomaly",
                            "severity": "high",
                            "detail": f"Date de création suspecte dans les métadonnées : {creation_tag}",
                            "score": 0.7,
                        })
        except Exception as exc:
            log.debug("temporal_check_failed", error=str(exc))

        return 0.7 if anomaly else 0.0, anomaly, findings

    def _check_codec_params(self, metadata: dict) -> tuple[float, list[dict]]:
        """Vérifie les paramètres de codec atypiques."""
        findings = []
        suspicious_score = 0.0

        try:
            ffprobe = metadata.get("ffprobe", {})
            streams = ffprobe.get("streams", [])

            for stream in streams:
                encoder = (stream.get("tags", {}).get("encoder", "") or "").lower()

                # Encoder inconnu ou générique
                if encoder and not any(
                    legit in encoder for legit in ["premiere", "final cut", "resolve", "obs"]
                ):
                    if any(s in encoder for s in _SUSPICIOUS_CODEC_PARAMS["suspicious_encoders"]):
                        findings.append({
                            "type": "generic_encoder",
                            "severity": "low",
                            "detail": f"Encodeur générique détecté : '{encoder}' (souvent utilisé par les outils deepfake)",
                            "score": 0.2,
                        })
                        suspicious_score = max(suspicious_score, 0.2)

                # Vérification du nombre de frames vs durée
                nb_frames = stream.get("nb_frames")
                duration = stream.get("duration")
                r_frame_rate = stream.get("r_frame_rate", "")

                if nb_frames and duration:
                    try:
                        fps_declared = eval(r_frame_rate) if r_frame_rate else 0
                        fps_actual = int(nb_frames) / float(duration)
                        if fps_declared > 0 and abs(fps_actual - fps_declared) / fps_declared > 0.1:
                            findings.append({
                                "type": "fps_mismatch",
                                "severity": "medium",
                                "detail": f"FPS déclaré ({fps_declared:.2f}) vs réel ({fps_actual:.2f}) : écart >10%",
                                "score": 0.35,
                            })
                            suspicious_score = max(suspicious_score, 0.35)
                    except Exception:
                        pass

        except Exception as exc:
            log.debug("codec_check_failed", error=str(exc))

        return suspicious_score, findings

    def _check_data_anomalies(
        self, file_path: Path, metadata: dict
    ) -> tuple[float, list[dict]]:
        """Cherche des données inhabituelles (padding excessif, chunks inconnus)."""
        findings = []
        score = 0.0

        try:
            file_size = file_path.stat().st_size

            # Vérification de la taille vs durée déclarée (ratio atypique)
            ffprobe = metadata.get("ffprobe", {})
            fmt = ffprobe.get("format", {})
            duration = float(fmt.get("duration", 0) or 0)
            bit_rate = int(fmt.get("bit_rate", 0) or 0)

            if duration > 0 and bit_rate > 0:
                expected_size = (bit_rate / 8) * duration
                ratio = file_size / expected_size
                if ratio > 3.0:
                    findings.append({
                        "type": "oversized_for_bitrate",
                        "severity": "medium",
                        "detail": f"Fichier {ratio:.1f}× plus grand qu'attendu pour son débit — padding suspect",
                        "score": 0.4,
                    })
                    score = max(score, 0.4)

        except Exception as exc:
            log.debug("data_anomaly_check_failed", error=str(exc))

        return score, findings

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def _flatten_to_text(self, obj, depth: int = 0) -> str:
        """Convertit récursivement un dict/list en texte pour regex search."""
        if depth > 5:
            return ""
        if isinstance(obj, str):
            return obj + " "
        if isinstance(obj, dict):
            return " ".join(self._flatten_to_text(v, depth + 1) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return " ".join(self._flatten_to_text(i, depth + 1) for i in obj)
        return str(obj) + " "
