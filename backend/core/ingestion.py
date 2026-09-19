"""Pipeline d'ingestion de fichiers médias en 7 étapes — Chain of Custody."""
from __future__ import annotations

import json
import mimetypes
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import magic
import structlog
from sqlalchemy.orm import Session

from config import settings
from core.chain_of_custody import compute_hashes, stamp_file, sign_audit_entry
from models.media_file import MediaFile, MediaType, MediaStatus
from models.audit_log import AuditLog, AuditAction

log = structlog.get_logger(__name__)


class IngestionError(Exception):
    """Levée si le fichier est rejeté lors de l'ingestion."""


class IngestionResult:
    def __init__(self, media_file: MediaFile, warnings: list[str] | None = None) -> None:
        self.media_file = media_file
        self.warnings = warnings or []


def ingest_file(
    source_path: Path,
    original_filename: str,
    case_id: int,
    uploader_id: int,
    db: Session,
    ip_address: str = "",
) -> IngestionResult:
    """
    Pipeline d'ingestion complet en 7 étapes :
    1. Réception et validation de base
    2. Copie en quarantaine (isolation sandbox)
    3. Calcul des empreintes cryptographiques
    4. Extraction des métadonnées
    5. Horodatage TSA RFC 3161
    6. Enregistrement en base (MediaFile + AuditLog)
    7. Déplacement vers stockage définitif
    """
    warnings: list[str] = []
    file_uuid = str(uuid.uuid4())

    # ── Étape 1 : Validation ──────────────────────────────────────────────────
    log.info("ingestion_start", uuid=file_uuid, filename=original_filename, case_id=case_id)

    ext = Path(original_filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise IngestionError(f"Extension non autorisée : {ext}")

    if source_path.stat().st_size > settings.max_file_size_bytes:
        raise IngestionError(
            f"Fichier trop volumineux : {source_path.stat().st_size} octets "
            f"(max {settings.max_file_size_bytes})"
        )

    # ── Étape 2 : Quarantaine ─────────────────────────────────────────────────
    quarantine_dir = Path(settings.quarantine_dir) / file_uuid
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path = quarantine_dir / f"original{ext}"
    shutil.copy2(source_path, quarantine_path)
    log.info("file_quarantined", path=str(quarantine_path))

    # Détection MIME réelle (python-magic) — protection contre spoofing d'extension
    mime_type = _detect_mime(quarantine_path)
    media_type = _classify_media_type(mime_type, ext)
    if media_type is None:
        raise IngestionError(f"Type MIME non supporté : {mime_type}")

    # ── Étape 3 : Empreintes cryptographiques ────────────────────────────────
    bundle = compute_hashes(quarantine_path)
    log.info("hashes_computed", sha256=bundle.sha256[:16] + "...", blake3=bundle.blake3[:16] + "...")

    # Vérification de doublon (même hash dans la base)
    existing = db.query(MediaFile).filter(MediaFile.hash_sha256 == bundle.sha256).first()
    if existing:
        warnings.append(
            f"Fichier identique déjà enregistré (UUID: {existing.uuid}, Case: {existing.case_id})"
        )

    # ── Étape 4 : Métadonnées (FFmpeg) ───────────────────────────────────────
    metadata = _extract_metadata(quarantine_path, media_type)

    # ── Étape 5 : Horodatage TSA ─────────────────────────────────────────────
    tsa_token = stamp_file(quarantine_path)
    tsa_token_b64 = tsa_token.token_b64 if tsa_token else None
    tsa_authority = tsa_token.authority if tsa_token else None
    tsa_timestamp = tsa_token.timestamp if tsa_token else None

    if not tsa_token:
        warnings.append("Horodatage TSA indisponible — horodatage local utilisé")

    # ── Étape 6 : Enregistrement base de données ──────────────────────────────
    media_file = MediaFile(
        uuid=file_uuid,
        case_id=case_id,
        original_filename=original_filename,
        media_type=media_type,
        mime_type=mime_type,
        file_size_bytes=bundle.file_size,
        status=MediaStatus.verified,
        hash_sha256=bundle.sha256,
        hash_blake3=bundle.blake3,
        hash_md5=bundle.md5,
        tsa_token_b64=tsa_token_b64,
        tsa_authority=tsa_authority,
        tsa_timestamp=tsa_timestamp,
        media_metadata=metadata,
        ingested_by_id=uploader_id,
    )
    db.add(media_file)
    db.flush()  # Obtenir l'ID sans commit

    # Entrée dans le journal d'audit
    audit_data = {
        "action": AuditAction.FILE_INGESTED.value,
        "user_id": uploader_id,
        "resource_type": "MediaFile",
        "resource_id": str(media_file.id),
        "details": {
            "uuid": file_uuid,
            "filename": original_filename,
            "sha256": bundle.sha256,
            "blake3": bundle.blake3,
            "mime_type": mime_type,
            "size_bytes": bundle.file_size,
            "tsa_authority": tsa_authority,
            "tsa_timestamp": tsa_timestamp.isoformat() if tsa_timestamp else None,
            "warnings": warnings,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entry_hash, signature = sign_audit_entry(audit_data)

    audit = AuditLog(
        user_id=uploader_id,
        action=AuditAction.FILE_INGESTED,
        resource_type="MediaFile",
        resource_id=str(media_file.id),
        details=audit_data["details"],
        ip_address=ip_address,
        entry_hash=entry_hash,
        signature_b64=signature,
    )
    db.add(audit)
    db.commit()
    db.refresh(media_file)

    # ── Étape 7 : Déplacement vers stockage définitif ─────────────────────────
    processed_dir = Path(settings.processed_dir) / file_uuid
    processed_dir.mkdir(parents=True, exist_ok=True)
    final_path = processed_dir / f"original{ext}"
    shutil.move(str(quarantine_path), str(final_path))

    # Mise à jour du chemin de stockage local
    media_file.storage_key = str(final_path)
    db.commit()

    log.info("ingestion_complete", uuid=file_uuid, media_file_id=media_file.id)
    return IngestionResult(media_file=media_file, warnings=warnings)


def _detect_mime(path: Path) -> str:
    try:
        return magic.from_file(str(path), mime=True)
    except Exception:
        mime, _ = mimetypes.guess_type(str(path))
        return mime or "application/octet-stream"


def _classify_media_type(mime: str, ext: str) -> MediaType | None:
    video_mimes = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska", "video/webm"}
    audio_mimes = {"audio/wav", "audio/mpeg", "audio/aac", "audio/flac", "audio/ogg", "audio/mp4", "audio/x-wav"}
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    audio_exts = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"}

    # Validation stricte : MIME ET extension doivent être cohérents
    # Protection contre le spoofing d'extension (ex: malware.exe renommé en video.mp4)
    if mime in video_mimes and ext in video_exts:
        return MediaType.video
    if mime in audio_mimes and ext in audio_exts:
        return MediaType.audio
    return None


def _extract_metadata(path: Path, media_type: MediaType) -> dict:
    """Extraction des métadonnées via FFprobe (FFmpeg)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
        log.warning("metadata_extraction_failed", error=str(exc))

    return {"error": "ffprobe non disponible ou échec d'extraction"}
