"""Tâches Celery pour l'analyse deepfake asynchrone."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend/ is on the path regardless of how celery is launched
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from celery import Celery
from celery.utils.log import get_task_logger

from config import settings

celery_app = Celery(
    "deepfake_detector",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

logger = get_task_logger(__name__)


@celery_app.task(bind=True, name="analyze.run_deepfake_analysis", max_retries=2)
def run_deepfake_analysis(self, analysis_id: int, media_file_id: int) -> dict:
    """
    Tâche principale d'analyse deepfake.

    Pipeline :
    1. Récupération du fichier média
    2. Analyse vidéo (VideoEngine)
    3. Analyse audio (AudioEngine)
    4. Fusion des scores (FusionEngine)
    5. Mise à jour de l'enregistrement Analysis
    6. Entrée dans l'audit log
    """
    logger.info(f"Démarrage analyse #{analysis_id} pour média #{media_file_id}")
    start_time = time.time()

    from database import SessionLocal
    from models.analysis import Analysis, AnalysisStatus
    from models.media_file import MediaFile, MediaType
    from engines.video_engine import VideoEngine
    from engines.audio_engine import AudioEngine
    from engines.metadata_engine import MetadataEngine
    from engines.fusion import fuse_scores
    from models.audit_log import AuditLog, AuditAction
    from core.chain_of_custody import sign_audit_entry

    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            logger.error(f"Analyse #{analysis_id} introuvable")
            return {"error": "analysis_not_found"}

        analysis.status = AnalysisStatus.running
        db.commit()

        media_file = db.query(MediaFile).filter(MediaFile.id == media_file_id).first()
        if not media_file or not media_file.storage_key:
            raise ValueError(f"Fichier média #{media_file_id} introuvable ou sans chemin de stockage")

        file_path = Path(media_file.storage_key)

        # ── Analyse vidéo ─────────────────────────────────────────────────────
        video_scores = None
        if media_file.media_type == MediaType.video:
            logger.info("Lancement VideoEngine")
            engine = VideoEngine()
            video_scores = engine.analyze(file_path)

        # ── Analyse audio ─────────────────────────────────────────────────────
        audio_scores = None
        if media_file.media_type in (MediaType.video, MediaType.audio):
            logger.info("Lancement AudioEngine")
            engine = AudioEngine()
            audio_scores = engine.analyze(file_path)

        # ── Analyse métadonnées ───────────────────────────────────────────────
        logger.info("Lancement MetadataEngine")
        metadata_engine = MetadataEngine()
        metadata_result = metadata_engine.analyze(file_path)
        score_metadata = metadata_result.score if metadata_result.error is None else 0.0

        # ── Fusion ────────────────────────────────────────────────────────────
        result = fuse_scores(
            score_texture=video_scores.score_texture if video_scores else 0.0,
            score_temporal=video_scores.score_temporal if video_scores else 0.0,
            score_rppg=video_scores.score_rppg if video_scores else 0.0,
            score_biometrics=video_scores.score_biometrics if video_scores else 0.0,
            score_audio=audio_scores.score_model if audio_scores else 0.0,
            score_phase=audio_scores.score_phase if audio_scores else 0.0,
            score_metadata=score_metadata,
        )

        # ── Mise à jour Analysis ───────────────────────────────────────────────
        duration = int(time.time() - start_time)
        analysis.status = AnalysisStatus.completed
        analysis.verdict = result.verdict
        analysis.final_score = result.final_score
        analysis.confidence_low = result.confidence_low
        analysis.confidence_high = result.confidence_high
        analysis.score_video_texture = result.component_scores.get("texture")
        analysis.score_video_temporal = result.component_scores.get("temporal")
        analysis.score_rppg = result.component_scores.get("rppg")
        analysis.score_biometrics = result.component_scores.get("biometrics")
        analysis.score_audio_model = result.component_scores.get("audio")
        analysis.score_audio_phase = result.component_scores.get("phase")
        analysis.score_metadata = result.component_scores.get("metadata")
        analysis.model_far = result.model_far
        analysis.model_frr = result.model_frr
        analysis.model_eer = result.model_eer
        analysis.model_auc = result.model_auc
        analysis.xai_shap_values = result.shap_ranking
        analysis.xai_plain_explanation = result.plain_explanation
        analysis.xai_suspicious_timecodes = (
            video_scores.suspicious_timecodes if video_scores else []
        )
        analysis.xai_heatmap_path = video_scores.heatmap_path if video_scores else None
        analysis.xai_spectrogram_path = audio_scores.spectrogram_path if audio_scores else None
        analysis.models_used = (
            {**(video_scores.models_used if video_scores else {}),
             **(audio_scores.models_used if audio_scores else {})}
        )
        analysis.duration_seconds = duration
        analysis.completed_at = datetime.now(timezone.utc)
        db.commit()

        # ── Audit ─────────────────────────────────────────────────────────────
        data = {
            "action": AuditAction.ANALYSIS_COMPLETED.value,
            "analysis_id": analysis_id,
            "verdict": result.verdict.value,
            "final_score": result.final_score,
            "duration_seconds": duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry_hash, sig = sign_audit_entry(data)
        log_entry = AuditLog(
            action=AuditAction.ANALYSIS_COMPLETED,
            resource_type="Analysis",
            resource_id=str(analysis_id),
            details=data,
            entry_hash=entry_hash,
            signature_b64=sig,
        )
        db.add(log_entry)
        db.commit()

        logger.info(
            f"Analyse #{analysis_id} terminée — verdict={result.verdict.value} "
            f"score={result.final_score:.3f} durée={duration}s"
        )
        return {
            "analysis_id": analysis_id,
            "verdict": result.verdict.value,
            "final_score": result.final_score,
            "duration_seconds": duration,
        }

    except Exception as exc:
        logger.error(f"Erreur analyse #{analysis_id} : {exc}", exc_info=True)
        try:
            analysis.status = AnalysisStatus.failed
            analysis.error_message = str(exc)
            analysis.completed_at = datetime.now(timezone.utc)
            db.commit()
        except Exception:
            pass
        self.retry(exc=exc, countdown=60)

    finally:
        db.close()
