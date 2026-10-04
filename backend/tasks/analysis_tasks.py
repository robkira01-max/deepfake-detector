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


def _dispatch_webhooks(db, event: str, payload: dict) -> None:
    """Crée les WebhookDelivery et lance la livraison pour tous les hooks abonnés à cet événement."""
    try:
        from models.webhook import Webhook, WebhookDelivery, WebhookStatus
        from core.webhook_dispatcher import dispatch_webhook_sync

        hooks = db.query(Webhook).filter(
            Webhook.active == True  # noqa: E712
        ).all()
        for hook in hooks:
            if event not in (hook.events or []):
                continue
            delivery = WebhookDelivery(
                webhook_id=hook.id,
                event=event,
                payload=payload,
                status=WebhookStatus.pending,
            )
            db.add(delivery)
            db.commit()
            db.refresh(delivery)
            try:
                dispatch_webhook_sync(delivery.id)
            except Exception as exc:
                logger.warning(f"Webhook delivery #{delivery.id} erreur: {exc}")
    except Exception as exc:
        logger.warning(f"_dispatch_webhooks failed: {exc}")


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
    from engines.compression_preprocess import detect_compression, apply_compression_penalty
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

        # ── Compression detection (Phase 2 v2.0) ─────────────────────────────
        compression_profile = None
        if media_file.media_type == MediaType.video:
            compression_profile = detect_compression(file_path)
            if compression_profile.degraded:
                logger.warning(
                    f"Compression dégradée détectée — "
                    f"codec={compression_profile.codec} "
                    f"bitrate={compression_profile.bitrate_kbps}kbps "
                    f"qualité={compression_profile.quality_flag.value}"
                )

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

        # ── Vérification C2PA (provenance cryptographique — non ML) ──────────
        c2pa_result_dict = None
        try:
            from core.c2pa_verifier import C2PAVerifier
            c2pa_result = C2PAVerifier.verify(file_path)
            c2pa_result_dict = c2pa_result.to_dict()
            logger.info(
                "c2pa_verified",
                has_manifest=c2pa_result.has_manifest,
                is_valid=c2pa_result.is_cryptographically_valid,
                is_ai_generated=c2pa_result.is_ai_generated,
            )
        except Exception as exc:
            logger.warning("c2pa_verification_failed", error=str(exc))

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

        # ── Ajustement compression (Phase 2 v2.0) ────────────────────────────
        conf_low = result.confidence_low
        conf_high = result.confidence_high
        source_warning = None
        if compression_profile and compression_profile.degraded:
            conf_low, conf_high, _ = apply_compression_penalty(
                conf_low, conf_high, result.final_score, compression_profile
            )
            source_warning = compression_profile.source_warning

        # ── Mise à jour Analysis ───────────────────────────────────────────────
        duration = int(time.time() - start_time)
        analysis.status = AnalysisStatus.completed
        analysis.verdict = result.verdict
        analysis.final_score = result.final_score
        analysis.confidence_low = conf_low
        analysis.confidence_high = conf_high
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
        analysis.models_used = {
            **(video_scores.models_used if video_scores else {}),
            **(audio_scores.models_used if audio_scores else {}),
            **({"compression": compression_profile.as_dict()} if compression_profile else {}),
        }
        # Stocker le warning compression dans l'explication XAI
        if source_warning:
            existing = analysis.xai_plain_explanation or ""
            analysis.xai_plain_explanation = f"{source_warning}\n\n{existing}"
        analysis.c2pa_result = c2pa_result_dict
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

        # ── Dispatch webhooks (analysis.completed) ────────────────────────────
        _dispatch_webhooks(db, "analysis.completed", {
            "event": "analysis.completed",
            "analysis_id": analysis_id,
            "case_id": analysis.case_id,
            "verdict": result.verdict.value,
            "final_score": result.final_score,
            "duration_seconds": duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

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
            _dispatch_webhooks(db, "analysis.failed", {
                "event": "analysis.failed",
                "analysis_id": analysis_id,
                "error": str(exc)[:256],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass
        self.retry(exc=exc, countdown=60)

    finally:
        db.close()


@celery_app.task(bind=True, name="analyze.run_document_analysis", max_retries=1)
def run_document_analysis(self, doc_analysis_id: int, media_file_id: int) -> dict:
    """Tâche d'analyse forensique de document (v2.0).

    Pipeline :
    1. Récupération du fichier
    2. DocumentEngine (ELA + clone + métadonnées + polices)
    3. TextEngine (via DocumentEngine si PDF/DOCX)
    4. Persist résultat + audit log
    """
    logger.info(f"Démarrage analyse document #{doc_analysis_id} pour fichier #{media_file_id}")
    start_time = time.time()

    from database import SessionLocal
    from models.analysis import AnalysisStatus, Verdict
    from models.document_analysis import DocumentAnalysis
    from models.media_file import MediaFile
    from engines.document_engine import DocumentEngine
    from engines.compression_preprocess import detect_compression, CompressionQuality
    from models.audit_log import AuditLog, AuditAction
    from core.chain_of_custody import sign_audit_entry

    db = SessionLocal()
    try:
        doc = db.query(DocumentAnalysis).filter(DocumentAnalysis.id == doc_analysis_id).first()
        if not doc:
            logger.error(f"DocumentAnalysis #{doc_analysis_id} introuvable")
            return {"error": "doc_analysis_not_found"}

        doc.status = AnalysisStatus.running
        doc.started_at = datetime.now(timezone.utc)
        db.commit()

        media_file = db.query(MediaFile).filter(MediaFile.id == media_file_id).first()
        if not media_file or not media_file.storage_key:
            raise ValueError(f"Fichier #{media_file_id} introuvable ou sans chemin de stockage")

        file_path = Path(media_file.storage_key)

        # ── Analyse document ──────────────────────────────────────────────────
        engine = DocumentEngine()
        result = engine.analyze(file_path)

        # ── Verdict mapping ───────────────────────────────────────────────────
        label = result.verdict_label
        verdict_map = {
            "DOCUMENT FALSIFIÉ": Verdict.deepfake,
            "DOCUMENT AUTHENTIQUE": Verdict.authentic,
            "INDÉTERMINÉ": Verdict.undetermined,
        }
        verdict = verdict_map.get(label, Verdict.undetermined)

        # ── Explication lisible ───────────────────────────────────────────────
        score = result.final_score
        explanation_parts = [f"Verdict : {label} (score {score:.2f}/1.00)"]
        if result.score_ela and result.score_ela > 0.5:
            explanation_parts.append(f"• ELA : anomalies de retouche JPEG ({result.score_ela:.2f})")
        if result.score_clone and result.score_clone > 0.3:
            explanation_parts.append(f"• Clone : régions copy-move détectées ({result.score_clone:.2f})")
        if result.score_metadata and result.score_metadata > 0.3:
            explanation_parts.append(f"• Métadonnées : incohérences ({result.score_metadata:.2f})")
        if result.score_font and result.score_font > 0.3:
            explanation_parts.append(f"• Polices : mélange typographique ({result.score_font:.2f})")
        if result.score_text_ai and result.score_text_ai > 0.5:
            explanation_parts.append(f"• Texte : rédaction IA probable ({result.score_text_ai:.2f})")

        # ── Mise à jour DocumentAnalysis ──────────────────────────────────────
        duration = int(time.time() - start_time)
        doc.status = AnalysisStatus.completed
        doc.verdict = verdict
        doc.verdict_label = label
        doc.final_score = result.final_score
        doc.score_ela = result.score_ela
        doc.score_clone = result.score_clone
        doc.score_metadata_doc = result.score_metadata
        doc.score_font = result.score_font
        doc.score_text_ai = result.score_text_ai
        doc.anomalies = {"anomalies": result.anomalies}
        doc.xai_plain_explanation = "\n".join(explanation_parts)
        doc.text_preview = result.text_content_preview
        doc.models_used = result.models_used
        doc.completed_at = datetime.now(timezone.utc)
        doc.duration_seconds = duration
        if result.error:
            doc.error_message = result.error
        db.commit()

        # ── Audit ─────────────────────────────────────────────────────────────
        data = {
            "action": AuditAction.DOCUMENT_ANALYSIS_COMPLETED.value,
            "doc_analysis_id": doc_analysis_id,
            "verdict_label": label,
            "final_score": result.final_score,
            "duration_seconds": duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry_hash, sig = sign_audit_entry(data)
        log_entry = AuditLog(
            action=AuditAction.DOCUMENT_ANALYSIS_COMPLETED,
            resource_type="DocumentAnalysis",
            resource_id=str(doc_analysis_id),
            details=data,
            entry_hash=entry_hash,
            signature_b64=sig,
        )
        db.add(log_entry)
        db.commit()

        logger.info(
            f"Analyse document #{doc_analysis_id} terminée — "
            f"verdict={label} score={result.final_score:.3f} durée={duration}s"
        )
        return {
            "doc_analysis_id": doc_analysis_id,
            "verdict_label": label,
            "final_score": result.final_score,
            "duration_seconds": duration,
        }

    except Exception as exc:
        logger.error(f"Erreur analyse document #{doc_analysis_id} : {exc}", exc_info=True)
        try:
            doc.status = AnalysisStatus.failed
            doc.error_message = str(exc)
            doc.completed_at = datetime.now(timezone.utc)
            db.commit()
        except Exception:
            pass
        self.retry(exc=exc, countdown=30)

    finally:
        db.close()
