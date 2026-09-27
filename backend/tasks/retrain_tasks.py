"""Tâche Celery — détection de dérive et déclenchement réentraînement."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from celery import shared_task
from celery.utils.log import get_task_logger

from tasks.analysis_tasks import celery_app  # noqa: F401 — registers the Celery app

logger = get_task_logger(__name__)

_DRIFT_THRESHOLD = 0.20  # 20% de feedbacks incorrects déclenche une alerte


@shared_task(name="retrain_models", bind=True)
def retrain_models(self) -> dict:
    """
    Détecte la dérive des modèles à partir des feedbacks accumulés.

    Retourne : {feedbacks_processed, drift_score, models_flagged}
    Note : le réentraînement PyTorch réel est hors scope de cette tâche ;
    elle détecte la dérive et crée une nouvelle ModelVersion "draft" avec
    les métriques de dérive pour traçabilité.
    """
    from database import SessionLocal
    from models.feedback import AnalysisFeedback, FeedbackType
    from models.model_version import ModelVersion
    from sqlalchemy import func

    db = SessionLocal()
    try:
        feedbacks = db.query(AnalysisFeedback).all()
        total = len(feedbacks)

        if total == 0:
            logger.info("retrain_no_feedbacks")
            return {"feedbacks_processed": 0, "drift_score": 0.0, "models_flagged": []}

        incorrect = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.incorrect)
        drift_score = round(incorrect / total, 4)

        models_flagged: list[str] = []
        if drift_score > _DRIFT_THRESHOLD:
            logger.warning(
                "model_drift_detected",
                drift_score=drift_score,
                threshold=_DRIFT_THRESHOLD,
                incorrect=incorrect,
                total=total,
            )
            # Récupérer les noms de modèles actifs et les marquer pour réentraînement
            active_versions = db.query(ModelVersion).filter(ModelVersion.is_active.is_(True)).all()
            models_flagged = [v.model_name for v in active_versions]

            # Créer une entrée de dérive dans le registre pour traçabilité
            for av in active_versions:
                drift_record = ModelVersion(
                    model_name=av.model_name,
                    version=f"{av.version}-drift-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    hf_repo_id=None,
                    is_active=False,
                    metrics={
                        "drift_score": drift_score,
                        "feedbacks_total": total,
                        "feedbacks_incorrect": incorrect,
                        "flagged_at": datetime.now(timezone.utc).isoformat(),
                        "needs_retrain": True,
                    },
                    registered_by_id=av.registered_by_id,
                )
                db.add(drift_record)

            db.commit()
            logger.warning("drift_records_created", count=len(active_versions))
        else:
            logger.info(
                "retrain_drift_ok",
                drift_score=drift_score,
                threshold=_DRIFT_THRESHOLD,
            )

        return {
            "feedbacks_processed": total,
            "drift_score": drift_score,
            "models_flagged": models_flagged,
        }
    finally:
        db.close()
