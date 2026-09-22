"""Routes dashboard — statistiques globales, audit log, santé détaillée."""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import settings
from core.security import require_admin, require_any
from database import get_db
from models.analysis import Analysis, AnalysisStatus, Verdict
from models.audit_log import AuditLog, AuditAction
from models.case import Case, CaseStatus
from models.media_file import MediaFile
from models.report import Report
from models.user import User

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# Temps de démarrage du processus (approximatif)
_START_TIME = time.time()


# ── Schémas ───────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_cases: int
    cases_by_status: dict[str, int]
    total_analyses: int
    analyses_by_verdict: dict[str, int]
    total_media_files: int
    total_reports: int
    deepfake_rate: float
    avg_confidence: float
    analyses_last_7_days: int
    generated_at: datetime


class AuditLogEntry(BaseModel):
    id: int
    user_username: str | None
    action: str
    resource_type: str | None
    resource_id: str | None
    timestamp: datetime
    details: dict | None

    model_config = {"from_attributes": True}


class ComponentHealth(BaseModel):
    status: str
    latency_ms: float | None = None
    detail: str | None = None


class HealthDetailedResponse(BaseModel):
    status: str  # ok | degraded | critical
    components: dict[str, Any]
    version: str
    uptime_seconds: float


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse, summary="Statistiques globales")
def get_stats(
    _: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
) -> StatsResponse:
    # Dossiers
    total_cases = db.query(func.count(Case.id)).scalar() or 0
    cases_by_status: dict[str, int] = {s.value: 0 for s in CaseStatus}
    for status, count in db.query(Case.status, func.count(Case.id)).group_by(Case.status).all():
        cases_by_status[status.value] = count

    # Analyses
    total_analyses = db.query(func.count(Analysis.id)).scalar() or 0
    analyses_by_verdict: dict[str, int] = {v.value: 0 for v in Verdict}
    analyses_by_verdict["pending"] = 0
    for verdict, count in (
        db.query(Analysis.verdict, func.count(Analysis.id))
        .filter(Analysis.verdict.isnot(None))
        .group_by(Analysis.verdict)
        .all()
    ):
        analyses_by_verdict[verdict.value] = count

    # Taux deepfake
    completed = db.query(func.count(Analysis.id)).filter(
        Analysis.verdict.isnot(None)
    ).scalar() or 0
    deepfake_count = analyses_by_verdict.get(Verdict.deepfake.value, 0)
    deepfake_rate = round(deepfake_count / completed, 4) if completed > 0 else 0.0

    # Confiance moyenne
    avg_conf_raw = db.query(func.avg(Analysis.final_score)).filter(
        Analysis.final_score.isnot(None)
    ).scalar()
    avg_confidence = round(float(avg_conf_raw), 4) if avg_conf_raw is not None else 0.0

    # Analyses 7 derniers jours
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    analyses_last_7_days = db.query(func.count(Analysis.id)).filter(
        Analysis.started_at >= cutoff
    ).scalar() or 0

    # Médias et rapports
    total_media_files = db.query(func.count(MediaFile.id)).scalar() or 0
    total_reports = db.query(func.count(Report.id)).scalar() or 0

    return StatsResponse(
        total_cases=total_cases,
        cases_by_status=cases_by_status,
        total_analyses=total_analyses,
        analyses_by_verdict=analyses_by_verdict,
        total_media_files=total_media_files,
        total_reports=total_reports,
        deepfake_rate=deepfake_rate,
        avg_confidence=avg_confidence,
        analyses_last_7_days=analyses_last_7_days,
        generated_at=datetime.now(timezone.utc),
    )


@router.get(
    "/audit-log",
    response_model=list[AuditLogEntry],
    summary="Journal d'audit (admin)",
)
def get_audit_log(
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 50,
    user_id: int | None = None,
    action: AuditAction | None = None,
) -> list[AuditLog]:
    q = db.query(AuditLog).order_by(AuditLog.timestamp.desc())
    if user_id is not None:
        q = q.filter(AuditLog.user_id == user_id)
    if action is not None:
        q = q.filter(AuditLog.action == action)
    return q.offset(skip).limit(min(limit, 500)).all()


@router.get(
    "/health/detailed",
    response_model=HealthDetailedResponse,
    summary="Santé détaillée des composants",
)
def get_health_detailed(
    _: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
) -> HealthDetailedResponse:
    components: dict[str, Any] = {}
    degraded = False

    # ── Database ──────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_latency = round((time.monotonic() - t0) * 1000, 2)
        components["database"] = {"status": "ok", "latency_ms": db_latency}
    except Exception as exc:
        components["database"] = {"status": "critical", "detail": str(exc)}
        # DB down = critical (overrides degraded)
        return HealthDetailedResponse(
            status="critical",
            components=components,
            version=settings.app_version,
            uptime_seconds=round(time.time() - _START_TIME, 1),
        )

    # ── Redis ─────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        import redis as _redis
        r = _redis.from_url(settings.redis_url, socket_connect_timeout=1)
        r.ping()
        redis_latency = round((time.monotonic() - t0) * 1000, 2)
        components["redis"] = {"status": "ok", "latency_ms": redis_latency}
    except Exception:
        components["redis"] = {"status": "unavailable"}
        degraded = True

    # ── Celery ────────────────────────────────────────────────────────────────
    try:
        from tasks.analysis_tasks import celery_app
        inspector = celery_app.control.inspect(timeout=1.0)
        active = inspector.active()
        if active is None:
            raise RuntimeError("no workers")
        worker_count = len(active)
        components["celery"] = {"status": "ok", "workers": worker_count}
    except Exception:
        components["celery"] = {"status": "unavailable", "workers": 0}
        degraded = True

    # ── JWT keys ──────────────────────────────────────────────────────────────
    priv_ok = Path(settings.jwt_private_key_path).exists()
    pub_ok = Path(settings.jwt_public_key_path).exists()
    if priv_ok and pub_ok:
        components["jwt_keys"] = {"status": "ok", "algorithm": settings.jwt_algorithm}
    else:
        missing = []
        if not priv_ok:
            missing.append("private")
        if not pub_ok:
            missing.append("public")
        components["jwt_keys"] = {
            "status": "missing",
            "algorithm": settings.jwt_algorithm,
            "missing": missing,
        }
        degraded = True

    # ── Storage ───────────────────────────────────────────────────────────────
    upload_dir = Path(settings.upload_dir)
    if upload_dir.exists():
        components["storage"] = {"status": "ok", "type": "local", "path": settings.upload_dir}
    else:
        # Tenter MinIO
        try:
            from minio import Minio
            client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_root_user,
                secret_key=settings.minio_root_password,
                secure=settings.minio_use_ssl,
            )
            client.list_buckets()
            components["storage"] = {"status": "ok", "type": "minio"}
        except Exception:
            components["storage"] = {"status": "unavailable", "type": "unknown"}
            degraded = True

    overall = "degraded" if degraded else "ok"
    return HealthDetailedResponse(
        status=overall,
        components=components,
        version=settings.app_version,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )
