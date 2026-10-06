"""Tableau de bord analytique — tendances temporelles, métriques moteurs, Prometheus."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.security import require_analyst
from database import get_db
from models.analysis import Analysis, AnalysisStatus, Verdict
from models.case import Case, CaseStatus
from models.user import User

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ── Prometheus registry (isolé pour éviter les collisions de tests) ───────────
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

_registry = CollectorRegistry()

prom_analyses_total = Counter(
    "deepfake_analyses_total",
    "Total analyses by verdict",
    ["verdict"],
    registry=_registry,
)
prom_analyses_duration = Histogram(
    "deepfake_analysis_duration_seconds",
    "Analysis duration in seconds",
    buckets=[10, 30, 60, 120, 300, 600, 1800],
    registry=_registry,
)
prom_cases_open = Gauge(
    "deepfake_cases_open_total",
    "Number of open cases",
    registry=_registry,
)
prom_final_score = Histogram(
    "deepfake_final_score",
    "Distribution of final deepfake scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=_registry,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/verdicts",
    summary="Répartition globale des verdicts",
    responses={401: {"description": "Non authentifié"}},
)
def verdict_distribution(
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    rows = (
        db.query(Analysis.verdict, func.count(Analysis.id).label("count"))
        .filter(Analysis.status == AnalysisStatus.completed)
        .group_by(Analysis.verdict)
        .all()
    )
    total = sum(r.count for r in rows)
    distribution = {
        (r.verdict.value if r.verdict else "NULL"): {
            "count": r.count,
            "pct": round(r.count / total * 100, 1) if total else 0.0,
        }
        for r in rows
    }
    return {"total": total, "distribution": distribution}


@router.get(
    "/trends",
    summary="Tendances temporelles — analyses par période",
    responses={401: {"description": "Non authentifié"}},
)
def trends(
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
    period: str = Query(default="monthly", pattern="^(daily|weekly|monthly)$"),
    limit: int = Query(default=12, ge=1, le=52),
) -> dict:
    """Retourne les comptages d'analyses groupés par période (daily/weekly/monthly)."""
    completed = (
        db.query(Analysis)
        .filter(
            Analysis.status == AnalysisStatus.completed,
            Analysis.completed_at.isnot(None),
        )
        .order_by(Analysis.completed_at.desc())
        .all()
    )

    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0,
        "deepfake": 0,
        "authentic": 0,
        "undetermined": 0,
    })

    for a in completed:
        dt = a.completed_at
        if period == "daily":
            key = dt.strftime("%Y-%m-%d")
        elif period == "weekly":
            key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        else:
            key = dt.strftime("%Y-%m")

        buckets[key]["total"] += 1
        if a.verdict == Verdict.deepfake:
            buckets[key]["deepfake"] += 1
        elif a.verdict == Verdict.authentic:
            buckets[key]["authentic"] += 1
        else:
            buckets[key]["undetermined"] += 1

    # Trier et limiter
    sorted_keys = sorted(buckets.keys(), reverse=True)[:limit]
    data = [{"period": k, **buckets[k]} for k in reversed(sorted_keys)]

    return {"period": period, "data": data}


@router.get(
    "/engines",
    summary="Distribution des scores par moteur ML",
    responses={401: {"description": "Non authentifié"}},
)
def engine_scores(
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Moyennes et écarts-types des 7 scores composantes pour les analyses complétées."""
    import statistics

    completed = (
        db.query(Analysis)
        .filter(Analysis.status == AnalysisStatus.completed)
        .all()
    )

    engines = {
        "video_texture": [],
        "video_temporal": [],
        "rppg": [],
        "biometrics": [],
        "audio_model": [],
        "audio_phase": [],
        "metadata": [],
    }

    for a in completed:
        if a.score_video_texture is not None:
            engines["video_texture"].append(a.score_video_texture)
        if a.score_video_temporal is not None:
            engines["video_temporal"].append(a.score_video_temporal)
        if a.score_rppg is not None:
            engines["rppg"].append(a.score_rppg)
        if a.score_biometrics is not None:
            engines["biometrics"].append(a.score_biometrics)
        if a.score_audio_model is not None:
            engines["audio_model"].append(a.score_audio_model)
        if a.score_audio_phase is not None:
            engines["audio_phase"].append(a.score_audio_phase)
        if a.score_metadata is not None:
            engines["metadata"].append(a.score_metadata)

    result = {}
    for name, vals in engines.items():
        if not vals:
            result[name] = {"count": 0, "mean": None, "stdev": None, "min": None, "max": None}
        else:
            result[name] = {
                "count": len(vals),
                "mean": round(statistics.mean(vals), 4),
                "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }

    return {"total_analyses": len(completed), "engines": result}


@router.get(
    "/cases-summary",
    summary="Résumé des dossiers par statut et juridiction",
    responses={401: {"description": "Non authentifié"}},
)
def cases_summary(
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    by_status = (
        db.query(Case.status, func.count(Case.id).label("count"))
        .group_by(Case.status)
        .all()
    )
    by_jurisdiction = (
        db.query(Case.jurisdiction, func.count(Case.id).label("count"))
        .group_by(Case.jurisdiction)
        .all()
    )
    return {
        "by_status": {r.status.value: r.count for r in by_status},
        "by_jurisdiction": {r.jurisdiction.value: r.count for r in by_jurisdiction},
    }


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Endpoint Prometheus — scrape metrics",
    include_in_schema=False,
)
def prometheus_metrics(
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> PlainTextResponse:
    """Métriques Prometheus compatibles avec prometheus scrape."""
    # Rafraîchir les métriques depuis la DB
    rows = (
        db.query(Analysis.verdict, func.count(Analysis.id).label("count"))
        .filter(Analysis.status == AnalysisStatus.completed)
        .group_by(Analysis.verdict)
        .all()
    )
    for r in rows:
        label = r.verdict.value if r.verdict else "null"
        try:
            prom_analyses_total.labels(verdict=label)._value.set(r.count)
        except Exception:
            pass

    open_count = db.query(func.count(Case.id)).filter(
        Case.status.in_([CaseStatus.open, CaseStatus.in_analysis])
    ).scalar() or 0
    prom_cases_open.set(open_count)

    durations = (
        db.query(Analysis.duration_seconds)
        .filter(
            Analysis.status == AnalysisStatus.completed,
            Analysis.duration_seconds.isnot(None),
        )
        .all()
    )
    for (dur,) in durations:
        prom_analyses_duration.observe(dur)

    scores = (
        db.query(Analysis.final_score)
        .filter(
            Analysis.status == AnalysisStatus.completed,
            Analysis.final_score.isnot(None),
        )
        .all()
    )
    for (score,) in scores:
        prom_final_score.observe(score)

    output = generate_latest(_registry)
    return PlainTextResponse(content=output.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
