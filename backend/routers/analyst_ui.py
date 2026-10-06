"""Interface analyste — Jinja2 SSR (cookie JWT, rôles analyst + admin).

Routes sous /analyst — accessible aux rôles analyst et admin.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import settings
from core.ingestion import ingest_file, IngestionError
from core.security import create_access_token, decode_token, verify_password
from database import get_db
from models.analysis import Analysis, AnalysisStatus, Verdict
from models.audit_log import AuditAction, AuditLog
from models.case import Case, CaseStatus, Jurisdiction
from models.document_analysis import DocumentAnalysis
from models.feedback import AnalysisFeedback, FeedbackType, TrueVerdict
from models.media_file import MediaFile, MediaType
from models.user import User, UserRole

router = APIRouter(prefix="/analyst", tags=["Interface Analyste"])

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_COOKIE_NAME  = "analyst_token"
_COOKIE_AGE   = settings.access_token_expire_minutes * 60
_ALLOWED_ROLES = {UserRole.admin, UserRole.analyst}

# ── Poids des composantes ─────────────────────────────────────────────────────
_COMPONENTS_AV = [
    {"key": "score_video_texture",  "label": "Texture vidéo",       "icon": "🎥", "weight": "20%"},
    {"key": "score_video_temporal", "label": "Cohérence temporelle", "icon": "⏱",  "weight": "15%"},
    {"key": "score_rppg",           "label": "Signal rPPG",          "icon": "❤️", "weight": "18%"},
    {"key": "score_biometrics",     "label": "Biométrie faciale",    "icon": "👤", "weight": "10%"},
    {"key": "score_audio_model",    "label": "Modèle audio",         "icon": "🎵", "weight": "20%"},
    {"key": "score_audio_phase",    "label": "Phase audio",          "icon": "〰", "weight": "10%"},
    {"key": "score_metadata",       "label": "Métadonnées",          "icon": "📋", "weight": "7%"},
]
_COMPONENTS_DOC = [
    {"key": "score_ela",          "label": "Analyse ELA",        "icon": "🔍", "weight": "40%"},
    {"key": "score_clone",        "label": "Clone detection",    "icon": "⎘",  "weight": "25%"},
    {"key": "score_metadata_doc", "label": "Métadonnées",        "icon": "📋", "weight": "30%"},
    {"key": "score_font",         "label": "Cohérence polices",  "icon": "🖋", "weight": "25%"},
    {"key": "score_text_ai",      "label": "Texte IA",           "icon": "🤖", "weight": "35%"},
]


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _redirect_login(msg: str = "") -> RedirectResponse:
    url = "/analyst/login"
    if msg:
        url += f"?error={msg}"
    resp = RedirectResponse(url, status_code=302)
    resp.delete_cookie(_COOKIE_NAME)
    return resp


def _get_analyst_from_cookie(
    analyst_token: Annotated[str | None, Cookie()] = None,
    db: Session = Depends(get_db),
) -> User:
    if not analyst_token:
        raise HTTPException(status_code=302, headers={"Location": "/analyst/login"})
    try:
        payload = decode_token(analyst_token)
        username: str = payload.get("sub", "")
        if not username:
            raise ValueError("invalid sub")
        user = db.query(User).filter(User.username == username).first()
        if not user or not user.is_active:
            raise ValueError("user inactive")
        if user.role not in _ALLOWED_ROLES:
            raise ValueError("insufficient role")
        return user
    except Exception:
        raise HTTPException(status_code=302, headers={"Location": "/analyst/login"})


def _audit(db: Session, user: User, action: AuditAction, detail: str = "") -> None:
    db.add(AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=action,
        details={"info": detail} if detail else None,
    ))
    db.commit()


# ── Login / Logout ─────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
def get_login(request: Request, error: str = "") -> HTMLResponse:
    return templates.TemplateResponse(request, "analyst/login.html", {"error": error})


@router.post("/login")
def post_login(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    db: Session = Depends(get_db),
) -> RedirectResponse:
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_active or not verify_password(password, user.hashed_password):
        return _redirect_login("Identifiants incorrects")
    if user.role not in _ALLOWED_ROLES:
        return _redirect_login("Accès réservé aux analystes")

    token = create_access_token(user.username, user.role.value)
    resp = RedirectResponse("/analyst/dashboard", status_code=302)
    resp.set_cookie(
        _COOKIE_NAME, token,
        max_age=_COOKIE_AGE,
        httponly=True,
        secure=not settings.debug,
        samesite="strict",
        path="/analyst",
    )
    _audit(db, user, AuditAction.USER_LOGIN, "analyst_ui")
    return resp


@router.post("/logout")
def post_logout() -> RedirectResponse:
    resp = RedirectResponse("/analyst/login", status_code=302)
    resp.delete_cookie(_COOKIE_NAME, path="/analyst")
    return resp


@router.get("/", response_class=HTMLResponse)
def root() -> RedirectResponse:
    return RedirectResponse("/analyst/dashboard", status_code=302)


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    flash: str = "",
    flash_type: str = "info",
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    q = db.query(Case)
    if user.role != UserRole.admin:
        q = q.filter(Case.created_by_id == user.id)
    cases = q.order_by(Case.created_at.desc()).limit(50).all()

    case_ids = [c.id for c in cases]
    counts_av = (
        db.query(Analysis.case_id, func.count(Analysis.id))
        .filter(Analysis.case_id.in_(case_ids))
        .group_by(Analysis.case_id)
        .all()
    ) if case_ids else []
    counts_doc = (
        db.query(DocumentAnalysis.case_id, func.count(DocumentAnalysis.id))
        .filter(DocumentAnalysis.case_id.in_(case_ids))
        .group_by(DocumentAnalysis.case_id)
        .all()
    ) if case_ids else []
    case_analysis_counts: dict[int, int] = {}
    for cid, cnt in counts_av + counts_doc:
        case_analysis_counts[cid] = case_analysis_counts.get(cid, 0) + cnt

    total  = len(cases)
    open_c = sum(1 for c in cases if c.status in (CaseStatus.open, CaseStatus.in_analysis))
    base_q = db.query(func.count(Analysis.id))
    if user.role != UserRole.admin:
        base_q = base_q.filter(Analysis.requested_by_id == user.id)
    deepfakes = base_q.filter(Analysis.verdict == Verdict.deepfake).scalar() or 0
    authentic = base_q.filter(Analysis.verdict == Verdict.authentic).scalar() or 0

    return templates.TemplateResponse(request, "analyst/dashboard.html", {
        "user": user,
        "cases": cases,
        "case_analysis_counts": case_analysis_counts,
        "stats": {
            "total": total, "open": open_c,
            "deepfakes": deepfakes, "authentic": authentic,
            "analyses": sum(case_analysis_counts.values()),
        },
        "flash": flash,
        "flash_type": flash_type,
    })


# ── Analytics ────────────────────────────────────────────────────────────────

@router.get("/analytics", response_class=HTMLResponse)
def analytics(
    request: Request,
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    import json, statistics

    # ── Verdicts ──────────────────────────────────────────────────────────────
    from collections import defaultdict
    from datetime import datetime as _dt

    completed = (
        db.query(Analysis)
        .filter(Analysis.status == AnalysisStatus.completed)
        .order_by(Analysis.completed_at.desc())
        .all()
    )

    deepfake_count = sum(1 for a in completed if a.verdict == Verdict.deepfake)
    authentic_count = sum(1 for a in completed if a.verdict == Verdict.authentic)
    undetermined_count = sum(1 for a in completed if a.verdict not in (Verdict.deepfake, Verdict.authentic))
    total_analyses = len(completed)
    deepfake_rate = round(deepfake_count / total_analyses * 100, 1) if total_analyses else 0.0

    verdict_dist = [
        {"label": "Deepfakes détectés", "count": deepfake_count,    "color": "#ff4757", "pct": round(deepfake_count / total_analyses * 100, 1) if total_analyses else 0},
        {"label": "Authentiques",        "count": authentic_count,   "color": "#00e676", "pct": round(authentic_count / total_analyses * 100, 1) if total_analyses else 0},
        {"label": "Indéterminés",        "count": undetermined_count, "color": "#ffc107", "pct": round(undetermined_count / total_analyses * 100, 1) if total_analyses else 0},
    ]
    verdict_chart = {
        "labels": [v["label"] for v in verdict_dist],
        "values": [v["count"] for v in verdict_dist],
        "colors": [v["color"] for v in verdict_dist],
        "total": total_analyses,
    }

    # ── Tendances (monthly / weekly / daily) ──────────────────────────────────
    def _bucket(analyses, fmt):
        buckets = defaultdict(lambda: {"total": 0, "deepfake": 0, "authentic": 0, "undetermined": 0})
        for a in analyses:
            if not a.completed_at:
                continue
            if fmt == "daily":
                k = a.completed_at.strftime("%Y-%m-%d")
            elif fmt == "weekly":
                iso = a.completed_at.isocalendar()
                k = f"{iso[0]}-W{iso[1]:02d}"
            else:
                k = a.completed_at.strftime("%Y-%m")
            buckets[k]["total"] += 1
            if a.verdict == Verdict.deepfake:
                buckets[k]["deepfake"] += 1
            elif a.verdict == Verdict.authentic:
                buckets[k]["authentic"] += 1
            else:
                buckets[k]["undetermined"] += 1
        return [{"period": k, **buckets[k]} for k in sorted(buckets.keys())[-12:]]

    trends_json = json.dumps({
        "monthly": _bucket(completed, "monthly"),
        "weekly":  _bucket(completed, "weekly"),
        "daily":   _bucket(completed, "daily"),
    })

    # ── Moteurs ───────────────────────────────────────────────────────────────
    _ENG = [
        ("score_video_texture",  "Texture vidéo",      20),
        ("score_video_temporal", "Cohérence temporelle", 15),
        ("score_rppg",           "Signal rPPG",         18),
        ("score_biometrics",     "Biométrie faciale",   10),
        ("score_audio_model",    "Modèle audio",        20),
        ("score_audio_phase",    "Phase audio",         10),
        ("score_metadata",       "Métadonnées",          7),
    ]
    engines_data = []
    for field, label, _w in _ENG:
        vals = [getattr(a, field) for a in completed if getattr(a, field) is not None]
        if vals:
            mean = statistics.mean(vals)
            color = "#ff4757" if mean >= 0.7 else ("#ffc107" if mean >= 0.5 else "#00e676")
            engines_data.append({
                "label": label,
                "mean_fmt": f"{mean:.2f}",
                "pct": round(mean * 100, 1),
                "color": color,
            })
        else:
            engines_data.append({"label": label, "mean_fmt": "—", "pct": 0, "color": "var(--text-dim)"})

    # ── Dossiers ──────────────────────────────────────────────────────────────
    all_cases = db.query(Case).all()
    cases_total = len(all_cases)
    status_counts: dict[str, int] = defaultdict(int)
    jur_counts: dict[str, int] = defaultdict(int)
    for c in all_cases:
        status_counts[c.status.value] += 1
        jur_counts[c.jurisdiction.value] += 1

    _STATUS_COLORS = {
        "open": "#00e676", "in_analysis": "#00cba4",
        "completed": "#5a7a99", "archived": "var(--text-dim)"
    }
    by_status = [
        {"label": k, "count": v, "pct": round(v / cases_total * 100) if cases_total else 0,
         "color": _STATUS_COLORS.get(k, "var(--accent)")}
        for k, v in sorted(status_counts.items(), key=lambda x: -x[1])
    ]
    by_jurisdiction = [
        {"label": k, "count": v, "pct": round(v / cases_total * 100) if cases_total else 0}
        for k, v in sorted(jur_counts.items(), key=lambda x: -x[1])
    ]

    # ── Dossiers ouverts ──────────────────────────────────────────────────────
    open_cases = sum(1 for c in all_cases if c.status in (CaseStatus.open, CaseStatus.in_analysis))

    return templates.TemplateResponse(request, "analyst/analytics.html", {
        "user": user,
        "total_analyses": total_analyses,
        "deepfake_count": deepfake_count,
        "authentic_count": authentic_count,
        "deepfake_rate": deepfake_rate,
        "open_cases": open_cases,
        "verdict_dist": verdict_dist,
        "verdict_chart_json": json.dumps(verdict_chart),
        "trends_json": trends_json,
        "engines": engines_data,
        "cases_total": cases_total,
        "by_status": by_status,
        "by_jurisdiction": by_jurisdiction,
    })


# ── Cases ─────────────────────────────────────────────────────────────────────

@router.get("/cases/new", response_class=HTMLResponse)
def get_new_case(
    request: Request,
    error: str = "",
    user: User = Depends(_get_analyst_from_cookie),
) -> HTMLResponse:
    return templates.TemplateResponse(request, "analyst/case_new.html", {
        "user": user, "error": error,
    })


@router.post("/cases/new")
def post_new_case(
    case_number: Annotated[str, Form()],
    title:       Annotated[str, Form()],
    jurisdiction: Annotated[str, Form()] = "federal",
    description: Annotated[str | None, Form()] = None,
    plaintiff:   Annotated[str | None, Form()] = None,
    defendant:   Annotated[str | None, Form()] = None,
    counsel:     Annotated[str | None, Form()] = None,
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    existing = db.query(Case).filter(Case.case_number == case_number).first()
    if existing:
        return RedirectResponse(
            "/analyst/cases/new?error=Numéro+de+dossier+déjà+utilisé",
            status_code=302,
        )
    try:
        jur = Jurisdiction(jurisdiction)
    except ValueError:
        jur = Jurisdiction.other

    case = Case(
        case_number=case_number,
        title=title,
        jurisdiction=jur,
        description=description or "",
        plaintiff=plaintiff or None,
        defendant=defendant or None,
        counsel=counsel or None,
        created_by_id=user.id,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    _audit(db, user, AuditAction.CASE_CREATED, f"case_id={case.id} number={case_number}")
    return RedirectResponse(
        f"/analyst/cases/{case.id}?flash=Dossier+créé+avec+succès&flash_type=success",
        status_code=302,
    )


@router.get("/cases/{case_id}", response_class=HTMLResponse)
def case_detail(
    case_id: int,
    request: Request,
    flash: str = "",
    flash_type: str = "info",
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        return RedirectResponse(
            "/analyst/dashboard?flash=Dossier+introuvable&flash_type=error", status_code=302,
        )
    if user.role != UserRole.admin and case.created_by_id != user.id:
        return RedirectResponse(
            "/analyst/dashboard?flash=Accès+refusé&flash_type=error", status_code=302,
        )

    files = (db.query(MediaFile).filter(MediaFile.case_id == case_id)
             .order_by(MediaFile.ingested_at.desc()).all())

    analyses: dict[int, Analysis | DocumentAnalysis | None] = {}
    for f in files:
        av = (db.query(Analysis).filter(Analysis.media_file_id == f.id)
              .order_by(Analysis.started_at.desc()).first())
        doc = (db.query(DocumentAnalysis).filter(DocumentAnalysis.media_file_id == f.id)
               .order_by(DocumentAnalysis.started_at.desc()).first())
        if av and doc:
            analyses[f.id] = av if av.status != AnalysisStatus.pending else doc
        else:
            analyses[f.id] = av or doc

    return templates.TemplateResponse(request, "analyst/case_detail.html", {
        "user": user,
        "case": case,
        "files": files,
        "analyses": analyses,
        "flash": flash,
        "flash_type": flash_type,
        "max_mb": settings.max_file_size_mb,
    })


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/cases/{case_id}/upload")
async def upload_file(
    case_id: int,
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        return RedirectResponse("/analyst/dashboard", status_code=302)
    if user.role != UserRole.admin and case.created_by_id != user.id:
        return RedirectResponse(
            f"/analyst/cases/{case_id}?flash=Accès+refusé&flash_type=error", status_code=302,
        )

    ext = Path(file.filename or "file").suffix.lower()
    all_exts = settings.allowed_extensions | settings.allowed_document_exts
    if ext not in all_exts:
        return RedirectResponse(
            f"/analyst/cases/{case_id}?flash=Extension+non+autorisée+{ext}&flash_type=error",
            status_code=302,
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        ip = request.client.host if request.client else ""
        result = ingest_file(
            source_path=tmp_path,
            original_filename=file.filename or "upload",
            case_id=case_id,
            uploader_id=user.id,
            db=db,
            ip_address=ip,
        )
        mf = result.media_file
        _audit(db, user, AuditAction.FILE_INGESTED, f"file_id={mf.id} name={mf.original_filename}")
        return RedirectResponse(
            f"/analyst/cases/{case_id}?flash=Fichier+{mf.original_filename}+ingéré&flash_type=success",
            status_code=302,
        )
    except IngestionError as exc:
        return RedirectResponse(
            f"/analyst/cases/{case_id}?flash={str(exc)[:80]}&flash_type=error",
            status_code=302,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Start analysis ────────────────────────────────────────────────────────────

@router.post("/cases/{case_id}/analyze/{file_id}")
def start_analysis(
    case_id: int,
    file_id: int,
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    mf = db.query(MediaFile).filter(
        MediaFile.id == file_id, MediaFile.case_id == case_id,
    ).first()
    if not mf:
        return RedirectResponse(
            f"/analyst/cases/{case_id}?flash=Fichier+introuvable&flash_type=error", status_code=302,
        )

    if mf.media_type == MediaType.document:
        doc = DocumentAnalysis(
            case_id=case_id, media_file_id=mf.id,
            status=AnalysisStatus.pending, requested_by_id=user.id,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        try:
            from tasks.analysis_tasks import run_document_analysis
            task = run_document_analysis.delay(doc.id, mf.id)
            doc.celery_task_id = task.id
            doc.status = AnalysisStatus.running
            db.commit()
        except Exception as exc:
            doc.status = AnalysisStatus.failed
            doc.error_message = str(exc)
            db.commit()
        _audit(db, user, AuditAction.DOCUMENT_ANALYSIS_STARTED, f"doc_analysis_id={doc.id}")
        return RedirectResponse(f"/analyst/doc/{doc.id}", status_code=302)

    analysis = Analysis(
        case_id=case_id, media_file_id=mf.id,
        status=AnalysisStatus.pending, requested_by_id=user.id,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    try:
        from tasks.analysis_tasks import run_deepfake_analysis
        task = run_deepfake_analysis.delay(analysis.id, mf.id)
        analysis.celery_task_id = task.id
        analysis.status = AnalysisStatus.running
        db.commit()
    except Exception as exc:
        analysis.status = AnalysisStatus.failed
        analysis.error_message = str(exc)
        db.commit()
    _audit(db, user, AuditAction.ANALYSIS_STARTED, f"analysis_id={analysis.id}")
    return RedirectResponse(f"/analyst/analyze/{analysis.id}", status_code=302)


# ── Results — Audio/Video ─────────────────────────────────────────────────────

def _build_av_components(analysis: Analysis) -> tuple[list[dict], list[str], list[float]]:
    comps, labels, raw_scores = [], [], []
    for c in _COMPONENTS_AV:
        score = getattr(analysis, c["key"], None) or 0.0
        comps.append({**c, "score": score})
        labels.append(c["label"])
        raw_scores.append(round(score, 4))
    return comps, labels, raw_scores


@router.get("/analyze/{analysis_id}", response_class=HTMLResponse)
def av_results(
    analysis_id: int,
    request: Request,
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        return RedirectResponse(
            "/analyst/dashboard?flash=Analyse+introuvable&flash_type=error", status_code=302,
        )
    case = db.query(Case).filter(Case.id == analysis.case_id).first()
    if user.role != UserRole.admin and (not case or case.created_by_id != user.id):
        return RedirectResponse(
            "/analyst/dashboard?flash=Accès+refusé&flash_type=error", status_code=302,
        )
    mf = db.query(MediaFile).filter(MediaFile.id == analysis.media_file_id).first()
    components, component_labels, component_raw_scores = _build_av_components(analysis)

    compression = None
    if analysis.xai_shap_values and "compression" in analysis.xai_shap_values:
        compression = analysis.xai_shap_values["compression"]

    return templates.TemplateResponse(request, "analyst/results.html", {
        "user": user,
        "analysis": analysis,
        "media_file": mf,
        "case": case,
        "components": components,
        "component_labels": component_labels,
        "component_raw_scores": component_raw_scores,
        "compression": compression,
        "analysis_type": "av",
    })


@router.get("/analyze/{analysis_id}/poll", response_class=HTMLResponse)
def av_poll(
    analysis_id: int,
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    status = analysis.status.value if analysis else "failed"
    results_url = f"/analyst/analyze/{analysis_id}"

    if status in ("completed", "failed"):
        return HTMLResponse(
            content=f'<div data-status="{status}" data-results-url="{results_url}"></div>'
        )
    return HTMLResponse(content=(
        f'<div id="poll-zone" hx-get="/analyst/analyze/{analysis_id}/poll" '
        f'hx-trigger="every 2s" hx-swap="outerHTML">'
        f'<div data-status="{status}" data-results-url="{results_url}">'
        f'<span class="badge badge-info pulse-ring">{status}…</span>'
        f'</div></div>'
    ))


# ── Results — Document ────────────────────────────────────────────────────────

def _build_doc_components(analysis: DocumentAnalysis) -> tuple[list[dict], list[str], list[float]]:
    comps, labels, raw_scores = [], [], []
    for c in _COMPONENTS_DOC:
        score = getattr(analysis, c["key"], None) or 0.0
        comps.append({**c, "score": score})
        labels.append(c["label"])
        raw_scores.append(round(score, 4))
    return comps, labels, raw_scores


@router.get("/doc/{doc_analysis_id}", response_class=HTMLResponse)
def doc_results(
    doc_analysis_id: int,
    request: Request,
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    analysis = db.query(DocumentAnalysis).filter(DocumentAnalysis.id == doc_analysis_id).first()
    if not analysis:
        return RedirectResponse(
            "/analyst/dashboard?flash=Analyse+introuvable&flash_type=error", status_code=302,
        )
    case = db.query(Case).filter(Case.id == analysis.case_id).first()
    if user.role != UserRole.admin and (not case or case.created_by_id != user.id):
        return RedirectResponse(
            "/analyst/dashboard?flash=Accès+refusé&flash_type=error", status_code=302,
        )
    mf = db.query(MediaFile).filter(MediaFile.id == analysis.media_file_id).first()
    components, component_labels, component_raw_scores = _build_doc_components(analysis)

    class _AnalysisProxy:
        """Adapts DocumentAnalysis to the shared results.html template interface."""
        def __init__(self, doc: DocumentAnalysis) -> None:
            self.id = doc.id
            self.case_id = doc.case_id
            self.status = doc.status
            self.final_score = doc.final_score
            self.confidence_low = None
            self.confidence_high = None
            self.started_at = doc.started_at
            self.completed_at = doc.completed_at
            self.duration_seconds = doc.duration_seconds
            self.error_message = doc.error_message
            self.xai_plain_explanation = doc.xai_plain_explanation
            self.xai_suspicious_timecodes = None
            self.xai_shap_values = None
            self.model_far = None
            self.model_frr = None
            self.model_eer = None
            self.model_auc = None
            self.models_used = doc.models_used
            score = doc.final_score or 0.0
            if score >= 0.65:
                self.verdict = Verdict.deepfake
            elif score <= 0.35:
                self.verdict = Verdict.authentic
            else:
                self.verdict = Verdict.undetermined

    return templates.TemplateResponse(request, "analyst/results.html", {
        "user": user,
        "analysis": _AnalysisProxy(analysis),
        "media_file": mf,
        "case": case,
        "components": components,
        "component_labels": component_labels,
        "component_raw_scores": component_raw_scores,
        "compression": None,
        "analysis_type": "document",
    })


@router.get("/doc/{doc_analysis_id}/poll", response_class=HTMLResponse)
def doc_poll(
    doc_analysis_id: int,
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    analysis = db.query(DocumentAnalysis).filter(DocumentAnalysis.id == doc_analysis_id).first()
    status = analysis.status.value if analysis else "failed"
    results_url = f"/analyst/doc/{doc_analysis_id}"

    if status in ("completed", "failed"):
        return HTMLResponse(
            content=f'<div data-status="{status}" data-results-url="{results_url}"></div>'
        )
    return HTMLResponse(content=(
        f'<div id="poll-zone" hx-get="/analyst/doc/{doc_analysis_id}/poll" '
        f'hx-trigger="every 2s" hx-swap="outerHTML">'
        f'<div data-status="{status}" data-results-url="{results_url}">'
        f'<span class="badge badge-info pulse-ring">{status}…</span>'
        f'</div></div>'
    ))


# ── Feedback active learning ──────────────────────────────────────────────────

@router.post("/feedback/{analysis_id}")
def post_feedback(
    analysis_id: int,
    request: Request,
    feedback_type: Annotated[str, Form()],
    true_verdict: Annotated[str | None, Form()] = None,
    analyst_notes: Annotated[str | None, Form()] = None,
    confidence_rating: Annotated[int | None, Form()] = None,
    doc: bool = False,
    user: User = Depends(_get_analyst_from_cookie),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    try:
        fb_type = FeedbackType(feedback_type)
    except ValueError:
        fb_type = FeedbackType.uncertain

    tv = None
    if true_verdict:
        try:
            tv = TrueVerdict(true_verdict)
        except ValueError:
            pass

    kwargs: dict = {
        "feedback_type": fb_type,
        "true_verdict": tv,
        "analyst_notes": analyst_notes or None,
        "confidence_rating": max(1, min(5, confidence_rating)) if confidence_rating else None,
        "submitted_by_id": user.id,
    }
    if doc:
        kwargs["document_analysis_id"] = analysis_id
        redirect_url = f"/analyst/doc/{analysis_id}?flash=Feedback+enregistr%C3%A9&flash_type=success"
    else:
        kwargs["analysis_id"] = analysis_id
        redirect_url = f"/analyst/results/{analysis_id}?flash=Feedback+enregistr%C3%A9&flash_type=success"

    db.add(AnalysisFeedback(**kwargs))
    db.commit()
    _audit(db, user, AuditAction.FEEDBACK_SUBMITTED, f"analysis_id={analysis_id} doc={doc} type={feedback_type}")
    return RedirectResponse(redirect_url, status_code=302)
