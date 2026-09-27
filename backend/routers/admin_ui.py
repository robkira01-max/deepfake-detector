"""Interface web d'administration — Jinja2 server-side rendering.

Routes sous /admin (cookie JWT httpOnly, rôle admin uniquement).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import settings
from core.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from database import get_db
from models.analysis import Analysis
from models.audit_log import AuditAction, AuditLog
from models.case import Case, CaseStatus
from models.media_file import MediaFile
from models.report import Report
from models.user import User, UserRole

router = APIRouter(prefix="/admin", tags=["Administration"])

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Cookie name pour la session admin
_COOKIE_NAME = "admin_token"
_COOKIE_MAX_AGE = settings.access_token_expire_minutes * 60


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _redirect_login(msg: str = "") -> RedirectResponse:
    url = "/admin/login"
    if msg:
        url += f"?error={msg}"
    resp = RedirectResponse(url, status_code=302)
    resp.delete_cookie(_COOKIE_NAME)
    return resp


def _get_admin_from_cookie(
    admin_token: Annotated[str | None, Cookie()] = None,
    db: Session = Depends(get_db),
) -> User:
    if not admin_token:
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})
    try:
        payload = decode_token(admin_token)
    except HTTPException:
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})

    if payload.get("type") != "access":
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})

    username = payload.get("sub", "")
    if username.startswith("mfa:"):
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})

    user = db.query(User).filter(User.username == username, User.is_active == True).first()
    if not user or user.role != UserRole.admin:
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})
    return user


AdminUser = Annotated[User, Depends(_get_admin_from_cookie)]


def _flash_params(msg: str, ftype: str = "success") -> str:
    return f"?flash={msg}&ftype={ftype}"


@dataclass
class Flash:
    message: str
    type: str  # success | error | info


def _get_flash(request: Request) -> Flash | None:
    msg = request.query_params.get("flash")
    if not msg:
        return None
    return Flash(message=msg, type=request.query_params.get("ftype", "info"))


# ── Redirect root ─────────────────────────────────────────────────────────────

@router.get("/", response_class=RedirectResponse, include_in_schema=False)
def admin_root(admin_token: Annotated[str | None, Cookie()] = None):
    if admin_token:
        return RedirectResponse("/admin/dashboard", status_code=302)
    return RedirectResponse("/admin/login", status_code=302)


# ── Login ─────────────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_form(request: Request, error: str | None = None):
    return templates.TemplateResponse(
        request,
        "admin/login.html",
        {"error": error},
    )


@router.post("/login", response_class=HTMLResponse, include_in_schema=False)
def login_submit(
    request: Request,
    db: Session = Depends(get_db),
    username: str = Form(...),
    password: str = Form(...),
):
    user = db.query(User).filter(User.username == username, User.is_active == True).first()

    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            request,
            "admin/login.html",
            {"error": "Identifiants invalides.", "username": username},
            status_code=401,
        )

    if user.role != UserRole.admin:
        return templates.TemplateResponse(
            request,
            "admin/login.html",
            {"error": "Accès réservé aux administrateurs.", "username": username},
            status_code=403,
        )

    token = create_access_token(subject=user.username, role=user.role.value)
    resp = RedirectResponse("/admin/dashboard", status_code=302)
    resp.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="strict",
        secure=not settings.debug,
        path="/admin",
    )
    return resp


@router.post("/logout", include_in_schema=False)
def logout():
    resp = RedirectResponse("/admin/login", status_code=302)
    resp.delete_cookie(_COOKIE_NAME, path="/admin")
    return resp


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard(
    request: Request,
    current_user: AdminUser,
    db: Session = Depends(get_db),
):
    from routers.dashboard import get_stats
    from database import SessionLocal
    # Réutiliser la logique de stats existante
    stats = get_stats(current_user, db)

    return templates.TemplateResponse(
        request,
        "admin/dashboard.html",
        {
            "active": "dashboard",
            "current_user": current_user,
            "stats": stats,
            "flash": _get_flash(request),
        },
    )


# ── Users ─────────────────────────────────────────────────────────────────────

@router.get("/users", response_class=HTMLResponse, include_in_schema=False)
def users_list(
    request: Request,
    current_user: AdminUser,
    db: Session = Depends(get_db),
):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return templates.TemplateResponse(
        request,
        "admin/users.html",
        {
            "active": "users",
            "current_user": current_user,
            "users": users,
            "flash": _get_flash(request),
        },
    )


@router.post("/users", include_in_schema=False)
def users_create(
    current_user: AdminUser,
    db: Session = Depends(get_db),
    username: str = Form(...),
    email: str = Form(...),
    role: str = Form(...),
    password: str = Form(...),
):
    if db.query(User).filter(User.username == username).first():
        return RedirectResponse(
            "/admin/users" + _flash_params(f"Utilisateur '{username}' existe déjà.", "error"),
            status_code=302,
        )
    if db.query(User).filter(User.email == email).first():
        return RedirectResponse(
            "/admin/users" + _flash_params(f"Email '{email}' déjà utilisé.", "error"),
            status_code=302,
        )

    try:
        role_enum = UserRole(role)
    except ValueError:
        return RedirectResponse(
            "/admin/users" + _flash_params("Rôle invalide.", "error"),
            status_code=302,
        )

    new_user = User(
        username=username,
        email=email,
        role=role_enum,
        hashed_password=hash_password(password),
        is_active=True,
    )
    db.add(new_user)

    log = AuditLog(
        user_id=current_user.id,
        user_username=current_user.username,
        action=AuditAction.USER_CREATED,
        resource_type="User",
        details={"created_username": username, "role": role, "via": "admin_ui"},
    )
    db.add(log)
    db.commit()

    return RedirectResponse(
        "/admin/users" + _flash_params(f"Utilisateur '{username}' créé avec succès."),
        status_code=302,
    )


@router.post("/users/{user_id}/toggle", include_in_schema=False)
def users_toggle(
    user_id: int,
    current_user: AdminUser,
    db: Session = Depends(get_db),
):
    if user_id == current_user.id:
        return RedirectResponse(
            "/admin/users" + _flash_params("Vous ne pouvez pas vous désactiver vous-même.", "error"),
            status_code=302,
        )
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return RedirectResponse(
            "/admin/users" + _flash_params("Utilisateur introuvable.", "error"),
            status_code=302,
        )
    user.is_active = not user.is_active
    db.commit()
    action = "activé" if user.is_active else "désactivé"
    return RedirectResponse(
        "/admin/users" + _flash_params(f"Utilisateur '{user.username}' {action}."),
        status_code=302,
    )


@router.post("/users/{user_id}/delete", include_in_schema=False)
def users_delete(
    user_id: int,
    current_user: AdminUser,
    db: Session = Depends(get_db),
):
    if user_id == current_user.id:
        return RedirectResponse(
            "/admin/users" + _flash_params("Vous ne pouvez pas vous supprimer vous-même.", "error"),
            status_code=302,
        )
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return RedirectResponse(
            "/admin/users" + _flash_params("Utilisateur introuvable.", "error"),
            status_code=302,
        )
    username = user.username
    db.delete(user)
    db.commit()
    return RedirectResponse(
        "/admin/users" + _flash_params(f"Utilisateur '{username}' supprimé."),
        status_code=302,
    )


# ── Cases ─────────────────────────────────────────────────────────────────────

@dataclass
class CaseRow:
    case: Case
    owner_username: str | None
    analysis_count: int
    report_count: int


@router.get("/cases", response_class=HTMLResponse, include_in_schema=False)
def cases_list(
    request: Request,
    current_user: AdminUser,
    db: Session = Depends(get_db),
    page: int = 1,
    status: str | None = None,
):
    PAGE_SIZE = 25
    q = db.query(Case)
    if status:
        try:
            q = q.filter(Case.status == CaseStatus(status))
        except ValueError:
            pass

    total = q.count()
    offset = (page - 1) * PAGE_SIZE
    cases_db = q.order_by(Case.created_at.desc()).offset(offset).limit(PAGE_SIZE).all()

    # Enrich avec owner username et counts
    rows: list[CaseRow] = []
    for c in cases_db:
        owner = db.query(User.username).filter(User.id == c.created_by_id).scalar()
        a_count = db.query(func.count(Analysis.id)).filter(Analysis.case_id == c.id).scalar() or 0
        r_count = db.query(func.count(Report.id)).filter(Report.case_id == c.id).scalar() or 0
        rows.append(CaseRow(case=c, owner_username=owner, analysis_count=a_count, report_count=r_count))

    return templates.TemplateResponse(
        request,
        "admin/cases.html",
        {
            "active": "cases",
            "current_user": current_user,
            "cases": rows,
            "total": total,
            "page": page,
            "page_size": PAGE_SIZE,
            "status_filter": status,
            "flash": _get_flash(request),
        },
    )


# ── Audit log ─────────────────────────────────────────────────────────────────

@router.get("/audit", response_class=HTMLResponse, include_in_schema=False)
def audit_log(
    request: Request,
    current_user: AdminUser,
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
    username: str | None = None,
    action: str | None = None,
):
    q = db.query(AuditLog).order_by(AuditLog.timestamp.desc())
    if username:
        q = q.filter(AuditLog.user_username == username)
    if action:
        try:
            q = q.filter(AuditLog.action == AuditAction(action))
        except ValueError:
            pass

    total = q.count()
    entries = q.offset(skip).limit(min(limit, 200)).all()
    all_actions = [a.value for a in AuditAction]

    return templates.TemplateResponse(
        request,
        "admin/audit.html",
        {
            "active": "audit",
            "current_user": current_user,
            "entries": entries,
            "total": total,
            "skip": skip,
            "limit": limit,
            "username_filter": username,
            "action_filter": action,
            "actions": all_actions,
            "flash": _get_flash(request),
        },
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_class=HTMLResponse, include_in_schema=False)
def health_view(
    request: Request,
    current_user: AdminUser,
    db: Session = Depends(get_db),
):
    from routers.dashboard import get_health_detailed
    health = get_health_detailed(current_user, db)

    return templates.TemplateResponse(
        request,
        "admin/health.html",
        {
            "active": "health",
            "current_user": current_user,
            "health": health,
            "now": datetime.now(timezone.utc),
            "flash": _get_flash(request),
        },
    )
