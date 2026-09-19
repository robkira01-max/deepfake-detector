"""Routes de gestion des dossiers (CRUD) — accessible analyst+."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from core.security import get_current_user, require_analyst, require_any
from database import get_db
from models.case import Case, CaseStatus, Jurisdiction
from models.user import User, UserRole

router = APIRouter(prefix="/cases", tags=["Dossiers"])


# ── Schémas ───────────────────────────────────────────────────────────────────

class CaseCreateRequest(BaseModel):
    title: str
    description: str | None = None
    jurisdiction: Jurisdiction = Jurisdiction.federal
    plaintiff: str | None = None
    defendant: str | None = None
    counsel: str | None = None


class CaseResponse(BaseModel):
    id: int
    case_number: str
    title: str
    description: str | None
    jurisdiction: Jurisdiction
    status: CaseStatus
    plaintiff: str | None
    defendant: str | None
    counsel: str | None
    created_by_id: int
    created_at: datetime
    updated_at: datetime
    retain_until: datetime | None
    media_count: int = 0

    model_config = {"from_attributes": True}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=CaseResponse, status_code=201, summary="Créer un dossier")
def create_case(
    body: CaseCreateRequest,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> Case:
    case_number = _generate_case_number()
    retain_until = datetime.now(timezone.utc) + timedelta(days=365 * settings.retention_years)

    case = Case(
        case_number=case_number,
        title=body.title,
        description=body.description,
        jurisdiction=body.jurisdiction,
        plaintiff=body.plaintiff,
        defendant=body.defendant,
        counsel=body.counsel,
        created_by_id=current_user.id,
        retain_until=retain_until,
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    _audit_case(db, current_user, case, "CASE_CREATED")
    response = CaseResponse.model_validate(case)
    response.media_count = len(case.media_files)
    return response


@router.get("/", response_model=list[CaseResponse], summary="Lister les dossiers")
def list_cases(
    current_user: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
    status_filter: CaseStatus | None = None,
    skip: int = 0,
    limit: int = 50,
) -> list[CaseResponse]:
    q = db.query(Case)
    if status_filter:
        q = q.filter(Case.status == status_filter)
    cases = q.order_by(Case.created_at.desc()).offset(skip).limit(limit).all()

    results = []
    for c in cases:
        r = CaseResponse.model_validate(c)
        r.media_count = len(c.media_files)
        results.append(r)
    return results


@router.get("/{case_id}", response_model=CaseResponse, summary="Détail d'un dossier")
def get_case(
    case_id: int,
    current_user: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
) -> CaseResponse:
    case = _get_or_404(db, case_id)
    r = CaseResponse.model_validate(case)
    r.media_count = len(case.media_files)
    return r


@router.patch("/{case_id}", response_model=CaseResponse, summary="Mettre à jour un dossier")
def update_case(
    case_id: int,
    body: CaseCreateRequest,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> CaseResponse:
    case = _get_or_404(db, case_id)
    # H5 — IDOR: vérification de propriété
    if current_user.role != UserRole.admin and case.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Accès refusé — vous n'êtes pas le créateur de ce dossier")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(case, field, value)
    db.commit()
    db.refresh(case)
    _audit_case(db, current_user, case, "CASE_UPDATED")
    r = CaseResponse.model_validate(case)
    r.media_count = len(case.media_files)
    return r


@router.post("/{case_id}/archive", response_model=CaseResponse, summary="Archiver un dossier")
def archive_case(
    case_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> CaseResponse:
    case = _get_or_404(db, case_id)
    # H5 — IDOR: vérification de propriété
    if current_user.role != UserRole.admin and case.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Accès refusé — vous n'êtes pas le créateur de ce dossier")
    case.status = CaseStatus.archived
    case.closed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(case)
    _audit_case(db, current_user, case, "CASE_ARCHIVED")
    r = CaseResponse.model_validate(case)
    r.media_count = len(case.media_files)
    return r


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _get_or_404(db: Session, case_id: int) -> Case:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Dossier {case_id} introuvable")
    return case


def _generate_case_number() -> str:
    now = datetime.now(timezone.utc)
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"CA-{now.year}-{now.month:02d}-{short_uuid}"


def _audit_case(db: Session, user: User, case: Case, action: str) -> None:
    from models.audit_log import AuditLog, AuditAction
    from core.chain_of_custody import sign_audit_entry
    data = {
        "action": action,
        "user_id": user.id,
        "case_id": case.id,
        "case_number": case.case_number,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entry_hash, sig = sign_audit_entry(data)
    log = AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction[action],
        resource_type="Case",
        resource_id=str(case.id),
        details={"case_number": case.case_number, "status": case.status.value},
        entry_hash=entry_hash,
        signature_b64=sig,
    )
    db.add(log)
    db.commit()
