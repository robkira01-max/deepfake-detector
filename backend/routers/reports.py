"""Routes de génération et téléchargement des rapports PDF/A-3."""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.security import require_analyst, require_any
from database import get_db
from models.analysis import Analysis, AnalysisStatus
from models.report import Report
from models.user import User

router = APIRouter(prefix="/reports", tags=["Rapports"])


class ReportResponse(BaseModel):
    id: int
    report_number: str
    case_id: int
    analysis_id: int
    is_signed: bool
    report_hash_sha256: str | None
    tsa_timestamp: str | None
    expert_username: str
    generated_at: str

    model_config = {"from_attributes": True}


class GenerateReportRequest(BaseModel):
    template_id: int | None = None
    expert_name: str | None = None
    expert_title: str | None = None
    expert_credentials: str | None = None
    lab_name: str | None = None
    notes_for_court: str | None = None


@router.post(
    "/generate/{analysis_id}",
    response_model=ReportResponse,
    status_code=201,
    summary="Générer le rapport PDF/A-3 conforme R. c. Mohan",
)
def generate_report(
    analysis_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
    body: GenerateReportRequest = GenerateReportRequest(),
) -> ReportResponse:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analyse introuvable")
    if analysis.status != AnalysisStatus.completed:
        raise HTTPException(status_code=409, detail="L'analyse n'est pas encore terminée")

    extra_context: dict[str, Any] = {}
    if body.expert_name:
        extra_context["expert_name"] = body.expert_name
    if body.expert_title:
        extra_context["expert_title"] = body.expert_title
    if body.expert_credentials:
        extra_context["expert_credentials"] = body.expert_credentials
    if body.lab_name:
        extra_context["lab_name"] = body.lab_name
    if body.notes_for_court:
        extra_context["notes_for_court"] = body.notes_for_court

    from reporting.pdf_generator import generate_pdf_report
    report = generate_pdf_report(
        analysis_id=analysis_id,
        expert=current_user,
        db=db,
        template_id=body.template_id,
        extra_context=extra_context or None,
    )

    return ReportResponse(
        id=report.id,
        report_number=report.report_number,
        case_id=report.case_id,
        analysis_id=report.analysis_id,
        is_signed=report.is_signed,
        report_hash_sha256=report.report_hash_sha256,
        tsa_timestamp=report.tsa_timestamp.isoformat() if report.tsa_timestamp else None,
        expert_username=report.expert_username,
        generated_at=report.generated_at.isoformat(),
    )


@router.get(
    "/{report_id}/download",
    summary="Télécharger le rapport PDF",
)
def download_report(
    report_id: int,
    current_user: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
) -> FileResponse:
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Rapport introuvable")
    if not report.pdf_path:
        raise HTTPException(status_code=404, detail="Fichier PDF non disponible")

    from pathlib import Path
    pdf_path = Path(report.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Fichier PDF introuvable sur le disque")

    _audit_download(db, current_user, report)
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{report.report_number}.pdf",
    )


@router.get(
    "/case/{case_id}",
    response_model=list[ReportResponse],
    summary="Lister les rapports d'un dossier",
)
def list_case_reports(
    case_id: int,
    current_user: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
) -> list[ReportResponse]:
    reports = (
        db.query(Report)
        .filter(Report.case_id == case_id)
        .order_by(Report.generated_at.desc())
        .all()
    )
    return [
        ReportResponse(
            id=r.id,
            report_number=r.report_number,
            case_id=r.case_id,
            analysis_id=r.analysis_id,
            is_signed=r.is_signed,
            report_hash_sha256=r.report_hash_sha256,
            tsa_timestamp=r.tsa_timestamp.isoformat() if r.tsa_timestamp else None,
            expert_username=r.expert_username,
            generated_at=r.generated_at.isoformat(),
        )
        for r in reports
    ]


def _audit_download(db: Session, user: User, report: Report) -> None:
    from datetime import datetime, timezone
    from models.audit_log import AuditLog, AuditAction
    from core.chain_of_custody import sign_audit_entry
    data = {
        "action": AuditAction.REPORT_DOWNLOADED.value,
        "user_id": user.id,
        "report_id": report.id,
        "report_number": report.report_number,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entry_hash, sig = sign_audit_entry(data)
    log_entry = AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=AuditAction.REPORT_DOWNLOADED,
        resource_type="Report",
        resource_id=str(report.id),
        details={"report_number": report.report_number},
        entry_hash=entry_hash,
        signature_b64=sig,
    )
    db.add(log_entry)
    db.commit()
