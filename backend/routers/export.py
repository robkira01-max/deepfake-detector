"""Export batch — PDF en lot (ZIP) + CSV des analyses par dossier."""
from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.security import require_analyst
from database import get_db
from models.analysis import Analysis, AnalysisStatus
from models.audit_log import AuditLog, AuditAction
from models.case import Case
from models.responses import ANALYST_ERRORS
from models.user import User

router = APIRouter(prefix="/export", tags=["Export"])


class BatchPdfRequest(BaseModel):
    analysis_ids: list[int]
    template_id: int | None = None


class BatchPdfResponse(BaseModel):
    generated: int
    skipped: int
    zip_size_bytes: int


@router.post(
    "/batch-pdf",
    response_class=StreamingResponse,
    summary="Générer PDF pour plusieurs analyses — retourne un fichier ZIP",
    responses={
        200: {"description": "Archive ZIP contenant les PDFs"},
        **ANALYST_ERRORS,
    },
)
def batch_pdf_export(
    body: BatchPdfRequest,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> StreamingResponse:
    if not body.analysis_ids:
        raise HTTPException(status_code=422, detail="analysis_ids ne peut pas être vide")
    if len(body.analysis_ids) > 50:
        raise HTTPException(status_code=422, detail="Maximum 50 analyses par lot")

    from reporting.pdf_generator import generate_pdf_report
    from models.report import Report

    buf = io.BytesIO()
    generated = 0
    skipped = 0

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for analysis_id in body.analysis_ids:
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            if not analysis or analysis.status != AnalysisStatus.completed:
                skipped += 1
                continue
            try:
                report = generate_pdf_report(
                    analysis_id=analysis_id,
                    expert=current_user,
                    db=db,
                    template_id=body.template_id,
                )
                # Lire le fichier PDF généré
                from pathlib import Path
                pdf_path = Path(report.file_path)
                if pdf_path.exists():
                    zf.write(pdf_path, arcname=f"{report.report_number}.pdf")
                    generated += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    zip_bytes = buf.getvalue()

    # Audit
    entry = AuditLog(
        user_id=current_user.id,
        user_username=current_user.username,
        action=AuditAction.BATCH_EXPORT_GENERATED,
        resource_type="batch_pdf",
        details={
            "analysis_ids": body.analysis_ids,
            "generated": generated,
            "skipped": skipped,
            "zip_size_bytes": len(zip_bytes),
        },
    )
    try:
        from core.chain_of_custody import sign_audit_entry
        entry.entry_hash, entry.signature_b64 = sign_audit_entry(entry)
    except Exception:
        pass
    db.add(entry)
    db.commit()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"deepfake_batch_{timestamp}.zip"

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/cases/{case_id}/analyses.csv",
    response_class=StreamingResponse,
    summary="Exporter toutes les analyses d'un dossier en CSV",
    responses={
        200: {"description": "Fichier CSV des analyses"},
        **ANALYST_ERRORS,
    },
)
def export_case_csv(
    case_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> StreamingResponse:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Dossier introuvable")

    analyses = (
        db.query(Analysis)
        .filter(Analysis.case_id == case_id)
        .order_by(Analysis.started_at.asc())
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)

    # En-têtes
    writer.writerow([
        "analysis_id",
        "case_number",
        "case_title",
        "status",
        "verdict",
        "final_score",
        "confidence_low",
        "confidence_high",
        "score_video_texture",
        "score_video_temporal",
        "score_rppg",
        "score_biometrics",
        "score_audio_model",
        "score_audio_phase",
        "score_metadata",
        "duration_seconds",
        "started_at",
        "completed_at",
    ])

    for a in analyses:
        writer.writerow([
            a.id,
            case.case_number,
            case.title,
            a.status.value if a.status else "",
            a.verdict.value if a.verdict else "",
            f"{a.final_score:.4f}" if a.final_score is not None else "",
            f"{a.confidence_low:.4f}" if a.confidence_low is not None else "",
            f"{a.confidence_high:.4f}" if a.confidence_high is not None else "",
            f"{a.score_video_texture:.4f}" if a.score_video_texture is not None else "",
            f"{a.score_video_temporal:.4f}" if a.score_video_temporal is not None else "",
            f"{a.score_rppg:.4f}" if a.score_rppg is not None else "",
            f"{a.score_biometrics:.4f}" if a.score_biometrics is not None else "",
            f"{a.score_audio_model:.4f}" if a.score_audio_model is not None else "",
            f"{a.score_audio_phase:.4f}" if a.score_audio_phase is not None else "",
            f"{a.score_metadata:.4f}" if a.score_metadata is not None else "",
            a.duration_seconds or "",
            a.started_at.isoformat() if a.started_at else "",
            a.completed_at.isoformat() if a.completed_at else "",
        ])

    csv_bytes = output.getvalue().encode("utf-8-sig")  # BOM pour Excel

    # Audit
    entry = AuditLog(
        user_id=current_user.id,
        user_username=current_user.username,
        action=AuditAction.CSV_EXPORT_GENERATED,
        resource_type="case",
        resource_id=str(case_id),
        details={
            "case_number": case.case_number,
            "analysis_count": len(analyses),
        },
    )
    try:
        from core.chain_of_custody import sign_audit_entry
        entry.entry_hash, entry.signature_b64 = sign_audit_entry(entry)
    except Exception:
        pass
    db.add(entry)
    db.commit()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{case.case_number}_analyses_{timestamp}.csv"

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
