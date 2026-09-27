"""Routes d'analyse de documents forensiques (v2.0).

Pipeline :
1. POST /analyze/document/upload/{case_id}   — ingestion du fichier document
2. POST /analyze/document/start/{file_id}    — déclenchement analyse Celery
3. GET  /analyze/document/{doc_analysis_id} — résultat d'analyse
4. GET  /analyze/document/case/{case_id}     — liste analyses d'un dossier
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from core.ingestion import ingest_file
from core.security import require_analyst, require_any
from database import get_db
from models.analysis import AnalysisStatus, Verdict
from models.audit_log import AuditLog, AuditAction
from models.case import Case
from models.document_analysis import DocumentAnalysis
from models.media_file import MediaFile, MediaType, MediaStatus
from models.responses import ANALYST_ERRORS, CRUD_ERRORS, HTTP_401
from models.user import User

router = APIRouter(prefix="/analyze/document", tags=["Documents"])

_DOCUMENT_VERDICT_MAP = {
    "DOCUMENT FALSIFIÉ": Verdict.deepfake,
    "DOCUMENT AUTHENTIQUE": Verdict.authentic,
    "INDÉTERMINÉ": Verdict.undetermined,
}


# ── Schémas de réponse ────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    media_file_id: int
    uuid: str
    filename: str
    file_size_bytes: int
    media_type: str
    hash_sha256: str
    status: str
    message: str


class DocumentAnalysisResponse(BaseModel):
    id: int
    case_id: int
    media_file_id: int
    status: str
    celery_task_id: str | None = None
    verdict: str | None = None
    verdict_label: str | None = None
    final_score: float | None = None
    score_ela: float | None = None
    score_clone: float | None = None
    score_metadata_doc: float | None = None
    score_font: float | None = None
    score_text_ai: float | None = None
    anomalies: dict | None = None
    xai_plain_explanation: str | None = None
    text_preview: str | None = None
    models_used: dict | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: int | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}


def _build_explanation(doc: DocumentAnalysis) -> str:
    """Génère une explication lisible du verdict."""
    parts = []
    score = doc.final_score or 0.0
    label = doc.verdict_label or "INDÉTERMINÉ"

    parts.append(f"Verdict : {label} (score {score:.2f}/1.00)")

    if doc.score_ela and doc.score_ela > 0.5:
        parts.append(f"• ELA : anomalies de retouche JPEG détectées (score {doc.score_ela:.2f})")
    if doc.score_clone and doc.score_clone > 0.3:
        parts.append(f"• Clone detection : régions copy-move détectées (score {doc.score_clone:.2f})")
    if doc.score_metadata_doc and doc.score_metadata_doc > 0.3:
        parts.append(f"• Métadonnées : incohérences détectées (score {doc.score_metadata_doc:.2f})")
    if doc.score_font and doc.score_font > 0.3:
        parts.append(f"• Polices : mélange typographique suspect (score {doc.score_font:.2f})")
    if doc.score_text_ai and doc.score_text_ai > 0.5:
        parts.append(f"• Texte : rédaction par IA probable (score {doc.score_text_ai:.2f})")

    if len(parts) == 1:
        parts.append("Aucune anomalie significative détectée.")

    return "\n".join(parts)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/upload/{case_id}",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload d'un document pour analyse forensique",
    responses=CRUD_ERRORS,
)
def upload_document(
    case_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Dossier introuvable")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_document_exts:
        raise HTTPException(
            status_code=422,
            detail=f"Format non supporté. Formats acceptés : {settings.allowed_document_extensions}",
        )

    content = file.file.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=422,
            detail=f"Fichier trop volumineux (max {settings.max_file_size_mb} MB)",
        )

    media_file = ingest_file(
        db=db,
        content=content,
        original_filename=file.filename or "document",
        case_id=case_id,
        user_id=current_user.id,
        media_type=MediaType.document,
    )

    return DocumentUploadResponse(
        media_file_id=media_file.id,
        uuid=media_file.uuid,
        filename=media_file.original_filename,
        file_size_bytes=media_file.file_size_bytes,
        media_type=media_file.media_type.value,
        hash_sha256=media_file.hash_sha256,
        status=media_file.status.value,
        message="Document ingéré avec chaîne de possession Blake3 + TSA RFC3161",
    )


@router.post(
    "/start/{file_id}",
    response_model=DocumentAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Démarrer l'analyse forensique d'un document",
    responses=CRUD_ERRORS,
)
def start_document_analysis(
    file_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentAnalysis:
    media_file = db.query(MediaFile).filter(MediaFile.id == file_id).first()
    if not media_file:
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    if media_file.media_type != MediaType.document:
        raise HTTPException(status_code=422, detail="Ce fichier n'est pas un document")
    if media_file.status != MediaStatus.verified:
        raise HTTPException(status_code=422, detail="Fichier non vérifié — attendez la fin de l'ingestion")

    doc_analysis = DocumentAnalysis(
        case_id=media_file.case_id,
        media_file_id=file_id,
        status=AnalysisStatus.pending,
        requested_by_id=current_user.id,
    )
    db.add(doc_analysis)
    db.flush()

    from tasks.analysis_tasks import run_document_analysis
    task = run_document_analysis.delay(doc_analysis.id, file_id)
    doc_analysis.celery_task_id = task.id

    log = AuditLog(
        user_id=current_user.id,
        user_username=current_user.username,
        action=AuditAction.DOCUMENT_ANALYSIS_STARTED,
        resource_type="DocumentAnalysis",
        resource_id=str(doc_analysis.id),
        details={"file_id": file_id, "case_id": media_file.case_id},
    )
    db.add(log)
    db.commit()
    db.refresh(doc_analysis)
    return doc_analysis


@router.get(
    "/{doc_analysis_id}",
    response_model=DocumentAnalysisResponse,
    summary="Résultat d'une analyse de document",
    responses=CRUD_ERRORS,
)
def get_document_analysis(
    doc_analysis_id: int,
    current_user: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentAnalysis:
    doc = db.query(DocumentAnalysis).filter(DocumentAnalysis.id == doc_analysis_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Analyse de document introuvable")
    return doc


@router.get(
    "/case/{case_id}",
    response_model=list[DocumentAnalysisResponse],
    summary="Liste des analyses de documents d'un dossier",
    responses=CRUD_ERRORS,
)
def list_document_analyses(
    case_id: int,
    current_user: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 50,
) -> list[DocumentAnalysis]:
    return (
        db.query(DocumentAnalysis)
        .filter(DocumentAnalysis.case_id == case_id)
        .order_by(DocumentAnalysis.id.desc())
        .offset(skip)
        .limit(min(limit, 200))
        .all()
    )
