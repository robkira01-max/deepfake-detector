"""Routes d'upload et d'analyse deepfake."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from typing import Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import settings
from core.ingestion import ingest_file, IngestionError
from core.security import require_analyst, require_any, get_current_user
from database import get_db
from models.responses import ANALYST_ERRORS, CRUD_ERRORS, HTTP_401, HTTP_403
from models.analysis import Analysis, AnalysisStatus, Verdict
from models.case import Case
from models.media_file import MediaFile
from models.user import User, UserRole

router = APIRouter(prefix="/analyze", tags=["Analyse"])


# ── Schémas ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    media_file_id: int
    uuid: str
    filename: str
    sha256: str
    blake3: str
    size_bytes: int
    tsa_stamped: bool
    warnings: list[str]


class AnalysisResponse(BaseModel):
    analysis_id: int = Field(alias="id")
    case_id: int
    media_file_id: int
    status: AnalysisStatus
    celery_task_id: str | None
    verdict: Verdict | None
    final_score: float | None
    confidence_low: float | None
    confidence_high: float | None
    score_video_texture: float | None
    score_video_temporal: float | None
    score_rppg: float | None
    score_biometrics: float | None
    score_audio_model: float | None
    score_audio_phase: float | None
    score_metadata: float | None
    model_far: float | None
    model_frr: float | None
    model_eer: float | None
    model_auc: float | None
    xai_shap_values: Any | None
    xai_plain_explanation: str | None
    xai_suspicious_timecodes: list | None
    error_message: str | None

    model_config = {"from_attributes": True, "populate_by_name": True}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/upload/{case_id}",
    response_model=UploadResponse,
    status_code=201,
    summary="Déposer un fichier média dans un dossier",
    responses=CRUD_ERRORS,
)
async def upload_media(
    case_id: int,
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> UploadResponse:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Dossier {case_id} introuvable")

    # H5 — IDOR: vérification de propriété
    if current_user.role != UserRole.admin and case.created_by_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Accès refusé — vous n'êtes pas le créateur de ce dossier",
        )

    ext = Path(file.filename or "file").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(status_code=422, detail=f"Extension non autorisée : {ext}")

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        ip = request.client.host if request.client else ""
        result = ingest_file(
            source_path=tmp_path,
            original_filename=file.filename or "upload",
            case_id=case_id,
            uploader_id=current_user.id,
            db=db,
            ip_address=ip,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    mf = result.media_file
    return UploadResponse(
        media_file_id=mf.id,
        uuid=mf.uuid,
        filename=mf.original_filename,
        sha256=mf.hash_sha256,
        blake3=mf.hash_blake3,
        size_bytes=mf.file_size_bytes,
        tsa_stamped=mf.tsa_token_b64 is not None,
        warnings=result.warnings,
    )


@router.post(
    "/start/{media_file_id}",
    response_model=AnalysisResponse,
    status_code=202,
    summary="Démarrer l'analyse deepfake (asynchrone via Celery)",
    responses=CRUD_ERRORS,
)
def start_analysis(
    media_file_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> AnalysisResponse:
    mf = db.query(MediaFile).filter(MediaFile.id == media_file_id).first()
    if not mf:
        raise HTTPException(status_code=404, detail="Fichier média introuvable")

    # Créer l'enregistrement d'analyse
    analysis = Analysis(
        case_id=mf.case_id,
        media_file_id=mf.id,
        status=AnalysisStatus.pending,
        requested_by_id=current_user.id,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # Soumettre la tâche Celery
    try:
        from tasks.analysis_tasks import run_deepfake_analysis
        task = run_deepfake_analysis.delay(analysis.id, media_file_id)
        analysis.celery_task_id = task.id
        analysis.status = AnalysisStatus.running
        db.commit()
    except Exception as exc:
        analysis.status = AnalysisStatus.failed
        analysis.error_message = str(exc)
        db.commit()

    return AnalysisResponse.model_validate(analysis)


@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Résultat d'une analyse",
    responses=CRUD_ERRORS,
)
def get_analysis(
    analysis_id: int,
    current_user: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
) -> AnalysisResponse:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analyse introuvable")
    # SEC-04 FIX: IDOR — vérifier que l'utilisateur a accès au case parent
    if current_user.role != UserRole.admin:
        case = db.query(Case).filter(Case.id == analysis.case_id).first()
        if not case or case.created_by_id != current_user.id:
            raise HTTPException(status_code=403, detail="Accès refusé à cette analyse")
    return AnalysisResponse.model_validate(analysis)


@router.get(
    "/case/{case_id}",
    response_model=list[AnalysisResponse],
    summary="Toutes les analyses d'un dossier",
    responses={**HTTP_401, **HTTP_403},
)
def list_case_analyses(
    case_id: int,
    current_user: Annotated[User, Depends(require_any)],
    db: Annotated[Session, Depends(get_db)],
) -> list[AnalysisResponse]:
    # SEC-04 FIX: IDOR — vérifier que l'utilisateur a accès au case
    if current_user.role != UserRole.admin:
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case or case.created_by_id != current_user.id:
            raise HTTPException(status_code=403, detail="Accès refusé à ce dossier")
    analyses = (
        db.query(Analysis)
        .filter(Analysis.case_id == case_id)
        .order_by(Analysis.started_at.desc())
        .all()
    )
    return [AnalysisResponse.model_validate(a) for a in analyses]
