"""Routes KYC — vérification d'identité synchrone (face match + OCR + deepfake doc)."""
from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from core.ingestion import ingest_file, IngestionError
from core.security import require_analyst, get_current_user
from database import get_db
from engines.biometric_engine import FaceMatchingEngine
from engines.ocr_engine import OCREngine, ProvinceIDData
from models.audit_log import AuditLog, AuditAction
from models.kyc_verification import KYCVerification, KYCVerdict
from models.user import User

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/kyc", tags=["KYC"])

# Module-level singletons (chargés une fois)
_biometric: FaceMatchingEngine | None = None
_ocr: OCREngine | None = None


def _get_biometric() -> FaceMatchingEngine:
    global _biometric
    if _biometric is None:
        _biometric = FaceMatchingEngine()
    return _biometric


def _get_ocr() -> OCREngine:
    global _ocr
    if _ocr is None:
        _ocr = OCREngine()
    return _ocr


# ── Schémas Pydantic ──────────────────────────────────────────────────────────

class KYCVerificationResponse(BaseModel):
    verification_id: int
    kyc_verdict: str
    kyc_verdict_reasons: list[str]
    face_match_score: float | None
    face_match_verdict: str | None
    ocr_surname: str | None
    ocr_given_names: str | None
    ocr_document_number: str | None
    ocr_birth_date: str | None
    ocr_expiry_date: str | None
    ocr_country: str | None
    ocr_mrz_valid: bool | None
    document_deepfake_score: float | None
    duration_ms: int | None

    model_config = {"from_attributes": True}


def _to_response(v: KYCVerification) -> KYCVerificationResponse:
    return KYCVerificationResponse(
        verification_id=v.id,
        kyc_verdict=v.kyc_verdict.value,
        kyc_verdict_reasons=v.kyc_verdict_reasons or [],
        face_match_score=v.face_match_score,
        face_match_verdict=v.face_match_verdict,
        ocr_surname=v.ocr_surname,
        ocr_given_names=v.ocr_given_names,
        ocr_document_number=v.ocr_document_number,
        ocr_birth_date=v.ocr_birth_date,
        ocr_expiry_date=v.ocr_expiry_date,
        ocr_country=v.ocr_country,
        ocr_mrz_valid=v.ocr_mrz_valid,
        document_deepfake_score=v.document_deepfake_score,
        duration_ms=v.duration_ms,
    )


# ── POST /kyc/verify ─────────────────────────────────────────────────────────

@router.post("/verify", response_model=KYCVerificationResponse, status_code=status.HTTP_201_CREATED)
def verify_identity(
    request: Request,
    doc_file: Annotated[UploadFile, File(description="Document d'identité (passeport, permis, carte)")],
    selfie_file: Annotated[UploadFile | None, File(description="Photo selfie (optionnelle)")] = None,
    case_id: Annotated[int | None, Form()] = None,
    current_user: User = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> KYCVerificationResponse:
    """Vérification KYC synchrone — retour en temps réel (< 15 s)."""
    t_start = time.monotonic()

    # ── Ingestion document ────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_path = Path(tmpdir) / (doc_file.filename or "document")
        doc_path.write_bytes(doc_file.file.read())

        try:
            doc_result = ingest_file(
                source_path=doc_path,
                original_filename=doc_file.filename or "document",
                case_id=case_id or 0,
                uploader_id=current_user.id,
                db=db,
                ip_address=request.client.host if request.client else "",
            )
        except IngestionError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

        doc_mf = doc_result.media_file
        doc_local = Path(settings.processed_dir) / doc_mf.storage_key if doc_mf.storage_key else doc_path

        # ── Ingestion selfie (optionnelle) ────────────────────────────────────
        selfie_mf = None
        selfie_local: Path | None = None

        if selfie_file is not None:
            selfie_path = Path(tmpdir) / (selfie_file.filename or "selfie")
            selfie_path.write_bytes(selfie_file.file.read())
            try:
                selfie_result = ingest_file(
                    source_path=selfie_path,
                    original_filename=selfie_file.filename or "selfie",
                    case_id=case_id or 0,
                    uploader_id=current_user.id,
                    db=db,
                    ip_address=request.client.host if request.client else "",
                )
                selfie_mf = selfie_result.media_file
                selfie_local = Path(settings.processed_dir) / selfie_mf.storage_key if selfie_mf.storage_key else selfie_path
            except IngestionError:
                pass  # selfie invalide → verdict REVIEW

        # ── Analyse parallèle : face match + OCR ─────────────────────────────
        biometric = _get_biometric()
        ocr = _get_ocr()

        face_result = None
        province_data = None
        mrz_data = None

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}

            if selfie_local and doc_local.exists():
                futures["face"] = pool.submit(
                    biometric.compare, str(doc_local), str(selfie_local)
                )

            # Détection automatique : PDF417 (permis provinciaux) ou MRZ (passeports)
            futures["ocr"] = pool.submit(ocr.extract_document, str(doc_local))

            for key, future in futures.items():
                try:
                    result = future.result(timeout=12)
                    if key == "face":
                        face_result = result
                    elif key == "ocr":
                        province_data, mrz_data = result  # tuple (ProvinceIDData|None, MRZData|None)
                except Exception as exc:  # noqa: BLE001
                    log.warning("kyc_engine_error", engine=key, error=str(exc))

        # ── Construire KYCVerification ────────────────────────────────────────
        verif = KYCVerification(
            case_id=case_id,
            doc_media_file_id=doc_mf.id,
            selfie_media_file_id=selfie_mf.id if selfie_mf else None,
            requested_by_id=current_user.id,
        )

        # Face matching
        if face_result is not None:
            verif.face_match_score = face_result.similarity
            if face_result.error == "no_face_backend":
                verif.face_match_verdict = "UNAVAILABLE"
            elif not face_result.face_found_doc or not face_result.face_found_selfie:
                verif.face_match_verdict = "NO_FACE"
            elif face_result.is_match:
                verif.face_match_verdict = "MATCH"
            else:
                verif.face_match_verdict = "NO_MATCH"
        else:
            verif.face_match_verdict = "NO_FACE" if selfie_file is not None else "UNAVAILABLE"

        # OCR — permis provincial (PDF417 AAMVA) prioritaire sur MRZ
        if province_data is not None:
            verif.ocr_document_type   = "DL"
            verif.ocr_country         = province_data.country
            verif.ocr_surname         = province_data.surname
            verif.ocr_given_names     = province_data.given_names
            verif.ocr_document_number = province_data.document_number
            verif.ocr_birth_date      = province_data.birth_date
            verif.ocr_expiry_date     = province_data.expiry_date
            verif.ocr_nationality     = province_data.province_code
            verif.ocr_mrz_valid       = province_data.error is None
            verif.ocr_raw_fields      = {
                "source": "pdf417_aamva",
                "province": province_data.province_name,
                "province_code": province_data.province_code,
                "issuer_id": province_data.issuer_id,
                "aamva_version": province_data.aamva_version,
                "fields": province_data.raw_fields,
            }
            log.info("kyc_pdf417_decoded", province=province_data.province_code,
                     doc_num=province_data.document_number[:4] + "***")

        elif mrz_data is not None:
            verif.ocr_document_type   = mrz_data.document_type
            verif.ocr_country         = mrz_data.country
            verif.ocr_surname         = mrz_data.surname
            verif.ocr_given_names     = mrz_data.given_names
            verif.ocr_document_number = mrz_data.document_number
            verif.ocr_birth_date      = mrz_data.birth_date
            verif.ocr_expiry_date     = mrz_data.expiry_date
            verif.ocr_nationality     = mrz_data.nationality
            verif.ocr_mrz_valid       = mrz_data.mrz_check_valid
            verif.ocr_raw_fields      = {"source": "mrz", "raw_mrz": mrz_data.raw_mrz}

        # Calcul verdict + durée
        verif.compute_verdict()
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        verif.duration_ms = elapsed_ms
        verif.completed_at = datetime.now(timezone.utc)

        db.add(verif)
        db.flush()

        # AuditLog
        db.add(AuditLog(
            user_id=current_user.id,
            user_username=current_user.username,
            action=AuditAction.KYC_VERIFIED,
            resource_type="kyc_verification",
            resource_id=str(verif.id),
            details={
                "verdict": verif.kyc_verdict.value,
                "face_match": verif.face_match_verdict,
                "duration_ms": elapsed_ms,
                "case_id": case_id,
            },
            ip_address=request.client.host if request.client else None,
        ))

        db.commit()
        db.refresh(verif)

    return _to_response(verif)


# ── GET /kyc/{verification_id} ────────────────────────────────────────────────

@router.get("/{verification_id}", response_model=KYCVerificationResponse)
def get_verification(
    verification_id: int,
    current_user: User = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> KYCVerificationResponse:
    verif = db.get(KYCVerification, verification_id)
    if verif is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vérification introuvable")
    return _to_response(verif)


# ── GET /kyc/case/{case_id} ───────────────────────────────────────────────────

@router.get("/case/{case_id}", response_model=list[KYCVerificationResponse])
def list_case_verifications(
    case_id: int,
    current_user: User = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> list[KYCVerificationResponse]:
    verifs = (
        db.query(KYCVerification)
        .filter(KYCVerification.case_id == case_id)
        .order_by(KYCVerification.created_at.desc())
        .all()
    )
    return [_to_response(v) for v in verifs]
