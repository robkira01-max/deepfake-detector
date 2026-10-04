"""Routes C2PA — vérification de provenance cryptographique.

C2PA est une preuve non-ML. Ces endpoints sont distincts des endpoints d'analyse ML.
Avertissement : la vérification répond uniquement à « ce fichier a-t-il un manifeste C2PA valide ? »
Elle ne remplace pas une analyse forensique ML ni l'expertise humaine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.security import require_analyst
from database import get_db
from models.media_file import MediaFile
from models.user import User

router = APIRouter(prefix="/c2pa", tags=["c2pa"])


class C2PAVerificationResponse(BaseModel):
    file_id: int
    filename: str
    has_manifest: bool
    is_cryptographically_valid: bool | None
    producer: str | None
    tool: str | None
    created_at: str | None
    is_ai_generated: bool | None
    ai_training_opted_out: bool | None
    hard_bindings_count: int
    soft_bindings_count: int
    actions: list[str]
    validation_errors: list[str]
    raw_assertions_count: int
    disclaimer: str


_DISCLAIMER = (
    "La vérification C2PA confirme la présence d'un manifeste de provenance cryptographique "
    "et son intégrité. Elle ne prouve pas qu'un contenu est authentique ni qu'il a été manipulé. "
    "L'absence de manifeste ne signifie pas qu'un fichier est un deepfake. "
    "Ce résultat doit être interprété par un expert en forensique numérique."
)


@router.get(
    "/verify/{file_id}",
    response_model=C2PAVerificationResponse,
    summary="Vérifier le manifeste C2PA d'un fichier",
    description=(
        "Lit et vérifie le manifeste C2PA (Content Credentials) du fichier spécifié. "
        "Retourne les assertions de provenance, la validité cryptographique et les signaux IA. "
        "Requiert le rôle analyst ou supérieur."
    ),
)
def verify_c2pa(
    file_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Session = Depends(get_db),
) -> C2PAVerificationResponse:
    mf = db.query(MediaFile).filter(MediaFile.id == file_id).first()
    if mf is None:
        raise HTTPException(status_code=404, detail="Fichier non trouvé")

    if not mf.storage_key:
        raise HTTPException(status_code=422, detail="Fichier sans chemin de stockage")

    file_path = Path(mf.storage_key)
    if not file_path.exists():
        raise HTTPException(status_code=422, detail="Fichier introuvable sur le disque")

    from core.c2pa_verifier import C2PAVerifier
    result = C2PAVerifier.verify(file_path)

    return C2PAVerificationResponse(
        file_id=file_id,
        filename=mf.original_filename or mf.storage_key,
        has_manifest=result.has_manifest,
        is_cryptographically_valid=result.is_cryptographically_valid,
        producer=result.producer,
        tool=result.tool,
        created_at=result.created_at,
        is_ai_generated=result.is_ai_generated,
        ai_training_opted_out=result.ai_training_opted_out,
        hard_bindings_count=result.hard_bindings_count,
        soft_bindings_count=result.soft_bindings_count,
        actions=result.actions,
        validation_errors=result.validation_errors,
        raw_assertions_count=result.raw_assertions_count,
        disclaimer=_DISCLAIMER,
    )
