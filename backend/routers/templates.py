"""Routes de gestion des templates de rapport."""
from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import jinja2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from core.security import get_current_user, require_admin, require_analyst
from database import get_db
from models.report_template import ReportTemplate, TemplateType
from models.user import User, UserRole

router = APIRouter(prefix="/reports/templates", tags=["Templates de rapport"])

_CUSTOM_DIR = Path("/home/kali/deepfake_detector/data/templates")
_BUILTIN_DIR = Path(__file__).parent.parent / "reporting" / "templates"


# ── Schémas ───────────────────────────────────────────────────────────────────

class TemplateResponse(BaseModel):
    id: int
    name: str
    description: str
    template_type: TemplateType
    jurisdiction: str | None
    language: str
    is_active: bool
    is_default: bool
    created_by_id: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[TemplateResponse], summary="Lister les templates disponibles")
def list_templates(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    language: str | None = None,
    jurisdiction: str | None = None,
) -> list[ReportTemplate]:
    q = db.query(ReportTemplate).filter(ReportTemplate.is_active == True)
    if language:
        q = q.filter(ReportTemplate.language == language)
    if jurisdiction:
        q = q.filter(
            (ReportTemplate.jurisdiction == jurisdiction) |
            (ReportTemplate.jurisdiction == None)
        )
    return q.order_by(ReportTemplate.template_type, ReportTemplate.id).all()


@router.get("/{template_id}", response_model=TemplateResponse, summary="Détail d'un template")
def get_template(
    template_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ReportTemplate:
    tmpl = db.query(ReportTemplate).filter(ReportTemplate.id == template_id).first()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template introuvable")
    return tmpl


@router.post(
    "/upload",
    response_model=TemplateResponse,
    status_code=201,
    summary="Uploader un template Jinja2 personnalisé (analyst/admin)",
)
async def upload_custom_template(
    name: str,
    description: str = "",
    jurisdiction: str | None = None,
    language: str = "fr",
    file: UploadFile = File(...),
    current_user: Annotated[User, Depends(require_analyst)] = None,
    db: Annotated[Session, Depends(get_db)] = None,
) -> ReportTemplate:
    # Validation extension
    if not (file.filename or "").endswith(".html"):
        raise HTTPException(status_code=422, detail="Le template doit être un fichier .html")

    # Lecture et validation syntaxe Jinja2
    content = await file.read()
    try:
        content_str = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="Le fichier doit être encodé en UTF-8")

    _validate_jinja2_template(content_str)

    # Vérification sécurité basique (pas d'accès fichier, pas d'exec)
    _check_template_safety(content_str)

    # Sauvegarde
    _CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(name)
    dest = _CUSTOM_DIR / f"custom_{current_user.id}_{safe_name}.html"
    dest.write_bytes(content)

    # Enregistrement DB
    tmpl = ReportTemplate(
        name=name,
        description=description,
        template_type=TemplateType.custom,
        file_path=str(dest),
        jurisdiction=jurisdiction,
        language=language,
        is_active=True,
        is_default=False,
        created_by_id=current_user.id,
    )
    db.add(tmpl)
    db.commit()
    db.refresh(tmpl)
    return tmpl


@router.delete("/{template_id}", status_code=204, summary="Supprimer un template custom (admin ou créateur)")
def delete_template(
    template_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> None:
    tmpl = db.query(ReportTemplate).filter(ReportTemplate.id == template_id).first()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template introuvable")
    if tmpl.template_type == TemplateType.builtin:
        raise HTTPException(status_code=403, detail="Les templates intégrés ne peuvent pas être supprimés")
    if current_user.role != UserRole.admin and tmpl.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Accès refusé — vous n'êtes pas le créateur de ce template")

    # Soft-delete
    tmpl.is_active = False
    db.commit()


@router.post("/{template_id}/set-default", summary="Définir le template par défaut (admin)")
def set_default_template(
    template_id: int,
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    tmpl = db.query(ReportTemplate).filter(
        ReportTemplate.id == template_id,
        ReportTemplate.is_active == True,
    ).first()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template introuvable")

    # Retirer l'ancien défaut
    db.query(ReportTemplate).filter(ReportTemplate.is_default == True).update({"is_default": False})
    tmpl.is_default = True
    db.commit()
    return {"message": f"Template '{tmpl.name}' défini comme défaut"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_jinja2_template(content: str) -> None:
    """Valide que le contenu est du Jinja2 valide. Lève HTTPException si invalide."""
    try:
        env = jinja2.Environment(autoescape=True)
        env.parse(content)
    except jinja2.TemplateSyntaxError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Syntaxe Jinja2 invalide à la ligne {exc.lineno}: {exc.message}",
        )


_FORBIDDEN_TAGS = [
    "{% set",  # Peut permettre des injections complexes — vérification manuelle
]
_FORBIDDEN_FILTERS = ["|pprint", "tojson|", "safe"]  # évite XSS si utilisateur malveillant


def _check_template_safety(content: str) -> None:
    """
    Vérifications de sécurité basiques sur les templates uploadés.
    Refuse les patterns dangereux.
    """
    dangerous_patterns = [
        "__import__", "os.system", "subprocess", "eval(", "exec(",
        "open(", "file(", "config.secret", "settings.secret",
    ]
    lower = content.lower()
    for pat in dangerous_patterns:
        if pat.lower() in lower:
            raise HTTPException(
                status_code=422,
                detail=f"Template refusé — pattern dangereux détecté : '{pat}'",
            )


def _safe_filename(name: str) -> str:
    """Nettoie un nom pour usage comme nom de fichier."""
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
    return safe.strip().replace(" ", "_")[:60]


# ── Seeding des templates intégrés ───────────────────────────────────────────

BUILTIN_TEMPLATES_DATA = [
    {
        "name": "Rapport complet (défaut)",
        "description": "Rapport forensique complet 8 sections — conforme R. c. Mohan, LPC 31.1-31.6. Recommandé pour usage judiciaire général.",
        "file_path": "rapport_complet.html",
        "jurisdiction": None,
        "language": "fr-en",
        "is_default": True,
    },
    {
        "name": "Résumé exécutif",
        "description": "Rapport condensé 2-3 pages pour juges et avocats. Vue d'ensemble rapide avec verdict, scores et chaîne de possession abrégée.",
        "file_path": "rapport_executif.html",
        "jurisdiction": None,
        "language": "fr-en",
        "is_default": False,
    },
    {
        "name": "Rapport bilingue FR/EN",
        "description": "Rapport officiel bilingue français/anglais — requis par la Loi sur les langues officielles pour les cours fédérales du Canada.",
        "file_path": "rapport_bilingue.html",
        "jurisdiction": "federal",
        "language": "fr-en",
        "is_default": False,
    },
    {
        "name": "Affidavit technique (CPCivQ)",
        "description": "Affidavit technique conforme au Code de procédure civile du Québec (RLRQ c. C-25.01), art. 282-293. Inclut espace pour commissaire à l'assermentation.",
        "file_path": "rapport_affidavit.html",
        "jurisdiction": "quebec",
        "language": "fr-en",
        "is_default": False,
    },
]


def seed_builtin_templates(db: Session) -> None:
    """Insère les templates intégrés si absents — appelé au démarrage."""
    for data in BUILTIN_TEMPLATES_DATA:
        file_path = str(_BUILTIN_DIR / data["file_path"])
        existing = db.query(ReportTemplate).filter(
            ReportTemplate.file_path == file_path,
            ReportTemplate.template_type == TemplateType.builtin,
        ).first()
        if existing:
            if existing.language != data["language"]:
                existing.language = data["language"]
                db.commit()
            continue
        tmpl = ReportTemplate(
            name=data["name"],
            description=data["description"],
            template_type=TemplateType.builtin,
            file_path=file_path,
            jurisdiction=data["jurisdiction"],
            language=data["language"],
            is_active=True,
            is_default=data["is_default"],
            created_by_id=None,
        )
        db.add(tmpl)
    db.commit()
