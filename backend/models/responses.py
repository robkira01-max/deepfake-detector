"""Schémas Pydantic partagés pour les réponses d'erreur OpenAPI."""
from __future__ import annotations

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    detail: str

    model_config = {
        "json_schema_extra": {"example": {"detail": "Description de l'erreur"}}
    }


class ValidationErrorItem(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    detail: list[ValidationErrorItem]

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": [
                    {
                        "loc": ["body", "field_name"],
                        "msg": "field required",
                        "type": "missing",
                    }
                ]
            }
        }
    }


# Dictionnaires prêts à l'emploi pour le paramètre `responses=` de FastAPI

HTTP_401 = {
    401: {
        "model": ErrorDetail,
        "description": "Token absent, expiré ou invalide",
    }
}

HTTP_403 = {
    403: {
        "model": ErrorDetail,
        "description": "Accès refusé — droits insuffisants ou ressource appartenant à un autre utilisateur",
    }
}

HTTP_404 = {
    404: {
        "model": ErrorDetail,
        "description": "Ressource introuvable",
    }
}

HTTP_409 = {
    409: {
        "model": ErrorDetail,
        "description": "Conflit — la ressource existe déjà",
    }
}

HTTP_422 = {
    422: {
        "model": ValidationErrorResponse,
        "description": "Données invalides (validation Pydantic ou contrainte métier)",
    }
}

HTTP_429 = {
    429: {
        "model": ErrorDetail,
        "description": "Trop de requêtes — rate limit atteint ou compte verrouillé",
        "headers": {
            "Retry-After": {
                "description": "Secondes avant de pouvoir réessayer",
                "schema": {"type": "integer"},
            }
        },
    }
}

# Combinaisons courantes
ANALYST_ERRORS = {**HTTP_401, **HTTP_403, **HTTP_422}
ADMIN_ERRORS = {**HTTP_401, **HTTP_403}
AUTH_ERRORS = {**HTTP_401, **HTTP_422, **HTTP_429}
CRUD_ERRORS = {**HTTP_401, **HTTP_403, **HTTP_404, **HTTP_422}
