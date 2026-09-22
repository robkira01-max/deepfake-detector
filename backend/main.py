"""Point d'entrée FastAPI — DeepfakeDetector Canada."""
from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config import settings
from database import init_db
from routers import auth, cases, analyze, dashboard, reports, templates as templates_router

log = structlog.get_logger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup", app=settings.app_name, version=settings.app_version, env=settings.app_env)
    init_db()
    _ensure_keys()
    _seed_templates()
    yield
    log.info("shutdown", app=settings.app_name)


def _ensure_keys() -> None:
    """Génère les clés RSA-4096 JWT et audit si elles n'existent pas encore."""
    from pathlib import Path
    from core.security import generate_rsa_keypair
    priv = Path(settings.jwt_private_key_path)
    pub = Path(settings.jwt_public_key_path)
    if not priv.exists() or not pub.exists():
        log.warning("jwt_keys_missing", generating="RSA-4096")
        generate_rsa_keypair(priv.parent)

    audit_key = Path(settings.audit_signing_key_path)
    if not audit_key.exists():
        log.warning("audit_key_missing", generating="RSA-4096")
        _generate_audit_key(audit_key)


def _seed_templates() -> None:
    """Insère les templates intégrés au démarrage si absents."""
    from database import SessionLocal
    from routers.templates import seed_builtin_templates
    db = SessionLocal()
    try:
        seed_builtin_templates(db)
    finally:
        db.close()


def _generate_audit_key(audit_key_path: "Path") -> None:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    audit_key_path.parent.mkdir(parents=True, exist_ok=True)
    audit_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    audit_key_path.chmod(0o600)
    log.info("audit_signing_key_generated", path=str(audit_key_path))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injecte les headers de sécurité sur toutes les réponses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"  # Désactivé — CSP est la défense moderne
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'none';"
        )
        if not settings.debug:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        return response


_OPENAPI_DESCRIPTION = """
## DeepfakeDetector Canada — API Forensique

Plateforme d'analyse forensique deepfake audio/vidéo pour le système judiciaire canadien.

### Conformité légale
- **LPC art. 31.1–31.6** — Authenticité et intégrité des documents électroniques
- **R. c. Mohan [1994] 2 RCS 9** — Critères d'admissibilité des preuves d'expert
- **CAN/DGSI 120** — Forensique numérique — Meilleures pratiques

### Authentification
Toutes les routes protégées requièrent un token JWT RS256 via header :
```
Authorization: Bearer <access_token>
```
Obtenir un token via `POST /auth/token`.

### Rôles RBAC
| Rôle | Accès |
|------|-------|
| `admin` | Toutes les routes + gestion utilisateurs + audit log |
| `analyst` | Dossiers + analyses + rapports |
| `readonly` | Lecture seule (dossiers, statistiques) |

### Pipeline d'analyse
1. **Créer un dossier** → `POST /cases/`
2. **Téléverser un fichier média** → `POST /analyze/upload/{case_id}`
3. **Lancer l'analyse** → `POST /analyze/start/{media_file_id}`
4. **Suivre l'état** → `GET /analyze/{analysis_id}`
5. **Générer le rapport** → `POST /reports/{case_id}/{analysis_id}`

### Chaîne de possession
Chaque fichier ingéré reçoit automatiquement :
- Hash **Blake3 + SHA-256 + MD5** (triple vérification)
- Horodatage **TSA RFC 3161** (FreeTSA.org)
- Signature **RSA-4096** des entrées d'audit

### Moteurs ML (7 composants)
| Composant | Poids | Description |
|-----------|-------|-------------|
| Texture vidéo | 20% | EfficientNet-B4 — artefacts GAN/diffusion |
| Temporel | 15% | ResNet-50+LSTM — incohérences inter-frames |
| rPPG | 18% | Analyse pouls carotidien (ChromChrom) |
| Biométrie | 10% | MediaPipe FaceMesh — landmarks faciaux |
| Audio modèle | 20% | Wav2Vec2 — voix synthétique |
| Phase audio | 10% | STFT — discontinuités de phase |
| Métadonnées | 7% | Signatures outils deepfake (DeepFaceLab, etc.) |
"""

_OPENAPI_TAGS = [
    {
        "name": "Authentification",
        "description": (
            "Gestion des sessions JWT RS256, MFA TOTP, "
            "enregistrement et gestion des utilisateurs."
        ),
    },
    {
        "name": "Dossiers",
        "description": (
            "Gestion des dossiers judiciaires (cases). "
            "Un dossier regroupe les fichiers médias et analyses pour une affaire."
        ),
    },
    {
        "name": "Analyses",
        "description": (
            "Téléversement de fichiers médias, déclenchement d'analyses deepfake "
            "asynchrones via Celery, et consultation des résultats."
        ),
    },
    {
        "name": "Tableau de bord",
        "description": (
            "Statistiques globales, journal d'audit paginé (admin), "
            "et état de santé détaillé de l'infrastructure."
        ),
    },
    {
        "name": "Rapports",
        "description": (
            "Génération de rapports forensiques PDF signés RSA-4096 + TSA. "
            "Conformes aux exigences de l'art. 31.1-31.6 LPC."
        ),
    },
    {
        "name": "Templates",
        "description": (
            "Gestion des templates Jinja2 HTML pour la génération de rapports. "
            "4 templates intégrés : complet, exécutif, bilingue, affidavit Québec."
        ),
    },
    {
        "name": "Système",
        "description": "Health check et informations de version.",
    },
]

app = FastAPI(
    title="DeepfakeDetector Canada",
    description=_OPENAPI_DESCRIPTION,
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_tags=_OPENAPI_TAGS,
    contact={
        "name": "DeepfakeDetector Canada",
        "email": "forensic@deepfake-detector.ca",
    },
    license_info={
        "name": "Propriétaire — Usage judiciaire uniquement",
    },
    lifespan=lifespan,
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Middlewares ───────────────────────────────────────────────────────────────
# Host header protection — whitelist explicite en production
if not settings.debug:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Routeurs ──────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(cases.router)
app.include_router(analyze.router)
app.include_router(dashboard.router)
app.include_router(reports.router)
app.include_router(templates_router.router)


@app.get("/health", tags=["Système"])
def health() -> dict:
    resp: dict = {"status": "ok"}
    if settings.debug:
        resp.update({"app": settings.app_name, "version": settings.app_version, "env": settings.app_env})
    return resp


@app.get("/", tags=["Système"])
def root() -> dict:
    return {
        "message": "DeepfakeDetector Canada API",
        "docs": "/docs",
        "health": "/health",
        "legal": "Conforme LPC 31.1-31.6 | R. c. Mohan [1994] 2 RCS 9 | CAN/DGSI 120",
    }
