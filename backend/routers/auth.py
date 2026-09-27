"""Routes d'authentification — JWT RS256 + MFA TOTP + RBAC."""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock
from typing import Annotated

import redis as _redis_lib

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

_limiter = Limiter(key_func=get_remote_address)

# ── Lockout par username (Redis-backed, in-process fallback) ─────────────────
_MAX_ATTEMPTS = 5       # échecs avant verrouillage
_LOCKOUT_SECONDS = 300  # 5 minutes

_login_failures: dict[str, list[float]] = defaultdict(list)
_lock = Lock()
_lockout_redis: _redis_lib.Redis | None = None
_lockout_redis_checked: bool = False


def _get_lockout_redis() -> _redis_lib.Redis | None:
    """Connexion Redis partagée entre workers (lazy, cachée). None si indisponible."""
    global _lockout_redis, _lockout_redis_checked
    if _lockout_redis_checked:
        return _lockout_redis
    try:
        from config import settings as _s
        r = _redis_lib.from_url(_s.redis_url, decode_responses=True, socket_connect_timeout=1)
        r.ping()
        _lockout_redis = r
    except Exception:
        _lockout_redis = None
    _lockout_redis_checked = True
    return _lockout_redis


def _check_lockout(username: str) -> None:
    """Lève 429 si le compte est temporairement verrouillé."""
    r = _get_lockout_redis()
    if r is not None:
        count = r.get(f"login:fail:{username}")
        if count and int(count) >= _MAX_ATTEMPTS:
            ttl = max(r.ttl(f"login:fail:{username}"), 0)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Compte temporairement verrouillé. Réessayez dans {ttl}s.",
                headers={"Retry-After": str(ttl)},
            )
        return
    # Fallback in-process (un seul worker ; acceptable en dev sans Redis)
    now = time.monotonic()
    with _lock:
        attempts = [t for t in _login_failures[username] if now - t < _LOCKOUT_SECONDS]
        _login_failures[username] = attempts
        if len(attempts) >= _MAX_ATTEMPTS:
            retry_after = int(_LOCKOUT_SECONDS - (now - attempts[0]))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Compte temporairement verrouillé. Réessayez dans {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )


def _record_failure(username: str) -> None:
    r = _get_lockout_redis()
    if r is not None:
        key = f"login:fail:{username}"
        count = r.incr(key)
        if count == 1:
            r.expire(key, _LOCKOUT_SECONDS)
        return
    with _lock:
        _login_failures[username].append(time.monotonic())


def _clear_failures(username: str) -> None:
    r = _get_lockout_redis()
    if r is not None:
        r.delete(f"login:fail:{username}")
        return
    with _lock:
        _login_failures.pop(username, None)

from models.responses import (
    AUTH_ERRORS, ADMIN_ERRORS, HTTP_401, HTTP_403, HTTP_409, HTTP_422,
)
from core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_mfa_qr_base64,
    generate_mfa_secret,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
    verify_totp,
)
from core.encryption import encrypt_secret, decrypt_secret
from core.token_blocklist import revoke_token, is_revoked
from config import settings
from database import get_db
from models.audit_log import AuditLog, AuditAction
from models.user import User, UserRole

router = APIRouter(prefix="/auth", tags=["Authentification"])


# ── Schémas Pydantic ──────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    mfa_required: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "mfa_required": False,
            }
        }
    }


class MFAVerifyRequest(BaseModel):
    totp_code: str = Field(
        description="Code TOTP à 6 chiffres généré par l'application authenticator",
        pattern=r"^\d{6}$",
        examples=["123456"],
    )
    temp_token: str = Field(
        description="Token temporaire reçu lors de la connexion quand mfa_required=true",
    )


class UserCreateRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        examples=["expert_forensique"],
        description="Identifiant unique (lettres, chiffres, tirets, underscores)",
    )
    email: EmailStr = Field(examples=["expert@tribunal.gc.ca"])
    password: str = Field(
        min_length=12,
        description="Minimum 12 caractères : majuscule, minuscule, chiffre, caractère spécial",
        examples=["Forensic@2026!Secure"],
    )
    role: UserRole = Field(
        default=UserRole.readonly,
        description="Rôle RBAC : admin | analyst | readonly",
        examples=["analyst"],
    )

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        """Validation politique de mots de passe : 12+ chars, maj, min, chiffre, spécial."""
        import re
        errors = []
        if len(v) < 12:
            errors.append("12 caractères minimum")
        if not re.search(r"[A-Z]", v):
            errors.append("au moins une majuscule")
        if not re.search(r"[a-z]", v):
            errors.append("au moins une minuscule")
        if not re.search(r"\d", v):
            errors.append("au moins un chiffre")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", v):
            errors.append("au moins un caractère spécial")
        if errors:
            raise ValueError("Mot de passe insuffisant : " + ", ".join(errors))
        return v


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool
    mfa_enabled: bool
    created_at: datetime
    last_login: datetime | None

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": 42,
                "username": "expert_forensique",
                "email": "expert@tribunal.gc.ca",
                "role": "analyst",
                "is_active": True,
                "mfa_enabled": True,
                "created_at": "2026-01-15T09:30:00Z",
                "last_login": "2026-09-20T14:22:11Z",
            }
        },
    }


class MFASetupResponse(BaseModel):
    secret: str
    qr_code_base64: str
    provisioning_uri: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Connexion (OAuth2 Password Flow)",
    responses={**AUTH_ERRORS, **ADMIN_ERRORS},
)
@_limiter.limit("5/minute")
def login(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    username = form_data.username
    ip = request.client.host if request.client else ""

    _check_lockout(username)

    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        _record_failure(username)
        _audit(db, None, AuditAction.LOGIN_FAILED, ip=ip, details={"username": username})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiant ou mot de passe incorrect",
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Compte désactivé")

    if user.mfa_enabled:
        # Retourne un token temporaire de courte durée pour la vérification MFA
        temp_token = create_access_token(f"mfa:{user.username}", user.role.value)
        _audit(db, user, AuditAction.USER_LOGIN, ip=ip, details={"mfa_pending": True})
        return TokenResponse(access_token=temp_token, refresh_token="", mfa_required=True)

    # Login complet sans MFA
    _clear_failures(user.username)
    _complete_login(db, user, ip)
    return TokenResponse(
        access_token=create_access_token(user.username, user.role.value),
        refresh_token=create_refresh_token(user.username),
    )


@router.post("/mfa/verify", response_model=TokenResponse, summary="Vérification MFA TOTP", responses=AUTH_ERRORS)
@_limiter.limit("5/minute")
def verify_mfa(
    request: Request,
    body: MFAVerifyRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    payload = decode_token(body.temp_token)
    sub = payload.get("sub", "")
    if not sub.startswith("mfa:"):
        raise HTTPException(status_code=400, detail="Token MFA invalide")

    username = sub[4:]
    user = db.query(User).filter(User.username == username, User.is_active == True).first()
    if not user or not user.mfa_secret:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable")

    ip = request.client.host if request.client else ""
    if not verify_totp(decrypt_secret(user.mfa_secret), body.totp_code):
        _record_failure(username)
        _audit(db, user, AuditAction.LOGIN_FAILED, ip=ip, details={"reason": "invalid_totp"})
        raise HTTPException(status_code=401, detail="Code MFA invalide")

    _clear_failures(username)
    _complete_login(db, user, ip)
    _audit(db, user, AuditAction.MFA_VERIFIED, ip=ip)

    return TokenResponse(
        access_token=create_access_token(user.username, user.role.value),
        refresh_token=create_refresh_token(user.username),
    )


@router.post("/mfa/setup", response_model=MFASetupResponse, summary="Configurer MFA TOTP", responses=HTTP_401)
def setup_mfa(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> MFASetupResponse:
    from core.security import get_mfa_provisioning_uri
    secret = generate_mfa_secret()
    qr_b64 = generate_mfa_qr_base64(current_user.username, secret)
    uri = get_mfa_provisioning_uri(current_user.username, secret)

    current_user.mfa_secret = encrypt_secret(secret)
    db.commit()
    return MFASetupResponse(secret=secret, qr_code_base64=qr_b64, provisioning_uri=uri)


class MFAEnableRequest(BaseModel):
    totp_code: str

    @field_validator("totp_code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        if not v.isdigit() or len(v) not in (6, 8):
            raise ValueError("Code TOTP invalide (6 ou 8 chiffres requis)")
        return v


@router.post("/mfa/enable", summary="Activer MFA après vérification du premier code", responses={**HTTP_401, **HTTP_422})
def enable_mfa(
    body: MFAEnableRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    if not current_user.mfa_secret:
        raise HTTPException(status_code=400, detail="Lancez d'abord /auth/mfa/setup")
    if not verify_totp(decrypt_secret(current_user.mfa_secret), body.totp_code):
        raise HTTPException(status_code=400, detail="Code TOTP invalide")
    current_user.mfa_enabled = True
    db.commit()
    _audit(db, current_user, AuditAction.MFA_ENABLED)
    return {"message": "MFA activé avec succès"}


@router.post("/users", response_model=UserResponse, summary="Créer un utilisateur (admin)", responses={**ADMIN_ERRORS, **HTTP_422, **HTTP_409})
def create_user(
    body: UserCreateRequest,
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=409, detail="Nom d'utilisateur déjà pris")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="Email déjà enregistré")

    user = User(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _audit(db, user, AuditAction.USER_CREATED, details={"role": body.role.value})
    return user


@router.get("/users", response_model=list[UserResponse], summary="Lister les utilisateurs (admin)", responses=ADMIN_ERRORS)
def list_users(
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 50,
) -> list[User]:
    return db.query(User).order_by(User.id).offset(skip).limit(min(limit, 200)).all()


@router.get("/me", response_model=UserResponse, summary="Profil de l'utilisateur courant", responses=HTTP_401)
def me(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    return current_user


@router.post("/refresh", response_model=TokenResponse, summary="Renouveler le token d'accès", responses=AUTH_ERRORS)
@_limiter.limit("10/minute")
def refresh_token(
    request: Request,
    body: RefreshRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Token de rafraîchissement invalide")

    sub = payload.get("sub", "")
    user = db.query(User).filter(User.username == sub, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable ou désactivé")

    # Rotate: revoke old refresh token, issue a new one
    old_jti = payload.get("jti", "")
    if old_jti:
        revoke_token(old_jti, settings.refresh_token_expire_days * 86400)

    return TokenResponse(
        access_token=create_access_token(user.username, user.role.value),
        refresh_token=create_refresh_token(user.username),
    )


@router.post("/logout", summary="Révoquer les tokens (déconnexion)", responses=HTTP_401)
def logout(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    body: LogoutRequest | None = None,
) -> dict:
    from jose import jwt as _jwt

    def _revoke_raw(raw_token: str, ttl: int) -> None:
        try:
            p = _jwt.decode(
                raw_token,
                settings.jwt_public_key,
                algorithms=[settings.jwt_algorithm],
                options={"verify_exp": False},
            )
            jti = p.get("jti", "")
            if jti:
                revoke_token(jti, ttl)
        except Exception:
            pass

    # Revoke access token
    auth_header = request.headers.get("Authorization", "")
    access_raw = auth_header.removeprefix("Bearer ").strip()
    _revoke_raw(access_raw, settings.access_token_expire_minutes * 60)

    # Revoke refresh token if provided — prevents re-use after logout
    if body and body.refresh_token:
        _revoke_raw(body.refresh_token, settings.refresh_token_expire_days * 86400)

    ip = request.client.host if request.client else ""
    _audit(db, current_user, AuditAction.USER_LOGOUT, ip=ip)
    return {"message": "Déconnexion réussie"}


# ── Utilitaire audit ──────────────────────────────────────────────────────────

def _audit(
    db: Session,
    user: User | None,
    action: AuditAction,
    ip: str = "",
    details: dict | None = None,
) -> None:
    from core.chain_of_custody import sign_audit_entry
    data = {
        "action": action.value,
        "user_id": user.id if user else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    entry_hash, sig = sign_audit_entry(data)
    log = AuditLog(
        user_id=user.id if user else None,
        user_username=user.username if user else None,
        action=action,
        details=details,
        ip_address=ip,
        entry_hash=entry_hash,
        signature_b64=sig,
    )
    db.add(log)
    db.commit()


def _complete_login(db: Session, user: User, ip: str) -> None:
    user.last_login = datetime.now(timezone.utc)
    user.login_count += 1
    db.commit()
    _audit(db, user, AuditAction.USER_LOGIN, ip=ip)
