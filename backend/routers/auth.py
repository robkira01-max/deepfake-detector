"""Routes d'authentification — JWT RS256 + MFA TOTP + RBAC."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

_limiter = Limiter(key_func=get_remote_address)

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


class MFAVerifyRequest(BaseModel):
    totp_code: str
    temp_token: str


class UserCreateRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.readonly


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool
    mfa_enabled: bool
    created_at: datetime
    last_login: datetime | None

    model_config = {"from_attributes": True}


class MFASetupResponse(BaseModel):
    secret: str
    qr_code_base64: str
    provisioning_uri: str


class RefreshRequest(BaseModel):
    refresh_token: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/token", response_model=TokenResponse, summary="Connexion (OAuth2 Password Flow)")
@_limiter.limit("5/minute")
def login(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    user = db.query(User).filter(User.username == form_data.username).first()
    ip = request.client.host if request.client else ""

    if not user or not verify_password(form_data.password, user.hashed_password):
        _audit(db, None, AuditAction.LOGIN_FAILED, ip=ip, details={"username": form_data.username})
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
    _complete_login(db, user, ip)
    return TokenResponse(
        access_token=create_access_token(user.username, user.role.value),
        refresh_token=create_refresh_token(user.username),
    )


@router.post("/mfa/verify", response_model=TokenResponse, summary="Vérification MFA TOTP")
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

    if not verify_totp(decrypt_secret(user.mfa_secret), body.totp_code):
        ip = request.client.host if request.client else ""
        _audit(db, user, AuditAction.LOGIN_FAILED, ip=ip, details={"reason": "invalid_totp"})
        raise HTTPException(status_code=401, detail="Code MFA invalide")

    ip = request.client.host if request.client else ""
    _complete_login(db, user, ip)
    _audit(db, user, AuditAction.MFA_VERIFIED, ip=ip)

    return TokenResponse(
        access_token=create_access_token(user.username, user.role.value),
        refresh_token=create_refresh_token(user.username),
    )


@router.post("/mfa/setup", response_model=MFASetupResponse, summary="Configurer MFA TOTP")
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


@router.post("/mfa/enable", summary="Activer MFA après vérification du premier code")
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


@router.post("/users", response_model=UserResponse, summary="Créer un utilisateur (admin)")
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


@router.get("/users", response_model=list[UserResponse], summary="Lister les utilisateurs (admin)")
def list_users(
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 50,
) -> list[User]:
    return db.query(User).order_by(User.id).offset(skip).limit(min(limit, 200)).all()


@router.get("/me", response_model=UserResponse, summary="Profil de l'utilisateur courant")
def me(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    return current_user


@router.post("/refresh", response_model=TokenResponse, summary="Renouveler le token d'accès")
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

    new_access_token = create_access_token(user.username, user.role.value)
    return TokenResponse(
        access_token=new_access_token,
        refresh_token=body.refresh_token,  # stateless — do not rotate
    )


@router.post("/logout", summary="Révoquer les tokens (déconnexion)")
def logout(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    # Extract the raw Bearer token to get its jti for revocation
    from core.security import oauth2_scheme
    from jose import jwt as _jwt
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    try:
        payload = _jwt.decode(
            token,
            settings.jwt_public_key,
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False},
        )
        jti = payload.get("jti", "")
        if jti:
            ttl = settings.access_token_expire_minutes * 60
            revoke_token(jti, ttl)
    except Exception:
        pass  # Token already validated upstream by get_current_user; best-effort revocation

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
