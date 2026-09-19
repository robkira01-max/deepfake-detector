"""Authentification JWT RS256 + MFA TOTP + RBAC."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import bcrypt as _bcrypt
import pyotp
import qrcode
import io
import base64
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from models.user import User, UserRole
from core.token_blocklist import is_revoked

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

_ALGORITHM = settings.jwt_algorithm
_ACCESS_EXPIRE = timedelta(minutes=settings.access_token_expire_minutes)
_REFRESH_EXPIRE = timedelta(days=settings.refresh_token_expire_days)


# ── Mots de passe ─────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT ───────────────────────────────────────────────────────────────────────

def _private_key() -> str:
    key = settings.jwt_private_key
    if not key:
        raise RuntimeError("JWT private key not found at " + settings.jwt_private_key_path)
    return key


def _public_key() -> str:
    key = settings.jwt_public_key
    if not key:
        raise RuntimeError("JWT public key not found at " + settings.jwt_public_key_path)
    return key


def create_access_token(subject: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + _ACCESS_EXPIRE
    payload = {
        "sub": subject,
        "role": role,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _private_key(), algorithm=_ALGORITHM)


def create_refresh_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + _REFRESH_EXPIRE
    payload = {
        "sub": subject,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _private_key(), algorithm=_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, _public_key(), algorithms=[_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    if is_revoked(payload.get("jti", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token révoqué",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


# ── Utilisateur courant ───────────────────────────────────────────────────────

def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Token de type invalide")

    sub = payload["sub"]
    # Rejeter les tokens MFA temporaires (sub commence par "mfa:")
    if sub.startswith("mfa:"):
        raise HTTPException(status_code=401, detail="Authentification MFA non complétée")

    user = db.query(User).filter(User.username == sub).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable ou désactivé")

    # Enforce MFA si requis par la configuration
    if settings.mfa_required and not user.mfa_enabled:
        raise HTTPException(
            status_code=403,
            detail="MFA obligatoire — configurez-le via POST /auth/mfa/setup puis /auth/mfa/enable",
        )
    return user


def require_role(*roles: UserRole):
    """Dépendance FastAPI — vérifie que l'utilisateur a l'un des rôles requis."""
    def checker(current_user: Annotated[User, Depends(get_current_user)]) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Rôle requis : {[r.value for r in roles]}",
            )
        return current_user
    return checker


require_admin = require_role(UserRole.admin)
require_analyst = require_role(UserRole.admin, UserRole.analyst)
require_any = require_role(UserRole.admin, UserRole.analyst, UserRole.readonly)


# ── MFA TOTP ──────────────────────────────────────────────────────────────────

def generate_mfa_secret() -> str:
    return pyotp.random_base32()


def get_mfa_provisioning_uri(username: str, secret: str) -> str:
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=username, issuer_name=settings.mfa_issuer)


def generate_mfa_qr_base64(username: str, secret: str) -> str:
    uri = get_mfa_provisioning_uri(username, secret)
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def verify_totp(secret: str, code: str) -> bool:
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)


# ── Génération des clés RSA (utilitaire CLI) ──────────────────────────────────

def generate_rsa_keypair(output_dir: Path) -> None:
    """Génère une paire de clés RSA-4096 pour les JWT et signatures."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    output_dir.mkdir(parents=True, exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    (output_dir / "private.pem").write_bytes(private_pem)
    (output_dir / "public.pem").write_bytes(public_pem)
    print(f"Clés RSA-4096 générées dans {output_dir}")
