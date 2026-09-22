"""Fixtures partagées pour toute la suite de tests DeepfakeDetector."""
from __future__ import annotations

import os
import sys
import unittest.mock as mock
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker, Session

# ── 1. sys.path ────────────────────────────────────────────────────────────────
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

# ── 2. Env vars avant tout import app ─────────────────────────────────────────
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ.setdefault("JWT_PRIVATE_KEY_PATH", str(_backend / "keys" / "private.pem"))
os.environ.setdefault("JWT_PUBLIC_KEY_PATH", str(_backend / "keys" / "public.pem"))
os.environ.setdefault("AUDIT_SIGNING_KEY_PATH", str(_backend / "keys" / "audit_private.pem"))
os.environ["DEBUG"] = "true"
os.environ["MFA_REQUIRED"] = "false"
os.environ["APP_ENV"] = "test"

# ── 3. Patch create_engine pour SQLite (pas de pool_size/max_overflow) ─────────
import itertools as _itertools
import sqlalchemy
from sqlalchemy import create_engine as _real_ce

def _sqlite_engine(url, **kwargs):
    kwargs.pop("pool_size", None)
    kwargs.pop("max_overflow", None)
    kwargs.pop("pool_pre_ping", None)
    return _real_ce(url, connect_args={"check_same_thread": False}, **kwargs)

sqlalchemy.create_engine = _sqlite_engine

# ── Patch slowapi rate limiter en mode test ────────────────────────────────────
# Chaque test reçoit une "IP" unique → pas de partage de quota entre tests
_ip_counter = _itertools.count(1)

def _unique_test_ip(request) -> str:  # noqa: ANN001
    """Retourne une IP unique par appel — contourne le rate limiter entre tests."""
    return f"10.0.{(next(_ip_counter) >> 8) & 0xFF}.{next(_ip_counter) & 0xFF}"

import slowapi.util as _slowapi_util  # noqa: E402
_slowapi_util.get_remote_address = _unique_test_ip

# ── 4. Imports app (après les patches) ────────────────────────────────────────
from database import Base, get_db   # noqa: E402

# ── 5. Mock is_revoked ─────────────────────────────────────────────────────────
# core.security fait : from core.token_blocklist import is_revoked
# Cela crée un binding local dans core.security — il faut patcher CE binding.
# On doit d'abord importer core.security pour que le binding existe.
import core.security as _sec_module  # noqa: E402
# Patch l'attribut is_revoked dans le namespace de core.security (là où decode_token le lit)
mock.patch.object(_sec_module, "is_revoked", return_value=False).start()

from main import app                 # noqa: E402
from models.user import User, UserRole  # noqa: E402
from core.security import hash_password, create_access_token  # noqa: E402


# ── Base de données en mémoire ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def engine():
    """Moteur SQLite en mémoire partagé pour la session de test."""
    _engine = _sqlite_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=_engine)
    yield _engine
    _engine.dispose()


@pytest.fixture(scope="function")
def db(engine) -> Generator[Session, None, None]:
    """Session DB fraîche pour chaque test — rollback automatique."""
    connection = engine.connect()
    transaction = connection.begin()
    TestingSessionLocal = sessionmaker(bind=connection)
    session = TestingSessionLocal()
    yield session
    session.close()
    try:
        transaction.rollback()
    except Exception:
        pass
    connection.close()


# ── Client HTTP ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def client(db: Session) -> TestClient:
    """Client HTTP FastAPI avec DB injectée."""
    def _override_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = _override_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


# ── Utilisateurs de test ──────────────────────────────────────────────────────

def _make_user(db: Session, username: str, role: UserRole, password: str = "Test1234!") -> User:
    user = User(
        username=username,
        email=f"{username}@test.deepfake.ca",
        hashed_password=hash_password(password),
        role=role,
        is_active=True,
        mfa_enabled=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def admin_user(db: Session) -> User:
    return _make_user(db, "admin_test", UserRole.admin)


@pytest.fixture
def analyst_user(db: Session) -> User:
    return _make_user(db, "analyst_test", UserRole.analyst)


@pytest.fixture
def readonly_user(db: Session) -> User:
    return _make_user(db, "readonly_test", UserRole.readonly)


# ── Tokens JWT ────────────────────────────────────────────────────────────────

@pytest.fixture
def admin_token(admin_user: User) -> str:
    return create_access_token(admin_user.username, admin_user.role.value)


@pytest.fixture
def analyst_token(analyst_user: User) -> str:
    return create_access_token(analyst_user.username, analyst_user.role.value)


@pytest.fixture
def readonly_token(readonly_user: User) -> str:
    return create_access_token(readonly_user.username, readonly_user.role.value)


# ── Helpers d'authentification ────────────────────────────────────────────────

@pytest.fixture
def auth_admin(admin_token: str) -> dict:
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def auth_analyst(analyst_token: str) -> dict:
    return {"Authorization": f"Bearer {analyst_token}"}


@pytest.fixture
def auth_readonly(readonly_token: str) -> dict:
    return {"Authorization": f"Bearer {readonly_token}"}


# ── Fichiers de test ──────────────────────────────────────────────────────────

@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    """Fichier WAV minimal valide (silence 0.1s)."""
    wav_path = tmp_path / "test.wav"
    sample_rate = 16000
    channels = 1
    bits = 16
    data = b"\x00" * 3200  # 0.1s de silence

    header = (
        b"RIFF"
        + (36 + len(data)).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + (sample_rate * channels * bits // 8).to_bytes(4, "little")
        + (channels * bits // 8).to_bytes(2, "little")
        + bits.to_bytes(2, "little")
        + b"data"
        + len(data).to_bytes(4, "little")
    )
    wav_path.write_bytes(header + data)
    return wav_path


@pytest.fixture
def tiny_mp4(tmp_path: Path) -> Path:
    """Fichier MP4 minimal (magic bytes valides)."""
    mp4_path = tmp_path / "test.mp4"
    mp4_path.write_bytes(
        b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2avc1mp41"
    )
    return mp4_path
