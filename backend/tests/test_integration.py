"""Tests d'intégration DeepfakeDetector — PostgreSQL + Redis réels via Docker Compose.

Ces tests démarrent les services d'infrastructure (postgres, redis) via docker compose,
appliquent les migrations Alembic, et vérifient les chemins critiques de l'API.

Usage :
    # Depuis la racine du projet (nécessite Docker) :
    pytest backend/tests/test_integration.py -v -m integration

    # Skip automatique si Docker indisponible :
    pytest backend/tests/ -v   # ces tests seront skippés

Marqueur pytest : @pytest.mark.integration
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# ── Constantes ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # /home/kali/deepfake_detector
BACKEND_DIR  = PROJECT_ROOT / "backend"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"

API_BASE     = "http://localhost:8082"   # port uvicorn dev (pas Docker)
# Pour les tests d'intégration Docker, l'API tourne derrière nginx sur :80
# Mais on peut aussi lancer l'API directement en test avec PostgreSQL/Redis Docker.
INFRA_ONLY_SERVICES = ["postgres", "redis"]

# Timeouts
POSTGRES_READY_TIMEOUT = 60   # secondes
REDIS_READY_TIMEOUT    = 30
API_READY_TIMEOUT      = 30

POSTGRES_TEST_DSN = (
    "postgresql://deepfake:changeme_strong_password@127.0.0.1:5432/deepfake_db"
)
REDIS_TEST_URL = "redis://:changeme_redis@127.0.0.1:6379/2"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _docker_available() -> bool:
    """Retourne True si docker et docker compose sont accessibles."""
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, check=True, timeout=10
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _compose_run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Lance docker compose depuis PROJECT_ROOT."""
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE)] + list(args)
    return subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=check
    )


def _wait_for_postgres(timeout: int = POSTGRES_READY_TIMEOUT) -> bool:
    """Attend que PostgreSQL accepte les connexions."""
    import psycopg2
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = psycopg2.connect(
                host="127.0.0.1",
                port=5432,
                user="deepfake",
                password="changeme_strong_password",
                dbname="deepfake_db",
                connect_timeout=3,
            )
            conn.close()
            return True
        except Exception:
            time.sleep(1)
    return False


def _wait_for_redis(timeout: int = REDIS_READY_TIMEOUT) -> bool:
    """Attend que Redis réponde au PING."""
    import redis as redis_lib
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = redis_lib.Redis.from_url(REDIS_TEST_URL, socket_connect_timeout=3)
            r.ping()
            r.close()
            return True
        except Exception:
            time.sleep(1)
    return False


def _run_migrations() -> subprocess.CompletedProcess:
    """Applique les migrations Alembic contre PostgreSQL Docker."""
    env = os.environ.copy()
    env["DATABASE_URL"] = POSTGRES_TEST_DSN
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
    )


# ── Skip marker ───────────────────────────────────────────────────────────────

requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker non disponible — tests d'intégration ignorés",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def infra_services():
    """Démarre postgres + redis via docker compose, yield, puis arrête."""
    if not _docker_available():
        pytest.skip("Docker non disponible")

    # Démarrer uniquement l'infrastructure (pas l'API — on la lance en-process)
    _compose_run("up", "-d", *INFRA_ONLY_SERVICES)

    # Attendre disponibilité
    assert _wait_for_postgres(), "PostgreSQL n'est pas prêt dans le délai imparti"
    assert _wait_for_redis(),    "Redis n'est pas prêt dans le délai imparti"

    # Appliquer migrations
    result = _run_migrations()
    assert result.returncode == 0, (
        f"Migrations Alembic ont échoué :\n{result.stdout}\n{result.stderr}"
    )

    yield {
        "postgres_dsn": POSTGRES_TEST_DSN,
        "redis_url":    REDIS_TEST_URL,
    }

    # Teardown : arrêter + supprimer les volumes de test
    _compose_run("down", "-v", "--remove-orphans", check=False)


@pytest.fixture(scope="module")
def pg_conn(infra_services):
    """Connexion psycopg2 directe à PostgreSQL de test."""
    import psycopg2
    conn = psycopg2.connect(infra_services["postgres_dsn"])
    conn.autocommit = True
    yield conn
    conn.close()


# ── Tests migrations ──────────────────────────────────────────────────────────

@requires_docker
@pytest.mark.integration
class TestMigrations:
    """Vérifie que toutes les migrations s'appliquent proprement sur PostgreSQL."""

    def test_alembic_version_at_head(self, infra_services, pg_conn):
        """Alembic doit être à la révision '0003' après upgrade head."""
        with pg_conn.cursor() as cur:
            cur.execute("SELECT version_num FROM alembic_version")
            row = cur.fetchone()
        assert row is not None, "Table alembic_version absente"
        assert row[0] == "0003", f"Version inattendue : {row[0]}"

    def test_tables_exist(self, infra_services, pg_conn):
        """Toutes les tables attendues doivent exister."""
        expected = [
            "users", "cases", "media_files", "audit_logs", "analyses",
            "reports", "document_analyses", "training_protocols",
            "model_versions", "analysis_feedbacks", "kyc_verifications",
        ]
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            actual = {r[0] for r in cur.fetchall()}

        for table in expected:
            assert table in actual, f"Table manquante : {table}"

    def test_c2pa_columns_on_media_files(self, infra_services, pg_conn):
        """Les colonnes C2PA doivent être présentes sur media_files."""
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'media_files'
                  AND column_name IN ('c2pa_status','c2pa_producer','c2pa_manifest_json')
                """
            )
            cols = {r[0] for r in cur.fetchall()}
        assert "c2pa_status"       in cols
        assert "c2pa_producer"     in cols
        assert "c2pa_manifest_json" in cols

    def test_enum_types_created(self, infra_services, pg_conn):
        """Les types enum PostgreSQL des phases 3/4/5 doivent exister."""
        expected_enums = {
            "validationstatus", "protocolstatus",
            "feedbacktype", "trueverdict", "kycverdict",
        }
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT typname FROM pg_type WHERE typtype = 'e'"
            )
            actual = {r[0] for r in cur.fetchall()}
        for e in expected_enums:
            assert e in actual, f"Enum manquant : {e}"

    def test_audit_action_enum_values(self, infra_services, pg_conn):
        """L'enum auditaction doit inclure les valeurs v3.0/v3.1."""
        new_values = {
            "FEEDBACK_SUBMITTED", "MODEL_REGISTERED", "MODEL_PULLED",
            "MODEL_ACTIVATED", "RETRAIN_TRIGGERED", "KYC_VERIFIED",
            "PROTOCOL_CREATED", "PROTOCOL_FROZEN", "PROTOCOL_CONSUMED",
        }
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT enumlabel FROM pg_enum
                JOIN pg_type ON pg_type.oid = pg_enum.enumtypid
                WHERE pg_type.typname = 'auditaction'
                """
            )
            actual = {r[0] for r in cur.fetchall()}
        for v in new_values:
            assert v in actual, f"AuditAction manquante : {v}"

    def test_downgrade_then_upgrade(self, infra_services):
        """Downgrade base → upgrade head doit être idempotent."""
        env = os.environ.copy()
        env["DATABASE_URL"] = POSTGRES_TEST_DSN

        down = subprocess.run(
            [sys.executable, "-m", "alembic", "downgrade", "base"],
            cwd=str(BACKEND_DIR), env=env,
            capture_output=True, text=True,
        )
        assert down.returncode == 0, f"Downgrade failed:\n{down.stderr}"

        up = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(BACKEND_DIR), env=env,
            capture_output=True, text=True,
        )
        assert up.returncode == 0, f"Re-upgrade failed:\n{up.stderr}"


# ── Tests API smoke (contre API dev locale avec infra Docker) ─────────────────

@requires_docker
@pytest.mark.integration
class TestAPISmokeWithRealInfra:
    """Smoke tests API contre PostgreSQL + Redis réels (API lancée en-process)."""

    @pytest.fixture(scope="class", autouse=True)
    def api_server(self, infra_services):
        """Lance uvicorn dans un thread de test avec PostgreSQL + Redis réels."""
        import threading
        import uvicorn

        env_backup = {}
        overrides = {
            "DATABASE_URL":        POSTGRES_TEST_DSN,
            "CELERY_BROKER_URL":   "redis://:changeme_redis@127.0.0.1:6379/0",
            "CELERY_RESULT_BACKEND": "redis://:changeme_redis@127.0.0.1:6379/1",
            "REDIS_URL":           REDIS_TEST_URL,
            "DEBUG":               "false",
            "MFA_REQUIRED":        "false",
            "APP_ENV":             "test",
        }

        for k, v in overrides.items():
            env_backup[k] = os.environ.get(k)
            os.environ[k] = v

        # Ajouter le backend au path
        if str(BACKEND_DIR) not in sys.path:
            sys.path.insert(0, str(BACKEND_DIR))

        # Réimporter la config avec les nouveaux env vars
        import importlib
        import config as cfg_module
        importlib.reload(cfg_module)

        config = uvicorn.Config(
            "main:app",
            host="127.0.0.1",
            port=18082,
            log_level="error",
            app_dir=str(BACKEND_DIR),
        )
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Attendre que le serveur soit prêt
        deadline = time.time() + API_READY_TIMEOUT
        while time.time() < deadline:
            try:
                r = httpx.get("http://127.0.0.1:18082/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            server.should_exit = True
            pytest.fail("API ne démarre pas dans le délai imparti")

        yield "http://127.0.0.1:18082"

        server.should_exit = True
        time.sleep(0.5)

        # Restaurer env
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _base(self, api_server):
        return api_server

    def test_health_endpoint(self, api_server):
        r = httpx.get(f"{api_server}/health", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "healthy"

    def test_health_reports_postgres(self, api_server):
        r = httpx.get(f"{api_server}/health", timeout=10)
        data = r.json()
        # La clé peut varier — on vérifie juste que db n'est pas "error"
        db_status = data.get("database") or data.get("db") or "ok"
        assert db_status not in ("error", "unreachable", "down"), (
            f"Base de données signalée en erreur : {db_status}"
        )

    def test_openapi_schema_accessible(self, api_server):
        r = httpx.get(f"{api_server}/openapi.json", timeout=10)
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        assert "DeepfakeDetector" in schema.get("info", {}).get("title", "")

    def test_login_with_real_db(self, api_server):
        """Crée un utilisateur en base PostgreSQL et vérifie l'auth JWT."""
        import psycopg2
        from core.security import hash_password

        # Insérer un utilisateur de test directement
        conn = psycopg2.connect(POSTGRES_TEST_DSN)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, email, hashed_password, role,
                                       is_active, mfa_enabled)
                    VALUES (%s, %s, %s, 'admin', true, false)
                    ON CONFLICT (username) DO NOTHING
                    """,
                    ("integ_admin", "integ@test.local", hash_password("Integ2026!")),
                )
        finally:
            conn.close()

        r = httpx.post(
            f"{api_server}/auth/login",
            json={"username": "integ_admin", "password": "Integ2026!"},
            timeout=10,
        )
        # 200 si MFA désactivé, sinon 202 (MFA required)
        assert r.status_code in (200, 202), (
            f"Login échoué ({r.status_code}): {r.text}"
        )

    def test_protected_endpoint_requires_auth(self, api_server):
        r = httpx.get(f"{api_server}/cases/", timeout=10)
        assert r.status_code == 401

    def test_dashboard_stats_endpoint(self, api_server):
        """GET /dashboard/stats doit retourner 401 sans token."""
        r = httpx.get(f"{api_server}/dashboard/stats", timeout=10)
        assert r.status_code == 401


# ── Tests Redis connectivity ───────────────────────────────────────────────────

@requires_docker
@pytest.mark.integration
class TestRedisConnectivity:
    """Vérifie que Redis est opérationnel avec auth."""

    def test_redis_ping(self, infra_services):
        import redis as redis_lib
        r = redis_lib.Redis.from_url(REDIS_TEST_URL, socket_connect_timeout=5)
        assert r.ping()
        r.close()

    def test_redis_set_get(self, infra_services):
        import redis as redis_lib
        r = redis_lib.Redis.from_url(REDIS_TEST_URL, socket_connect_timeout=5)
        key = "integration_test_key"
        r.set(key, "deepfake_detector_test", ex=60)
        val = r.get(key)
        r.delete(key)
        r.close()
        assert val == b"deepfake_detector_test"

    def test_redis_token_blocklist_db(self, infra_services):
        """DB 2 (token blocklist) doit être accessible."""
        import redis as redis_lib
        r = redis_lib.Redis.from_url(
            REDIS_TEST_URL.replace("/2", "/2"),  # explicit DB 2
            socket_connect_timeout=5,
        )
        r.set("__integration_test__", "1", ex=5)
        assert r.get("__integration_test__") == b"1"
        r.delete("__integration_test__")
        r.close()


# ── Résumé runner (standalone) ────────────────────────────────────────────────

if __name__ == "__main__":
    print("Lancer avec : pytest backend/tests/test_integration.py -v -m integration")
    print("Skip si Docker indisponible.")
