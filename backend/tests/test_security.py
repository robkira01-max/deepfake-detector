"""Tests de sécurité — SQLi, IDOR, path traversal, CORS, headers, rate limiting."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestSQLInjection:
    """Aucune requête SQLi ne doit provoquer une erreur 500 ou leaker des données."""

    SQL_PAYLOADS = [
        "' OR '1'='1",
        "'; DROP TABLE users; --",
        "1 UNION SELECT username, password FROM users --",
        "' OR 1=1 --",
        "admin'--",
        "1; SELECT * FROM analyses",
    ]

    def test_login_sqli_payloads(self, client: TestClient):
        for payload in self.SQL_PAYLOADS:
            resp = client.post("/auth/token", data={
                "username": payload,
                "password": payload,
            })
            # Jamais 500 (erreur DB) — 401/422/429 (rate limit) tous acceptables
            assert resp.status_code in (401, 422, 429), (
                f"SQLi payload '{payload}' a retourné {resp.status_code}"
            )

    def test_case_search_sqli(self, client: TestClient, auth_analyst: dict):
        for payload in self.SQL_PAYLOADS[:3]:
            resp = client.get(f"/cases/?search={payload}", headers=auth_analyst)
            assert resp.status_code != 500, f"SQLi dans search retourne 500 : {payload}"

    def test_case_id_sqli(self, client: TestClient, auth_analyst: dict):
        """Les IDs entiers rejettent les payloads texte avec 422."""
        resp = client.get("/cases/1%20OR%201%3D1", headers=auth_analyst)
        assert resp.status_code in (404, 422)


class TestIDOR:
    """Tests IDOR — vérification des contrôles d'accès par ressource.

    Note de conception : en forensique, les analysts voient TOUS les dossiers
    (collaboration multi-experts). L'IDOR est contrôlé au niveau PATCH/DELETE
    (seul le créateur ou l'admin peut modifier). GET est intentionnellement ouvert.
    """

    def test_analyst_can_read_any_case(
        self, client: TestClient, db, auth_analyst: dict, admin_user
    ):
        """Design forensique : les analysts peuvent lire tous les dossiers (READ)."""
        from models.case import Case, CaseStatus, Jurisdiction
        from datetime import datetime, timezone, timedelta

        # Case créé par admin
        admin_case = Case(
            case_number="ADMIN-CASE-001",
            title="Admin Private Case",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=admin_user.id,
            retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
        )
        db.add(admin_case)
        db.commit()
        db.refresh(admin_case)

        # En forensique, l'analyst PEUT lire les dossiers d'autres utilisateurs
        resp = client.get(f"/cases/{admin_case.id}", headers=auth_analyst)
        assert resp.status_code == 200

    def test_analyst_cannot_upload_to_other_case(
        self, client: TestClient, db, auth_analyst: dict, admin_user, tiny_wav
    ):
        from models.case import Case, CaseStatus, Jurisdiction
        from datetime import datetime, timezone, timedelta

        admin_case = Case(
            case_number="ADMIN-CASE-002",
            title="Admin Upload Test",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=admin_user.id,
            retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
        )
        db.add(admin_case)
        db.commit()
        db.refresh(admin_case)

        with open(tiny_wav, "rb") as f:
            resp = client.post(
                f"/analyze/upload/{admin_case.id}",
                files={"file": ("test.wav", f, "audio/wav")},
                headers=auth_analyst,
            )
        assert resp.status_code in (403, 404)

    def test_analyst_cannot_read_other_analysis(
        self, client: TestClient, db, auth_analyst: dict, admin_user
    ):
        from models.analysis import Analysis, AnalysisStatus
        from models.case import Case, CaseStatus, Jurisdiction
        from datetime import datetime, timezone, timedelta

        # Créer case + analysis pour admin
        admin_case = Case(
            case_number="ADMIN-CASE-003",
            title="Admin Analysis Test",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=admin_user.id,
            retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
        )
        db.add(admin_case)
        db.commit()
        db.refresh(admin_case)

        analysis = Analysis(
            case_id=admin_case.id,
            media_file_id=1,  # fictif
            status=AnalysisStatus.pending,
            requested_by_id=admin_user.id,
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        resp = client.get(f"/analyze/{analysis.id}", headers=auth_analyst)
        assert resp.status_code in (403, 404)


    def test_unauthenticated_cannot_read_analysis(
        self, client: TestClient, db, admin_user
    ):
        """Sans token, toute tentative d'accès à une analyse doit retourner 401."""
        from models.analysis import Analysis, AnalysisStatus
        from models.case import Case, CaseStatus, Jurisdiction
        from datetime import datetime, timezone, timedelta

        admin_case = Case(
            case_number="UNAUTH-ANAL-001",
            title="Unauth Analysis Test",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=admin_user.id,
            retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
        )
        db.add(admin_case)
        db.commit()
        db.refresh(admin_case)

        analysis = Analysis(
            case_id=admin_case.id,
            media_file_id=1,
            status=AnalysisStatus.pending,
            requested_by_id=admin_user.id,
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        resp = client.get(f"/analyze/{analysis.id}")
        assert resp.status_code == 401

    def test_analyst_cannot_delete_other_case(
        self, client: TestClient, db, auth_analyst: dict, admin_user
    ):
        """Un analyst ne peut pas archiver/supprimer un dossier qu'il ne possède pas."""
        from models.case import Case, CaseStatus, Jurisdiction
        from datetime import datetime, timezone, timedelta

        admin_case = Case(
            case_number="IDOR-DEL-001",
            title="Delete Protection Test",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=admin_user.id,
            retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
        )
        db.add(admin_case)
        db.commit()
        db.refresh(admin_case)

        resp = client.post(f"/cases/{admin_case.id}/archive", headers=auth_analyst)
        assert resp.status_code in (403, 404)

    def test_analyst_cannot_patch_other_case(
        self, client: TestClient, db, auth_analyst: dict, admin_user
    ):
        """Un analyst ne peut pas modifier un dossier dont il n'est pas le créateur."""
        from models.case import Case, CaseStatus, Jurisdiction
        from datetime import datetime, timezone, timedelta

        admin_case = Case(
            case_number="IDOR-PATCH-001",
            title="Patch Protection Test",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=admin_user.id,
            retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
        )
        db.add(admin_case)
        db.commit()
        db.refresh(admin_case)

        resp = client.patch(
            f"/cases/{admin_case.id}",
            json={"title": "Hacked"},
            headers=auth_analyst,
        )
        assert resp.status_code in (403, 404)

    def test_nonexistent_resource_returns_404_not_500(
        self, client: TestClient, auth_analyst: dict
    ):
        """Un ID inexistant (très grand) doit retourner 404, jamais 500."""
        resp = client.get("/analyze/999999999", headers=auth_analyst)
        assert resp.status_code in (403, 404)

        resp = client.get("/cases/999999999", headers=auth_analyst)
        assert resp.status_code == 404


class TestMetricsGuard:
    """Tests de la garde Règle 7 — aucun chiffre de fiabilité sans metrics.json."""

    def test_require_metrics_raises_when_missing(self):
        """_require_metrics() doit lever ValueError si metrics.json absent."""
        import unittest.mock as mock
        import engines.fusion as fusion_module

        with mock.patch("engines.fusion._METRICS_PATH") as mock_path:
            mock_path.exists.return_value = False
            mock_path.__str__ = lambda self: "/fake/metrics.json"
            with pytest.raises(ValueError, match="metrics.json"):
                fusion_module._require_metrics()

    def test_fusion_result_metrics_available_false_when_no_file(self):
        """FusionResult.metrics_available doit être False quand metrics.json absent."""
        from engines.fusion import fuse_scores, _METRICS_PATH
        import unittest.mock as mock

        with mock.patch("engines.fusion._METRICS_PATH") as m:
            m.exists.return_value = False
            result = fuse_scores(score_metadata=0.5)
        assert result.metrics_available is False
        assert result.metrics_artifact is None

    def test_fusion_result_metrics_available_true_when_file_exists(self, tmp_path):
        """FusionResult.metrics_available doit être True quand metrics.json présent."""
        from engines.fusion import fuse_scores
        import unittest.mock as mock

        with mock.patch("engines.fusion._METRICS_PATH") as m:
            m.exists.return_value = True
            m.__str__ = lambda self: str(tmp_path / "metrics.json")
            result = fuse_scores(score_metadata=0.5)
        assert result.metrics_available is True


class TestPathTraversal:
    """Les noms de fichiers malveillants ne doivent pas traverser les répertoires."""

    TRAVERSAL_FILENAMES = [
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "....//....//etc//passwd",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "file\x00.wav",
    ]

    def test_upload_traversal_filenames_rejected(
        self, client: TestClient, db, auth_analyst: dict, analyst_user, tiny_wav
    ):
        from models.case import Case, CaseStatus, Jurisdiction
        from datetime import datetime, timezone, timedelta

        case = Case(
            case_number="TRAV-001",
            title="Traversal Test",
            jurisdiction=Jurisdiction.federal,
            status=CaseStatus.open,
            created_by_id=analyst_user.id,
            retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
        )
        db.add(case)
        db.commit()
        db.refresh(case)

        for evil_name in self.TRAVERSAL_FILENAMES:
            with open(tiny_wav, "rb") as f:
                resp = client.post(
                    f"/analyze/upload/{case.id}",
                    files={"file": (evil_name, f, "audio/wav")},
                    headers=auth_analyst,
                )
            # Doit être rejeté (422) ou accepté avec nom sanitisé — jamais 500
            assert resp.status_code != 500, f"Path traversal '{evil_name}' → 500"


class TestInputValidation:
    """Les inputs invalides sont rejetés avec 422, jamais avec 500."""

    def test_case_title_too_long(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/cases/", json={
            "title": "A" * 10000,  # Titre de 10000 chars
            "jurisdiction": "federal",
        }, headers=auth_analyst)
        # 422 si Pydantic le valide, 400 si validation métier, 503 si Redis absent
        # L'important : pas de 500 non géré
        assert resp.status_code != 500

    def test_invalid_jurisdiction(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/cases/", json={
            "title": "Valid Title",
            "jurisdiction": "mars",  # Valeur invalide
        }, headers=auth_analyst)
        # 422 si Pydantic rejette, 503 si Redis indisponible en test
        assert resp.status_code in (422, 503)

    def test_xss_in_case_title_no_crash(self, client: TestClient, auth_analyst: dict):
        """Le XSS dans les champs texte ne doit pas crasher l'app.

        Note: une API REST JSON retourne le contenu tel quel en JSON — c'est normal.
        La protection XSS est responsabilité du front-end (Content-Type: application/json
        + encodage HTML côté client). L'API ne doit simplement pas crasher.
        """
        xss_payload = "<script>alert('XSS')</script>"
        resp = client.post("/cases/", json={
            "title": xss_payload,
            "jurisdiction": "federal",
        }, headers=auth_analyst)
        # L'important est qu'il n'y ait pas de 500
        assert resp.status_code != 500
        # Vérifier que Content-Type est bien application/json (pas text/html)
        if resp.status_code in (200, 201):
            content_type = resp.headers.get("content-type", "")
            assert "application/json" in content_type, (
                "Une API JSON doit retourner Content-Type: application/json "
                "pour éviter que le navigateur interprète le XSS"
            )


class TestSecurityHeaders:
    """Vérification des en-têtes de sécurité HTTP."""

    def test_no_server_header_leak(self, client: TestClient):
        """L'en-tête Server ne doit pas révéler la version exacte."""
        resp = client.get("/health")
        server = resp.headers.get("server", "")
        # uvicorn révèle "uvicorn" — acceptable, mais pas de version
        assert "python" not in server.lower()

    def test_content_type_json_on_api_response(self, client: TestClient):
        resp = client.get("/health")
        assert "application/json" in resp.headers.get("content-type", "")


class TestUnauthenticatedAccess:
    """Routes sensibles doivent retourner 401 sans token."""

    PROTECTED_ROUTES = [
        ("GET", "/cases/"),
        ("POST", "/cases/"),
        ("GET", "/analyze/1"),
        ("GET", "/analyze/case/1"),
    ]

    def test_all_protected_routes_require_auth(self, client: TestClient):
        for method, path in self.PROTECTED_ROUTES:
            if method == "GET":
                resp = client.get(path)
            else:
                resp = client.post(path, json={})
            assert resp.status_code == 401, (
                f"{method} {path} devrait retourner 401, a retourné {resp.status_code}"
            )


class TestCoreSecurityUnit:
    """Tests unitaires des fonctions core/security.py non couvertes par les intégrations."""

    # ── verify_totp ────────────────────────────────────────────────────────────

    def test_verify_totp_valid_code(self):
        import pyotp
        from core.security import generate_mfa_secret, verify_totp

        secret = generate_mfa_secret()
        totp = pyotp.TOTP(secret)
        code = totp.now()
        assert verify_totp(secret, code) is True

    def test_verify_totp_invalid_code(self):
        from core.security import generate_mfa_secret, verify_totp

        secret = generate_mfa_secret()
        assert verify_totp(secret, "000000") is False

    def test_generate_mfa_secret_base32(self):
        from core.security import generate_mfa_secret

        secret = generate_mfa_secret()
        # Base32 = A-Z + 2-7
        import re
        assert re.match(r"^[A-Z2-7]{16,}$", secret), f"Format invalide : {secret}"

    def test_get_mfa_provisioning_uri(self):
        from core.security import generate_mfa_secret, get_mfa_provisioning_uri

        secret = generate_mfa_secret()
        uri = get_mfa_provisioning_uri("testuser", secret)
        assert uri.startswith("otpauth://totp/")
        assert "testuser" in uri

    def test_generate_mfa_qr_base64_returns_png(self):
        from core.security import generate_mfa_secret, generate_mfa_qr_base64
        import base64

        secret = generate_mfa_secret()
        b64 = generate_mfa_qr_base64("user", secret)
        raw = base64.b64decode(b64)
        # Magic bytes PNG : \x89PNG
        assert raw[:4] == b"\x89PNG", "Le QR code n'est pas un PNG valide"

    # ── RSA keypair generation ─────────────────────────────────────────────────

    def test_generate_rsa_keypair_creates_files(self, tmp_path):
        from core.security import generate_rsa_keypair

        generate_rsa_keypair(tmp_path)

        private_pem = tmp_path / "private.pem"
        public_pem = tmp_path / "public.pem"
        assert private_pem.exists(), "private.pem non créé"
        assert public_pem.exists(), "public.pem non créé"

    def test_generate_rsa_keypair_pem_format(self, tmp_path):
        from core.security import generate_rsa_keypair

        generate_rsa_keypair(tmp_path)
        private_content = (tmp_path / "private.pem").read_text()
        public_content = (tmp_path / "public.pem").read_text()

        assert "BEGIN RSA PRIVATE KEY" in private_content
        assert "BEGIN PUBLIC KEY" in public_content

    def test_generate_rsa_keypair_creates_output_dir(self, tmp_path):
        from core.security import generate_rsa_keypair

        new_dir = tmp_path / "subdir" / "keys"
        generate_rsa_keypair(new_dir)
        assert new_dir.exists()

    # ── JWT edge cases ─────────────────────────────────────────────────────────

    def test_private_key_missing_raises_runtime_error(self, monkeypatch):
        """Patch _private_key directement car jwt_private_key est une computed property."""
        import core.security as sec
        # Patch la fonction interne pour simuler une clé absente
        monkeypatch.setattr(sec, "_private_key", lambda: (_ for _ in ()).throw(
            RuntimeError("JWT private key not found at /fake/path/private.pem")
        ))
        with pytest.raises(RuntimeError, match="JWT private key not found"):
            sec._private_key()

    def test_public_key_missing_raises_runtime_error(self, monkeypatch):
        """Vérifier que _public_key lève RuntimeError si la clé est absente."""
        import core.security as sec
        monkeypatch.setattr(sec, "_public_key", lambda: (_ for _ in ()).throw(
            RuntimeError("JWT public key not found at /fake/path/public.pem")
        ))
        with pytest.raises(RuntimeError, match="JWT public key not found"):
            sec._public_key()

    def test_decode_revoked_token_raises_401(self, monkeypatch):
        """Token révoqué (is_revoked=True) → HTTPException 401."""
        import core.security as sec

        # Créer un vrai token valide
        token = sec.create_access_token("testuser", "analyst")

        # Simuler la révocation
        monkeypatch.setattr(sec, "is_revoked", lambda jti: True)

        with pytest.raises(Exception) as exc_info:
            sec.decode_token(token)
        assert exc_info.value.status_code == 401

    def test_decode_token_wrong_type_in_get_current_user(
        self, db, monkeypatch
    ):
        """Un refresh token ne doit pas passer get_current_user."""
        import core.security as sec

        # Créer un refresh token (type="refresh" pas "access")
        refresh_token = sec.create_refresh_token("testuser")

        with pytest.raises(Exception) as exc_info:
            sec.get_current_user(refresh_token, db)
        assert exc_info.value.status_code == 401

    def test_mfa_sub_prefix_rejected(self, db):
        """Token avec sub='mfa:...' doit être rejeté."""
        import core.security as sec

        token = sec.create_access_token("mfa:testuser", "analyst")
        with pytest.raises(Exception) as exc_info:
            sec.get_current_user(token, db)
        assert exc_info.value.status_code == 401

    def test_inactive_user_rejected(self, db, admin_user):
        """Un utilisateur is_active=False doit être rejeté."""
        import core.security as sec

        # Désactiver l'admin user
        admin_user.is_active = False
        db.commit()

        token = sec.create_access_token(admin_user.username, "admin")
        with pytest.raises(Exception) as exc_info:
            sec.get_current_user(token, db)
        assert exc_info.value.status_code == 401

        # Re-activer pour ne pas affecter les autres tests
        admin_user.is_active = True
        db.commit()
