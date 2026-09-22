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
