"""Tests des routes CRUD des dossiers — /cases/."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from models.user import User, UserRole


def _make_case(db: Session, owner: User, case_number: str = "CASE-001", title: str = "Test Case"):
    from models.case import Case, CaseStatus, Jurisdiction
    case = Case(
        case_number=case_number,
        title=title,
        jurisdiction=Jurisdiction.federal,
        status=CaseStatus.open,
        created_by_id=owner.id,
        retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


class TestCreateCase:
    """POST /cases/."""

    def test_analyst_can_create_case(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/cases/", json={
            "title": "Cas Forensique #1",
            "jurisdiction": "federal",
        }, headers=auth_analyst)
        assert resp.status_code == 201
        body = resp.json()
        assert body["title"] == "Cas Forensique #1"
        assert body["jurisdiction"] == "federal"
        assert "case_number" in body

    def test_admin_can_create_case(self, client: TestClient, auth_admin: dict):
        resp = client.post("/cases/", json={
            "title": "Admin Case",
            "jurisdiction": "quebec",
        }, headers=auth_admin)
        assert resp.status_code == 201

    def test_readonly_cannot_create_case(self, client: TestClient, auth_readonly: dict):
        resp = client.post("/cases/", json={
            "title": "Unauthorized",
            "jurisdiction": "federal",
        }, headers=auth_readonly)
        assert resp.status_code == 403

    def test_create_case_with_all_fields(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/cases/", json={
            "title": "Cas complet",
            "description": "Description détaillée",
            "jurisdiction": "ontario",
            "plaintiff": "Crown",
            "defendant": "John Doe",
            "counsel": "Me Tremblay",
        }, headers=auth_analyst)
        assert resp.status_code == 201
        body = resp.json()
        assert body["plaintiff"] == "Crown"
        assert body["defendant"] == "John Doe"

    def test_create_case_unauthenticated(self, client: TestClient):
        resp = client.post("/cases/", json={"title": "Test", "jurisdiction": "federal"})
        assert resp.status_code == 401


class TestListCases:
    """GET /cases/."""

    def test_list_cases_returns_list(self, client: TestClient, auth_analyst: dict):
        resp = client.get("/cases/", headers=auth_analyst)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_cases_with_status_filter(
        self, client: TestClient, db: Session, auth_analyst: dict, analyst_user: User
    ):
        _make_case(db, analyst_user, "LIST-001")
        resp = client.get("/cases/?status_filter=open", headers=auth_analyst)
        assert resp.status_code == 200
        cases = resp.json()
        assert all(c["status"] == "open" for c in cases)

    def test_list_cases_with_pagination(self, client: TestClient, auth_analyst: dict):
        resp = client.get("/cases/?skip=0&limit=10", headers=auth_analyst)
        assert resp.status_code == 200
        assert len(resp.json()) <= 10

    def test_readonly_can_list_cases(
        self, client: TestClient, db: Session, auth_readonly: dict, analyst_user: User
    ):
        _make_case(db, analyst_user, "LIST-RO-001")
        resp = client.get("/cases/", headers=auth_readonly)
        assert resp.status_code == 200


class TestGetCase:
    """GET /cases/{case_id}."""

    def test_get_existing_case(
        self, client: TestClient, db: Session, auth_analyst: dict, analyst_user: User
    ):
        case = _make_case(db, analyst_user, "GET-001", "My Case")
        resp = client.get(f"/cases/{case.id}", headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json()["title"] == "My Case"

    def test_get_nonexistent_case(self, client: TestClient, auth_analyst: dict):
        resp = client.get("/cases/99999", headers=auth_analyst)
        assert resp.status_code == 404

    def test_any_user_can_read_any_case(
        self, client: TestClient, db: Session, auth_readonly: dict, admin_user: User
    ):
        """Conception forensique: lecture ouverte à tous les users authentifiés."""
        case = _make_case(db, admin_user, "READ-ALL-001")
        resp = client.get(f"/cases/{case.id}", headers=auth_readonly)
        assert resp.status_code == 200


class TestUpdateCase:
    """PATCH /cases/{case_id}."""

    def test_owner_can_update_case(
        self, client: TestClient, db: Session, auth_analyst: dict, analyst_user: User
    ):
        case = _make_case(db, analyst_user, "UPD-001")
        resp = client.patch(f"/cases/{case.id}", json={
            "title": "Titre Mis à Jour",
            "jurisdiction": "federal",
        }, headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json()["title"] == "Titre Mis à Jour"

    def test_non_owner_cannot_update_case(
        self, client: TestClient, db: Session, auth_analyst: dict, admin_user: User
    ):
        """Un analyst ne peut pas modifier le dossier d'un autre (IDOR)."""
        case = _make_case(db, admin_user, "UPD-IDOR-001")
        resp = client.patch(f"/cases/{case.id}", json={
            "title": "Hack",
            "jurisdiction": "federal",
        }, headers=auth_analyst)
        assert resp.status_code == 403

    def test_admin_can_update_any_case(
        self, client: TestClient, db: Session, auth_admin: dict, analyst_user: User
    ):
        case = _make_case(db, analyst_user, "UPD-ADMIN-001")
        resp = client.patch(f"/cases/{case.id}", json={
            "title": "Admin Updated",
            "jurisdiction": "federal",
        }, headers=auth_admin)
        assert resp.status_code == 200

    def test_update_nonexistent_case(self, client: TestClient, auth_analyst: dict):
        resp = client.patch("/cases/99999", json={
            "title": "Ghost",
            "jurisdiction": "federal",
        }, headers=auth_analyst)
        assert resp.status_code == 404


class TestArchiveCase:
    """POST /cases/{case_id}/archive."""

    def test_owner_can_archive_case(
        self, client: TestClient, db: Session, auth_analyst: dict, analyst_user: User
    ):
        case = _make_case(db, analyst_user, "ARCH-001")
        resp = client.post(f"/cases/{case.id}/archive", headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"

    def test_non_owner_cannot_archive(
        self, client: TestClient, db: Session, auth_analyst: dict, admin_user: User
    ):
        case = _make_case(db, admin_user, "ARCH-IDOR-001")
        resp = client.post(f"/cases/{case.id}/archive", headers=auth_analyst)
        assert resp.status_code == 403

    def test_admin_can_archive_any_case(
        self, client: TestClient, db: Session, auth_admin: dict, analyst_user: User
    ):
        case = _make_case(db, analyst_user, "ARCH-ADMIN-001")
        resp = client.post(f"/cases/{case.id}/archive", headers=auth_admin)
        assert resp.status_code == 200
