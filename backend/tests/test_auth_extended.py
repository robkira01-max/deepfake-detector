"""Tests étendus auth : MFA setup, /me, /users list, refresh edge cases."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from models.user import User, UserRole


class TestAuthMe:
    """GET /auth/me — profil utilisateur courant."""

    def test_me_returns_current_user(self, client: TestClient, auth_analyst: dict, analyst_user: User):
        resp = client.get("/auth/me", headers=auth_analyst)
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == analyst_user.username
        assert body["role"] == "analyst"

    def test_me_unauthenticated(self, client: TestClient):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_admin(self, client: TestClient, auth_admin: dict, admin_user: User):
        resp = client.get("/auth/me", headers=auth_admin)
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"

    def test_me_readonly(self, client: TestClient, auth_readonly: dict, readonly_user: User):
        resp = client.get("/auth/me", headers=auth_readonly)
        assert resp.status_code == 200
        assert resp.json()["username"] == readonly_user.username


class TestUserManagement:
    """GET /auth/users — liste des utilisateurs (admin seulement)."""

    def test_admin_can_list_users(self, client: TestClient, auth_admin: dict, analyst_user: User):
        resp = client.get("/auth/users", headers=auth_admin)
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) >= 1

    def test_analyst_cannot_list_users(self, client: TestClient, auth_analyst: dict):
        resp = client.get("/auth/users", headers=auth_analyst)
        assert resp.status_code == 403

    def test_readonly_cannot_list_users(self, client: TestClient, auth_readonly: dict):
        resp = client.get("/auth/users", headers=auth_readonly)
        assert resp.status_code == 403

    def test_list_users_unauthenticated(self, client: TestClient):
        resp = client.get("/auth/users")
        assert resp.status_code == 401

    def test_list_users_pagination(self, client: TestClient, auth_admin: dict):
        resp = client.get("/auth/users?skip=0&limit=1", headers=auth_admin)
        assert resp.status_code == 200
        assert len(resp.json()) <= 1

    def test_create_user_duplicate_email(
        self, client: TestClient, auth_admin: dict, analyst_user: User
    ):
        resp = client.post("/auth/users", json={
            "username": "brand_new_user",
            "email": analyst_user.email,  # email déjà pris
            "password": "StrongPass123!",
            "role": "readonly",
        }, headers=auth_admin)
        assert resp.status_code == 409

    def test_create_user_invalid_username_chars(self, client: TestClient, auth_admin: dict):
        resp = client.post("/auth/users", json={
            "username": "bad user!@#",
            "email": "bad@test.ca",
            "password": "StrongPass123!",
            "role": "readonly",
        }, headers=auth_admin)
        assert resp.status_code == 422

    def test_create_user_password_no_special(self, client: TestClient, auth_admin: dict):
        resp = client.post("/auth/users", json={
            "username": "nospecial_u",
            "email": "nospec@test.ca",
            "password": "StrongPass1234",  # pas de caractère spécial
            "role": "readonly",
        }, headers=auth_admin)
        assert resp.status_code == 422

    def test_create_user_password_no_digit(self, client: TestClient, auth_admin: dict):
        resp = client.post("/auth/users", json={
            "username": "nodigit_user",
            "email": "nodigit@test.ca",
            "password": "StrongPassword!",  # pas de chiffre
            "role": "readonly",
        }, headers=auth_admin)
        assert resp.status_code == 422


class TestMFASetup:
    """POST /auth/mfa/setup — configuration MFA TOTP."""

    def test_mfa_setup_returns_secret_and_qr(self, client: TestClient, auth_analyst: dict):
        resp = client.post("/auth/mfa/setup", headers=auth_analyst)
        assert resp.status_code == 200
        body = resp.json()
        assert "secret" in body
        assert "qr_code_base64" in body
        assert "provisioning_uri" in body
        assert len(body["secret"]) > 10

    def test_mfa_setup_unauthenticated(self, client: TestClient):
        resp = client.post("/auth/mfa/setup")
        assert resp.status_code == 401

    def test_mfa_setup_admin(self, client: TestClient, auth_admin: dict):
        resp = client.post("/auth/mfa/setup", headers=auth_admin)
        assert resp.status_code == 200

    def test_mfa_enable_without_setup_fails(self, client: TestClient, auth_analyst: dict):
        """Activer MFA sans avoir d'abord appelé /setup → 400."""
        resp = client.post("/auth/mfa/enable", json={"totp_code": "123456"}, headers=auth_analyst)
        # 400 car pas de mfa_secret, ou 400 car code invalide selon l'état
        assert resp.status_code in (400, 422)

    def test_mfa_enable_invalid_code_format(self, client: TestClient, auth_analyst: dict):
        """Code TOTP non numérique → 422 validation Pydantic."""
        resp = client.post("/auth/mfa/enable", json={"totp_code": "abcdef"}, headers=auth_analyst)
        assert resp.status_code == 422


class TestRefreshEdgeCases:
    """POST /auth/refresh — cas limites."""

    def test_refresh_with_access_token_fails(self, client: TestClient, analyst_user: User):
        """Utiliser un access token comme refresh token → 400."""
        from core.security import create_access_token
        access_tok = create_access_token(analyst_user.username, analyst_user.role.value)
        resp = client.post("/auth/refresh", json={"refresh_token": access_tok})
        assert resp.status_code == 400

    def test_refresh_invalid_token(self, client: TestClient):
        resp = client.post("/auth/refresh", json={"refresh_token": "garbage.token.value"})
        assert resp.status_code == 401

    def test_refresh_missing_body(self, client: TestClient):
        resp = client.post("/auth/refresh", json={})
        assert resp.status_code == 422

    def test_refresh_valid_works(self, client: TestClient, analyst_user: User):
        """Un refresh token valide doit retourner un nouveau access token."""
        from core.security import create_refresh_token
        refresh_tok = create_refresh_token(analyst_user.username)
        resp = client.post("/auth/refresh", json={"refresh_token": refresh_tok})
        assert resp.status_code == 200
        assert "access_token" in resp.json()
