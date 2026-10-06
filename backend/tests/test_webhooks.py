"""Tests pour la gestion des webhooks (CRUD + livraison)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from models.webhook import WebhookEvent, WebhookStatus


class TestWebhookCRUD:
    """Tests CRUD /webhooks/."""

    def test_list_webhooks_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/webhooks/")
        assert resp.status_code == 401

    def test_list_webhooks_empty(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/webhooks/", headers=auth_analyst)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_webhook_minimal(self, client: TestClient, auth_analyst: dict) -> None:
        payload = {
            "name": "Test SIEM",
            "url": "https://siem.example.com/deepfake",
            "events": ["analysis.completed"],
        }
        resp = client.post("/webhooks/", json=payload, headers=auth_analyst)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test SIEM"
        assert data["active"] is True
        assert "analysis.completed" in data["events"]

    def test_create_webhook_with_secret(self, client: TestClient, auth_analyst: dict, db) -> None:
        payload = {
            "name": "Secure Hook",
            "url": "https://secure.example.com/hook",
            "secret": "my-hmac-secret",
            "events": ["analysis.completed", "analysis.failed"],
        }
        # Secret needs AES-256-GCM key — skip if not configured
        from config import settings
        if not settings.encryption_key:
            pytest.skip("ENCRYPTION_KEY not configured")

        resp = client.post("/webhooks/", json=payload, headers=auth_analyst)
        assert resp.status_code == 201
        data = resp.json()
        # Secret must NOT be returned in response
        assert "secret" not in data
        assert "secret_encrypted" not in data

    def test_create_webhook_empty_events_returns_422(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        payload = {
            "name": "Bad Hook",
            "url": "https://example.com/hook",
            "events": [],
        }
        resp = client.post("/webhooks/", json=payload, headers=auth_analyst)
        assert resp.status_code == 422

    def test_get_webhook_by_id(self, client: TestClient, auth_analyst: dict) -> None:
        # Create first
        create_resp = client.post(
            "/webhooks/",
            json={"name": "GetTest", "url": "https://x.example.com", "events": ["analysis.completed"]},
            headers=auth_analyst,
        )
        assert create_resp.status_code == 201
        wid = create_resp.json()["id"]

        resp = client.get(f"/webhooks/{wid}", headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json()["id"] == wid

    def test_get_webhook_not_found(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/webhooks/99999", headers=auth_analyst)
        assert resp.status_code == 404

    def test_update_webhook_name(self, client: TestClient, auth_analyst: dict) -> None:
        create_resp = client.post(
            "/webhooks/",
            json={"name": "OldName", "url": "https://y.example.com", "events": ["analysis.completed"]},
            headers=auth_analyst,
        )
        wid = create_resp.json()["id"]

        resp = client.patch(f"/webhooks/{wid}", json={"name": "NewName"}, headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json()["name"] == "NewName"

    def test_update_webhook_deactivate(self, client: TestClient, auth_analyst: dict) -> None:
        create_resp = client.post(
            "/webhooks/",
            json={"name": "ToDisable", "url": "https://z.example.com", "events": ["analysis.completed"]},
            headers=auth_analyst,
        )
        wid = create_resp.json()["id"]

        resp = client.patch(f"/webhooks/{wid}", json={"active": False}, headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json()["active"] is False

    def test_delete_webhook(self, client: TestClient, auth_analyst: dict) -> None:
        create_resp = client.post(
            "/webhooks/",
            json={"name": "ToDelete", "url": "https://d.example.com", "events": ["analysis.completed"]},
            headers=auth_analyst,
        )
        wid = create_resp.json()["id"]

        del_resp = client.delete(f"/webhooks/{wid}", headers=auth_analyst)
        assert del_resp.status_code == 204

        get_resp = client.get(f"/webhooks/{wid}", headers=auth_analyst)
        assert get_resp.status_code == 404

    def test_delete_webhook_not_found(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.delete("/webhooks/99999", headers=auth_analyst)
        assert resp.status_code == 404


class TestWebhookDeliveries:
    """Tests des livraisons de webhooks."""

    def test_list_deliveries_empty(self, client: TestClient, auth_analyst: dict) -> None:
        create_resp = client.post(
            "/webhooks/",
            json={"name": "DeliveriesTest", "url": "https://e.example.com", "events": ["analysis.completed"]},
            headers=auth_analyst,
        )
        wid = create_resp.json()["id"]

        resp = client.get(f"/webhooks/{wid}/deliveries", headers=auth_analyst)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_deliveries_not_found(self, client: TestClient, auth_analyst: dict) -> None:
        resp = client.get("/webhooks/99999/deliveries", headers=auth_analyst)
        assert resp.status_code == 404

    def test_webhook_test_endpoint_bad_url(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        """Test endpoint should return 202 even if delivery fails (bad URL)."""
        create_resp = client.post(
            "/webhooks/",
            json={
                "name": "TestEndpoint",
                "url": "http://localhost:1/unreachable",
                "events": ["analysis.completed"],
            },
            headers=auth_analyst,
        )
        assert create_resp.status_code == 201
        wid = create_resp.json()["id"]

        resp = client.post(f"/webhooks/{wid}/test", headers=auth_analyst)
        assert resp.status_code == 202
        data = resp.json()
        assert "delivery_id" in data
        assert "status" in data
        assert data["status"] in ("delivered", "failed")

    def test_webhook_test_creates_delivery_record(
        self, client: TestClient, auth_analyst: dict
    ) -> None:
        create_resp = client.post(
            "/webhooks/",
            json={
                "name": "DeliveryRecord",
                "url": "http://localhost:1/unreachable",
                "events": ["analysis.completed"],
            },
            headers=auth_analyst,
        )
        wid = create_resp.json()["id"]

        client.post(f"/webhooks/{wid}/test", headers=auth_analyst)

        deliveries_resp = client.get(f"/webhooks/{wid}/deliveries", headers=auth_analyst)
        assert deliveries_resp.status_code == 200
        deliveries = deliveries_resp.json()
        assert len(deliveries) >= 1
        assert deliveries[0]["event"] == "webhook.test"


class TestWebhookDispatcher:
    """Tests unitaires du dispatcher HMAC."""

    def test_sign_payload_deterministic(self) -> None:
        from core.webhook_dispatcher import _sign_payload
        payload = b'{"event": "test"}'
        sig1 = _sign_payload("mysecret", payload)
        sig2 = _sign_payload("mysecret", payload)
        assert sig1 == sig2

    def test_sign_payload_different_secrets(self) -> None:
        from core.webhook_dispatcher import _sign_payload
        payload = b'{"event": "test"}'
        sig1 = _sign_payload("secret1", payload)
        sig2 = _sign_payload("secret2", payload)
        assert sig1 != sig2

    def test_encrypt_decrypt_roundtrip(self) -> None:
        import base64
        from unittest import mock
        import core.encryption as enc_module
        aes_key = base64.urlsafe_b64encode(b"\xCD" * 32).decode()
        with mock.patch.object(enc_module, "settings") as m:
            m.encryption_key = aes_key
            from core.webhook_dispatcher import _encrypt_secret, _decrypt_secret
            original = "my-webhook-secret-12345"
            encrypted = _encrypt_secret(original)
            assert encrypted != original
            decrypted = _decrypt_secret(encrypted)
            assert decrypted == original
