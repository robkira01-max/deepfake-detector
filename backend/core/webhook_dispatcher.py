"""Dispatcher webhook — HMAC-SHA256 signé, retry 3×, httpx async."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone

import httpx

from config import settings


def _sign_payload(secret: str, payload_bytes: bytes) -> str:
    """Retourne la signature HMAC-SHA256 hex du payload."""
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


def _encrypt_secret(plain: str) -> str:
    """Chiffre un secret webhook avec AES-256-GCM."""
    from core.encryption import encrypt_secret
    return encrypt_secret(plain)


def _decrypt_secret(encrypted: str) -> str:
    """Déchiffre un secret webhook avec AES-256-GCM."""
    from core.encryption import decrypt_secret
    return decrypt_secret(encrypted)


def dispatch_webhook_sync(
    delivery_id: int,
) -> None:
    """Appelé par la tâche Celery — effectue la livraison et met à jour WebhookDelivery."""
    from database import SessionLocal
    from models.webhook import WebhookDelivery, WebhookStatus

    db = SessionLocal()
    try:
        delivery = db.query(WebhookDelivery).filter(WebhookDelivery.id == delivery_id).first()
        if not delivery:
            return

        webhook = delivery.webhook
        if not webhook or not webhook.active:
            return

        payload_bytes = json.dumps(delivery.payload, default=str).encode()

        secret = ""
        if webhook.secret_encrypted:
            try:
                secret = _decrypt_secret(webhook.secret_encrypted)
            except Exception:
                pass

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": f"DeepfakeDetector/{settings.app_version}",
            "X-DeepfakeDetector-Event": delivery.event,
            "X-DeepfakeDetector-Delivery": str(delivery.id),
        }
        if secret:
            headers["X-Hub-Signature-256"] = f"sha256={_sign_payload(secret, payload_bytes)}"

        max_attempts = 3
        last_error: str | None = None
        last_code: int | None = None
        last_body: str | None = None

        for attempt in range(1, max_attempts + 1):
            delivery.attempt_count = attempt
            db.commit()
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(webhook.url, content=payload_bytes, headers=headers)
                last_code = resp.status_code
                last_body = resp.text[:512]
                if 200 <= resp.status_code < 300:
                    delivery.status = WebhookStatus.delivered
                    delivery.response_status_code = last_code
                    delivery.response_body = last_body
                    delivery.delivered_at = datetime.now(timezone.utc)
                    db.commit()
                    return
                else:
                    last_error = f"HTTP {resp.status_code}"
            except Exception as exc:
                last_error = str(exc)[:256]

            if attempt < max_attempts:
                time.sleep(2 ** attempt)  # backoff: 2s, 4s

        delivery.status = WebhookStatus.failed
        delivery.response_status_code = last_code
        delivery.response_body = last_body
        delivery.error_message = last_error
        db.commit()
    finally:
        db.close()
