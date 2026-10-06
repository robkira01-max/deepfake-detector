"""Gestion des abonnements webhook — CRUD admin + test de livraison."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from config import settings
from core.security import require_analyst
from database import get_db
from models.audit_log import AuditLog, AuditAction
from models.responses import ANALYST_ERRORS
from models.user import User
from models.webhook import Webhook, WebhookDelivery, WebhookEvent, WebhookStatus

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# ── Schémas Pydantic ─────────────────────────────────────────────────────────

class WebhookCreate(BaseModel):
    name: str
    url: str
    secret: str | None = None
    events: list[WebhookEvent] = [WebhookEvent.analysis_completed]

    @field_validator("events")
    @classmethod
    def events_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("Au moins un événement requis")
        return v


class WebhookUpdate(BaseModel):
    name: str | None = None
    url: str | None = None
    secret: str | None = None
    events: list[WebhookEvent] | None = None
    active: bool | None = None


class WebhookResponse(BaseModel):
    id: int
    name: str
    url: str
    events: list[str]
    active: bool
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class DeliveryResponse(BaseModel):
    id: int
    webhook_id: int
    event: str
    status: str
    attempt_count: int
    response_status_code: int | None
    error_message: str | None
    created_at: str
    delivered_at: str | None

    model_config = {"from_attributes": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _webhook_to_response(w: Webhook) -> WebhookResponse:
    return WebhookResponse(
        id=w.id,
        name=w.name,
        url=w.url,
        events=w.events,
        active=w.active,
        created_at=w.created_at.isoformat(),
        updated_at=w.updated_at.isoformat(),
    )


def _add_audit(db: Session, user: User, action: AuditAction, resource_id: str, details: dict) -> None:
    entry = AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=action,
        resource_type="webhook",
        resource_id=resource_id,
        details=details,
    )
    try:
        from core.chain_of_custody import sign_audit_entry
        entry.entry_hash, entry.signature_b64 = sign_audit_entry(entry)
    except Exception:
        pass
    db.add(entry)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=list[WebhookResponse],
    summary="Lister les webhooks",
    responses=ANALYST_ERRORS,
)
def list_webhooks(
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> list[WebhookResponse]:
    hooks = db.query(Webhook).order_by(Webhook.id).all()
    return [_webhook_to_response(h) for h in hooks]


@router.post(
    "/",
    response_model=WebhookResponse,
    status_code=201,
    summary="Créer un abonnement webhook",
    responses=ANALYST_ERRORS,
)
def create_webhook(
    body: WebhookCreate,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> WebhookResponse:
    from core.webhook_dispatcher import _encrypt_secret
    from config import settings

    secret_enc: str | None = None
    if body.secret:
        if not settings.fernet_key:
            raise HTTPException(status_code=503, detail="Chiffrement non configuré (FERNET_KEY manquante)")
        secret_enc = _encrypt_secret(body.secret)

    hook = Webhook(
        name=body.name,
        url=body.url,
        secret_encrypted=secret_enc,
        events=[e.value for e in body.events],
        active=True,
        created_by_id=current_user.id,
    )
    db.add(hook)
    db.flush()

    _add_audit(db, current_user, AuditAction.WEBHOOK_CREATED, str(hook.id), {
        "name": hook.name,
        "url": hook.url,
        "events": hook.events,
    })
    db.commit()
    db.refresh(hook)
    return _webhook_to_response(hook)


@router.get(
    "/{webhook_id}",
    response_model=WebhookResponse,
    summary="Détail d'un webhook",
    responses=ANALYST_ERRORS,
)
def get_webhook(
    webhook_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> WebhookResponse:
    hook = db.query(Webhook).filter(Webhook.id == webhook_id).first()
    if not hook:
        raise HTTPException(status_code=404, detail="Webhook introuvable")
    return _webhook_to_response(hook)


@router.patch(
    "/{webhook_id}",
    response_model=WebhookResponse,
    summary="Mettre à jour un webhook",
    responses=ANALYST_ERRORS,
)
def update_webhook(
    webhook_id: int,
    body: WebhookUpdate,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> WebhookResponse:
    from core.webhook_dispatcher import _encrypt_secret
    from config import settings

    hook = db.query(Webhook).filter(Webhook.id == webhook_id).first()
    if not hook:
        raise HTTPException(status_code=404, detail="Webhook introuvable")

    if body.name is not None:
        hook.name = body.name
    if body.url is not None:
        hook.url = body.url
    if body.active is not None:
        hook.active = body.active
    if body.events is not None:
        hook.events = [e.value for e in body.events]
    if body.secret is not None:
        if not settings.fernet_key:
            raise HTTPException(status_code=503, detail="Chiffrement non configuré")
        hook.secret_encrypted = _encrypt_secret(body.secret)
    hook.updated_at = datetime.now(timezone.utc)

    _add_audit(db, current_user, AuditAction.WEBHOOK_UPDATED, str(webhook_id), {
        "fields_updated": [k for k, v in body.model_dump().items() if v is not None],
    })
    db.commit()
    db.refresh(hook)
    return _webhook_to_response(hook)


@router.delete(
    "/{webhook_id}",
    status_code=204,
    summary="Supprimer un webhook",
    responses=ANALYST_ERRORS,
)
def delete_webhook(
    webhook_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> None:
    hook = db.query(Webhook).filter(Webhook.id == webhook_id).first()
    if not hook:
        raise HTTPException(status_code=404, detail="Webhook introuvable")

    _add_audit(db, current_user, AuditAction.WEBHOOK_DELETED, str(webhook_id), {
        "name": hook.name,
        "url": hook.url,
    })
    db.delete(hook)
    db.commit()


@router.get(
    "/{webhook_id}/deliveries",
    response_model=list[DeliveryResponse],
    summary="Historique des livraisons d'un webhook",
    responses=ANALYST_ERRORS,
)
def list_deliveries(
    webhook_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = 50,
) -> list[DeliveryResponse]:
    hook = db.query(Webhook).filter(Webhook.id == webhook_id).first()
    if not hook:
        raise HTTPException(status_code=404, detail="Webhook introuvable")

    deliveries = (
        db.query(WebhookDelivery)
        .filter(WebhookDelivery.webhook_id == webhook_id)
        .order_by(WebhookDelivery.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        DeliveryResponse(
            id=d.id,
            webhook_id=d.webhook_id,
            event=d.event,
            status=d.status.value,
            attempt_count=d.attempt_count,
            response_status_code=d.response_status_code,
            error_message=d.error_message,
            created_at=d.created_at.isoformat(),
            delivered_at=d.delivered_at.isoformat() if d.delivered_at else None,
        )
        for d in deliveries
    ]


@router.post(
    "/{webhook_id}/test",
    status_code=202,
    summary="Envoyer un événement de test au webhook",
    responses=ANALYST_ERRORS,
)
def test_webhook(
    webhook_id: int,
    current_user: Annotated[User, Depends(require_analyst)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    hook = db.query(Webhook).filter(Webhook.id == webhook_id).first()
    if not hook:
        raise HTTPException(status_code=404, detail="Webhook introuvable")

    test_payload = {
        "event": "webhook.test",
        "webhook_id": webhook_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Test de livraison DeepfakeDetector",
    }

    delivery = WebhookDelivery(
        webhook_id=webhook_id,
        event="webhook.test",
        payload=test_payload,
        status=WebhookStatus.pending,
    )
    db.add(delivery)
    db.commit()
    db.refresh(delivery)

    # Livraison synchrone inline — utilise la session courante (compatible SQLite in-memory)
    from core.webhook_dispatcher import _decrypt_secret, _sign_payload

    secret = ""
    if hook.secret_encrypted:
        try:
            secret = _decrypt_secret(hook.secret_encrypted)
        except Exception:
            pass

    payload_bytes = json.dumps(test_payload, default=str).encode()
    req_headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": f"DeepfakeDetector/{settings.app_version}",
        "X-DeepfakeDetector-Event": "webhook.test",
        "X-DeepfakeDetector-Delivery": str(delivery.id),
    }
    if secret:
        req_headers["X-Hub-Signature-256"] = f"sha256={_sign_payload(secret, payload_bytes)}"

    try:
        with httpx.Client(timeout=5.0) as hclient:
            resp_http = hclient.post(hook.url, content=payload_bytes, headers=req_headers)
        if 200 <= resp_http.status_code < 300:
            delivery.status = WebhookStatus.delivered
            delivery.response_status_code = resp_http.status_code
            delivery.response_body = resp_http.text[:512]
            delivery.delivered_at = datetime.now(timezone.utc)
        else:
            delivery.status = WebhookStatus.failed
            delivery.response_status_code = resp_http.status_code
            delivery.error_message = f"HTTP {resp_http.status_code}"
    except Exception as exc:
        delivery.status = WebhookStatus.failed
        delivery.error_message = str(exc)[:256]

    delivery.attempt_count = 1
    db.commit()

    return {
        "delivery_id": delivery.id,
        "status": delivery.status.value,
        "response_status_code": delivery.response_status_code,
    }
