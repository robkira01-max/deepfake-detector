"""Abonnements webhook pour notifications SIEM / systèmes externes."""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from sqlalchemy import String, Text, Boolean, DateTime, Enum as SAEnum, ForeignKey, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class WebhookEvent(str, enum.Enum):
    analysis_completed = "analysis.completed"
    analysis_failed = "analysis.failed"
    case_closed = "case.closed"
    batch_export_done = "batch_export.done"


class WebhookStatus(str, enum.Enum):
    pending = "pending"
    delivered = "delivered"
    failed = "failed"
    retrying = "retrying"


class Webhook(Base):
    __tablename__ = "webhooks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    # HMAC-SHA256 secret (stocké chiffré via Fernet dans le service)
    secret_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    events: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    deliveries: Mapped[list["WebhookDelivery"]] = relationship(
        "WebhookDelivery", back_populates="webhook", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Webhook id={self.id} name={self.name} active={self.active}>"


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    webhook_id: Mapped[int] = mapped_column(ForeignKey("webhooks.id"), nullable=False, index=True)
    webhook: Mapped["Webhook"] = relationship("Webhook", back_populates="deliveries")

    event: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[WebhookStatus] = mapped_column(
        SAEnum(WebhookStatus), default=WebhookStatus.pending, nullable=False
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    response_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
