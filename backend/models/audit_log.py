"""Journal d'audit immuable — conçu pour soutenir l'admissibilité en preuve (LPC 31.3 | CAN/DGSI 120 [À VALIDER])."""
import enum
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Enum as SAEnum, ForeignKey, JSON, event
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import InvalidRequestError
from database import Base


class AuditAction(str, enum.Enum):
    # Authentification
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    USER_CREATED = "USER_CREATED"
    MFA_ENABLED = "MFA_ENABLED"
    MFA_VERIFIED = "MFA_VERIFIED"
    LOGIN_FAILED = "LOGIN_FAILED"
    # Dossiers
    CASE_CREATED = "CASE_CREATED"
    CASE_UPDATED = "CASE_UPDATED"
    CASE_ARCHIVED = "CASE_ARCHIVED"
    # Médias
    FILE_UPLOADED = "FILE_UPLOADED"
    FILE_INGESTED = "FILE_INGESTED"
    FILE_VERIFIED = "FILE_VERIFIED"
    FILE_REJECTED = "FILE_REJECTED"
    # Analyses
    ANALYSIS_STARTED = "ANALYSIS_STARTED"
    ANALYSIS_COMPLETED = "ANALYSIS_COMPLETED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    # Rapports
    REPORT_GENERATED = "REPORT_GENERATED"
    REPORT_SIGNED = "REPORT_SIGNED"
    REPORT_DOWNLOADED = "REPORT_DOWNLOADED"
    # Chaîne de possession
    HASH_COMPUTED = "HASH_COMPUTED"
    TSA_STAMPED = "TSA_STAMPED"
    INTEGRITY_VERIFIED = "INTEGRITY_VERIFIED"
    INTEGRITY_FAILED = "INTEGRITY_FAILED"
    # Documents & feedback (v2.0)
    DOCUMENT_ANALYSIS_STARTED = "DOCUMENT_ANALYSIS_STARTED"
    DOCUMENT_ANALYSIS_COMPLETED = "DOCUMENT_ANALYSIS_COMPLETED"
    DOCUMENT_ANALYSIS_FAILED = "DOCUMENT_ANALYSIS_FAILED"
    FEEDBACK_SUBMITTED = "FEEDBACK_SUBMITTED"
    # Model Registry (v3.0)
    MODEL_REGISTERED = "MODEL_REGISTERED"
    MODEL_PULLED = "MODEL_PULLED"
    MODEL_ACTIVATED = "MODEL_ACTIVATED"
    # Active Learning (v3.0)
    RETRAIN_TRIGGERED = "RETRAIN_TRIGGERED"
    # KYC (v3.1)
    KYC_VERIFIED = "KYC_VERIFIED"
    # Protocoles d'entraînement (Brief v3 §0)
    PROTOCOL_CREATED  = "PROTOCOL_CREATED"
    PROTOCOL_FROZEN   = "PROTOCOL_FROZEN"
    PROTOCOL_CONSUMED = "PROTOCOL_CONSUMED"
    # Statut de validation des engines (Brief v3 §6)
    ENGINE_STATUS_CHANGED = "ENGINE_STATUS_CHANGED"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Qui
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    user_username: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Quoi
    action: Mapped[AuditAction] = mapped_column(SAEnum(AuditAction), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Détails
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Intégrité de l'entrée
    entry_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signature_b64: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Quand
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action} user={self.user_username}>"


@event.listens_for(AuditLog, "before_update")
def _block_audit_update(mapper, connection, target):
    raise InvalidRequestError("AuditLog entries are immutable — updates are forbidden.")


@event.listens_for(AuditLog, "before_delete")
def _block_audit_delete(mapper, connection, target):
    raise InvalidRequestError("AuditLog entries are immutable — deletes are forbidden.")
