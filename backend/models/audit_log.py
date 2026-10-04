"""Journal d'audit immuable — conçu pour soutenir l'admissibilité en preuve (LPC art. 31.1-31.6 | R. c. Mohan [1994] 2 RCS 9)."""
import enum
import hashlib
import json
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Enum as SAEnum, ForeignKey, JSON, event, text
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
    # Export batch (v3.1)
    BATCH_EXPORT_GENERATED = "BATCH_EXPORT_GENERATED"
    CSV_EXPORT_GENERATED = "CSV_EXPORT_GENERATED"
    # Webhooks (v3.1)
    WEBHOOK_CREATED = "WEBHOOK_CREATED"
    WEBHOOK_UPDATED = "WEBHOOK_UPDATED"
    WEBHOOK_DELETED = "WEBHOOK_DELETED"
    WEBHOOK_FIRED = "WEBHOOK_FIRED"


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

    # Intégrité de l'entrée — chaîne SHA-256
    # entry_hash = SHA-256(previous_entry_hash|timestamp|action|user_id|user_username|resource_type|resource_id|details_json)
    # Calculé automatiquement par le listener before_insert — ne pas définir manuellement.
    previous_entry_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
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


def _compute_chain_hash(
    previous_hash: str | None,
    timestamp: datetime,
    action: "AuditAction",
    user_id: int | None,
    user_username: str | None,
    resource_type: str | None,
    resource_id: str | None,
    details: dict | None,
) -> str:
    """Calcule le hash de chaîne SHA-256 pour une entrée d'audit.

    Le timestamp est normalisé en UTC naive avant isoformat() pour garantir
    la cohérence entre l'insertion (aware) et la lecture depuis SQLite (naive).
    """
    # Normalise en UTC naive : SQLite stocke sans tzinfo, on veut le même résultat
    ts_naive = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
    details_json = json.dumps(details, sort_keys=True, default=str) if details else "null"
    content = "|".join([
        str(previous_hash or ""),
        ts_naive.isoformat(),
        str(action.value if action else ""),
        str(user_id or ""),
        str(user_username or ""),
        str(resource_type or ""),
        str(resource_id or ""),
        details_json,
    ])
    return hashlib.sha256(content.encode()).hexdigest()


@event.listens_for(AuditLog, "before_insert")
def _chain_audit_entry(mapper, connection, target: "AuditLog") -> None:
    """Chaîne chaque entrée à la précédente avant INSERT.

    Garantit que entry_hash = SHA-256(previous_entry_hash + contenu).
    Résistance à la falsification : modifier une entrée casse tous les hash suivants.
    Limite connue : les INSERTs concurrents peuvent lire le même previous_entry_hash.
    Pour la production haute-concurrence, ajouter un verrou consultatif PostgreSQL.
    """
    if target.timestamp is None:
        target.timestamp = datetime.now(timezone.utc)

    row = connection.execute(
        text("SELECT entry_hash FROM audit_logs ORDER BY id DESC LIMIT 1")
    ).first()
    prev_hash: str | None = row[0] if row else None
    target.previous_entry_hash = prev_hash

    target.entry_hash = _compute_chain_hash(
        previous_hash=prev_hash,
        timestamp=target.timestamp,
        action=target.action,
        user_id=target.user_id,
        user_username=target.user_username,
        resource_type=target.resource_type,
        resource_id=target.resource_id,
        details=target.details,
    )


@event.listens_for(AuditLog, "before_update")
def _block_audit_update(mapper, connection, target):
    raise InvalidRequestError("AuditLog entries are immutable — updates are forbidden.")


@event.listens_for(AuditLog, "before_delete")
def _block_audit_delete(mapper, connection, target):
    raise InvalidRequestError("AuditLog entries are immutable — deletes are forbidden.")


def verify_audit_chain(db_session) -> tuple[bool, list[str]]:
    """Vérifie l'intégrité de la chaîne du journal d'audit.

    Retourne (True, []) si la chaîne est intacte.
    Retourne (False, [erreurs]) si une entrée a été falsifiée ou un lien rompu.
    """
    from sqlalchemy.orm import Session

    entries = db_session.query(AuditLog).order_by(AuditLog.id).all()
    errors: list[str] = []
    prev_hash: str | None = None

    for entry in entries:
        if entry.previous_entry_hash != prev_hash:
            errors.append(
                f"Entrée {entry.id} : lien rompu "
                f"(attendu={prev_hash!r}, reçu={entry.previous_entry_hash!r})"
            )

        if entry.timestamp is None:
            errors.append(f"Entrée {entry.id} : timestamp absent")
            prev_hash = entry.entry_hash
            continue

        expected = _compute_chain_hash(
            previous_hash=prev_hash,
            timestamp=entry.timestamp,
            action=entry.action,
            user_id=entry.user_id,
            user_username=entry.user_username,
            resource_type=entry.resource_type,
            resource_id=entry.resource_id,
            details=entry.details,
        )
        if entry.entry_hash != expected:
            errors.append(
                f"Entrée {entry.id} : entry_hash falsifié "
                f"(attendu={expected[:16]}…, reçu={str(entry.entry_hash)[:16]}…)"
            )

        prev_hash = entry.entry_hash

    return len(errors) == 0, errors
