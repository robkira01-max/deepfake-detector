"""Rapport d'expertise forensique généré — conçu pour soutenir l'admissibilité en preuve (R. c. Mohan [1994] 2 RCS 9)."""
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Liaisons
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), nullable=False, index=True)
    case: Mapped["Case"] = relationship("Case", back_populates="reports")  # noqa: F821
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), nullable=False)

    # Rapport
    report_number: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    pdf_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    storage_bucket: Mapped[str | None] = mapped_column(String(128), nullable=True)
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Intégrité du rapport lui-même
    report_hash_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Signature numérique RSA-4096
    is_signed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    signature_b64: Mapped[str | None] = mapped_column(Text, nullable=True)
    signer_cert_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Horodatage TSA du rapport
    tsa_token_b64: Mapped[str | None] = mapped_column(Text, nullable=True)
    tsa_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Expert signataire
    expert_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    expert_username: Mapped[str] = mapped_column(String(64), nullable=False)

    # Dates
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Report id={self.id} number={self.report_number} signed={self.is_signed}>"
