"""Dossier judiciaire / professionnel contenant un ou plusieurs médias."""
import enum
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Enum as SAEnum, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class CaseStatus(str, enum.Enum):
    open = "open"
    in_analysis = "in_analysis"
    completed = "completed"
    archived = "archived"


class Jurisdiction(str, enum.Enum):
    federal = "federal"
    ontario = "ontario"
    quebec = "quebec"
    british_columbia = "british_columbia"
    alberta = "alberta"
    other = "other"


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    case_number: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    jurisdiction: Mapped[Jurisdiction] = mapped_column(
        SAEnum(Jurisdiction), default=Jurisdiction.federal, nullable=False
    )
    status: Mapped[CaseStatus] = mapped_column(
        SAEnum(CaseStatus), default=CaseStatus.open, nullable=False
    )

    # Parties impliquées (optionnel, usage légal)
    plaintiff: Mapped[str | None] = mapped_column(String(256), nullable=True)
    defendant: Mapped[str | None] = mapped_column(String(256), nullable=True)
    counsel: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # Propriétaire du dossier
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    # Dates
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Rétention légale (10 ans par défaut)
    retain_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relations
    media_files: Mapped[list["MediaFile"]] = relationship("MediaFile", back_populates="case", cascade="all, delete-orphan")  # noqa: F821
    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="case", cascade="all, delete-orphan")  # noqa: F821
    reports: Mapped[list["Report"]] = relationship("Report", back_populates="case", cascade="all, delete-orphan")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Case id={self.id} number={self.case_number} status={self.status}>"
