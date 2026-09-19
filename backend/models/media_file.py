"""Fichier média ingéré avec empreintes cryptographiques et métadonnées."""
import enum
from datetime import datetime, timezone
from sqlalchemy import String, Text, BigInteger, DateTime, Enum as SAEnum, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class MediaType(str, enum.Enum):
    video = "video"
    audio = "audio"


class MediaStatus(str, enum.Enum):
    quarantine = "quarantine"
    verified = "verified"
    corrupted = "corrupted"
    rejected = "rejected"


class MediaFile(Base):
    __tablename__ = "media_files"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)

    # Dossier parent
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), nullable=False)
    case: Mapped["Case"] = relationship("Case", back_populates="media_files")  # noqa: F821

    # Fichier
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    media_type: Mapped[MediaType] = mapped_column(SAEnum(MediaType), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[MediaStatus] = mapped_column(
        SAEnum(MediaStatus), default=MediaStatus.quarantine, nullable=False
    )

    # Empreintes cryptographiques (Chain of Custody)
    hash_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    hash_blake3: Mapped[str] = mapped_column(String(64), nullable=False)
    hash_md5: Mapped[str] = mapped_column(String(32), nullable=False)

    # Horodatage TSA RFC 3161
    tsa_token_b64: Mapped[str | None] = mapped_column(Text, nullable=True)
    tsa_authority: Mapped[str | None] = mapped_column(String(256), nullable=True)
    tsa_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Stockage MinIO
    storage_bucket: Mapped[str | None] = mapped_column(String(128), nullable=True)
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Métadonnées extraites (FFmpeg / EXIF)
    media_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Dates
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    ingested_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    # Relations
    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="media_file")  # noqa: F821

    def __repr__(self) -> str:
        return f"<MediaFile id={self.id} uuid={self.uuid} type={self.media_type}>"
