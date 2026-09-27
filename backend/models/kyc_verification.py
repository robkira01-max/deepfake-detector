"""KYCVerification — résultat d'une vérification KYC (face match + OCR + deepfake)."""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

from config import settings


class KYCVerdict(str, enum.Enum):
    PASS   = "PASS"
    FAIL   = "FAIL"
    REVIEW = "REVIEW"


class KYCVerification(Base):
    __tablename__ = "kyc_verifications"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Dossier associé (optionnel)
    case_id: Mapped[int | None] = mapped_column(ForeignKey("cases.id"), nullable=True, index=True)

    # Fichiers analysés
    doc_media_file_id: Mapped[int] = mapped_column(ForeignKey("media_files.id"), nullable=False)
    selfie_media_file_id: Mapped[int | None] = mapped_column(ForeignKey("media_files.id"), nullable=True)

    # Face matching
    face_match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_match_verdict: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )  # MATCH | NO_MATCH | NO_FACE | UNAVAILABLE

    # OCR — champs extraits du document
    ocr_document_type: Mapped[str | None] = mapped_column(String(8), nullable=True)
    ocr_country: Mapped[str | None] = mapped_column(String(8), nullable=True)
    ocr_surname: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ocr_given_names: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ocr_document_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ocr_birth_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    ocr_expiry_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    ocr_nationality: Mapped[str | None] = mapped_column(String(8), nullable=True)
    ocr_mrz_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ocr_raw_fields: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Deepfake document
    document_deepfake_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    document_deepfake_verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Verdict global
    kyc_verdict: Mapped[KYCVerdict] = mapped_column(
        SAEnum(KYCVerdict), nullable=False, default=KYCVerdict.REVIEW
    )
    kyc_verdict_reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Méta
    requested_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def compute_verdict(self) -> None:
        """Calcule kyc_verdict + kyc_verdict_reasons à partir des scores."""
        reasons: list[str] = []
        verdict = KYCVerdict.PASS

        # ── FAIL ──────────────────────────────────────────────────────────────
        if self.face_match_verdict == "NO_MATCH":
            verdict = KYCVerdict.FAIL
            reasons.append("Face matching: le visage du document ne correspond pas au selfie")

        df_fail = settings.kyc_document_deepfake_threshold_fail
        if self.document_deepfake_score is not None and self.document_deepfake_score > df_fail:
            verdict = KYCVerdict.FAIL
            reasons.append(
                f"Document potentiellement falsifié (score {self.document_deepfake_score:.2f} > {df_fail})"
            )

        # ── REVIEW (seulement si pas déjà FAIL) ──────────────────────────────
        if verdict != KYCVerdict.FAIL:
            if self.face_match_verdict in ("NO_FACE", "UNAVAILABLE"):
                verdict = KYCVerdict.REVIEW
                reasons.append(f"Face matching indisponible: {self.face_match_verdict}")

            df_review = settings.kyc_document_deepfake_threshold_review
            if (
                self.document_deepfake_score is not None
                and self.document_deepfake_score > df_review
                and verdict != KYCVerdict.FAIL
            ):
                verdict = KYCVerdict.REVIEW
                reasons.append(
                    f"Document suspect — révision manuelle recommandée (score {self.document_deepfake_score:.2f})"
                )

        self.kyc_verdict = verdict
        self.kyc_verdict_reasons = reasons

    def __repr__(self) -> str:
        return f"<KYCVerification id={self.id} verdict={self.kyc_verdict}>"
