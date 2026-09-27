"""OCREngine — extraction MRZ + PDF417 AAMVA + champs textuels pour documents KYC Canada."""
from __future__ import annotations

import re
import shutil
import structlog
from dataclasses import dataclass, field

log = structlog.get_logger(__name__)

# ── Patterns regex champs libres ──────────────────────────────────────────────
_RE_DATE      = re.compile(r'\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4}|\d{4}[/\-\.]\d{2}[/\-\.]\d{2})\b')
_RE_DOC_NUM   = re.compile(r'\b([A-Z]{2}\d{6,9}|[A-Z]{1,3}\d{5,8})\b')
_RE_COUNTRY   = re.compile(r'\b(CAN|USA|FRA|GBR|DEU|BEL|CHE|AUS|NZL|IRL)\b')

# ── AAMVA 2020 — codes émetteurs provinciaux canadiens ───────────────────────
_AAMVA_PROVINCE_ISSUERS: dict[str, tuple[str, str]] = {
    "636012": ("ON", "Ontario"),
    "636032": ("QC", "Québec"),
    "636028": ("BC", "Colombie-Britannique"),
    "636023": ("AB", "Alberta"),
    "636044": ("SK", "Saskatchewan"),
    "636024": ("MB", "Manitoba"),
    "636013": ("NS", "Nouvelle-Écosse"),
    "636031": ("NB", "Nouveau-Brunswick"),
    "636042": ("PE", "Île-du-Prince-Édouard"),
    "636033": ("NL", "Terre-Neuve-et-Labrador"),
    "636048": ("YT", "Yukon"),
    "636063": ("NT", "Territoires du Nord-Ouest"),
    "636085": ("NU", "Nunavut"),
}

# Champs AAMVA DL/ID standard (code 3 chars → clé sémantique)
_AAMVA_FIELDS: dict[str, str] = {
    "DAQ": "document_number",
    "DCS": "surname",
    "DCT": "given_names",
    "DAD": "middle_name",
    "DBB": "birth_date_raw",   # MMDDYYYY
    "DBA": "expiry_date_raw",  # MMDDYYYY
    "DBC": "sex_code",         # 1=M 2=F 9=X
    "DAY": "eye_color",
    "DAU": "height",
    "DAG": "address_street",
    "DAH": "address_street2",
    "DAI": "city",
    "DAJ": "province_code",
    "DAK": "postal_code",
    "DCG": "country",
    "DCB": "restrictions",
    "DCD": "endorsements",
    "DCF": "document_discriminator",
    "DDE": "surname_truncated",
    "DDF": "first_name_truncated",
    "DDG": "middle_name_truncated",
}


@dataclass
class ProvinceIDData:
    """Données extraites d'un permis de conduire provincial canadien (PDF417 AAMVA 2020)."""
    province_code: str          # ON, QC, BC…
    province_name: str
    document_number: str
    surname: str
    given_names: str
    birth_date: str             # YYYYMMDD normalisé
    expiry_date: str            # YYYYMMDD normalisé
    sex: str                    # M, F, X
    country: str = "CAN"
    address_street: str = ""
    city: str = ""
    postal_code: str = ""
    eye_color: str = ""
    height: str = ""
    restrictions: str = ""
    endorsements: str = ""
    raw_fields: dict = field(default_factory=dict)
    aamva_version: str = ""
    issuer_id: str = ""
    is_canadian: bool = True
    error: str | None = None


@dataclass
class MRZData:
    document_type: str
    country: str
    surname: str
    given_names: str
    document_number: str
    nationality: str
    birth_date: str      # YYMMDD
    sex: str             # M, F, <
    expiry_date: str     # YYMMDD
    personal_number: str
    mrz_check_valid: bool
    raw_mrz: str
    error: str | None = None


class OCREngine:
    """Moteur OCR pour documents d'identité KYC Canada.

    Stratégie de détection automatique :
      1. PDF417 barcode (pyzbar) → permis provinciaux canadiens (AAMVA 2020)
      2. MRZ OCR (tesseract) → passeports TD3 + cartes TD1 (NEXUS, RP)
      3. OCR général → extraction regex de champs libres

    Dégradation gracieuse si pyzbar ou tesseract ne sont pas disponibles.
    """

    def __init__(self) -> None:
        self._tesseract_available = shutil.which("tesseract") is not None
        if not self._tesseract_available:
            log.warning("tesseract_unavailable", msg="OCR MRZ disabled — install tesseract-ocr")

        try:
            from pyzbar.pyzbar import decode as _pyzbar_decode  # noqa: F401
            self._pyzbar_available = True
        except ImportError:
            self._pyzbar_available = False
            log.warning("pyzbar_unavailable", msg="PDF417 disabled — pip install pyzbar")

    def is_available(self) -> bool:
        return self._tesseract_available or self._pyzbar_available

    def extract_document(self, image_path: str) -> tuple[ProvinceIDData | None, MRZData | None]:
        """Détection automatique du type de document.

        Essaie PDF417 en premier (permis provinciaux), puis MRZ (passeports).
        Retourne (ProvinceIDData, None) ou (None, MRZData) selon ce qui est trouvé.
        """
        province_data = self.extract_pdf417(image_path)
        if province_data is not None:
            return province_data, None

        mrz_data = self.extract_mrz(image_path)
        return None, mrz_data

    def extract_pdf417(self, image_path: str) -> ProvinceIDData | None:
        """Décode le code-barres PDF417 AAMVA 2020 d'un permis provincial canadien.

        Retourne None si aucun code-barres trouvé ou si le document n'est pas AAMVA canadien.
        """
        if not self._pyzbar_available:
            return None

        try:
            from PIL import Image
            from pyzbar.pyzbar import decode, ZBarSymbol

            img = Image.open(image_path).convert("RGB")
            barcodes = decode(img, symbols=[ZBarSymbol.PDF417])

            if not barcodes:
                # Essai sans filtre symbole (certains scanners encodent en QR ou DataMatrix)
                barcodes = decode(img)

            for bc in barcodes:
                try:
                    raw = bc.data.decode("utf-8", errors="replace")
                except Exception:
                    continue

                result = self._parse_aamva(raw)
                if result is not None:
                    return result

            return None

        except Exception as exc:
            log.warning("pdf417_extraction_failed", error=str(exc))
            return None

    def _parse_aamva(self, raw: str) -> ProvinceIDData | None:
        """Parse le contenu brut AAMVA 2020 d'un PDF417.

        Format attendu : commence par '@' + LF + LF + 'ANSI ' ou '@\r\n\r\nANSI '
        """
        # Normalisation des séparateurs de ligne
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")

        # Validation en-tête AAMVA
        if "ANSI " not in raw:
            return None

        ansi_pos = raw.index("ANSI ")
        header = raw[ansi_pos + 5:]

        if len(header) < 10:
            return None

        issuer_id  = header[:6]
        aamva_ver  = header[6:8]

        # Vérifier que c'est un émetteur canadien
        if issuer_id not in _AAMVA_PROVINCE_ISSUERS:
            log.debug("aamva_non_canadian_issuer", issuer_id=issuer_id)
            return None

        province_code, province_name = _AAMVA_PROVINCE_ISSUERS[issuer_id]

        # Extraction des champs — chaque champ est "XXXvaleur\n"
        fields: dict[str, str] = {}
        for line in raw.splitlines():
            line = line.strip()
            if len(line) >= 4:
                code = line[:3]
                if code in _AAMVA_FIELDS:
                    fields[_AAMVA_FIELDS[code]] = line[3:].strip()

        if "document_number" not in fields:
            return None

        # Normalisation des dates MMDDYYYY → YYYYMMDD
        def _normalize_date(raw_date: str) -> str:
            raw_date = re.sub(r'\D', '', raw_date)
            if len(raw_date) == 8:
                # MMDDYYYY → YYYYMMDD
                return raw_date[4:8] + raw_date[0:2] + raw_date[2:4]
            return raw_date

        sex_map = {"1": "M", "2": "F", "9": "X"}
        sex_raw = fields.get("sex_code", "9").strip()
        sex = sex_map.get(sex_raw, "X")

        given_raw = fields.get("given_names", "")
        middle    = fields.get("middle_name", "")
        if middle and middle not in given_raw:
            given_names = f"{given_raw} {middle}".strip()
        else:
            given_names = given_raw

        return ProvinceIDData(
            province_code=province_code,
            province_name=province_name,
            document_number=fields.get("document_number", ""),
            surname=fields.get("surname", ""),
            given_names=given_names,
            birth_date=_normalize_date(fields.get("birth_date_raw", "")),
            expiry_date=_normalize_date(fields.get("expiry_date_raw", "")),
            sex=sex,
            country=fields.get("country", "CAN"),
            address_street=fields.get("address_street", ""),
            city=fields.get("city", ""),
            postal_code=fields.get("postal_code", "").strip(),
            eye_color=fields.get("eye_color", ""),
            height=fields.get("height", ""),
            restrictions=fields.get("restrictions", ""),
            endorsements=fields.get("endorsements", ""),
            raw_fields=fields,
            aamva_version=aamva_ver,
            issuer_id=issuer_id,
            is_canadian=True,
        )

    def extract_mrz(self, image_path: str) -> MRZData | None:
        """Extrait et parse la MRZ depuis le bas de l'image (25% inférieur)."""
        if not self._tesseract_available:
            return None

        try:
            from PIL import Image
            import pytesseract

            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            # Crop 25% bas — zone MRZ des passeports TD3 et cartes TD1
            crop = img.crop((0, int(h * 0.75), w, h))

            raw = pytesseract.image_to_string(
                crop,
                config=(
                    "--psm 6 --oem 1 "
                    "-c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
                ),
            ).strip()

            return self._parse_mrz(raw)

        except Exception as exc:  # noqa: BLE001
            log.warning("mrz_extraction_failed", error=str(exc))
            return None

    def _parse_mrz(self, raw: str) -> MRZData | None:
        """Parse le texte MRZ brut via la librairie mrz."""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        if len(lines) < 2:
            return None

        # TD3 : 2 lignes × 44 chars (passeport)
        try:
            from mrz.checker.td3 import TD3CodeChecker
            # Pad with < (not space — mrz library rejects spaces in MRZ charset)
            line1 = (lines[0] + "<" * 44)[:44]
            line2 = (lines[1] + "<" * 44)[:44]
            td3_raw = f"{line1}\n{line2}"
            chk = TD3CodeChecker(td3_raw)
            f = chk.fields()
            return MRZData(
                document_type=f.document_type or "P",
                country=f.country or "",
                surname=f.surname or "",
                given_names=f.name or "",
                document_number=f.document_number or "",
                nationality=f.nationality or "",
                birth_date=f.birth_date or "",
                sex=f.sex or "<",
                expiry_date=f.expiry_date or "",
                personal_number=f.optional_data or "",
                mrz_check_valid=bool(chk.result),
                raw_mrz=td3_raw,
            )
        except Exception:  # noqa: BLE001
            pass

        # TD1 : 3 lignes × 30 chars (carte ID)
        if len(lines) >= 3:
            try:
                from mrz.checker.td1 import TD1CodeChecker
                line1 = (lines[0] + "<" * 30)[:30]
                line2 = (lines[1] + "<" * 30)[:30]
                line3 = (lines[2] + "<" * 30)[:30]
                td1_raw = f"{line1}\n{line2}\n{line3}"
                chk = TD1CodeChecker(td1_raw)
                f = chk.fields()
                return MRZData(
                    document_type=f.document_type or "ID",
                    country=f.country or "",
                    surname=f.surname or "",
                    given_names=f.name or "",
                    document_number=f.document_number or "",
                    nationality=f.nationality or "",
                    birth_date=f.birth_date or "",
                    sex=f.sex or "<",
                    expiry_date=f.expiry_date or "",
                    personal_number=f.optional_data or "",
                    mrz_check_valid=bool(chk.result),
                    raw_mrz=td1_raw,
                )
            except Exception:  # noqa: BLE001
                pass

        return None

    def extract_fields(self, image_path: str) -> dict[str, str]:
        """OCR général sur tout le document + extraction regex des champs clés."""
        if not self._tesseract_available:
            return {}

        try:
            from PIL import Image
            import pytesseract

            img = Image.open(image_path).convert("RGB")
            text = pytesseract.image_to_string(img, config="--psm 3 --oem 1")

            fields: dict[str, str] = {}

            # Numéro de document
            doc_match = _RE_DOC_NUM.search(text.upper())
            if doc_match:
                fields["document_number"] = doc_match.group(1)

            # Pays / nationalité
            country_match = _RE_COUNTRY.search(text.upper())
            if country_match:
                fields["country"] = country_match.group(1)

            # Dates (DDN, expiration)
            dates = _RE_DATE.findall(text)
            if dates:
                fields["date_1"] = dates[0]
            if len(dates) > 1:
                fields["date_2"] = dates[1]

            # Nom (heuristique : ligne longue en majuscules)
            for line in text.splitlines():
                stripped = line.strip()
                if len(stripped) > 10 and stripped.isupper() and stripped.replace(" ", "").isalpha():
                    fields.setdefault("name_line", stripped)
                    break

            return fields

        except Exception as exc:  # noqa: BLE001
            log.warning("ocr_fields_failed", error=str(exc))
            return {}
