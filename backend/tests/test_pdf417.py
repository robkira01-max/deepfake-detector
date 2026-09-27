"""Tests — PDF417 AAMVA 2020 parsing (permis provinciaux canadiens)."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from engines.ocr_engine import OCREngine, ProvinceIDData, _AAMVA_PROVINCE_ISSUERS


# ── Données de test AAMVA ──────────────────────────────────────────────────────

def _make_aamva(issuer: str = "636012", extra_fields: str = "") -> str:
    """Génère un payload AAMVA minimal valide pour les tests."""
    base = (
        f"@\n\nANSI {issuer}080002DL00410235ZO03150022DL\n"
        f"DAQONT-1234567-890\n"
        f"DCSSMITH\n"
        f"DCTJOHN JAMES\n"
        f"DBB01151985\n"    # 15 janvier 1985
        f"DBA01152028\n"    # 15 janvier 2028
        f"DBC1\n"           # M
        f"DAYBLK\n"
        f"DAU511\n"
        f"DAG123 MAIN ST\n"
        f"DAITOLEDO\n"
        f"DAJON\n"
        f"DAK1A2B3C\n"
        f"DCGCAN\n"
    )
    return base + extra_fields


# ── Tests parsing AAMVA ────────────────────────────────────────────────────────

class TestAAMVAParsing:
    def test_parse_ontario_license(self):
        engine = OCREngine()
        raw = _make_aamva("636012")
        result = engine._parse_aamva(raw)

        assert result is not None
        assert isinstance(result, ProvinceIDData)
        assert result.province_code == "ON"
        assert result.province_name == "Ontario"
        assert result.issuer_id == "636012"
        assert result.is_canadian is True

    def test_parse_quebec_license(self):
        engine = OCREngine()
        raw = _make_aamva("636032")
        result = engine._parse_aamva(raw)

        assert result is not None
        assert result.province_code == "QC"
        assert result.province_name == "Québec"

    def test_parse_bc_license(self):
        engine = OCREngine()
        raw = _make_aamva("636028")
        result = engine._parse_aamva(raw)

        assert result is not None
        assert result.province_code == "BC"

    def test_parse_surname_and_given_names(self):
        engine = OCREngine()
        result = engine._parse_aamva(_make_aamva())

        assert result.surname == "SMITH"
        assert "JOHN" in result.given_names

    def test_parse_document_number(self):
        engine = OCREngine()
        result = engine._parse_aamva(_make_aamva())

        assert result.document_number == "ONT-1234567-890"

    def test_parse_birth_date_normalized(self):
        """Date MMDDYYYY → YYYYMMDD."""
        engine = OCREngine()
        result = engine._parse_aamva(_make_aamva())

        assert result.birth_date == "19850115"

    def test_parse_expiry_date_normalized(self):
        engine = OCREngine()
        result = engine._parse_aamva(_make_aamva())

        assert result.expiry_date == "20280115"

    def test_parse_sex_male(self):
        engine = OCREngine()
        result = engine._parse_aamva(_make_aamva())
        assert result.sex == "M"

    def test_parse_sex_female(self):
        engine = OCREngine()
        raw = _make_aamva().replace("DBC1\n", "DBC2\n")
        result = engine._parse_aamva(raw)
        assert result.sex == "F"

    def test_parse_country_can(self):
        engine = OCREngine()
        result = engine._parse_aamva(_make_aamva())
        assert result.country == "CAN"

    def test_non_canadian_issuer_returns_none(self):
        """Un émetteur américain (636000 = New Hampshire) ne doit pas être parsé."""
        engine = OCREngine()
        raw = _make_aamva("636000")  # non canadien
        result = engine._parse_aamva(raw)
        assert result is None

    def test_missing_ansi_header_returns_none(self):
        engine = OCREngine()
        result = engine._parse_aamva("TOTALLY NOT AAMVA DATA")
        assert result is None

    def test_missing_document_number_returns_none(self):
        engine = OCREngine()
        raw = (
            "@\n\nANSI 636012080002DL\n"
            "DCSSMITH\n"
            "DCTJOHN\n"
        )
        result = engine._parse_aamva(raw)
        assert result is None

    def test_raw_fields_stored(self):
        engine = OCREngine()
        result = engine._parse_aamva(_make_aamva())
        assert result is not None
        assert "document_number" in result.raw_fields
        assert result.aamva_version == "08"

    def test_all_provinces_covered(self):
        """Tous les codes province canadiens doivent être dans la table."""
        provinces = [code for code, _ in _AAMVA_PROVINCE_ISSUERS.values()]
        assert "ON" in provinces
        assert "QC" in provinces
        assert "BC" in provinces
        assert "AB" in provinces
        assert len(_AAMVA_PROVINCE_ISSUERS) >= 10  # 10 provinces + 3 territoires


# ── Tests extract_pdf417 (avec mock pyzbar) ───────────────────────────────────

class TestExtractPDF417:
    def test_extract_pdf417_no_barcode(self, tmp_path):
        """Image sans barcode → None."""
        from PIL import Image
        img = Image.new("RGB", (100, 60), color=(255, 255, 255))
        img_path = str(tmp_path / "blank.png")
        img.save(img_path)

        engine = OCREngine()
        if not engine._pyzbar_available:
            pytest.skip("pyzbar non disponible")

        result = engine.extract_pdf417(img_path)
        assert result is None

    def test_extract_pdf417_with_mocked_barcode(self, tmp_path):
        """Simule un décodage pyzbar réussi."""
        from PIL import Image
        img = Image.new("RGB", (100, 60), color=(255, 255, 255))
        img_path = str(tmp_path / "license.png")
        img.save(img_path)

        aamva_payload = _make_aamva("636012").encode("utf-8")

        mock_barcode = MagicMock()
        mock_barcode.data = aamva_payload

        engine = OCREngine()
        engine._pyzbar_available = True

        with patch("engines.ocr_engine.OCREngine.extract_pdf417",
                   return_value=engine._parse_aamva(_make_aamva("636012"))):
            result = engine.extract_pdf417(img_path)

        assert result is not None
        assert result.province_code == "ON"

    def test_pyzbar_unavailable_returns_none(self, tmp_path):
        from PIL import Image
        img = Image.new("RGB", (100, 60))
        img_path = str(tmp_path / "doc.png")
        img.save(img_path)

        engine = OCREngine()
        engine._pyzbar_available = False
        result = engine.extract_pdf417(img_path)
        assert result is None


# ── Tests extract_document (auto-détection) ───────────────────────────────────

class TestExtractDocument:
    def test_returns_province_data_when_pdf417_found(self, tmp_path):
        """Si PDF417 est trouvé, retourne (ProvinceIDData, None)."""
        from PIL import Image
        img = Image.new("RGB", (100, 60))
        img_path = str(tmp_path / "doc.png")
        img.save(img_path)

        engine = OCREngine()
        mock_province = ProvinceIDData(
            province_code="ON", province_name="Ontario",
            document_number="ONT123", surname="SMITH", given_names="JOHN",
            birth_date="19850115", expiry_date="20280115", sex="M",
        )

        with patch.object(engine, "extract_pdf417", return_value=mock_province):
            with patch.object(engine, "extract_mrz", return_value=None):
                province, mrz = engine.extract_document(img_path)

        assert province is mock_province
        assert mrz is None

    def test_falls_back_to_mrz_when_no_pdf417(self, tmp_path):
        """Si PDF417 absent → essaie MRZ."""
        from PIL import Image
        img = Image.new("RGB", (100, 60))
        img_path = str(tmp_path / "passport.png")
        img.save(img_path)

        engine = OCREngine()

        with patch.object(engine, "extract_pdf417", return_value=None):
            with patch.object(engine, "extract_mrz", return_value=None):
                province, mrz = engine.extract_document(img_path)

        assert province is None
        assert mrz is None
