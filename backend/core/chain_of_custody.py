"""Chaîne de possession numérique — LPC art. 31.1-31.6 + CAN/DGSI 120 [À VALIDER].

Responsabilités :
  - Calcul d'empreintes SHA-256, BLAKE3, MD5
  - Horodatage certifié RFC 3161 (TSA)
  - Signature des entrées d'audit (RSA-4096)
  - Vérification d'intégrité à la demande
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import blake3 as _blake3
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

from config import settings
import structlog

log = structlog.get_logger(__name__)

# Taille de lecture par chunk (4 MB) pour les gros fichiers
_CHUNK_SIZE = 4 * 1024 * 1024


class HashBundle:
    """Empreintes cryptographiques d'un fichier."""

    def __init__(self, sha256: str, blake3: str, md5: str, file_size: int) -> None:
        self.sha256 = sha256
        self.blake3 = blake3
        self.md5 = md5
        self.file_size = file_size

    def to_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "blake3": self.blake3,
            "md5": self.md5,
            "file_size_bytes": self.file_size,
        }


def compute_hashes(file_path: Path) -> HashBundle:
    """Calcule SHA-256, BLAKE3 et MD5 en un seul passage sur le fichier."""
    h_sha256 = hashlib.sha256()
    h_blake3 = _blake3.blake3()
    h_md5 = hashlib.md5(usedforsecurity=False)  # nosec B324 — MD5 pour interopérabilité légale uniquement, pas pour sécurité
    total = 0

    with file_path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            h_sha256.update(chunk)
            h_blake3.update(chunk)
            h_md5.update(chunk)
            total += len(chunk)

    bundle = HashBundle(
        sha256=h_sha256.hexdigest(),
        blake3=h_blake3.hexdigest(),
        md5=h_md5.hexdigest(),
        file_size=total,
    )
    log.info("hashes_computed", sha256=bundle.sha256[:16] + "...", size_bytes=total)
    return bundle


def verify_file_integrity(file_path: Path, expected_sha256: str) -> bool:
    """Vérifie l'intégrité d'un fichier contre son hash SHA-256 enregistré."""
    bundle = compute_hashes(file_path)
    match = hmac.compare_digest(bundle.sha256, expected_sha256)
    if not match:
        log.error(
            "integrity_check_failed",
            path=str(file_path),
            expected=expected_sha256[:16],
            got=bundle.sha256[:16],
        )
    return match


class TSAToken:
    """Représente un jeton d'horodatage RFC 3161."""

    def __init__(self, token_bytes: bytes, authority: str, timestamp: datetime) -> None:
        self.token_bytes = token_bytes
        self.token_b64 = base64.b64encode(token_bytes).decode()
        self.authority = authority
        self.timestamp = timestamp


def request_tsa_timestamp(data_hash: bytes, hash_algorithm: str = "sha256") -> TSAToken | None:
    """
    Envoie une requête d'horodatage à la TSA configurée (RFC 3161).

    En production : Entrust Canada (https://timestamp.entrust.net/TSS/RFC3161sha2TS)
    En développement : FreeTSA.org (https://freetsa.org/tsr)
    """
    try:
        # Construction de la requête TSA selon RFC 3161
        # Structure simplifiée : TimeStampReq ::= SEQUENCE { version, messageImprint, nonce, certReq }
        nonce = int.from_bytes(hashlib.sha256(data_hash).digest()[:8], "big")
        tsa_req = _build_tsa_request(data_hash, nonce)

        response = requests.post(
            settings.tsa_url,
            data=tsa_req,
            headers={"Content-Type": "application/timestamp-query"},
            timeout=15,
        )
        response.raise_for_status()

        tsa_token = TSAToken(
            token_bytes=response.content,
            authority=settings.tsa_url,
            timestamp=datetime.now(timezone.utc),
        )
        log.info("tsa_stamp_obtained", authority=settings.tsa_url, nonce=nonce)
        return tsa_token

    except Exception as exc:
        log.warning("tsa_stamp_failed", error=str(exc), tsa_url=settings.tsa_url)
        return None


def _build_tsa_request(message_hash: bytes, nonce: int) -> bytes:
    """
    Construit une requête TimeStampReq DER minimale (RFC 3161 §2.4.1).

    Structure ASN.1 simplifiée encodée manuellement :
    TimeStampReq ::= SEQUENCE {
        version      INTEGER { v1(1) },
        messageImprint MessageImprint,
        nonce        INTEGER OPTIONAL,
        certReq      BOOLEAN DEFAULT FALSE
    }
    MessageImprint ::= SEQUENCE {
        hashAlgorithm AlgorithmIdentifier,
        hashedMessage OCTET STRING
    }
    """
    # OID SHA-256 : 2.16.840.1.101.3.4.2.1
    sha256_oid = bytes([
        0x30, 0x0d,  # SEQUENCE
        0x06, 0x09,  # OID
        0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01,  # SHA-256
        0x05, 0x00,  # NULL
    ])
    hash_octet = bytes([0x04, len(message_hash)]) + message_hash
    msg_imprint = bytes([0x30, len(sha256_oid) + len(hash_octet)]) + sha256_oid + hash_octet

    nonce_bytes = nonce.to_bytes((nonce.bit_length() + 8) // 8, "big")
    nonce_asn1 = bytes([0x02, len(nonce_bytes)]) + nonce_bytes

    version = bytes([0x02, 0x01, 0x01])  # INTEGER 1
    cert_req = bytes([0x01, 0x01, 0xff])  # BOOLEAN TRUE

    inner = version + msg_imprint + nonce_asn1 + cert_req
    return bytes([0x30, len(inner)]) + inner


class AuditSigner:
    """Signe les entrées du journal d'audit avec RSA-4096."""

    def __init__(self) -> None:
        self._private_key = None
        self._load_key()

    def _load_key(self) -> None:
        # Utilise la clé de signature d'audit dédiée (distincte des clés JWT)
        key_path = Path(settings.audit_signing_key_path)
        if not key_path.exists():
            # Fallback sur la clé JWT uniquement si la clé d'audit n'existe pas encore
            key_path = Path(settings.jwt_private_key_path)
        if key_path.exists():
            with key_path.open("rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )

    def sign_entry(self, entry_data: dict) -> tuple[str, str]:
        """
        Signe une entrée d'audit. Retourne (entry_hash, signature_b64).
        """
        serialized = json.dumps(entry_data, sort_keys=True, default=str).encode()
        entry_hash = hashlib.sha256(serialized).hexdigest()

        if self._private_key is None:
            return entry_hash, ""

        signature = self._private_key.sign(
            serialized,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return entry_hash, base64.b64encode(signature).decode()


_signer = AuditSigner()


def sign_audit_entry(entry_data: dict) -> tuple[str, str]:
    """Interface publique pour signer une entrée d'audit."""
    return _signer.sign_entry(entry_data)


def stamp_file(file_path: Path) -> TSAToken | None:
    """Calcule le hash SHA-256 d'un fichier et obtient un tampon TSA."""
    bundle = compute_hashes(file_path)
    hash_bytes = bytes.fromhex(bundle.sha256)
    return request_tsa_timestamp(hash_bytes)
