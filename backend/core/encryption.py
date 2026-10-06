"""Chiffrement symétrique AES-256-GCM pour les données sensibles at-rest.

Format du ciphertext : base64url( nonce[12] || tag[16] || ciphertext )
Nonce : 96 bits aléatoires, unique par opération (NIST SP 800-38D §8.2).
"""
from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config import settings


def _get_key() -> bytes | None:
    raw = settings.encryption_key
    if not raw:
        return None
    key_bytes = base64.urlsafe_b64decode(raw + "==")
    if len(key_bytes) != 32:
        raise ValueError(
            f"encryption_key doit faire 32 octets (AES-256), reçu {len(key_bytes)}. "
            "Générer avec : python3 -c "
            "\"import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())\""
        )
    return key_bytes


def encrypt_secret(plaintext: str) -> str:
    """Chiffre avec AES-256-GCM. Retourne le plaintext si aucune clé configurée."""
    key = _get_key()
    if key is None:
        return plaintext
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.urlsafe_b64encode(nonce + ct).decode()


def decrypt_secret(ciphertext: str) -> str:
    """Déchiffre AES-256-GCM. Retourne le ciphertext si aucune clé configurée."""
    key = _get_key()
    if key is None:
        return ciphertext
    try:
        data = base64.urlsafe_b64decode(ciphertext + "==")
        nonce, ct = data[:12], data[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None).decode()
    except Exception:
        return ciphertext
