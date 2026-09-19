"""Chiffrement symétrique Fernet pour les données sensibles at-rest."""
from cryptography.fernet import Fernet
from config import settings


def encrypt_secret(plaintext: str) -> str:
    """Chiffre une valeur sensible. Retourne le plaintext si Fernet non configuré."""
    f = settings.fernet
    if f is None:
        return plaintext
    return f.encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    """Déchiffre une valeur sensible. Retourne le ciphertext si Fernet non configuré."""
    f = settings.fernet
    if f is None:
        return ciphertext
    try:
        return f.decrypt(ciphertext.encode()).decode()
    except Exception:
        return ciphertext
