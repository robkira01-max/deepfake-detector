"""Blocklist Redis pour la révocation de tokens JWT."""
import logging
import redis
from config import settings

logger = logging.getLogger(__name__)

_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis | None:
    global _redis
    if _redis is None:
        try:
            _redis = redis.from_url(settings.redis_url, decode_responses=True)
            # Validate connection
            _redis.ping()
        except Exception as exc:
            logger.warning("Redis unavailable for token blocklist: %s", exc)
            _redis = None
    return _redis


def revoke_token(jti: str, ttl_seconds: int) -> None:
    """Ajoute un token à la blocklist pour ttl_seconds secondes."""
    r = _get_redis()
    if r is None:
        logger.warning("Token revocation skipped — Redis unavailable (jti=%s)", jti)
        return
    try:
        r.setex(f"blocklist:{jti}", ttl_seconds, "1")
    except Exception as exc:
        logger.warning("Failed to revoke token jti=%s: %s", jti, exc)


def is_revoked(jti: str) -> bool:
    """Vérifie si un token est révoqué."""
    if not jti:
        return False
    r = _get_redis()
    if r is None:
        logger.warning("Cannot check token revocation — Redis unavailable (jti=%s)", jti)
        return False
    try:
        return r.exists(f"blocklist:{jti}") > 0
    except Exception as exc:
        logger.warning("Failed to check revocation for jti=%s: %s", jti, exc)
        return False
