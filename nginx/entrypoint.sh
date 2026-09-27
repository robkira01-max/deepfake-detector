#!/bin/sh
# Entrypoint Nginx production — crée les symlinks vers les certs Let's Encrypt.
# En dev, le répertoire /etc/nginx/ssl contient déjà des certs self-signed.
# En prod, on symlinke depuis /etc/letsencrypt/live/$CERTBOT_DOMAIN/.

set -e

DOMAIN="${CERTBOT_DOMAIN:-deepfake-detector.ca}"
LE_DIR="/etc/letsencrypt/live/${DOMAIN}"
SSL_DIR="/etc/nginx/ssl"

mkdir -p "${SSL_DIR}"

# Si un cert Let's Encrypt existe, on l'utilise (override self-signed)
if [ -f "${LE_DIR}/fullchain.pem" ] && [ -f "${LE_DIR}/privkey.pem" ]; then
    echo "[entrypoint] Let's Encrypt cert found for ${DOMAIN} — symlinking..."
    ln -sf "${LE_DIR}/fullchain.pem" "${SSL_DIR}/cert.pem"
    ln -sf "${LE_DIR}/privkey.pem"   "${SSL_DIR}/key.pem"
else
    echo "[entrypoint] No Let's Encrypt cert found — using existing SSL dir content."
fi

exec nginx -g "daemon off;"
