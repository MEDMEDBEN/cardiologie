#!/usr/bin/env bash
# Installation de SYSTOLE sur un VPS Ubuntu. Usage : sudo EMAIL=vous@exemple.com bash deploy/setup.sh
set -euo pipefail

DOMAIN="cardiologue.mohamedbennabi.tech"
EMAIL="${EMAIL:?Définissez EMAIL=vous@exemple.com}"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Paquets système"
apt-get update -y
apt-get install -y nginx certbot python3-certbot-nginx git curl ufw

if ! command -v docker >/dev/null 2>&1; then
  echo "==> Docker"
  curl -fsSL https://get.docker.com | sh
fi

cd "$APP_DIR"
mkdir -p models

echo "==> Construction de l'API"
docker compose up -d --build

echo "==> Entraînement du modèle clinique (plis 1–8)"
docker compose exec -T api python -m training.train_clinique
docker compose restart api

echo "==> Nginx"
if [ "$APP_DIR" != "/opt/systole" ]; then
  sed -i "s#/opt/systole/web#$APP_DIR/web#" deploy/nginx/$DOMAIN.conf
fi
cp deploy/nginx/$DOMAIN.conf /etc/nginx/sites-available/$DOMAIN.conf
ln -sf /etc/nginx/sites-available/$DOMAIN.conf /etc/nginx/sites-enabled/$DOMAIN.conf
nginx -t
systemctl reload nginx

echo "==> Pare-feu"
ufw allow OpenSSH || true
ufw allow 'Nginx Full' || true
ufw --force enable || true

echo "==> HTTPS (Let's Encrypt)"
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect

echo "==> Terminé : https://$DOMAIN  ·  santé : https://$DOMAIN/api/health"
[ -f models/ecg_cnn_v2.pt ] || echo "⚠  models/ecg_cnn_v2.pt absent : le CNN ECG sera indisponible tant qu'il n'est pas copié (puis : docker compose restart api)."
