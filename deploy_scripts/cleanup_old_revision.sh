#!/bin/bash
set -euo pipefail

echo "[cleanup] Removing old app files from /opt/ppe-app"

# If your venv is /opt/ppe-venv (outside ppe-app), it's safe to clear everything under /opt/ppe-app
sudo rm -rf /opt/ppe-app/*
