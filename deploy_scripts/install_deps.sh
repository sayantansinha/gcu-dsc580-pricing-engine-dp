#!/bin/bash
set -euo pipefail

echo "[install_deps] Dependencies baked into AMI; skipping heavy install."
# optionally run: no needed as its baked into the AMI
# source /opt/ppe-venv/bin/activate
# export TMPDIR=/opt/ppe-tmp
# pip install -r /opt/ppe-app/requirements.txt --no-deps
exit 0
