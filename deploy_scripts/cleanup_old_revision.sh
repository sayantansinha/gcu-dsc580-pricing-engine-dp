#!/bin/bash
set -euo pipefail

echo "[cleanup] Removing old app files from /opt/ppe-app"

# Clear everything under /opt/ppe-app
sudo rm -rf /opt/ppe-app/* /opt/ppe-app/.[!.]* /opt/ppe-app/..?*
