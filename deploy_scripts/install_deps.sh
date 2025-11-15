#!/bin/bash
set -euo pipefail
cd /opt/ppe/current
# Python
command -v python3.11 >/dev/null 2>&1 || dnf -y install python3.11 python3.11-pip
/usr/bin/python3.11 -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  /usr/bin/python3.11 -m pip install -r requirements.txt
fi

# systemd unit (only if you keep it versioned)
install -m 0644 deploy_scripts/ppe.service /etc/systemd/system/ppe.service || true
systemctl daemon-reload
