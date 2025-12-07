#!/bin/bash
set -euo pipefail

sudo systemctl daemon-reload
sudo systemctl restart ppe.service
sudo systemctl status ppe.service --no-pager -l || true
