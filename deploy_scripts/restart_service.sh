#!/bin/bash
set -euo pipefail
systemctl enable ppe.service
systemctl restart ppe.service
