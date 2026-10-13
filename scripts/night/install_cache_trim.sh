#!/usr/bin/env bash
# Install the page-cache trimmer as a root system service (needs sudo once).
#   sudo scripts/night/install_cache_trim.sh [--uninstall]
# The script is copied to a root-owned path, so later edits in the checkout
# never run as root without a reinstall.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl disable --now sepalith-cache-trim.service || true
  rm -f /etc/systemd/system/sepalith-cache-trim.service /usr/local/sbin/sepalith-cache-trim
  systemctl daemon-reload
  echo "removed sepalith-cache-trim"
  exit 0
fi
install -m 0755 -o root -g root "$here/../../packages/sepalith/src/sepalith/ops/cache_trim.py" /usr/local/sbin/sepalith-cache-trim
install -m 0644 -o root -g root "$here/sepalith-cache-trim.service" /etc/systemd/system/sepalith-cache-trim.service
systemctl daemon-reload
systemctl enable --now sepalith-cache-trim.service
sleep 25
systemctl --no-pager status sepalith-cache-trim.service | head -12
