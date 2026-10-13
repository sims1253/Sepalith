#!/usr/bin/env bash
set -euo pipefail
test -n "${SEPALITH_CPT_BINDING:-}"
packet_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python -B "$packet_dir/cloud_entry.py" "$SEPALITH_CPT_BINDING"
