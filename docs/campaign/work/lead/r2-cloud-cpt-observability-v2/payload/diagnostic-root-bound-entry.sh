#!/usr/bin/env bash
set -euo pipefail
test -n "${SEPALITH_CPT_DIAGNOSTIC_BINDING:-}"
packet_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python -B "$packet_dir/bootstrap_diagnostic.py" "$SEPALITH_CPT_DIAGNOSTIC_BINDING"
