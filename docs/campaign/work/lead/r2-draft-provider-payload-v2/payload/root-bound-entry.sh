#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
printf '%s\n' '{"event":"phase_start","phase":"entry-shell"}'
if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' '{"event":"phase_terminal","phase":"entry-shell","status":"failed","reason":"python3 unavailable"}' '{"event":"entry_terminal","status":"failed","training_success":false,"upload_success":false}'
  exit 1
fi
printf '%s\n' '{"event":"phase_terminal","phase":"entry-shell","status":"succeeded"}'
exec python3 -u -B cloud_entry.py payload-binding.json
