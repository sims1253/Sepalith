#!/usr/bin/env bash
# Entry point for the nightly GPU window (systemd service and Windows tasks).
#   night-runner.sh run|check|hold|window [options]
# Loads the three campaign secrets from ~/.zshrc into this process only; jobs
# receive a secret only when their env allow-list names it.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -r "$HOME/.zshrc" ]]; then
  eval "$(grep -E '^export (HF_TOKEN|ZAI_API_KEY|KAGGLE_API_TOKEN)=' "$HOME/.zshrc" || true)"
fi
export PATH="/usr/lib/wsl/lib:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$repo/packages/sepalith/src"
export PYTHONDONTWRITEBYTECODE=1
export TZ=Europe/Berlin
exec /usr/bin/python3 -B -s -m sepalith.ops.night_runner "$@"
