#!/usr/bin/env bash
# C02 dry run: two compressed windows in a separate state root.
#   dry_run.sh cuda|cpu [extra night_runner run options]
# Phase A (about 5 min): a job that honours the deadline (4 min) must save and
#   exit 75 before the graceful stop (5 min) and the hard stop (6 min).
# Phase B (15 min): a job that ignores everything must be killed at 15 min.
# CUDA mode is GPU work: run it only with the user's OK or inside the window.
set -euo pipefail
device="${1:?usage: dry_run.sh cuda|cpu [runner options]}"; shift
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
state="$HOME/.local/state/sepalith/night-dryrun-$(date +%Y%m%d-%H%M%S)-$device"
python=/usr/bin/python3
[[ "$device" == cuda ]] && python=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
if [[ "$device" == cuda ]]; then windows=("4 5 6" "10 12.5 15"); else windows=("1 1.5 2" "1 1.5 2"); fi
export PYTHONPATH="$repo/packages/sepalith/src"
queue() { /usr/bin/python3 -B -m sepalith.ops.night_queue --root "$state/night-queue" "$@"; }
runner() { /usr/bin/python3 -B -m sepalith.ops.night_runner --state-root "$state" "$@"; }
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo n/a; }
echo "state root: $state"
echo "gpu MiB before: $(gpu)"
queue enqueue --id "honour-$device" --cwd "$here" --resumable --estimated-minutes 1 -- \
  "$python" -u "$here/synthetic_job.py" --mode honour --device "$device"
# shellcheck disable=SC2086
runner run --test-window ${windows[0]} --min-start-minutes 0.5 --poll-seconds 1 "$@" || true
echo "gpu MiB after phase A: $(gpu)"
# Priority 0 so it runs before the deferred phase A job.
queue enqueue --id "ignore-$device" --cwd "$here" --resumable --priority 0 --estimated-minutes 1 -- \
  "$python" -u "$here/synthetic_job.py" --mode ignore --device "$device"
# shellcheck disable=SC2086
runner run --test-window ${windows[1]} --min-start-minutes 0.5 --poll-seconds 1 "$@" || true
sleep 3
echo "gpu MiB after phase B: $(gpu)"
queue list
echo "reports:"; ls "$state/night-reports"
