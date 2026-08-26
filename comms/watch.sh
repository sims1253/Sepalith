#!/usr/bin/env bash
# watch.sh — session-bound comms watcher (comms.md protocol tooling).
#
# Replaces scheduled/automated polling (retired 2026-08-27 after scheduler
# reliability issues, user directive; see board 2026-08-27T00:06). Pattern
# adopted from zcode-ddot-poc's watcher: fingerprint the comms files and
# git HEAD; exit as soon as ANY of them changes. The calling agent runs
# this as a session-bound background task — the task-completion wake-up
# IS the notification — then reads the news, acts per protocol, and
# starts the next watcher instance.
#
# Usage: comms/watch.sh [interval_s] [max_s]
#   interval_s  poll cadence          (default 10)
#   max_s       give-up lifetime      (default 21600 = 6h; exits 2)
# Exit codes: 0 = change detected (summary on stdout), 2 = lifetime
# expired with no change, 3 = repo missing.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${1:-10}"
MAX_S="${2:-21600}"

[ -d "$REPO/comms" ] || { echo "no comms/ under $REPO"; exit 3; }

fingerprint() {
    {
        for f in comms.md comms/board.md comms/gpu.md; do
            if [ -f "$REPO/$f" ]; then
                printf '%s %s %s\n' "$f" \
                    "$(stat -c %s "$REPO/$f")" "$(stat -c %Y "$REPO/$f")"
            else
                printf '%s MISSING\n' "$f"
            fi
        done
        git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NO-HEAD
    } | sha1sum | cut -c1-12
}

BASE="$(fingerprint)"
START=$SECONDS
while :; do
    sleep "$INTERVAL"
    NOW="$(fingerprint)"
    if [ "$NOW" != "$BASE" ]; then
        WHAT="comms files"
        git -C "$REPO" rev-parse HEAD >/dev/null 2>&1 && \
            WHAT="git HEAD and/or comms files"
        echo "CHANGE $NOW (was $BASE): $WHAT — read the board tail, act per protocol, relaunch watcher"
        exit 0
    fi
    if [ $((SECONDS - START)) -ge "$MAX_S" ]; then
        echo "NO-CHANGE after ${MAX_S}s — relaunch if still on duty"
        exit 2
    fi
done
