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
# Usage: comms/watch.sh [interval_s] [max_s] [mention_egrep]
#   interval_s    poll cadence          (default 10)
#   max_s         give-up lifetime      (default 21600 = 6h; exits 2)
#   mention_egrep optional noise trim (e.g. 'zcode-pvf-poc|pvf'): board.md
#                 changes only wake when a NEW matching line appears;
#                 gpu.md / comms.md / git HEAD changes always wake.
# Exit codes: 0 = change detected (summary on stdout), 2 = lifetime
# expired with no change, 3 = repo missing.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${1:-10}"
MAX_S="${2:-21600}"
MENTION="${3:-}"

[ -d "$REPO/comms" ] || { echo "no comms/ under $REPO"; exit 3; }

snap_board() {  # board lines that count when MENTION is set
    if [ -n "$MENTION" ]; then
        grep -E "$MENTION" "$REPO/comms/board.md" 2>/dev/null || true
    else
        cat "$REPO/comms/board.md" 2>/dev/null || true
    fi
}

STATE="$(mktemp -d)"
trap 'rm -rf "$STATE"' EXIT

fingerprint() {
    {
        stat -c '%s %Y' "$REPO/comms.md" "$REPO/comms/gpu.md" 2>/dev/null
        git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NO-HEAD
    } | sha1sum | cut -c1-12
}

snap_board > "$STATE/board_base"
BASE="$(fingerprint)"
START=$SECONDS
while :; do
    sleep "$INTERVAL"
    snap_board > "$STATE/board_now"
    NOW="$(fingerprint)"
    if [ "$NOW" != "$BASE" ] || \
       ! cmp -s "$STATE/board_base" "$STATE/board_now"; then
        WHAT="comms files"
        [ "$NOW" != "$BASE" ] && WHAT="gpu.md/comms.md and/or git HEAD"
        [ ! -s "$STATE/board_now" ] || \
            cmp -s "$STATE/board_base" "$STATE/board_now" || \
            WHAT="$WHAT + board mention"
        echo "CHANGE: $WHAT — read the board tail, act per protocol, relaunch watcher"
        exit 0
    fi
    if [ $((SECONDS - START)) -ge "$MAX_S" ]; then
        echo "NO-CHANGE after ${MAX_S}s — relaunch if still on duty"
        exit 2
    fi
done
