#!/usr/bin/env bash
# Render and install the night-runner systemd user units.
#   install_systemd_units.sh [--enable]
# Run it from the checkout the runner should use for good, normally
# /home/m0hawk/Documents/Sepalith on main. Worktrees under ~/.t3 get cleaned.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
units="$HOME/.config/systemd/user"
case "$repo" in
  */.t3/worktrees/*) echo "warning: $repo is a disposable worktree; install from a stable checkout" >&2 ;;
esac
mkdir -p "$units"
for unit in sepalith-night-runner.service sepalith-night-runner.timer; do
  sed "s|@REPO@|$repo|g" "$here/$unit.in" > "$units/$unit"
  echo "wrote $units/$unit"
done
systemctl --user daemon-reload
if [[ "${1:-}" == "--enable" ]]; then
  systemctl --user enable --now sepalith-night-runner.timer
  systemctl --user list-timers sepalith-night-runner.timer --no-pager
else
  echo "not enabled; run: systemctl --user enable --now sepalith-night-runner.timer"
fi
