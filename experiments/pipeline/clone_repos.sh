#!/usr/bin/env bash
# Clone selected repos shallow-since 2026-05-01 into /mnt/h/sepalith/git/<owner>__<repo>.
# Concurrency <= 6 (resource-polite). Skips existing dirs. Logs failures.
set -u
SEL=${1:-/mnt/h/sepalith/meta/selected_repos.json}
GITDIR=/mnt/h/sepalith/git
LOG=/mnt/h/sepalith/logs/clone_failures.log
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python

mkdir -p "$GITDIR" /mnt/h/sepalith/logs
: > "$LOG"

clone_one() {
  slug="$1"; url="$2"
  name="${slug//\//__}"
  dest="$GITDIR/$name"
  if [ -d "$dest/.git" ]; then
    echo "SKIP $slug (exists)"
    return 0
  fi
  rm -rf "$dest"  # remove failed partial
  if nice -n 10 git clone --quiet --shallow-since=2026-05-01 "$url" "$dest" >>"$LOG" 2>&1; then
    # basic sanity: repo must have at least one commit since the shallow date
    if [ -n "$(git -C "$dest" log --since=2026-05-01 --format=%H -1 2>/dev/null)" ]; then
      echo "OK $slug"
    else
      echo "EMPTY $slug"
      rm -rf "$dest"
    fi
  else
    echo "FAIL $slug"
    rm -rf "$dest"
  fi
}
export -f clone_one
export LOG GITDIR

"$PY" -c "
import json
sel = json.load(open('$SEL'))
for s in sel:
    print(s['slug'] + ' ' + s['url'])
" | xargs -n 2 -P 6 bash -c 'clone_one "$0" "$1"' > /mnt/h/sepalith/logs/clone_results.log 2>&1
echo "clone pass finished: $(grep -c '^OK' /mnt/h/sepalith/logs/clone_results.log 2>/dev/null || echo 0) ok"
