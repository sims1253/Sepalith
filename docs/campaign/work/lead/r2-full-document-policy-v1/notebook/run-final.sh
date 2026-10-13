#!/usr/bin/env bash
set -u
PREFIX="$HOME/.local/state/sepalith/campaign-20260915/run06-full-document-policy-v1-final-20260915T0026Z"
mkdir -p "$PREFIX"
tar -xf /tmp/run06-full-document-policy-v1-final-20260915T0026Z.tar -C "$PREFIX"
cd "$PREFIX" || exit 90
date -u +%FT%TZ > start.txt
node --version > runtime.txt
taskset -c 0,2 timeout --signal=TERM 300 node --no-warnings=ExperimentalWarning --experimental-strip-types test_policy.ts policy-fixtures.json policy-results.json > stdout.txt 2> stderr.txt &
pid=$!
printf '%s\n' "$pid" > pid.txt
trap 'kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' TERM INT HUP
wait "$pid"
rc=$?
printf '%s\n' "$rc" > exit.txt
date -u +%FT%TZ > terminal.txt
sha256sum policy-results.json > output.sha256
pgrep -af "$PREFIX" > processes-after.txt || true
exit "$rc"
