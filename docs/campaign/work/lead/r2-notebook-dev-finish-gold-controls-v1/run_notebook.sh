#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
packet="$root/packet"
results="$root/results"
child=""
cleanup() {
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  child=""
}
trap cleanup EXIT INT TERM HUP
[[ ! -e "$results" ]] || { echo "results path must be fresh" >&2; exit 21; }
mkdir "$results"
printf '%s\n' "$$" > "$results/runner.pid"
(cd "$packet" && sha256sum -c payload.sha256) > "$results/payload-rehash.txt"
[[ $(sha256sum "$root/parse_only.R" | cut -d' ' -f1) == 4c16d0feef3555efae5a578627fee4b3859b912d4f78be6e2fa93328a1b89c1e ]]
Rscript --version > "$results/r-version.txt" 2>&1
start_utc=$(date -u +%FT%TZ); start_ns=$(date +%s%N)
set +e
setsid timeout --signal=TERM --kill-after=5s 300 taskset -c 0,2 \
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  Rscript --vanilla "$root/parse_only.R" "$packet/parse-controls.tsv" "$results/results-controls.tsv" \
  > "$results/stdout-controls.txt" 2> "$results/stderr-controls.txt" &
child=$!; printf '%s\n' "$child" > "$results/active.pid"; wait "$child"; code=$?
set -e; child=""; end_ns=$(date +%s%N); end_utc=$(date -u +%FT%TZ)
printf '{"batch":"finish-controls","start_utc":"%s","end_utc":"%s","elapsed_ns":%s,"exit_code":%s,"cpu_affinity":"0,2","maximum_threads":2,"timeout_seconds":300}\n' "$start_utc" "$end_utc" "$((end_ns-start_ns))" "$code" > "$results/timing-controls.json"
[[ "$code" -eq 0 ]]
owned_after=$({ pgrep -af "Rscript.*$root/parse_only.R" || true; } | wc -l)
printf '{"payload_rehash":"pass","owned_parse_processes":%s,"checked_utc":"%s"}\n' "$owned_after" "$(date -u +%FT%TZ)" > "$results/cleanup.json"
[[ "$owned_after" -eq 0 ]]
(cd "$results" && sha256sum results-controls.tsv timing-controls.json stdout-controls.txt stderr-controls.txt r-version.txt cleanup.json) > "$results/result-hashes.sha256"
