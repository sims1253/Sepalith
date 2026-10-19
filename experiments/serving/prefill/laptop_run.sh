#!/bin/bash
# Laptop (m0pad) run driver for the frozen PRM03 session bench.
#
# Portable counterpart of flag_sweep.sh / quant_speed_table.sh: everything is
# env-driven so the same rig runs on the desktop proxy or on the product
# laptop.  One bench invocation = one server-restart pass, which is what makes
# the A/B/A/B interleaving below possible (drift-aware: each arm is measured in
# both halves of the run, and each run starts from a comparable thermal state
# thanks to the fixed pre-burn).
#
# Usage: bash laptop_run.sh <config.json>:<label>[:<passes>] ...
#   e.g. bash laptop_run.sh configs/warmup-full-eager-6t.json:m0pad-eager-6t-p0
set -uo pipefail

REPO="${SEPALITH_REPO:-$HOME/Documents/Sepalith}"
ARTIFACTS="${SEPALITH_ARTIFACTS:-$HOME/sepalith-prefill/artifacts}"
PY="${SEPALITH_PY:-python3}"
PORT="${PORT:-18071}"
BURNS="${SEPALITH_PREBURN:-1}"       # 1 = thermal pre-burn before every run
BURN_N="${SEPALITH_PREBURN_N:-20}"   # prefill reps used as the pre-burn (~2.5 min)
TH="${SEPALITH_TCTL:-/sys/class/hwmon/hwmon5/temp1_input}"  # kestrel: Tctl

mkdir -p "$ARTIFACTS"
cd "$REPO"

state() {  # host state line: load1, Tctl (C), mean cpu MHz, free mem (MB)
  local tctl="$( [ -r "$TH" ] && awk '{printf "%.1f", $1/1000}' "$TH" || echo NA )"
  local mhz="$(awk '/cpu MHz/{s+=$4;n++} END{if(n) printf "%.0f", s/n; else print "NA"}' /proc/cpuinfo)"
  local mem="$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)"
  echo "load1=$(cut -d' ' -f1 /proc/loadavg) tctl_C=$tctl mean_mhz=$mhz mem_avail_mb=$mem"
}

preburn() {  # drive the package to a steady thermal state before each timed run
  [ "$BURNS" = "1" ] || return 0
  echo "-- pre-burn ($BURN_N x pp512 @6T) $(state)"
  nice -n 10 "$SEPALITH_LLAMA_BENCH" \
    -m "${SEPALITH_MODELS_DIR}/arm-a-Q8_0.gguf" -t 6 -fa on -r "$BURN_N" -p 512 -n 0 \
    >/dev/null 2>&1
  echo "-- post-burn $(state)"
}

sampler() {  # 30 s host-state samples across the timed run (thermal drift log)
  local out="$1"
  echo "epoch,load1,tctl_c,mean_mhz,mem_avail_mb" > "$out"
  while :; do
    echo "$(date +%s),$(cut -d' ' -f1 /proc/loadavg),$( [ -r "$TH" ] && awk '{printf "%.1f", $1/1000}' "$TH" || echo NA),$(awk '/cpu MHz/{s+=$4;n++} END{if(n) printf "%.0f", s/n; else print 0}' /proc/cpuinfo),$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)" >> "$out"
    sleep 30
  done
}

for spec in "$@"; do
  cfg="${spec%%:*}"; rest="${spec#*:}"; label="${rest%%:*}"
  passes="${rest##*:}"; [ "$passes" = "$label" ] && passes=1
  echo "=== $(date -Is) run $label (passes=$passes) cfg=$cfg"
  echo "-- pre-run $(state)"
  preburn
  sampler "$ARTIFACTS/${label}.hoststate.csv" & sampler_pid=$!
  start=$(date +%s)
  nice -n 10 "$PY" experiments/serving/prefill/bench.py \
    --config "$cfg" --port "$PORT" --passes "$passes" --label "$label" \
    > "$ARTIFACTS/${label}.stdout.log" 2>&1
  rc=$?
  kill "$sampler_pid" 2>/dev/null
  echo "-- run $label rc=$rc wall_s=$(( $(date +%s) - start )) $(state)"
  cp -f "$ARTIFACTS/server-pass0.log" "$ARTIFACTS/server-${label}.log" 2>/dev/null
  grep -E '^(METRIC|SUMMARY)' "$ARTIFACTS/${label}.stdout.log" | grep -v '^SUMMARY' || true
done
echo "=== $(date -Is) laptop_run.sh complete"