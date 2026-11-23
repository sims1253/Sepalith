#!/usr/bin/env bash
# Unmeasured device probe: the measured server command plus `-lv 4`, so the
# log shows device, buffer and offload lines that verbosity 3 hides.
# Loads the model, waits for /health, records iGPU memory, stops.
#   probe_config.sh <tier> <backend>
set -euo pipefail
tier=$1 backend=$2
W=$HOME/sepalith-c17-latency
case $tier in
  Q6_K) model=$W/models/c19-100-imat-Q6_K.gguf ;;
  Q4_K_M-mixB-e4) model=$W/models/c19-100-mixb-e4-Q4_K_M.gguf ;;
  *) exit 2 ;;
esac
case $backend in
  cpu) bin=$W/llama.cpp/build-cpu/bin/llama-server; ngl=0 ;;
  vulkan) bin=$W/llama.cpp/build-vulkan/bin/llama-server; ngl=99 ;;
  *) exit 2 ;;
esac
tag=$tier-$backend
G=/sys/class/drm/card1/device
mem() { echo "$1 gpu_busy=$(cat $G/gpu_busy_percent) vram_used=$(cat $G/mem_info_vram_used) gtt_used=$(cat $G/mem_info_gtt_used)"; }
if curl -s -o /dev/null http://127.0.0.1:18099/health; then echo "port busy" >&2; exit 1; fi
out=$W/logs/$tag.probe.txt
mem before > "$out"
"$bin" -m "$model" --host 127.0.0.1 --port 18099 -c 8192 -t 8 -np 1 -ngl "$ngl" -lv 4 > "$W/logs/$tag.probe-server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true' EXIT
for _ in $(seq 120); do
  [[ $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18099/health || true) == 200 ]] && break
  kill -0 $pid || exit 1
  sleep 1
done
mem loaded >> "$out"
kill $pid; wait $pid 2>/dev/null || true
sleep 2
mem after >> "$out"
cat "$out"
