#!/usr/bin/env bash
# Run one C17 latency configuration on the notebook: start one llama-server,
# wait for health, run the frozen benchmark, stop the server.
#   run_config.sh <tier> <backend>
set -euo pipefail
tier=$1 backend=$2
W=$HOME/sepalith-c17-latency
case $tier in
  Q6_K) model=$W/models/c19-100-imat-Q6_K.gguf; sha=6593785330b99076b80ae3ac679bda76c5fcff1ebd1873f46f944c1ec43a3261 ;;
  Q4_K_M-mixB-e4) model=$W/models/c19-100-mixb-e4-Q4_K_M.gguf; sha=e24fbfbc5ea47c88647445ec682196fc67af9134d4fa36e826beeda045f1c71b ;;
  *) echo "bad tier" >&2; exit 2 ;;
esac
case $backend in
  cpu) bin=$W/llama.cpp/build-cpu/bin/llama-server; ngl=0 ;;
  vulkan) bin=$W/llama.cpp/build-vulkan/bin/llama-server; ngl=99 ;;
  *) echo "bad backend" >&2; exit 2 ;;
esac
tag=$tier-$backend
log=$W/logs/$tag.server.log
out=$W/results/$tag.json
cmd=("$bin" -m "$model" --host 127.0.0.1 --port 18099 -c 8192 -t 8 -np 1 -ngl "$ngl")
printf -v cmdstr '%q ' "${cmd[@]}"; cmdstr=${cmdstr% }

echo "$(date -Is) verify $model"
test "$(sha256sum "$model" | cut -d' ' -f1)" = "$sha"
if curl -s -o /dev/null http://127.0.0.1:18099/health; then echo "port 18099 busy" >&2; exit 1; fi

{ date -Is; uptime; powerprofilesctl get 2>/dev/null || true; cat /sys/class/power_supply/AC*/online 2>/dev/null || true; } > "$W/logs/$tag.env-before.txt"
"${cmd[@]}" > "$log" 2>&1 &
pid=$!
G=/sys/class/drm/card1/device
( echo -e "time\tgpu_busy_percent\tvram_used\tgtt_used"
  while sleep 2; do echo -e "$(date +%s.%N)\t$(cat $G/gpu_busy_percent)\t$(cat $G/mem_info_vram_used)\t$(cat $G/mem_info_gtt_used)"; done ) > "$W/logs/$tag.gpu.tsv" &
sampler=$!
trap 'kill $pid $sampler 2>/dev/null || true; wait $pid 2>/dev/null || true' EXIT
for _ in $(seq 300); do
  if ! kill -0 $pid 2>/dev/null; then echo "server exited" >&2; tail -40 "$log" >&2; exit 1; fi
  [[ $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18099/health || true) == 200 ]] && break
  sleep 1
done
[[ $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18099/health) == 200 ]]
echo "$(date -Is) server ready pid=$pid; benchmark"
curl -s http://127.0.0.1:18099/props > "$W/logs/$tag.props.json" || true
cd "$W/code"
PYTHONPATH=packages/sepalith/src python3 scripts/packaging/notebook_latency.py \
  --backend "$backend" --tier "$tier" --runtime-commit 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70 \
  --model-sha256 "$sha" --server-command "$cmdstr" --output "$out"
{ date -Is; uptime; } > "$W/logs/$tag.env-after.txt"
echo "$(date -Is) done $out"
