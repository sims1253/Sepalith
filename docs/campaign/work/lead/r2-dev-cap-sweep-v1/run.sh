set -euo pipefail
W=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-dev-cap-sweep-v1
pid=
cleanup() { if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
env CUDA_VISIBLE_DEVICES=0 GGML_CUDA_GRAPH_OPT=0 /home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server -m /home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf --host 127.0.0.1 --port 18414 -c 4096 -b 256 -ub 256 --parallel 1 -t 6 -tb 6 --threads-http 2 -ngl 99 -lv 4 > "$W/server.log" 2>&1 &
pid=$!
echo "$pid" > "$W/server.pid"
ready=0
for i in {1..60}; do
 if curl -fsS --max-time 2 http://127.0.0.1:18414/health > "$W/health.json" 2>/dev/null; then ready=1; break; fi
 kill -0 "$pid" || exit 2
 sleep 1
done
[[ "$ready" == 1 ]]
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B "$W/run_client.py"
