#!/bin/bash
# Quant candidate → CPU KLD vs bf16. Emits METRIC lines. See .auto/prompt.md.
set -euo pipefail
cd "$(dirname "$0")/.."

WORK=/mnt/e/sepalith/campaign-20260915/quant-work/autoresearch-20261017
BF16=/mnt/e/sepalith/campaign-20260915/quant-work/arm-a/arm-a-bf16.gguf
IMAT=/mnt/e/sepalith/campaign-20260915/quant-work/arm-a/arm-a-imatrix.gguf
CAL=/mnt/e/sepalith/campaign-20260915/quant-work/calibration-v1
HF=/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536
LLAMA=/home/m0hawk/.local/opt/llama.cpp-b10453/build-cpu/bin
PY=/home/m0hawk/sandbox/Sepalith/.venv/bin/python
BASE_MB_LIMIT=1759769728   # Q4_K_M imatrix baseline bytes

CAND=${1:-.auto/candidate.json}
label=$($PY -c "import json;print(json.load(open('$CAND'))['label'])")
tier=$($PY -c "import json;print(json.load(open('$CAND'))['tier'])")
chunks=$($PY -c "import json;print(json.load(open('$CAND')).get('kld_chunks',64))")
existing=$($PY -c "import json;print(json.load(open('$CAND')).get('existing_gguf','') or '')")

# Be a good citizen on a shared box: if a latency-sensitive prefill bench is
# running or the box is saturated, wait a bounded time before starting.
for _ in $(seq 1 30); do
  load=$(cut -d' ' -f1 /proc/loadavg)
  busy=$(pgrep -fc "serving/prefill/bench.py" || true)
  awk -v l="$load" -v b="$busy" 'BEGIN{exit !(l>15 || b>0)}' || break
  sleep 60
done

mkdir -p "$WORK/ggufs" "$WORK/kld" "$WORK/logs"
export CUDA_VISIBLE_DEVICES="" NO_LORA=1 SEPALITH_LLAMA_CPP=/home/m0hawk/.local/opt/llama.cpp-b10453

# 1. One-time CPU bf16 KLD base + token-stream check.
if [ ! -f "$WORK/kld/bf16-base-cpu.kld" ]; then
  echo "computing CPU bf16 KLD base (one-time)..." >&2
  nice -n 10 "$LLAMA/llama-perplexity" -m "$BF16" -f "$CAL/kld-heldout.txt" \
    --kl-divergence-base "$WORK/kld/bf16-base-cpu.kld" \
    -c 1024 -b 1024 --chunks 64 --no-escape -t 8 -ngl 0 \
    > "$WORK/logs/kld-base-cpu.log" 2>&1
fi
if [ ! -f "$WORK/kld/bf16-base-cpu.check.json" ]; then
  PYTHONPATH=packages/sepalith/src "$PY" - "$WORK/kld/bf16-base-cpu.kld" "$CAL/kld-heldout.ids.json" > "$WORK/kld/bf16-base-cpu.check.json" <<'EOF'
import json, sys
sys.path.insert(0, "packages/sepalith/src")
from sepalith.quant.llama_tools import read_kld_base_tokens, first_mismatch
stored = read_kld_base_tokens(sys.argv[1])
expected = json.loads(open(sys.argv[2]).read())[: stored["n_chunk"] * stored["n_ctx"]]
mm = first_mismatch(stored["tokens"], expected)
print(json.dumps({"identical": mm is None, "first_mismatch": mm, "n_chunk": stored["n_chunk"]}))
EOF
fi
$PY -c "import json;d=json.load(open('$WORK/kld/bf16-base-cpu.check.json'));assert d['identical'],d"

# 2. Quantize the candidate unless it reuses an existing GGUF.
if [ -n "$existing" ]; then
  gguf="$existing"
else
  out_type=$($PY -c "import json;print(json.load(open('$CAND')).get('output_type','q8_0'))")
  emb_type=$($PY -c "import json;print(json.load(open('$CAND')).get('embedding_type','q8_0'))")
  use_imat=$($PY -c "import json;print(json.load(open('$CAND')).get('imatrix',True))")
  gguf="$WORK/ggufs/$label-$tier.gguf"
  if [ ! -f "$gguf" ]; then
    flags=(unused "$HF" "$label" --source-gguf "$BF16" --out-dir "$WORK/ggufs"
           --tiers "$tier" --threads 8 --output-tensor-type "$out_type"
           --token-embedding-type "$emb_type")
    if [ "$use_imat" = "True" ]; then flags+=(--imatrix "$IMAT"); else flags+=(--uncalibrated); fi
    tt=$($PY -c "import json;print('\n'.join(json.load(open('$CAND')).get('tensor_types',[])))")
    if [ -n "$tt" ]; then
      while IFS= read -r pat; do flags+=(--tensor-type "$pat"); done <<< "$tt"
    fi
    nice -n 10 "$PY" experiments/training/export_gguf.py "${flags[@]}" \
      > "$WORK/logs/quantize-$label.log" 2>&1
  fi
fi

# 3. KLD vs the CPU base.
log="$WORK/logs/kld-$label-c$chunks.log"
if [ ! -f "$WORK/kld/$label-c$chunks.json" ]; then
  nice -n 10 "$LLAMA/llama-perplexity" -m "$gguf" \
    --kl-divergence-base "$WORK/kld/bf16-base-cpu.kld" --kl-divergence \
    -c 1024 -b 1024 --chunks "$chunks" --no-escape -t 8 -ngl 0 \
    > "$log" 2>&1
  PYTHONPATH=packages/sepalith/src "$PY" - "$log" "$label" "$chunks" "$gguf" > "$WORK/kld/$label-c$chunks.json" <<'EOF'
import json, sys
sys.path.insert(0, "packages/sepalith/src")
from sepalith.quant.llama_tools import parse_kld_log
stats = parse_kld_log(open(sys.argv[1]).read())
print(json.dumps({"label": sys.argv[2], "chunks": int(sys.argv[3]), "gguf": sys.argv[4],
                  "statistics": stats}, indent=2))
EOF
fi

# 4. Emit metrics.
$PY - "$WORK/kld/$label-c$chunks.json" <<'EOF'
import json, os, sys
r = json.load(open(sys.argv[1]))
s = r["statistics"]
b = os.path.getsize(r["gguf"])
print(f"METRIC kld={s['mean_kld']:.6f}")
print(f"METRIC mb={b/1e6:.1f}")
print(f"METRIC top1={s.get('same_top_p_pct', float('nan')):.3f}")
print(f"METRIC p99={s.get('p99_kld', float('nan')):.6f}")
print(f"METRIC ppl_gap={s.get('mean_ppl_q', 0) - s.get('mean_ppl_base', 0):+.4f}")
print(f"METRIC over_budget_mb={max(0.0, (b - 1759769728)/1e6):.1f}")
print(f"METRIC chunks={r['chunks']}")
EOF
