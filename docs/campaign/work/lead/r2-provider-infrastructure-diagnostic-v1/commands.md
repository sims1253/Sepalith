# DAT-10 diagnostic commands

These commands document the bounded probes that produced the packet. They are reproducible case commands; no full shard or campaign rerun is included.

```bash
BASE=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-provider-infrastructure-diagnostic-v1/diagnostic
SEM=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic9535-provider-preparation-v1
NOOP=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-noop4100-provider-preparation-v1
SEM10948=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic10948-provider-materialization-v2
TOKENIZER=/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python

# Stream only to each approved row; the extractor stops immediately after it writes the row.
python3 "$BASE/extract_target_row.py" "$SEM/render-inputs-v1/shard-0027.jsonl" b15446c8d06df1c77e9caac6 "$BASE/inputs/semantic-0027.jsonl"
python3 "$BASE/extract_target_row.py" "$SEM/render-inputs-v1/shard-0028.jsonl" b777001bc9e4d6462cb2a8c1 "$BASE/inputs/semantic-0028.jsonl"
python3 "$BASE/extract_target_row.py" /mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-reconstruction-v1/inputs-full41/shard-0036.jsonl e677ee6a8da38436f4bdb6b6 "$BASE/inputs/noop-0036.jsonl"

# Semantic9535 full-render probes, one case per process.
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 120s taskset -c 8 nice -n 10 ionice -c 3 \
  node --no-warnings=ExperimentalWarning --experimental-strip-types "$BASE/semantic/render_shard.ts" \
  "$BASE/inputs/semantic-0027.jsonl" "$BASE/outputs/semantic-0027.jsonl" \
  "$PYTHON" "$BASE/semantic/tokenize_bridge.py" "$TOKENIZER" "$BASE/semantic/source/namespace_evidence.R" 16384 2048 \
  > "$BASE/logs/semantic-0027.log" 2>&1

env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 120s taskset -c 10 nice -n 10 ionice -c 3 \
  node --no-warnings=ExperimentalWarning --experimental-strip-types "$BASE/semantic/render_shard.ts" \
  "$BASE/inputs/semantic-0028.jsonl" "$BASE/outputs/semantic-0028.jsonl" \
  "$PYTHON" "$BASE/semantic/tokenize_bridge.py" "$TOKENIZER" "$BASE/semantic/source/namespace_evidence.R" 16384 2048 \
  > "$BASE/logs/semantic-0028.log" 2>&1

# Noop4100 full-render probe. The copied Noop runtime uses its local source-import helper;
# its namespace helper and tokenizer bridge are the same frozen Semantic10948 dependencies
# used by r2-noop4100-provider-preparation-v1/run_shard.sh.
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 120s taskset -c 8 nice -n 10 ionice -c 3 \
  node --no-warnings=ExperimentalWarning --experimental-strip-types "$BASE/noop/render_shard.ts" \
  "$BASE/inputs/noop-0036.jsonl" "$BASE/outputs/noop-0036.jsonl" \
  "$PYTHON" "$BASE/noop/tokenize_bridge.py" "$TOKENIZER" "$BASE/noop/source/namespace_evidence.R" 16384 2048 \
  > "$BASE/logs/noop-0036.log" 2>&1
```

