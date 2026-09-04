#!/usr/bin/env bash
# Cloud SFT entrypoint for Anyscale jobs (single GPU node, auto-terminating).
# Runbook: docs/research/2026-09-04-anyscale-sft-cloud-runbook.md
#
# Job env (set in the job yaml config.env_vars — secrets NEVER in the repo):
#   HF_TOKEN   – hub access (private dataset + base model pulls)
#   MODEL      – HF model id (default Qwen/Qwen3.5-0.8B-Base)
#   STEPS      – max_steps for train_sft.py (default 60 for the smoke)
#   DATA_DIR   – where pull_data.py writes the mixture (default /root/data/sft_v7)
#   OUT_DIR    – trainer output dir (default /root/run_sft)
#   SFT_TARGETS / UNSLOTH_COMPILE_DISABLE / UNSLOTH_DISABLE_AUTO_PADDING_FREE
#             – banked B4-safe recipe values (set by the job yaml; see runbook)
set -euo pipefail
T0=$(date +%s)
ts() { echo "[CLOUD T+$(( $(date +%s) - T0 ))s] $*"; }

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B-Base}"
STEPS="${STEPS:-60}"
DATA_DIR="${DATA_DIR:-/root/data/sft_v7}"
OUT_DIR="${OUT_DIR:-/root/run_sft}"

ts "node probe: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
ts "python $(python -V 2>&1) nproc=$(nproc) mem: $(free -g | awk '/^Mem:/{print $2}')GiB"

# 1) env: uv-managed py3.10 venv mirroring .venv-sft pins
pip install -q uv
uv venv /root/.venv-sft --python 3.10 --quiet
export VIRTUAL_ENV=/root/.venv-sft
export PATH="$VIRTUAL_ENV/bin:$PATH"
ts "venv created; installing pins (~3GB wheels)"
uv pip install -q -r scripts/cloud/requirements-cloud-sft.txt
ts "env ready: torch $(python -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))')"

# 2) data: private HF dataset -> byte-identical local mixture
python scripts/cloud/pull_data.py "$DATA_DIR"
ts "data ready: $(du -sh "$DATA_DIR" | cut -f1)"

# 3) train: the repo's own trainer, verbatim (smoke = small STEPS)
ts "train start: $MODEL $STEPS steps"
cd experiments/training
python train_sft.py "$MODEL" "$STEPS" "$DATA_DIR" "$OUT_DIR" "" 2>&1 | tee /root/train.log
ts "train done"

# 4) metrics: tok/s over the exact trained subset + throughput summary
python scripts/cloud/report_toks.py
ts "CLOUD RUN COMPLETE"
