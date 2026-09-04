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
REPO_ROOT="$(pwd)"
ts() { echo "[CLOUD T+$(( $(date +%s) - T0 ))s] $*"; }

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B-Base}"
STEPS="${STEPS:-60}"
DATA_DIR="${DATA_DIR:-/tmp/data/sft_v7}"
OUT_DIR="${OUT_DIR:-/tmp/run_sft}"
TRAIN_LOG="${TRAIN_LOG:-/tmp/train.log}"

ts "node probe: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
ts "python $(python -V 2>&1) nproc=$(nproc) mem: $(free -g | awk '/^Mem:/{print $2}')GiB user=$(whoami) HOME=$HOME"

# 1) env: uv-managed py3.10 venv mirroring .venv-sft pins (job user is NOT
# root — everything lives under $HOME or /tmp)
pip install -q uv

setup_venv() {  # $1 = python selector for uv venv
  rm -rf "$HOME/.venv-sft"
  uv venv "$HOME/.venv-sft" --python "$1" --quiet
  export VIRTUAL_ENV="$HOME/.venv-sft"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  ts "venv created ($1); installing pins (~3GB wheels)"
  uv pip install -q -r scripts/cloud/requirements-cloud-sft.txt
}

# triton JIT-compiles cuda_utils (gcc, needs Python.h) at first kernel
# launch, but the image ships NO python3.10 dev headers, sudo is restricted,
# and `uv venv --python 3.10` silently binds the header-less SYSTEM 3.10.
# Fix: force a uv-managed standalone CPython (python-build-standalone
# bundles full headers), venv from it explicitly, and export the venv's
# sysconfig include path via C_INCLUDE_PATH (gcc reads it from the env).
uv python install 3.10
M310="$(ls -d "$HOME"/.local/share/uv/python/cpython-3.10*/bin/python3 2>/dev/null | head -1)"
[ -z "$M310" ] && M310=3.10
setup_venv "$M310"
PYINC="$("$HOME/.venv-sft/bin/python" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [ ! -f "$PYINC/Python.h" ]; then
  PYINC="$(dirname "$(dirname "$(readlink -f "$HOME/.venv-sft/bin/python")")")/include/python3.10"
fi
if [ -f "$PYINC/Python.h" ]; then
  export C_INCLUDE_PATH="$PYINC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
  ts "headers: $PYINC/Python.h"
else
  # last resort: the image's own default python keeps headers with it
  ts "managed 3.10 headers missing — trying image default python"
  setup_venv "$(command -v python)"
  PYINC="$("$HOME/.venv-sft/bin/python" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
  export C_INCLUDE_PATH="$PYINC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
fi
printf '#include <Python.h>\nint main(void){return 0;}\n' > /tmp/hdrtest.c
gcc -fsyntax-only /tmp/hdrtest.c || { ts "FATAL: no usable Python.h — triton cannot JIT"; exit 3; }
ts "env ready: torch $(python -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))')"

# 2) data: private HF dataset -> byte-identical local mixture
python scripts/cloud/pull_data.py "$DATA_DIR"
ts "data ready: $(du -sh "$DATA_DIR" | cut -f1)"

# 3) train: the repo's own trainer, verbatim (smoke = small STEPS)
ts "train start: $MODEL $STEPS steps"
cd "$REPO_ROOT/experiments/training"
python train_sft.py "$MODEL" "$STEPS" "$DATA_DIR" "$OUT_DIR" "" 2>&1 | tee "$TRAIN_LOG"
ts "train done"
cd "$REPO_ROOT"

# 4) push the adapter back to the HF hub (the transfer path for real
#    rungs — nodes die with their disks). Opt-in via LORA_REPO/RUN_NAME.
if [ -n "${LORA_REPO:-}" ] && [ -d "$OUT_DIR/final_lora" ]; then
  RUN_NAME="${RUN_NAME:-sepalith-run}"
  LORA_REPO="$LORA_REPO" RUN_NAME="$RUN_NAME" python - "$OUT_DIR/final_lora" <<'PY'
import os
import sys
from huggingface_hub import HfApi

repo, run = os.environ["LORA_REPO"], os.environ["RUN_NAME"]
folder = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=folder, path_in_repo=f"{run}/final_lora",
                  repo_id=repo, repo_type="model",
                  commit_message=f"final_lora: {run}")
print(f"LORA PUSHED -> https://huggingface.co/{repo}/tree/main/{run}/final_lora")
PY
  ts "final_lora pushed: $LORA_REPO/$RUN_NAME"
fi

# 5) metrics: tok/s over the exact trained subset + throughput summary
python scripts/cloud/report_toks.py
ts "CLOUD RUN COMPLETE"
