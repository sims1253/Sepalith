#!/usr/bin/env bash
# Kaggle SFT entrypoint — the Anyscale sft_entry.sh pattern adapted to
# Kaggle notebooks (T4x2 "Save & Run All" batch sessions).
# Runbook: docs/research/2026-09-06-kaggle-compute-integration.md
#
# Launch convention: the pushed kernel (script type) is a ~15-line bootstrap
# that extracts the sepalith-repo tarball (attached private dataset) to
# /kaggle/working/Sepalith, exports the arm's env knobs, then execs this
# script with cwd = the extracted repo root. Env knobs (set by the
# bootstrap's os.environ.setdefault block — HF_TOKEN lives ONLY there,
# never in the repo; same convention as the Anyscale job yaml):
#   HF_TOKEN   – hub access (private dataset + base model pulls)
#   MODEL      – HF model id (default the staged b4 base text variant)
#   STEPS      – max_steps for train_sft.py (default 60)
#   SFT_LR     – LoRA learning rate (default 2e-4 = banked b4 literal;
#                the LR-sweep channel, see train_sft.py)
#   DATA_DIR/OUT_DIR – staging + trainer output (defaults under /tmp and
#                /kaggle/working — the latter is the saved kernel output)
#   SKIP_TRAIN=1 – CPU smoke mode: env + data + audit, no GPU train
#                  (zero GPU quota; proves packaging/egress/guard)
# Everything else (SFT_TARGETS regex, EXPECT_TRAINABLE, LORA_REPO,
# RUN_NAME, UNSLOTH_* knobs, SFT_PD_BATCH/SFT_GRAD_ACCUM) passes through
# to the same code paths as the Anyscale jobs.
set -euo pipefail
T0=$(date +%s)
REPO_ROOT="$(pwd)"
ts() { echo "[KAGGLE T+$(( $(date +%s) - T0 ))s] $*"; }

MODEL="${MODEL:-scholzmx/sepalith-base-qwen35-2b-text}"
STEPS="${STEPS:-60}"
SFT_LR="${SFT_LR:-2e-4}"
DATA_DIR="${DATA_DIR:-/tmp/data/sft_v7}"
OUT_DIR="${OUT_DIR:-/kaggle/working/run_sft}"
TRAIN_LOG="${TRAIN_LOG:-/kaggle/working/train.log}"
export MODEL STEPS DATA_DIR OUT_DIR TRAIN_LOG
export SFT_DATA_REPO="${SFT_DATA_REPO:-scholzmx/sepalith-sft-v7}"
export HF_HOME="${HF_HOME:-/tmp/hf}"   # keep the 4.6GB hub cache off the saved output volume

ts "node probe: $(uname -r) python$(python -V 2>&1 | cut -d' ' -f2) nproc=$(nproc) mem: $(free -g | awk '/^Mem:/{print $2}')GiB user=$(whoami)"
WORK_FREE="$(df -h /kaggle/working 2>/dev/null | tail -1 | awk '{print $4}')"
TMP_FREE="$(df -h /tmp 2>/dev/null | tail -1 | awk '{print $4}')"
ts "disk: working=${WORK_FREE:-?}B free tmp=${TMP_FREE:-?}B free"

# GPU gate: on GPU sessions require a T4 (sm75). P100 is sm60 — our pinned
# torch cu130 stack cannot run on it; the push API cannot select the GPU
# type, so fail fast with the fix (flip accelerator in kernel settings)
# instead of burning quota on a crash loop.
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
  ts "gpu probe: ${GPU_NAME:-none}"
  case "$GPU_NAME" in
    *T4*) : ;;
    "")   ts "FATAL: no GPU on a train session"; exit 5 ;;
    *)    ts "FATAL: $GPU_NAME unsupported (P100=sm60 vs cu130 pins) — set accelerator to GPU T4 x2 in the kernel settings and re-push"; exit 5 ;;
  esac
  # single-GPU trainer: pin to the first T4 (T4x2 sessions expose both;
  # the second idles — quota is per-session, so 2-arms-per-session is the
  # documented densification option, not wired here)
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

# 1) env: the Kaggle image (python 3.12, user root, dedicated ephemeral
# container) has NO working ensurepip — `python3 -m venv` dies (cost one
# smoke iteration). The Kaggle-native pattern is plain SYSTEM pip: the
# container is ours alone and dies with the session. PINS=audit installs
# only the CPU-audit subset (fast smoke path).
if [ "${PINS:-full}" = "audit" ]; then
  ts "installing audit pins (transformers/peft/hub/torchao only)"
  # torchao pin required: the image ships 0.10.0 and peft 0.20's lora
  # dispatcher hard-fails below 0.16 (cost one smoke iteration)
  pip install -q "transformers==5.5.0" "peft==0.20.0" "accelerate==1.14.0" \
    "huggingface_hub==1.27.0" "numpy==2.2.6" "torchao==0.18.0"
else
  ts "installing full cloud pins (~3GB wheels; Kaggle pipe is fast)"
  pip install -q -r scripts/cloud/requirements-cloud-sft.txt
fi
# triton JIT-compiles cuda_utils at first kernel launch and needs Python.h
# (Anyscale gotcha #2); Kaggle images ship headers — keep the cheap gate.
PYINC="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
printf '#include <Python.h>\nint main(void){return 0;}\n' > /tmp/hdrtest.c
gcc -fsyntax-only -I"$PYINC" /tmp/hdrtest.c || ts "WARN: no usable Python.h at $PYINC — triton JIT may fail"
ts "env ready: torch $(python -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo '(not installed — audit pins)')"

# 2) data: private HF dataset -> byte-identical local mixture (gz pull+gunzip
# ~9s on AWS Xet; proves egress + token path on Kaggle)
python scripts/cloud/pull_data.py "$DATA_DIR"
ts "data ready: $(du -sh "$DATA_DIR" | cut -f1)"

# 3) pre-flight: LoRA attachment audit (B3 under-attach rule). EXPECT_TRAINABLE
#    + the regex-form SFT_TARGETS re-audit on CPU exactly like the local gate:
#    96 modules / 21,823,488 trainable for the b4 regex on this base.
if [ -n "${EXPECT_TRAINABLE:-}" ]; then
  ts "target audit: expecting ${EXPECT_TRAINABLE} trainable params"
  EXPECT_TRAINABLE="$EXPECT_TRAINABLE" \
    python scripts/cloud/audit_targets.py "$MODEL" "${SFT_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
    || { ts "FATAL: audit failed — not training"; exit 4; }
  ts "audit pass"
fi

# 3b) train: the repo's own trainer, verbatim (SKIP_TRAIN=1 = CPU smoke stops here).
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  ts "SMOKE COMPLETE (SKIP_TRAIN=1): packaging + egress + data + audit proven, no GPU burn"
  exit 0
fi
# The repo SHIPS unsloth_compiled_cache/ generated on the 5090 (bf16 GPU);
# on no-bf16 hosts (T4) its compiled dtype dispatch mixes bf16/fp16 casts
# (BFloat16 != Half at q_proj — cost two GPU iterations) while
# UNSLOTH_COMPILE_DISABLE=1 does not prevent USING an existing cache.
# Purge it: with compile disabled unsloth falls back to its runtime
# (non-compiled, dtype-consistent) patch path.
rm -rf "$REPO_ROOT/unsloth_compiled_cache" \
       "$REPO_ROOT/experiments/training/unsloth_compiled_cache"
ts "train start: $MODEL $STEPS steps lr=$SFT_LR"
cd "$REPO_ROOT/experiments/training"
python train_sft.py "$MODEL" "$STEPS" "$DATA_DIR" "$OUT_DIR" "" 2>&1 | tee "$TRAIN_LOG"
ts "train done"
cd "$REPO_ROOT"

# 4) artifact return, path A: final_lora lands in /kaggle/working (the saved
#    kernel output — `kaggle kernels output <slug>` downloads it; works even
#    with internet OFF). Path B: push to the private HF hub (Anyscale-proven
#    transfer path; needs egress).
mkdir -p /kaggle/working/outputs
if [ -d "$OUT_DIR/final_lora" ]; then
  cp -r "$OUT_DIR/final_lora" /kaggle/working/outputs/
  ts "final_lora copied to kernel output"
fi
if [ -n "${LORA_REPO:-}" ] && [ -d "$OUT_DIR/final_lora" ]; then
  RUN_NAME="${RUN_NAME:-sepalith-kaggle-run}"
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

# 5) metrics: tok/s over the exact trained subset
python scripts/cloud/report_toks.py || ts "WARN: report_toks failed (non-fatal)"
ts "KAGGLE RUN COMPLETE"
