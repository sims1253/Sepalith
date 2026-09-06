#!/usr/bin/env bash
# Kaggle push driver — stages the repo tarball as a private Kaggle dataset,
# generates the bootstrap kernel (script type) with the arm's env block, and
# pushes it ("Save & Run All" batch session). Kaggle counterpart of the
# Anyscale job-yaml FIRE commands; runbook:
# docs/research/2026-09-06-kaggle-compute-integration.md
#
# The GPU type is set via machine_shape/--accelerator (NvidiaTeslaT4 = the
# T4x2 shape) on a kaggle CLI 2.x binary — API default is P100 (sm60, wrong
# for our cu130 pins; the entry script hard-gates it). One-time setup:
#   uv venv /tmp/k2venv --python 3.12 && uv pip install --python
#   /tmp/k2venv/bin/python kaggle==2.2.4    # + export KAGGLE_API_TOKEN
# (2.x also provides `kaggle quota` — live weekly GPU/TPU state.)
#
# Usage (from repo root, tree COMMITTED — the tarball is `git archive HEAD`):
#   bash scripts/cloud/kaggle_push.sh <slug> <steps> <lr> <mode> [--fire]
#     slug  – kernel slug, e.g. sepalith-sft-lr2e-4  (kernel id m0hawk/<slug>)
#     steps – train_sft.py max_steps
#     lr    – SFT_LR (LoRA learning rate; banked b4 anchor = 2e-4)
#     mode  – cpu-smoke | gpu   (cpu-smoke = SKIP_TRAIN=1, zero GPU quota;
#             gpu REQUIRES --fire: interactive confirmation of GPU burn)
#     --fire – required for mode=gpu (the PARKED-until-FIRE convention)
#
# Env: HF_TOKEN sourced from ~/.zshrc (same convention as the Anyscale
# FIRE commands); KAGGLE auth via ~/.kaggle/kaggle.json or the zshrc token.
#
# Watch:  kaggle kernels status m0hawk/<slug>
#         kaggle kernels output   m0hawk/<slug> -p /tmp/kout/<slug>
#         (web log: kaggle.com/code/m0hawk/<slug>/...)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SLUG="${1:?slug}"; STEPS="${2:?steps}"; LR="${3:?lr}"; MODE="${4:?cpu-smoke|gpu}"; FIRE="${5:-}"
[ "$MODE" = "gpu" ] && [ "$FIRE" != "--fire" ] && { echo "REFUSED: mode=gpu needs --fire (quota burn)"; exit 2; }
[ "$MODE" = "cpu-smoke" ] && [ "$STEPS" != "0" ] && echo "NOTE: cpu-smoke ignores steps (no train)"

eval "$(grep -E '^export HF_TOKEN=' ~/.zshrc)"
[ -n "${HF_TOKEN:-}" ] || { echo "FATAL: HF_TOKEN not found in ~/.zshrc"; exit 2; }
# KAGGLE_REPO_SHA (optional): stage/fire at that commit instead of HEAD —
# e.g. when only the driver changed since the dataset's version (avoids the
# 15-20min version re-ingest), or to fire a banked tree exactly.
SHA="${KAGGLE_REPO_SHA:-$(git rev-parse --short HEAD)}"
PKG=/tmp/kaggle_pkg; REPO_DS="$PKG/repo"; KDIR="$PKG/kernel-$SLUG"
rm -rf "$REPO_DS" "$KDIR"; mkdir -p "$REPO_DS" "$KDIR"

# 1) repo tarball -> private dataset m0hawk/sepalith-repo (12MB). Kaggle
#    AUTO-EXTRACTS the uploaded tar.gz, so the dataset CONTENT is the repo
#    tree (bootstrap copytrees it — no extraction step in the kernel). A
#    REPO_SHA sentinel carries the commit for dedup: skip the upload when
#    the latest version already carries this SHA — a fresh version goes
#    through processing and a kernel pushed meanwhile mounts a STALE/EMPTY
#    /kaggle/input/sepalith-repo (race, cost two iterations).
TARBALL="sepalith-repo.tar.gz"
git archive --format=tar.gz -o "$REPO_DS/$TARBALL" HEAD
touch "$REPO_DS/REPO_SHA_$SHA"
cat > "$REPO_DS/dataset-metadata.json" <<EOF
{
  "title": "sepalith-repo",
  "id": "m0hawk/sepalith-repo",
  "isPrivate": true,
  "licenses": [{"name": "MIT"}]
}
EOF
ds_files() { kaggle datasets files m0hawk/sepalith-repo --page-size 200 -v 2>/dev/null; }
if ds_files | grep "REPO_SHA_$SHA" >/dev/null; then
  echo "repo dataset already at $SHA (no re-upload)"
else
  if kaggle datasets files m0hawk/sepalith-repo >/dev/null 2>&1; then
    kaggle datasets version -p "$REPO_DS" -m "repo $SHA" >/dev/null
  else
    kaggle datasets create -p "$REPO_DS" >/dev/null
  fi
  # `datasets status` can read "ready" from the PREVIOUS version while the
  # new one processes (kernel then attaches a STALE tree — cost one
  # iteration). The real readiness signal: this SHA's sentinel appears in
  # the FILES listing of the latest version. Observed flip latency: 15-20
  # min per version re-ingest of the 423-file tree — poll 45 min.
  ok=""
  for i in $(seq 1 135); do
    if ds_files | grep "REPO_SHA_$SHA" >/dev/null; then ok=1; break; fi
    if [ $((i % 6)) -eq 0 ]; then echo "  ...waiting for version flip ($((i*20/60))min)"; fi
    sleep 20
  done
  [ -n "$ok" ] || { echo "FATAL: REPO_SHA_$SHA not in dataset after 45min"; exit 3; }
fi
echo "repo staged: m0hawk/sepalith-repo @ $SHA (auto-extracted tree + REPO_SHA sentinel)"

# 2) bootstrap kernel: extract tarball -> export env -> exec the entry script.
#    HF_TOKEN lives ONLY here (generated, never committed) — the exact
#    Anyscale job-yaml convention.
SKIP=0; GPU=false
[ "$MODE" = "cpu-smoke" ] && { SKIP=1; GPU=false; }
[ "$MODE" = "gpu" ] && { SKIP=0; GPU=true; }
RUN_NAME="${RUN_NAME:-kaggle-${SLUG#sepalith-}-s${STEPS}}"
# b4 banked target regex (single-quoted: NO escape processing; the heredoc
# ${B4_REGEX} expansion passes it through verbatim). Source of truth:
# /mnt/h/sepalith/runs/b4_qwen35_2b/final_lora/adapter_config.json — CPU-
# audited 96 modules / 21,823,488 trainable against the staged base.
B4_REGEX='(?:.*?(?:language|text).*?(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer).*?(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj))|(?:\bmodel\.layers\.[\d]{1,}\.(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer)\.(?:(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)))'
cat > "$KDIR/kernel.py" <<EOF
import os, shutil, subprocess, sys

# arm env (generated by scripts/cloud/kaggle_push.sh — token redacted in no
# copy of this file; the kernel itself is private to the account)
ENV = {
    "HF_TOKEN": "${HF_TOKEN}",
    "MODEL": "scholzmx/sepalith-base-qwen35-2b-text",
    "STEPS": "${STEPS}",
    "SFT_LR": "${LR}",
    "SFT_DATA_REPO": "scholzmx/sepalith-sft-v7",
    "SKIP_TRAIN": "${SKIP}",
    "PINS": "$([ "$MODE" = "cpu-smoke" ] && echo audit || echo full)",
    "SFT_TARGETS": r"regex:${B4_REGEX}",
    "EXPECT_TRAINABLE": "21823488",
    # T4 16GB geometry: bs2 x ga8 = the B13-sanctioned OOM-safe fallback,
    # identical effective 16 to b4's bs4xga4 (optimizer math unchanged)
    "SFT_PD_BATCH": "2",
    "SFT_GRAD_ACCUM": "8",
    "LORA_REPO": "scholzmx/sepalith-lora",
    "RUN_NAME": "${RUN_NAME}",
    "UNSLOTH_COMPILE_DISABLE": "1",
    "UNSLOTH_DISABLE_AUTO_PADDING_FREE": "1",
    # T4 (Turing) has no bf16 — fp16 channel on GPU arms (train_sft SFT_FP16)
    "SFT_FP16": "$([ "$MODE" = "gpu" ] && echo 1 || echo 0)",
}
os.environ.update(ENV)
# Kaggle auto-extracted the repo tarball into the dataset mount. THREE
# observed layouts: classic /kaggle/input/<slug>; nested <slug>/<slug> when
# the REPO_SHA sentinel sits at the mount root; and GPU-worker images
# mounting under /kaggle/input/datasets/<owner>/<slug>[/<slug>] (diag
# kernel 2026-09-06). Probe them all, then a bounded walk for run.py.
cands = [
    "/kaggle/input/sepalith-repo",
    "/kaggle/input/sepalith-repo/sepalith-repo",
    "/kaggle/input/datasets/m0hawk/sepalith-repo",
    "/kaggle/input/datasets/m0hawk/sepalith-repo/sepalith-repo",
]
src = next((c for c in cands if os.path.isfile(os.path.join(c, "run.py"))), None)
if src is None:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if "run.py" in filenames:
            src = dirpath
            break
if src is None:
    tops = os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else []
    print("FATAL: no repo tree (run.py) under /kaggle/input (tops:", tops[:8],
          ") — dataset not attached or a stale version", flush=True)
    sys.exit(6)
print("bootstrap: repo tree at", src, flush=True)
dst = "/kaggle/working/Sepalith"
shutil.copytree(src, dst)
r = subprocess.run(["bash", "scripts/cloud/kaggle_sft_entry.sh"], cwd=dst)
sys.exit(r.returncode)
EOF

# 3) push as a private script kernel (Save & Run All starts automatically)
#    GPU pushes route through a kaggle CLI 2.x binary when available
#    (KAGGLE2_BIN, default /tmp/k2venv/bin/kaggle): only 2.x carries the
#    --accelerator/machine_shape field. Without it the API default is P100
#    (sm60 — incompatible with the cu130 pins; the entry script gates it).
#    NvidiaTeslaT4 = the "GPU T4 x2" shape. CPU pushes stay on the 1.x CLI.
cat > "$KDIR/kernel-metadata.json" <<EOF
{
  "id": "m0hawk/${SLUG}",
  "title": "${SLUG}",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": ${GPU},
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["m0hawk/sepalith-repo"]$( [ "$GPU" = "true" ] && echo -n ',
  "machine_shape": "NvidiaTeslaT4"' )
}
EOF
KAGGLE2="${KAGGLE2_BIN:-/tmp/k2venv/bin/kaggle}"
if [ "$GPU" = "true" ] && [ -x "$KAGGLE2" ]; then
  "$KAGGLE2" kernels push -p "$KDIR" --accelerator NvidiaTeslaT4
else
  kaggle kernels push -p "$KDIR"
fi
echo
echo "kernel pushed: m0hawk/${SLUG} (mode=${MODE} steps=${STEPS} lr=${LR} run=${RUN_NAME})"
echo "watch: kaggle kernels status m0hawk/${SLUG}"
echo "pull:  kaggle kernels output m0hawk/${SLUG} -p /tmp/kout/${SLUG}"
