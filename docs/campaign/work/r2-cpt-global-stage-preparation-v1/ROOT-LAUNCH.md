This is an unadmitted second CPT stage. Root must select a terminal full checkpoint from source `5149a4065285c681c2230352e5847b861e2ad2e416c8c263131a557e48bd23ca`, review its unchanged validation panel, and debit the existing four-hour aggregate CPT budget before a merge or launch. The current broad stage and its files are unchanged.

The candidate consumes all 18,991 global rows in a seeded permutation, then replays its first row once to fill the final effective batch. `exposure.json` names that row: 126 code tokens and 127 loss tokens. Total: 18,992 draws, 1,187 effective batches, 24,696,954 supervised tokens, zero omitted rows. This is one complete pass plus an explicit one-row replay, not an exact one-pass exposure.

The new policy is `new_lora_on_merged_cpt_parent`. The first attempt has `resume_from: null`, fresh rank-32/alpha-64 LoRA and fresh fused AdamW. Parameters remain 2,048 tokens, microbatch 2, accumulation 8, BF16, seed 3407, peak LR 1e-4 and zero weight decay. The new cosine horizon is 1,187 steps with 36 warmup steps (ceil of 3%). It does not inherit the prior stage's optimizer, scheduler or step counter. The first root decision is step 250. A later continuation may resume only an exact full checkpoint of this new identity, preserving the 1,187-step horizon; the existing exact-identity verifier rejects the prior-stage checkpoint.

The same 661,360-loss-token, 50-package validation panel remains pinned at SHA `efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8`. The existing step-zero evaluator runs on the merged parent before any new update. Compare both this boundary measurement and later loss with the previous checkpoint; BF16 merging can change numerical behavior. CPT loss is diagnostic and cannot establish editing quality. No optional new validation panel was materialized in this packet.

Root must capture/admit the exact 14-file `source/` tree with its source-relative paths and readonly mode 0444. Its Runner-compatible content identity is in `source-manifest.json`. Ten inherited files are byte-identical; two CPT modules are amended and two helpers are added. The installed training environment remains the accepted current environment; this packet does not replace its source/library admission.

Commands below are preparation and root-only launch instructions. This worker ran only the CPU tests, syntax checks and streaming schedule validator. Set each root variable explicitly; no file discovery or checkpoint selection is implied.

```sh
PACKET=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-cpt-global-stage-preparation-v1
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
# SRC = root-captured source directory with source-manifest.json's exact identity.
# PREVIOUS_RECIPE = exact admitted recipe that owns the selected full checkpoint.
# CHECKPOINT = root-selected terminal full checkpoint; do not use a live directory.
# MERGED = fresh native-ext4 model directory under campaign models.
# ROOT_WORK = fresh root-owned preparation/output directory.

# Root CPU merge, only after host-memory admission and previous job release.
# Use the existing host monitor around this bounded foreground command.
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$SRC/experiments/training:$SRC/packages/sepalith/src" \
  /usr/bin/timeout --signal=TERM --kill-after=30s 570s \
  "$PY" -B "$SRC/experiments/training/merge_cpt_cpu.py" \
  --recipe "$PREVIOUS_RECIPE" --checkpoint "$CHECKPOINT" --output "$MERGED"

# Root independently verifies the resulting manifest and model-byte hash.
# MERGED_MANIFEST_SHA = independently reviewed preparation-manifest SHA256.
# SOURCE_ID = the exact admitted candidate source identity.
# TRAIN_OUTPUT / TRAIN_ARCHIVE = fresh absolute campaign directories.
# DEADLINE = root-approved cutoff bounded by remaining aggregate CPT allowance.
# ATTEMPT_SECONDS <= 5400, including evaluation/checkpoint work; first stop is 250.
"$PY" -B "$PACKET/bind_after_merge.py" \
  --merged-manifest "$MERGED/parent-manifest.preparation.json" \
  --merged-manifest-sha256 "$MERGED_MANIFEST_SHA" --source-id "$SOURCE_ID" \
  --output-dir "$TRAIN_OUTPUT" --archive-dir "$TRAIN_ARCHIVE" \
  --deadline "$DEADLINE" --max-seconds "$ATTEMPT_SECONDS" \
  --output-recipe "$ROOT_WORK/recipe.json"

# This is a full data/weights preflight; root alone runs it after memory admission.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$SRC/experiments/training:$SRC/packages/sepalith/src" \
  "$PY" -B "$SRC/experiments/training/campaign_cpt.py" \
  "$ROOT_WORK/recipe.json" --preflight-only --verify-model-files

# Prepare the same accepted Runner/host-guard route with new paths and identity.
"$PY" -B "$PACKET/make_runner_recipe.py" \
  --recipe "$ROOT_WORK/recipe.json" --output "$ROOT_WORK/runner-recipe.json"
# Root reviews/admits and enqueues runner-recipe.json into its selected Runner.
# Root invokes Runner.run_next() through its existing owned supervisor.
```

The generated runner recipe preserves offline/thread settings, the nonblocking global `cuda0.lock`, Windows host monitor with 8,192 MiB admission floor, cleanup, the selected merged-file cache release, and the campaign timeout. It changes only recipe/source/run identities and the guard duration to attempt seconds plus 60. No worker allocation, state/lease edit or enqueue occurred. `launch_authorized: false` is preparation metadata, not an operating-system security boundary; root's existing admission and resource controls remain authoritative.

At the supplied observed rate of about 5,600 supervised tokens/s, scheduled global updates alone need about 4,410 seconds (73.5 minutes). This is an estimate, not a new throughput measurement. Add startup, validation, checkpoint, cleanup and CPU merge time; 5,400 seconds is a candidate training-attempt upper envelope, not newly granted budget. The prior accepted comparable CPU merge finished between its 15:09:34 admission and 15:13:25 review and had a 480-second ceiling; propose at most 600 seconds here, two CPU cores, no concurrent training, and the existing host-memory monitor with an 8 GiB free floor. FP32 merge expansion needs root to confirm real host headroom. No memory peak was measured in this preparation.

The referenced `merge_sft_cpu-v1.py` was a failed arithmetic version (receipt RUN-01-cpu-merge-a-terminal.json). This candidate explicitly incorporates the accepted `merge_sft_cpu.py` SHA e11ca338eaa5e080f81d9b75193f9d73d4cf3c5573311a1830de0259cbe70c3f correction: cast model to FP32, merge, then cast once to BF16. It still checks all 588 adapter tensors and 294 independent matrix merges, preserves original tokenizer JSON/config bytes, and emits a new CPT-named parent manifest with checkpoint sampler lineage. CPU tests establish metadata rejection and the arithmetic/source policy, not an actual model merge. Actual merge, tokenizer reload, source freeze, model hash, same-panel baseline, first-step supervision and budget admission remain root gates.
