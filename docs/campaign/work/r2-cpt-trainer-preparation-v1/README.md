# SFT-11 raw-R CPT preparation

This directory contains a CPU reviewed candidate for a fresh raw causal-LM
continued-pretraining stage. The entrypoint is
`source/experiments/training/campaign_cpt.py`; its import and `--preflight-only`
paths do not import CUDA, load model weights, or launch a process. The root
supervisor must provide the global CUDA lock, host guard, deadlines, and the
final source/admission pins before `run` is callable.

The training input contract is `{id, split: "train", package_id, input_ids,
source}`. The materialized profile emitted by the frozen raw packer is accepted
as an explicit adapter input and is normalized only after its raw fields,
labels, masks, document spans, and provenance have been checked. The initial
profile has 2,048 total IDs per row: manual BOS `0`, at most one masked
continuation token, owned raw-R tokens, and manual EOS/PAD `1`. Internal EOS
labels are masked; only an EOS at the actual document end is supervised. The
owned spans are contiguous and complete, and package plus document SHA sets are
disjoint between train and the package-held-out TRAIN validation split.

The candidate uses a new LoRA on the pinned Midtrain base, BF16,
`adamw_torch_fused`, rank 32/alpha 64, learning rate `1e-4`, microbatch 2 with
gradient accumulation 8 (effective batch 16), and context 2,048. The raw CPT
validation loss is a diagnostic only. SFT continuation and edit quality require
a later fresh stage. A fresh admitted run records both a step-zero holdout
causal-loss baseline and an actual Dataset/collator supervision witness. A
resume validates the full checkpoint identity before framework/model loading,
records the verified resume step, and does not relabel the resumed adapter as a
step-zero baseline.

The frozen 2K profile inputs are outside this owned directory:

* `profile-shard-v2-2k/cpt_train.jsonl`: 2,526 rows, 1,055 documents, 745
  packages, 3,867,176 supervised loss targets, SHA256
  `d707a61ccfabc058dc1215cd1b2345e17edf77e2797cc5fa6f6f53daebf31da9`.
* `profile-shard-v2-2k/cpt_validation.jsonl`: 499 rows, 294 documents, 50
  packages, 661,360 supervised loss targets, SHA256
  `efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8`.
* `profile-shard-v2-2k/manifest.json`: SHA256
  `f54e1c24eb44dfc18bfc944e575d3569247b671932b3259a6f48cee9ad593a7c`.

Root can replay the CPU contract against the prepared recipe with:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-cpt-trainer-preparation-v1/source/packages/sepalith/src:/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-cpt-trainer-preparation-v1/source/experiments/training \
python3 /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-cpt-trainer-preparation-v1/source/experiments/training/campaign_cpt.py \
  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-smoke-a/recipe.prepared.json \
  --preflight-only
```

The focused CPU suite is under `tests/`; it includes canonical and
materialized mask/holdout rejection cases and the real frozen-shard
Dataset/collator probe. The receipt records the exact source/input hashes,
preflight result, and the fact that this worker performed no CUDA/model
launch.
