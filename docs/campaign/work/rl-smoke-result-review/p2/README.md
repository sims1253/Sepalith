# RL-03 P2 independent review

`verify_rl03_p2.py` is a standard-library-only, CPU review of the P2 smoke.
It checks the immutable supervisor/result-receipt split, the canonical P2
identity, BF16/4096 loader audit, and the admitted source schedule.  For each
optimizer update it requires four generation calls, each containing two
logical groups of four candidates: 4 calls × 8 groups × 4 rows = 32 rows.

Generation has no source ID by design.  The verifier joins ordered generation
and reward rows by `output_ids_sha256`; the reward row supplies the source ID
and the generation row supplies prompt hash and P2 call/group geometry.  It
checks the admitted source-prefix audit, finite rewards, positive finite LoRA
gradients, full checkpoints 1 and 2, and both 14-case DEV evaluations when
those artifacts are present.  The maximal DEV row must retain its 2,619-token
prompt and the 512-token evaluation cap.

Run the current continuous review with:

```text
python3 docs/campaign/work/rl-smoke-result-review/p2/verify_rl03_p2.py \
  --run-root /home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-03-two-update-p2-a/uninterrupted \
  --source-prefix-audit docs/campaign/work/rl-smoke-live-preparation/generated-rl03-step1000/source-prefix-audit.json \
  --development-panel docs/campaign/work/rl-smoke-live-preparation/generated-rl03-step1000/development-panel-small.jsonl \
  --json-out docs/campaign/receipts/RL-03-p2-uninterrupted-independent-review.json
```

The continuous arm is bound to recipe SHA
`e4cc42dd72fec1ea9c41d831c896171a3317472416426f0253e4e3a7d5a91ba3`.  When
split roots are supplied, the first arm is bound to
`1a65dd4c3652eb22c526876eb3206a1f942602dd86ec2ff04c0bbe992c874de7` and the
resumed arm to
`f6c33a3eac6f0fcd86248d61e8d8b1000c9c8e23cc90e5a41800420c63af5879`.

The receipt is `pass` for the uninterrupted P2 result and records split
resume proof as `pending` until root supplies both split roots.  Supply
`--split-first <step-1-root> --split-resume <step-2-root>` after both runs
finish.  The comparison requires all three reviews to pass, then compares the
ordered source/prompt/output/reward stream, exact adapter tensor bytes,
optimizer, scheduler, RNG bytes, and sampler/source fields.  Run-specific
paths, timing, trainer state, manifests, tokenizers, and other checkpoint
metadata are retained in an explicit allowlist audit and do not fail equality.
If an optimizer container hash differs, the verifier lazily loads only the
trusted owned state files on CPU with `weights_only=True` and records tensor
value equality separately from storage-alias/zip-container differences.  It
never loads the model or allocates CUDA memory.  A host-guard stop remains an
incomplete arm even when its step-2 checkpoint and ordered rollout rows are
already verified.
