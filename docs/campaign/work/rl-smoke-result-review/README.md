# RL-03 result review

`verify_rl03_results.py` is a standard-library-only verifier for the bounded
RL-03 smoke. It reads the entry and host-supervision receipts, control
telemetry, generation and reward JSONL, LoRA gradient JSONL, full checkpoint
metadata, and the 14-case development results. It does not import a model,
CUDA, Torch, Transformers, TRL, or the project training modules.

Run the current continuous attempt with:

```text
python3 docs/campaign/work/rl-smoke-result-review/verify_rl03_results.py \
  --run-root /home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-03-two-update-a/uninterrupted \
  --source-prefix-audit docs/campaign/work/rl-smoke-live-preparation/generated-rl03-step1000/source-prefix-audit.json \
  --development-panel docs/campaign/work/rl-smoke-live-preparation/generated-rl03-step1000/development-panel-small.jsonl \
  --json-out docs/campaign/receipts/RL-03-uninterrupted-independent-review.json
```

The generation file has no source ID by design. The verifier joins each
ordered generation row to the ordered reward row by
`output_ids_sha256`; the reward row supplies the source ID and the generation
row supplies `prompt_ids_sha256`. It checks 8 groups × 4 candidates × 2
updates, reward source-ID prefix, finite/nonzero LoRA gradients, 1/2 full
checkpoint state, and 14-case DEV results including the 2619-token case.

After root materializes both split attempts, add the two run roots:

```text
--split-first <first-run-root> --split-resume <resume-run-root>
```

The split comparison requires every constituent artifact review to pass. It
then compares the ordered joined generation/reward sequence and compact
checkpoint sampler state at optimizer steps 1 and 2. Missing split artifacts
remain pending and cannot produce resume proof.
