# RL-08 independent review: full checkpoint 100

Review date: 2026-09-13 UTC  
Run: `/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-c`  
Checkpoint: `archive/full/checkpoint-100` and `output/checkpoint-100`  
Review result: `pass_mechanical_quality_pending`

The independent verifier is [verify_rl08_step100.py](./verify_rl08_step100.py). It imports no CUDA, model, Transformers, TRL, or Unsloth code. It parses only projected manifest/state JSON, small checkpoint metadata, the three record streams, the source sequence/row metadata, and the complete 75-case DEV100 artifact. It streams all manifest-listed checkpoint files, including the adapter file, and does not read the large live telemetry stream.

## Accepted evidence

- Both checkpoint directories are full `step=100` artifacts. Archive and output manifests have identical SHA256 `2fa678487c5a760a2d04b798fc29787cd300752e7b6bd3445fdb8a9b09399ebd`; projected state and byte SHA256 are identical at `930385c7715c91d3be3e9a570ee8de752d12f04ef1e6db55b10aab05900b9ed8`.
- The manifest contains all 12 expected files. Every manifest-listed file matched its expected byte count and SHA256 in both locations. The adapter model is `100544848` bytes and SHA256 `d3d59f0812a598230c14d194dfdf93638a0ab31236bf41715780aa4739be9c94`. Logical bytes per checkpoint directory are `311925200`.
- The full state records `consumed_rows=25600`, `consumed_prompt_copies=6400`, `current_index=25600`, `source_draw_cursor=800`, `selected_id_index=425`, `shuffle=false`, and the frozen G4/P2 geometry (`per_device_train_batch_size=4`, `gradient_accumulation_steps=8`, `steps_per_generation=8`, `generation_batch_size=32`). The sampler and source identities match the v6 sequence and admitted RL-02 data identities.
- Optimizer (`53243243` bytes, SHA256 `43f1038021e7d1f8be878f5c56b42d968ccb2b9c469e7fa2d4ec59b0d2f26fb8`), scheduler (`1465` bytes, SHA256 `7a193aff509c34ca8b68d5ba51f3a617898f4292b6176f52d376f4fbd437a50f`), RNG (`14645` bytes, SHA256 `697816ed22d8c4fa867dd8ef46d40de6923908ccf8c60353bbc830d16cbe5097`), tokenizer JSON (`9894271` bytes, SHA256 `d5ede0bcd21e0676a58b176937262fb80a06a32b5cb4ed8bfed8b7d11e45b0e1`), and tokenizer config (`94399` bytes, SHA256 `71e726da77f3f0bbfde07d916acae8bd5a316d1857a00979cdc2e9ed3061034c`) all matched archive/output and the manifest.
- The complete record streams contain 2400 generation and 2400 reward records. The accepted window contains 1600 records for global steps 50–99, exactly 32 per step, and 1600 rewards selected by the corresponding generation indices. All generation/reward output-token hashes and token counts join.
- The window contains 400 source IDs, exactly sequence rows 400–799, with four candidates per source and eight source groups per update. Source sequence artifact SHA256 is the schedule hash `2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132`; its declared internal sequence SHA256 is `dc052dc99347eef5e348fe4621b73c3a1fb0137d1aca765a02edaf5b486cfea6`.
- All gradient records for updates 51–100 are finite with 588 trainable tensors and 588 present gradients. Zero-gradient updates are 55, 59, 60, and 86; the corresponding zero reward-variance updates match exactly. No gradient record is missing or non-finite.
- Generation termination in the window is 1528 canonical EOS and 72 length-cap records. Reward records report 72 `missing_canonical_eos` failures and 72 cap hits; these are retained as observed denominators. Family counts are finish_block 252, format_propagation 400, na_rm_propagation 28, no_op 320, pipe_rewrite 160, rename_propagation 400, and roxygen_drafting 40.
- The recipe SHA256 is `fc1de4c4b4e9b1da9617edbc99162d1db1df31080de95eed2bc75c695d5725c3`; source snapshot, identity, schedule, parent, renderer, and tokenizer projections match the pinned full5-c contract. Runtime evidence records preflight pass, BF16, 294 adapter attachments, rank/alpha 16, 25,116,672 trainable parameters, allocator fraction 0.75, requested/observed load capacity 4096, and verified tokenizer contract.
- DEV100 is complete and has 75 exact panel IDs. Independent pinned-parser reclassification passes. Denominators are 75 cases, 43 edits, 32 strict no-ops; counts are 26 exact edits, 25 correct no-ops, 6 strict-no-op false suggestions, 70 protocol-valid, and 5 caps. The maximal prompt case `dat07-existing-719cd49683667d0fb86fb2fa` retains 2619 prompt tokens. Artifact SHA256 is `fa24d3bfbd4c5fa73c7710eacc77157edf5cedadbb476ee7122f223c3c567fe4`.

## Limits

The `resume_validation` field remains the production sentinel text requiring a matched interruption/resume receipt. This review verifies the complete state and cursor at step 100, but it does not claim interruption/resume equivalence. The supplied identity SHA is recorded and all projected fields match; canonical identity serialization is not independently re-derived here. The large terminal/supervision JSON was not parsed, so this receipt does not independently certify the host guard's later terminal status. These checks are mechanical artifact evidence and do not promote model quality or establish R semantic validity.

Validation run:

```text
python3 -m py_compile docs/campaign/work/rl-step100-review/verify_rl08_step100.py
PYTHONHASHSEED=0 python3 docs/campaign/work/rl-step100-review/verify_rl08_step100.py --json-out docs/campaign/receipts/RL-08-step100-independent-review.json
```

The command returned `pass_mechanical_quality_pending` with 50 passing checks and zero failed checks. The JSON receipt contains the complete check list, file hashes, record-stream hashes, state values, and unresolved items.
