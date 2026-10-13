# RUN-08 cloud profile review

This review inspected the new profile orchestrator, target runtime loader,
warm-start trainer, profile configuration, and the named watchdog/uploader
references. It used source and receipt metadata only. It did not open model
weights, TRAIN rows, provider APIs, or initialize CUDA.

The stage command map is structurally correct. It runs target-cache generation,
the relocatable wrapper around the pinned native DeepSpec smoke, and
`DeepSpec/train.py --config profile_config.py` in sequence. The trainer output
path resolves to
`<run>/checkpoints/sepalith-r2/dspark_minicpm5_2b_train_only`, which matches
`profile_config.py`. The telemetry path is owned by the run directory, and the
validator rejects missing or nonfinite samples. The checkpoint validator looks
for `step_8` and inventories files below that exact root.

The following issues remain before cloud admission.

## Findings

1. **High — target loader flags are incomplete.**

   `r2-draft-target-runtime-v2/target_runtime.py` passes `dtype` and
   `attn_implementation="sdpa"` to `from_pretrained`, but does not pass
   `trust_remote_code=False` or `use_safetensors=True`. The profile preflight
   protects the canonical five filenames, but the loader contract should still
   fail closed independently. Add both explicit arguments and retain the
   post-load dtype/SDPA checks.

2. **High — public draft config is not preflighted.**

   `warmstart_trainer.py` reads `Path(public_draft_weights).with_name("config.json")`
   and checks `PUBLIC_CONFIG_SHA256 =
   bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b`.
   The profile currently checks only the public weight size and SHA. Before
   launching the trainer, require a non-symlink sibling `config.json`, verify
   this exact hash, and record its path/hash in the durable input binding.

3. **High — no cumulative 7,200-second budget.**

   The stage limits are 1,800 + 900 + 2,400 = 5,100 seconds, leaving 2,100
   seconds for setup, upload, and final refresh. `execute_profile()` applies
   each stage limit independently and has no profile start deadline, so three
   full stages can consume the entire provider window before persistence.
   Add a 7,200-second monotonic profile deadline, clamp every stage to the
   remaining time, and reserve an explicit 2,100-second setup/upload/refresh
   budget. Put cumulative elapsed and remaining seconds in every terminal
   receipt.

4. **Medium — revision is recorded but not enforced.**

   `_safe_manifest_fields()` returns the manifest's `deepspec_revision` but
   does not require it to equal the pinned revision
   `005e03b81cec38b7da6399833d609ee89a2587f2`. Add that equality gate before
   model staging.

5. **Medium — alternative model files are only partially rejected.**

   The profile rejects `.index.json` and `model-*.safetensors`, but leaves
   other loader alternatives such as `pytorch_model.bin` or `flax_model.*` in
   the directory. Enforce an explicit allowed model-file set, or reject all
   additional weight/index names after requiring the five canonical files.

6. **Medium — step parsing accepts an incomplete log.**

   `validate_trainer_stage()` requires only that the maximum parsed value is
   `8` and every denominator is `8`. A log containing only `step=8/8` would
   pass. Require the observed optimizer steps to be exactly the contiguous
   set `1..8` (and reject duplicates or regressions) before accepting the
   checkpoint.

7. **Medium — workstation cache locations are not fully contained.**

   The child environment sets the profile output/cache directories, but leaves
   `HOME`, `TMPDIR`, `HF_HOME`, `XDG_CACHE_HOME`, and `TORCH_HOME` inherited.
   Set these to run-owned subdirectories. The pinned upstream smoke also uses
   an explicit `/tmp` temporary cache roundtrip and deletes it; record that
   exception or wrap it if the provider policy requires every transient file
   under the supplied run directory.

8. **Low — durable upload is intentionally delegated but lacks a final
   handoff receipt.**

   The orchestrator writes `durable/persistence-manifest.json` and excludes
   raw cache shards, but does not invoke the existing uploader or write a
   provider readback receipt. This is compatible with root retaining uploader,
   paid admission, and watchdog control, but the launch handoff needs an
   explicit contract: upload `durable/` only, then attach the provider commit
   and independent readback receipt to the profile terminal record.

## Verified facts

- Stage limits sum to 5,100 seconds; the required total window is 7,200
  seconds, leaving 2,100 seconds for non-stage work.
- The profile fixes target generation to eight rows and trainer configuration
  to eight optimizer steps; there is no larger-run flag in this packet.
- The trainer's `allow_resume=False` and the fresh run-directory check prevent
  implicit checkpoint resume.
- The expected checkpoint root matches `profile_config.py`'s
  `SEPALITH_DRAFT_OUTPUT_ROOT/project_name/exp_name` expansion.
- The profile's canonical target pins are the task-global merged target:
  `model.safetensors=b862986475d8b7f9dd74639e53c2b79b7d85abe30e79af5763efdcc9a1ed6fc4`,
  config `f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991`,
  generation config `7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269`,
  tokenizer `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`,
  tokenizer config `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`.

No cloud or GPU execution was performed by this review.
