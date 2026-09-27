# Ordinary checkpoint 330 continuation preparation

This packet permits a root-reviewed transition from the immutable ordinary canary checkpoint 330 to the existing ordinary production training identity. It does not authorize a launch or select checkpoint 330 over checkpoint 322.

The checkpoint remains at `/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-varlen-canary-322-ordinary_reference-v1/runtime/checkpoint-330`. Its manifest SHA-256 is `f7bd7b819584a8abeadc37c677914bd8775f350bd097479d326425292ba8e943`. The transition reads the original checkpoint directly. It neither copies nor rewrites its model, optimizer, scheduler, RNG, trainer state, tokenizer, or campaign state.

The source checkpoint is verified under its actual canary identity `759714fa7e57b4168029ed3e7bb43867bbc79b4b07020a01a2b03f906adc3e3e`. The only accepted identity delta is the seven ordinary-canary execution fields recorded in `ordinary330-transition-admission.template.json`. The destination identity is the unchanged production identity `730c2aba916ae9ab8a97a1584207e7c1d7095e23ce1988cdd968df7fd17844b6`. Packed attention is rejected.

The trainer validates the root transition admission before the ordinary full-checkpoint verifier. The full verifier then reads and hashes every original checkpoint payload against the immutable manifest. Hugging Face Trainer still receives the original checkpoint path through `resume_from_checkpoint`, preserving optimizer, scheduler, RNG, and trainer state. Campaign-state validation requires global step 330, stage cursor 4224, global offset 66, effective batch 16, and the frozen draw schedule. New checkpoints use the production identity and carry a `resume_lineage` record with the original checkpoint manifest, source identity, destination identity, and transition-admission hash.

Root must provide four independent gates in order:

1. Admit the new runtime source through a fresh copy of `runtime-source-migration-admission.template.json`.
2. Build a fresh runtime recipe with `prepare_runtime_recipe.py`. This changes only the runtime source binding and fresh output paths; its production identity must remain unchanged.
3. Bind and admit fresh copies of `ordinary330-transition-admission.template.json`, `continuation-admission-330.template.json`, and `execution-stop-450.template.json` to that runtime recipe SHA.
4. Run the CPU preflight through `native_launch_attestation.py`. Root may construct a guarded CUDA launch only after it passes and after checkpoint selection.

Checkpoint 322 and its prior production continuation remain valid fallback inputs. This packet does not alter them. The next proposed full checkpoint stop is global step 450 because `(450 - 66)` is the next 128-update stage-local cadence after step 330.

`checkpoint-metadata-review.json` records the small metadata inspection. Large checkpoint payloads were not loaded or rehashed during preparation. `native_launch_attestation.py` and `verify_checkpoint` perform the required full payload verification immediately before an admitted run.
