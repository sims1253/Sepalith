# RL-08 cap80 full100 derivative independent review

Status: **accepted as an explicit resource derivative; live replay remains pending**.

I independently checked the original full100 checkpoint and the new cap80 derivative. No source, recipe, checkpoint, CLI, or state files were changed by this review. The check used one CPU process, no CUDA, SSH, cloud, model launch, or framework training import.

The source checkpoint is:

`/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-c/archive/full/checkpoint-100`

The derivative is:

`/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/RL-primary-full100-cap80/full/checkpoint-100`

Both manifests are full step 100 checkpoints. Canonical identity hashes are:

- original: `48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2`
- derivative: `ebf964b456eff1c2df2b3064277afe346db9a358eb47b5a902a203822b5b4818`

A recursive identity comparison found exactly one change: `policy.cuda_memory_fraction`, `0.75 → 0.80`. Parent, tokenizer, renderer, data, source, schedule, optimizer geometry, and all other policy fields match.

Every original inventory entry except the rewritten `campaign-state.json` is byte and hash identical in the derivative. The derivative adds exactly one bound provenance file, `resource-cap-migration.json` (2,766 bytes, SHA256 `db25d3322bac3a98920c408087f35c112e9d964b46a41aedff2fe6a8841ba609`). The derivative manifest and state hashes are respectively `2fbd0fad2b2fc38dab9bf306fdfb5dac0be88bfd21889462954328e39bed4a8b` and `68ac4013e530cd210470b4184b031c6b4b26ac8926f0ec001337e6888ddf156e`; the source manifest and state remain `2fa678487c5a760a2d04b798fc29787cd300752e7b6bd3445fdb8a9b09399ebd` and `930385c7715c91d3be3e9a570ee8de752d12f04ef1e6db55b10aab05900b9ed8`.

The state JSON excluding its identity is canonical-hash identical. The sampler is canonical-hash identical, with source draw cursor `800`, selected ID index `425`, consumed rows `25,600`, consumed prompt copies `6,400`, and epoch `0`. The state identity matches each corresponding manifest identity.

The frozen `campaign_checkpoint.verify_checkpoint(..., require_full=True)` independently passed for both the original and derivative. This verifies the complete destination inventory and exact derivative identity under the production verifier. It does not prove that a `.80` live allocation will complete training.

The matching e recipe is present at `docs/campaign/work/main-rl-preparation/primary-mb4-full5-e.recipe.json`, SHA256 `31254c2c1b50cb87a68a563cc1e996423a2241c747630c536c58e98b6963aff8`. Its canonical identity is the derivative hash, both cap fields are `.80`, and `resume_from` points to the derivative. Pure recipe validation passed with G4, microbatch 4, accumulation 8, and no torch/Transformers/TRL/Unsloth imports.

The derivative is suitable for Root’s guarded cap80 retry. The remaining gate is actual replay, including the formerly failing update104 boundary, with host and driver supervision. This is a resource and resume-integrity result only; it makes no quality or promotion claim.
