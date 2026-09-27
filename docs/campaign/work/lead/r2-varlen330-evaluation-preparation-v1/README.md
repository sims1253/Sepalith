# Checkpoint-330 paired canary evaluation preparation

This packet prepares the same frozen 2K, 8K, and 16K causal-NLL evaluator for the future ordinary-reference and block-diagonal-varlen checkpoint-330 canary outputs. It does not read model or optimizer payloads until the root canary controller and both host guards report clean completion.

`prepare.py` then verifies every one of the 12 checkpoint files in both the native hot checkpoint and durable E archive. It requires identical manifests and bytes, full optimizer/scheduler/RNG/trainer/sampler state, step 330, cursor 4224, source step 322, source cursor 4096, eight updates, 128 exact draws, and the correct arm-specific scientific identity. Arm swaps, source changes, draw/denominator changes, or incomplete terminals fail closed. Both arms must have identical draw IDs, loss denominators, and common scientific bindings.

The prepared binding stays `prepared_requires_root_admission`. Root writes a fresh admission against `payload-review.json` and both binding-review hashes, then runs `admit.py` to create executable bindings. `commands.json` runs the arm evaluations serially under separate CUDA guards. The frozen evaluator executes the 499-row 2K, 20-row 8K, and 6-row 16K panels separately and preserves their denominators.

No CUDA command, evaluation, promotion, or large payload hash ran during this preparation. `test_prepare.py` uses tiny 12-file checkpoints to test native/durable equality, payload corruption, arm swaps, cursor changes, and draw-count changes. The live preterminal and unadmitted-template controls both exit 1.
