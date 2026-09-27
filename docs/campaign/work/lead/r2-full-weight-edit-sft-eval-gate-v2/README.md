# Full-weight editing SFT with mandatory DEV gates and saved-dtype restoration

This v2 packet preserves the v1 target-only TRAIN cohort, full-coverage schedule choices, full checkpoint/optimizer/RNG/sampler resume semantics, mandatory optimizer-boundary DEV stops, corrected DEV75 panel, tokenizer contract, generation termination accounting, and configurable output cap.

It adds one narrow correction for mixed-dtype full-weight parents and checkpoints. The frozen `saved_precision.py` helper inspects the pinned safetensors header, restores only saved F32 parameters after framework dtype transformations, and verifies exact values. Training calls it after `FastLanguageModel.for_training` and before parameter inventory or optimizer construction. DEV generation calls it immediately after `AutoModelForCausalLM.from_pretrained` and before runtime audit or generation. All saved BF16 parameters remain untouched. Both run results include the precision audit.

The template remains parent-null and launch-refusing. Root must select and bind the parent, learning rate, batching, schedule, milestones, output cap, and admission. V1 checkpoints do not silently resume under the v2 source identity.
