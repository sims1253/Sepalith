# Full-weight saved-dtype audit

This packet is a CPU-only inspection of full-weight model load boundaries. It does not load the model payload or alter frozen training/evaluation sources.

The merged CPT250 parent stores all 381 tensors as BF16. The LR3e-5 checkpoint24 stores 85 norm weights as F32 and the other 296 tensors as BF16. A loader forced to BF16 can round those 85 saved F32 tensors before an evaluation or a fresh training stage begins.

Trainer resume is separate. The full-weight runtime first constructs the MiniCPM model through Unsloth, whose training preparation exposes the norm parameters as F32, then `Trainer.train(resume_from_checkpoint=...)` restores the saved checkpoint and optimizer/scheduler/RNG state. Existing interrupted-versus-uninterrupted evidence covers that mechanism. This audit found no evidence of prior gradient or resume-state loss.

Fresh parent and evaluation loads require a header-driven restoration step. After the framework has completed its dtype transformations, load only tensors declared F32 in the pinned safetensors header, require exact name/shape/finiteness and a bounded tensor size, make the corresponding parameter F32, copy the exact saved value, and verify equality. Keep all BF16 tensors untouched. Bind and verify the checkpoint file hash before this helper is called, and record restored names, counts, element counts, changed counts, and per-tensor hashes.

Required integration points:

* `r2-cpt-long-eval-root-v1/evaluate.py`: affected checkpoint evaluation load. Root's fresh v2 places the helper after `FastLanguageModel.for_training` and before evaluation.
* `r2-full-weight-edit-sft-eval-gate-v1/source/experiments/training/run_dev_generation.py`: affected DEV checkpoint load. Restore immediately after `AutoModelForCausalLM.from_pretrained`, before tokenizer/runtime audit and `model.eval()`.
* `r2-full-weight-edit-sft-eval-gate-v1/source/experiments/training/full_weight_edit_sft.py`: affected when its chosen fresh parent contains F32 tensors. Restore after `FastLanguageModel.for_training` and before parameter inventory, optimizer creation, baseline use, or training.
* `r2-full-weight-cpt-lr-pilot-v3/source/experiments/training/full_weight_cpt_trainer.py`: the current fresh parent250 has no F32 tensors, so its load introduces no additional precision loss. Apply the same helper if a later stage selects checkpoint24 or another mixed-dtype checkpoint as a fresh parent. Its current baseline and step evaluation use the live in-process model and do not reload checkpoint weights.

Do not add the helper to the established Trainer resume path without a separate regression proof. It would duplicate a distinct restoration mechanism and could invalidate optimizer/model identity assumptions.
