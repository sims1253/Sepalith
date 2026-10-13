# Native CPT trainer v2

This fresh packet preserves the v1 scientific trainer and adds two runtime safety properties for a checkpoint-194 continuation.

The full-weight Transformers 5.5 proof uses the production Aurora/Muon/AdamW composite optimizer. It interrupts after update 2, resumes from the atomically published full checkpoint, and matches an uninterrupted update 3 in model, FP32 optimizer, scheduler, CPU RNG at the optimizer boundary, and draw cursor. The checkpoint contains real optimizer, scheduler, RNG, Trainer state, and safetensors bytes.

Trainer keeps two native checkpoints during serialization and cross-filesystem publication. It prunes to one only after the new checkpoint is durable on E. An injected mid-copy failure proves the prior native and durable checkpoint remain and the failed final E name is absent.

An execution-stop admission can stop a resumed run at global step 322 without changing scientific identity. It is bound to the recipe hash and resume step, must be after the resume and at or before the horizon, and must land on the stage-local full-checkpoint cadence. The callback saves and stops at that optimizer boundary.

The staged reusable data receipt is pinned at `a72c1b699fe931e0db0fa2ddfc96a43486e74fc962802e8a541540e16596b88d`. Checkpoint 194, its continuation admission, and the execution-stop admission remain root-bound launch inputs. No launch is authorized by this packet.
