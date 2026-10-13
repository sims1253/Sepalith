# Full-weight CPT trainer preparation v1

This packet prepares one complete, already materialized CPT cohort: 30,421 unique TRAIN rows, 30,432 deterministic draws, and 1,902 effective-batch-16 updates. It does not claim that this cohort is the entire eligible corpus. The cohort registry keeps 8,092 known upstream groups queued for separate materialization and review. The 499 package-held-out validation rows remain excluded from training and provide causal-loss diagnostics only.

The template has no admitted parent, optimizer rates, micro batch, scheduler, or checkpoint/evaluation cadence. Root must record those choices in a fresh admission using `root-admission.template.json`; the binder rejects an admission for any other template. The proposed parent is the saved global CPT-250 merged model.

The live path loads Unsloth before torch, trains all 381 parameter tensors through the reviewed composite optimizer, preserves a two-rate scheduler state, uses the exact sequential schedule, and verifies every draw position. It applies only the reviewed Transformers 5.5 EOS-alignment repair and checks the full tokenizer vocabulary, BOS/EOS/PAD, native EOG set, and embedding identity through training, serialization, evaluation, and terminal state.

A full Trainer checkpoint contains dense weights, FP32 optimizer state, scheduler, CPU/CUDA RNG, Trainer state, and sampler cursor. Trainer writes it under the hidden runtime path. The callback seals and flushes it, computes one exact SHA-256 inventory, then atomically renames the directory into the durable archive on the same filesystem. Resume and terminal acceptance independently rehash all bytes. No dense light checkpoint is written. Selected milestones and the latest two full checkpoints survive retention.

Root owns the CUDA lease, host guard, scientific admission, and launch. Commands in `root-commands.json` are argument arrays; the run/resume arrays must be wrapped by the reviewed root supervisor and resource guard.
