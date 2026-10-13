# Full-weight CPT trainer preparation v2

This packet prepares one complete, already materialized CPT cohort: 30,421 unique TRAIN rows, 30,432 deterministic draws, and 1,902 effective-batch-16 updates. It does not claim that this cohort is the entire eligible corpus. The cohort registry keeps 8,092 known upstream groups queued for separate materialization and review. The 499 package-held-out validation rows remain excluded from training and provide the same fixed 2,048-token causal-loss comparator.

TRAIN context comes only from `cohort.max_sequence_tokens`; the loader, row validator, model, collator, and renderer identity share that bound. Prepared values are 2,048 through 32,768 powers of two. A separate all-token CPT resource fixture must establish the chosen context before root admission. The template currently binds the existing 2,048-token cohort, while retaining the longer-context interface for later materialized cohorts.

`run()` repeats the complete cohort and validation scan before it creates outputs or imports Unsloth. It proves exact row/token/draw denominators, every unique row before replay, and zero package or document overlap with the 499-row holdout. Template and bound preflights apply the same check.

The live path loads Unsloth before torch, trains all 381 parameter tensors through the reviewed composite optimizer, preserves a two-rate scheduler state, uses the exact sequential schedule, and verifies every draw position. Immediately after model load it narrowly restores the observed id-130559 AddedToken flag mutation plus the native BOS/EOS/PAD/EOG contract. It repeats the repair after Trainer construction and at train start, and verifies exact vocabulary, AddedToken metadata, embeddings, serialization, evaluation, and terminal state. Unknown metadata mutations fail closed.

A full Trainer checkpoint contains dense weights, FP32 optimizer state, scheduler, CPU/CUDA RNG, Trainer state, and sampler cursor. Trainer writes it under the runtime path. The callback seals and flushes it, computes one exact SHA-256 inventory, then atomically renames the directory into the durable archive on the same E filesystem. Resume and terminal acceptance independently rehash all bytes. No dense light checkpoint is written. Selected milestones and the latest two full checkpoints survive retention.

The measured E full-checkpoint path took about 423 seconds from initial metadata to sealed manifest and 614 seconds until the next update began for a 17.25 GB payload. This packet does not use C hot staging: the C measurement authorized a separate probe, not production migration or retention. Root still must select the parent, optimizer rates, micro batch, scheduler, checkpoint cadence, evaluations, and milestones in a fresh admission.

Root owns the CUDA lease, host guard, scientific admission, and launch. Commands in `root-commands.json` are argument arrays; the run/resume arrays must be wrapped by the reviewed root supervisor and resource guard.
