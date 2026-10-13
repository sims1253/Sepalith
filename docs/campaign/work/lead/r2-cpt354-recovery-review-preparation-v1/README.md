# Checkpoint 354 recovery review

This packet remains dormant until root supplies the successful controller, guard, and model-child terminal handles. `prepare.py` then verifies updates 331 through 354, exactly 384 draws at positions 4224 through 4607, cursor 4608, and 381 finite nonzero gradient tensors at every update. It verifies the recovery recipe and source, packed330 lineage, destination identity, all 12 full-state files in both native and durable checkpoints, sampler state, trainer step, tokenizer bytes, and the internal package-held-out result.

The trainer already evaluates the exact 499-row, 661,360-loss-token 2K anchor at checkpoint354. The prepared external evaluator therefore runs only the matched 8K and 16K panels. It does not repeat the internal 2K panel. Evaluation remains unadmitted and requires a later root-owned exclusive CUDA guard.

The current runtime wrapper already releases clean model and optimizer cache pages after attestation. If root needs more host headroom after checkpoint354, a separate operational receipt may use `posix_fadvise(DONTNEED)` only on independently verified immutable source/checkpoint or staged-data files, with before/after inode, size, and timestamps. This packet makes no cache or trainer-source changes.
