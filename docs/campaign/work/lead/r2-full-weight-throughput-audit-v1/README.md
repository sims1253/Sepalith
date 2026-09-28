# Full-weight CPT throughput audit

The observed run rate is about 100.5 compute hours for 11,443 updates before
checkpoint and evaluation time. This projection uses nine actual log-to-log
intervals, not Transformers' incorrect terminal rate derived from the
configured horizon.

The ranked changes are:

1. **Profile isolation-preserving varlen packing inside each 16-row optimizer
   update.** Current microbatch-1 executes 183,088 physical forward/backward
   calls. First-fit grouping with a 16,384-token physical cap needs 34,396
   calls while keeping every row within its original optimizer update. Reset
   `position_ids` can drive Transformers 5.5 FlashAttention's varlen path, so
   members retain separate causal attention. The preparation module preserves
   every token, label, boundary and draw position. This is an 81.2% call-count
   reduction, not an 81.2% runtime claim. The actual patched MiniCPM/Unsloth
   path must pass standalone-versus-packed logits, gradient denominator,
   memory and checkpoint/resume tests first.

2. **Use a production full-state checkpoint cadence based on measured I/O.**
   Checkpoints 70 and 71 took 363 and 305 seconds. The temporary four-update
   cadence is intentionally diagnostic. Even cadence 128 would spend about
   8.26 hours on 89 full saves if these durations persist. Cadence 256 or 512,
   plus root-selected evaluation milestones and latest-two retention, would
   trade at most about 2.25 or 4.50 compute hours of restart exposure for about
   4.1 or 6.2 hours less save time. A C hot-stage benchmark may reduce the
   tradeoff, but no storage change is admitted here.

3. **Profile the existing GPU optimizer before changing Aurora/Muon math.**
   The optimizer averages 6.10 seconds, 19.3% of the measured update interval.
   Its 210 Muon, 84 Aurora and 87 AdamW tensors execute sequential Python loops
   with FP32 states and stochastic BF16 updates. Kernel launch fusion or
   compilation could help. CPU offload would introduce transfers and is not a
   speed candidate. TF32, fewer Newton-Schulz iterations, lower-precision
   state, or altered Aurora K change the reviewed algorithm and need a separate
   numerical and heldout comparison.

4. **Reduce restart-only redundant reads after the compute path is fixed.** A
   continuation hashes the 17GB checkpoint, hashes 3.84GB of cache files,
   hashes/loads the parent model, then Trainer reloads model and optimizer.
   A bound one-pass verification receipt plus a resume-aware model loader can
   remove redundant full-file passes while preserving immutable inode and
   digest evidence. Seven minutes per restart is material for short pilots but
   small against a 100-hour uninterrupted pass.

5. **Treat fused loss and multi-GPU sharding as verification items.** Unsloth
   2026.8.18 installs fused lm-head plus cross-entropy by default, but the
   frozen trainer does not record which path actually ran. Record it and A/B
   only if the current path is unfused. FSDP/ZeRO could shard 12.207GB of FP32
   optimizer state, but no compatible multi-GPU allocation, custom optimizer
   state-dict proof, or deterministic resume proof exists.

Ordinary length-paired microbatch-2 is not recommended for this schedule. It
adds 17.9% padded tokens, raises the sum-of-squared-length attention proxy by
35.9%, and creates 9,335 pairs wider than 16,384 tokens. Forward/data/backward
time correlates 0.923 with that attention proxy in the observed updates.

No active process, recipe, dataset, checkpoint or CUDA resource was changed.
