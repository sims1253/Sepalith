# A100 full-weight continuation preparation

This packet binds the accepted local checkpoint-90 metadata and proves the core 8-rank mechanics on CPU. It does not authorize or implement a cloud launch.

The proposed layout gives each rank two consecutive schedule rows. Eight ranks therefore preserve each original 16-row optimizer window. Transformers 5.5 `average_tokens_across_devices` gathers the supervised-label denominator and scales each local loss before DDP gradient averaging.

The hybrid optimizer needs an extra distributed rule. Its BF16 update uses stochastic rounding. Forward passes advance rank RNG streams differently, so ordinary replicated optimizer calls could produce different BF16 weights even after DDP synchronizes gradients. `a100_distributed.py` assigns a deterministic rounding seed to each global optimizer step inside a forked RNG scope. It restores the rank-specific model RNG after the optimizer call.

A distributed Trainer writes one RNG file per rank. The initial resume view hard-links the accepted canonical checkpoint RNG to all eight rank names without editing checkpoint-90. Future distributed checkpoints retain all eight RNG files and add `rng_state.pth` as the rank-0 alias needed for a later single-device transition.

Five CPU tests pass, including an actual eight-process Gloo DDP interrupted/resumed update using the campaign full-weight optimizer. This does not establish A100 memory fit, NCCL speed, Unsloth DDP compatibility, or MiniCPM numerical parity. The `run` command remains fail-closed until those checks and a root admission are wired into a fresh production runner.
