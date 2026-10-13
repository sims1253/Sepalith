# Varlen numerical attribution probe

This is an executable, no-update GPU diagnostic derived from the reviewed v4 actual-model harness. It loads selected representative checkpoint 66 once. It validates the small campaign manifest and stable model file metadata, relying on the accepted root hash receipt instead of hashing the 5 GB weight file again.

The probe first compares layer-0 split calls with one concatenated call at RMSNorm, q/k/v, gate and up projections. It repeats q projection in FP32. It then holds projected Q/K/V fixed and compares sixteen one-block xformers calls with one block-diagonal call. This separates dense shape effects from the attention kernel.

The full-model phase measures sequential microbatch-one twice, one packed sequence, and one ordinary right-padded batch. Every path uses the same 9,681 supervised-token denominator. No optimizer is created or stepped. The report records peak CUDA allocation and verifies that the source weight inode, size, timestamps, and device stay unchanged.

The model-independent verifier applies the predeclared gate. Isolation must be exact. Same-mode gradient repeatability must be within `1e-7`. The FP32 q projection control must be within `1e-6`. Packed gradient median and maximum must be no worse than `max(0.01, 1.25 × padded control)` and `max(0.05, 1.25 × padded control)`. A pass establishes only mechanical parity for further review.

Root must run the guard command after the active training lane releases CUDA. The output and host-supervision paths must still be fresh.
