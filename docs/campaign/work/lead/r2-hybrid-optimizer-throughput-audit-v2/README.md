# SFT-11 live optimizer dtype correction

This v2 packet corrects the BF16 hot-path census from the frozen v1 audit for
the live native continuation. The v1 dispatch manifest records the model
before checkpoint restore, when all 381 parameter rows are BF16. The native
trainer calls `restore_saved_fp32` before optimizer construction; that helper
selects F32 entries from the safetensors header, verifies their saved values,
converts the parameter storage to FP32, and copies the values back.

The checkpoint-322 runtime and durable `model.safetensors` files were inspected
through their eight-byte length prefix and JSON header only. Both headers are
43,368 bytes and have SHA-256
`7548160639b6226e1e75987742106d6809f5c355a46c6e7d89e46e2095a3876d`. Their
campaign manifests are identical and hash to the root-provided checkpoint
manifest identity `81dfdfb4af6d0690ebb114f68910fdbe270d20ad3393a84c47c1c337ce2dbbcf`.
No model payload bytes were read or hashed by this packet.

The live inventory remains 381 tensors and 2,516,756,480 parameters, but it is
85 F32 norm tensors (174,080 elements) plus 296 BF16 tensors
(2,516,582,400 elements). Dispatch names and counts remain 87 AdamW, 210
Muon, and 84 Aurora. With the one-million-element chunk, corrected per-step
counts are 2,442 stochastic-round chunks and `rand_like` calls, 804 stochastic
rounder invocations, and 1,608 +/-infinity scalar tensor allocations. The 85
F32 norm updates take the direct `param.sub_` branch: 85 update calls, zero
random draws, and zero infinity-scalar allocations.

The original v1 packet is unchanged. This v2 correction is accounting only,
remains prepared and not admitted, and does not alter the optimizer or running
training.
