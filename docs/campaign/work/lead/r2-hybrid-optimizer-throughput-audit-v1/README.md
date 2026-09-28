# SFT-11 hybrid optimizer throughput audit

This packet is a CPU-only, model-free audit of the frozen full-weight
optimizer. It does not change the optimizer, launch training, load a model or
checkpoint, or admit an optimizer candidate. The frozen source is bound by
SHA-256 in `static-analysis.json`.

The native Aurora configuration has 381 BF16 parameter tensors: 87 AdamW, 210
Muon, and 84 Aurora. At the configured one-million-element rounding chunk,
one step traverses 595 AdamW chunks, 2,527 BF16 rounding chunks, and 889 calls
to `apply_fp32_update_`. The custom matrix path expands to 5,670 matrix-multiply
operator dispatches: 3,150 Muon and 2,520 Aurora. These are source and
manifest counts, not isolated kernel timings.

The highest-confidence candidate is bounded workspace and metadata reuse. It
can preserve parameter/chunk order, FP32 arithmetic, the current
`torch.rand_like` calls, and checkpoint RNG behavior while removing repeated
scalar and pointwise temporary allocations. A fused round-and-store kernel
could have a larger effect, but changing its generator mapping or chunk
grouping changes exact stochastic-resume semantics until proven otherwise.
Foreach AdamW and compiled or batched Newton-Schulz paths are bounded
candidates that require parity and memory tests. Lowering Newton-Schulz or
Aurora iterations, changing matmul precision or state precision, and replacing
stochastic rounding are scientific changes.

The timing input reports a 21.7685-second mean update interval and a
5.9298-second mean pre-optimizer-to-log segment. That segment begins after all
381 gradient validity checks and includes optimizer, scheduler, and logging;
it is not an isolated optimizer-kernel measurement. A real CUDA A/B must
instrument those boundaries before assigning a speedup.

`static_audit.py` reads only source and small manifests/receipts. The frozen
optimizer tests ran on CPU with 13 passing tests. The test output is a control
for dispatch, FP32 state, stochastic-rounding, and exact-resume behavior; it
does not admit a throughput change.
