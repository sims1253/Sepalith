# Offline performance pass — 2026-09-06

Recommendation: use pi-autoresearch for the first bounded optimization trials.
Keep ShinkaEvolve for a later kernel search with a stable scorer. The immediate
constraint is trustworthy, uncontended evaluation, not a lack of proposal agents.
The harness comparison and runnable commands are in
[the performance guide](../../../experiments/performance/README.md).

## Work completed

The three requested reviewers ran in tandem:

- Astra (`gpt-6-astra`, medium): scratch/production training and kernel audit;
  implemented and reviewed the bounded evaluation projection.
- Pi `zai/glm-5.3 --thinking max`: custom kernel/training review, completed
  successfully. Raw response: [GLM](glm-review.txt).
- Pi `opencode/muse-spark-1.3-contributor-free --thinking max`: experiment and
  serving review, completed successfully. Raw response: [Muse](muse-review.txt).

The Pi calls used only read/grep/find/ls tools, with extensions and skills disabled.
Both providers reported ready. `max` records the requested Pi setting; no separate
claim is made about the providers' internal reasoning-token allocation. No local
models or CUDA workloads were launched. The live S1 server was pinned to CPUs
0–15; our tiny CPU checks used CPU 16, one thread and nice priority 15.

The shared checkout contains other sessions' uncommitted work. All code changes
are isolated in `Sepalith-performance`, branch `perf/offline-optimization-20260906`,
starting at `73d0bfe`. Shared registration is the only change made in the main
checkout. The production recipe, live scripts and experiment queue are unchanged.

## Candidate 1: bounded evaluation logits

The scratch quick evaluator and ladder evaluators built full batch × sequence ×
vocabulary logits before chunking CE. Projection now happens inside the chunk
loop. One helper preserves fp32 CE, token counts, existing autocast scopes and
Python accumulation order. It serves scratch quick-eval, ladder quick-eval,
ladder BPB, and A2 exit/MTP loss evaluation. The generic `evaluate.py` path still
uses its model's `forward`; changing that dispatch requires a separate model
compatibility check.

At 8 × 1,024 tokens and vocabulary 130,560, one fp32 logit tensor is 3.984 GiB.
The default 4,096-token chunk bounds that tensor to 1.992 GiB. Full-evaluator
peak memory includes other tensors and can differ further because the old loop
retained previous logits while projecting the next batch. These are allocation
calculations, not measured GPU savings or an achieved speedup.

The smoke receipt shows identical aggregate CE on the tiny CPU fixture:
724.1060361862183 for 130 tokens. Golden tests cover chunk boundaries, tails,
counts, ignored labels, preserved inputs, no gradients and accumulation order.
The paired benchmark records all samples and source hashes; it has not run in
timing mode. Full real-data GPU numerical, allocation and runtime gates remain.

## Fix 2: diffusion loss across multiple chunks

Independent source inspection found a correctness defect behind larger-batch
optimization. `_ce_chunk_weighted` returns a vector, but `mdlm_loss` added chunk
vectors together. Once selected tokens exceed one chunk, it can either encounter
incompatible shapes or give `scatter_add_` fewer values than indices.

A deterministic CPU regression with five masked positions reproduced failures at
chunk sizes 1, 2, 3 and 4. The fix concatenates the small per-position loss vectors
in original order before their existing per-example scatter. It preserves the
loss estimator and checkpointed vocabulary projection. Dense-reference tests
check both loss and hidden/head gradients across six chunk sizes. Before/after
logs are retained beside this report. This is a correctness fix, not a measured
throughput improvement or permission to reinterpret historical runs.

Changing `MICRO_PAD_TOK` still requires separate validation. `train_md.py` gives
each microbatch equal weight, while `mdlm_loss` averages examples within each
microbatch. An example in a microbatch of size B has weight 1/(M×B), where M is
the number of microbatches. Changing grouping can change these weights and random
mask draws. Keeping the total row/token count fixed is insufficient to claim
identical optimizer updates. No estimator or grouping change was made.

## Prioritized next work

| Priority | Candidate | Evidence and gate |
|---|---|---|
| 1 | Measure the evaluation patch on real shapes, then complete evaluator | Allocation established in code; preserve CE/counts and measure total time + peak memory in one quiet window. |
| 2 | Reduce per-micro `loss.item()`/statistics synchronization | Calls verified in scratch/MD/A2 loops; aggregate detached scalars and check logs, gradients and full optimizer-step time. Magnitude unmeasured. |
| 3 | Stage optimizer batches and prefetch data | Per-micro conversion/transfer verified; preserve sample order, resume and RNG behavior. Pageable `non_blocking=True` alone does not establish overlap. |
| 4 | Grouped QK maximum and tensorized clipping | Expanded K and host scalar reads verified. Preserve unmasked all-position maximum, GQA sharing, first-micro sampling and post-step clipping. |
| 5 | Fused linear/CE and batched Muon kernels | Requires profiler evidence and golden gradient/update tests. Compilation already helped the trunk; compiled CE previously caused excessive intermediates. |
| 6 | HTTP keep-alive and content-addressed data caches | Muse's source findings are plausible; likely smaller than decode. Verify request/row bytes and cache identities before real-model comparisons. |
| 7 | CPU GGML Q4 dequant | Existing P12 estimates ~16% remaining Q4 decode upside on its tested model. Keep its 13B/order gate; aggregate roofline data does not isolate one offending function. |

Production GDN SFT already uses Unsloth. First profile its actual executed loss
kernel and memory peaks; do not transfer scratch-kernel findings to it by analogy.
Packing, length grouping, precision changes, attention-backend swaps, prompt
reordering and adaptive output limits need scientific/quality validation. They
are not transparent speed fixes. Serving needs separate prefill, decode and
keystroke-cycle measurements on the target CPU/GPU hardware.

## Review corrections and evidence limits

Raw agent responses are hypotheses and observations, not authoritative results.
The synthesis above corrects these errors or overstatements:

- GLM's 4,096 × 130,560 fp32 allocation is **1.992 GiB**, not ~0.5 GB;
  a 2,048-token chunk is ~0.996 GiB, not ~0.25 GB.
- No measured evidence supports GLM's proposed ≥1.3× evaluator speedup or its
  attribution of a precise fraction of micro-model overhead to synchronization.
  Equal throughput across models alone cannot isolate that cause.
- Pageable transfers with `non_blocking=True` are not adequately described as a
  universal no-op. Useful overlap must be demonstrated with the actual staging.
- A population's mixed-candidate MAD is not repeated-baseline measurement noise.
- Muse's claim that reducing workers necessarily doubles runtime is unsupported.
  Connection reuse also needs retry/failure-accounting and server-cache checks;
  same request bytes alone do not establish timing equivalence.
- Float accumulation changes can alter telemetry; this patch retains Python sum
  order rather than combining GPU fp32 loss sums as GLM suggested.

pi-autoresearch 1.7.0 loaded successfully through Pi RPC. Its autonomous loop was
not started. Both harness source trees were inspected at pinned revisions listed
in the guide, without installing or changing training environments. Shinka's
exact Pi provider/max integration remains untested. The linked X post could not
be retrieved, so no workflow claim is attributed to it.

Final CPU validation: **17 checks pass**, including the six existing diffusion
objective tests. Four benchmark preflight rejection checks pass. `git diff --check`
is clean. The environment lacks pytest; `experiments/performance/check_cpu.py`
runs the same plain-function objective tests and new unittest suites directly.
Ladder integration checks stub CUDA transfers/autocast and do not validate GPU
precision. See [test output](cpu-checks.txt).

Remaining validation requires the existing manager's uncontended resource window.
No new dispatcher, recurring task, remote training job or background optimizer
was armed. Both Pi review processes exited successfully.
