# Performance work

Worktree: `Sepalith-performance`, branch `perf/offline-optimization-20260906`.
This branch prepares candidates while the shared checkout runs experiments.
The first candidate bounds scratch and ladder evaluator logits memory. It has CPU golden
checks; target-GPU numerical and speed validation is pending.

## First candidate: evaluation projection

`poc_twin/train.py::quick_eval` previously projected all tokens to the vocabulary
before chunking cross entropy. It now calls `model.py::chunked_eval_ce`, which
projects at most 4,096 tokens at once. The default chunk size, fp32 loss,
projection precision, token denominator and Python accumulation order are
preserved. Different matrix shapes can change floating-point rounding; this is
not a claim of bitwise GPU equivalence.

For an 8 × 1,024-token batch and vocabulary 130,560, fp32 logits alone formerly
occupied 3.984 GiB. The new maximum is 1.992 GiB at the existing chunk size.
These are tensor-size calculations, not measurements of total VRAM or speed.
The model, CE workspace and allocator add memory. Smaller chunks are candidates
for a later sweep; they can also slow the projection down.

Run CPU correctness checks from this worktree, using an existing environment
with PyTorch. Keep CUDA hidden and use one otherwise unused CPU:

```sh
sepalith_py=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 15 taskset -c 16 "$sepalith_py" experiments/performance/check_cpu.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 15 taskset -c 16 "$sepalith_py" experiments/performance/bench_eval_ce.py
```

The CPU check command runs 17 checks, including the existing diffusion objective
tests, a multi-chunk loss/gradient regression and historical evaluator comparisons.
The training environment has no pytest, so the runner uses unittest directly.
CUDA transfers/autocast are stubbed in ladder integration tests; GPU precision
remains a separate gate.

The benchmark defaults to a tiny CPU correctness smoke and prints no timing
metric. It independently retains the historical full-projection reference.
It rejects numerical disagreement and changed token counts. Timed runs alternate
reference/candidate order after warmup and retain every timing sample, source
hashes, settings and CUDA peak allocations. Output receipts cannot be overwritten.

After coordinating a quiet window and claiming the GPU through the existing
manager/ledger, the representative projection measurement is:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 15 taskset -c 16 "$sepalith_py" experiments/performance/bench_eval_ce.py \
  --device cuda --measure --resource-window YOUR_RECORDED_WINDOW_ID \
  --batch 8 --seq 1024 --hidden 768 --vocab 130560 --chunk 4096 --repeats 7 \
  --output /absolute/path/to/new-receipt.json
```

`--resource-window` records operator coordination; it is not a lock or a resource
reservation. Passing it does not authorize starting work during another claim.
This projection benchmark is not a full evaluation or training-step benchmark.
Before adoption, run the complete evaluator on frozen real hidden states/targets
and record CE, token counts, total time and peak memory. A faster microbenchmark
alone is insufficient. Preserve a memory-saving candidate separately if runtime
regresses; the acceptance rule must be declared before comparing candidates.

## Harness decision

Start with **pi-autoresearch**, one isolated worktree per active writer and one
measurement worker across this machine. It already provides measurement commands,
correctness checks, result logs and keep/revert iteration. Source review at
`41820794dcdc1a1fa9f3f19593052f7978f0de5e` (package 1.7.0) confirmed that keep
stages all changes and discard restores tracked files and removes untracked files.
Use a clean, dedicated checkout. Never point it at the shared research checkout.
Its extension loaded successfully through Pi RPC in this session, without model
requests, installation into global settings, or an optimization loop.
[Source](https://github.com/davebcn87/pi-autoresearch).

Use **ShinkaEvolve** when a stable, independently scored kernel warrants searching
many implementations. Its population/archive and separate proposal/evaluation
concurrency are useful then. Set evaluation concurrency to one here; multiple
remote proposals need not imply multiple local benchmarks. Its local-compatible
provider uses Chat Completions; its headless adapter accepts effort through xhigh,
not max, at inspected revision `9912af12d423504b8d580f4179fd15f5f88b8c50`.
The exact requested Pi provider/model/max routes therefore still need an adapter
check. Unknown model pricing can return zero in that provider: a reported cost
cap is not a complete billing safeguard for unpriced routes.
[Source](https://github.com/SakanaAI/ShinkaEvolve).

For the first pi-autoresearch session, create its `.auto` session in a fresh
candidate worktree after baseline checks are committed. Set `maxIterations: 5`.
Use the measurement command above in `.auto/measure.sh` and the golden test
command in `.auto/checks.sh`. Keep the evaluator, reference, fixtures and tests
outside agent-editable scope and verify their hashes independently before each
run. Scope candidate edits to `chunked_eval_ce` only. The current benchmark
imports the model module: it executes candidate code and is not a security sandbox.
A general-purpose coding agent with shell access can change its own tests;
worktree isolation alone does not enforce evaluator integrity.

The primary metric can be `candidate_ms` only after setting a hard memory ceiling
and correctness tolerances. For a memory-first trial, use measured peak allocation
as primary with a predeclared runtime-regression limit instead. Do not accept
speedup by shortening inputs, weakening correctness, changing loss normalization,
changing model precision or skipping scored rows. Validate winners on separate
shapes and real workload data. Repeat the same baseline to estimate noise;
pi-autoresearch's mixed-candidate MAD score is not an A/B confidence interval.

No loop is armed on this branch. The live S1/S2/V1c measurement ownership remains
with its current manager. The report under `docs/validation/performance-20260906`
records the first audit and the remaining candidates.
