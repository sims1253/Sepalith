# Latent module-memory gate: implementation and current evidence

2026-09-06. **Executable implementation delivered; the pretrained learnability
experiment has not run because the required resource window is occupied.**
No scientific accuracy, training-speed, or editing-improvement claim follows
from the CPU fixtures.

| Evidence | Result | Scope |
|---|---|---|
| CPU contracts | 13 tests pass | Tiny randomly initialized Qwen3.5 model using the pinned Transformers implementation, with both GDN and full attention |
| Core regression suite | 51 tests pass (`python3 scripts/check_core.py`) | Includes unchanged product/legacy prompt bytes and existing memory/runner contracts |
| Local decoder inspection | 114 checked projections; 7,382,016 rank-16 adapter parameters | Actual banked B4 config and safetensors tensor headers; no full weight load |
| Synthetic-data audit | 512 training / 160 held-out packages; no split overlap | Actual pinned decoder tokenizer; 134 encoder bytes, at most 23 query tokens / 66 retrieval tokens / 3 scored target tokens |
| Banked-decoder numeric contract | Pending | Must pass on the real bf16 decoder before training |
| Learned information recovery | **Not tested** | Relevant, absent and shuffled learned memory have no scientific scores yet |
| Actual editing improvement | **Not tested** | Stage B is disabled |

The contracts cover token-ID/embedding equivalence and three-token greedy output,
correct target-only causal loss alignment, gradients into learned slots and every
adapter target, incompatible/stale/corrupt payload rejection, disabled cache
reuse, parent-commit reads, masked-answer exclusion, target/view separation,
package separation, training/checkpoint round trips, all six evaluation arms,
paired-verdict logic, compute-cap rejection and an absolute-path worker launch
that rejects an unpinned preparation. Tiny fixture checkpoints are temporary test
artifacts, not learned B4 memory checkpoints.

Machine-readable evidence:

- [contracts-receipt.json](contracts-receipt.json): test output, exact code hashes,
  environment versions and elapsed CPU test time.
- [decoder-inspection.json](decoder-inspection.json): concrete names, dimensions,
  saved-to-runtime mapping, config and implementation hashes.
- [data-audit.json](data-audit.json): exact token budgets, training-order and split
  identities, and 3,072 scored tokens per trained arm (9,216 across three arms).
- [preparation-receipt.json](preparation-receipt.json): source, dataset, recipe,
  tokenizer and backend identity. `weights_pinned: false` is deliberate and
  blocks launch; the full weight digest must be frozen inside the resource window.
- [resource-observation.json](resource-observation.json): the observed scheduling
  dependency and relevant live process/claim evidence.

The complete synthetic dataset is stored outside Git under
`/home/m0hawk/.local/share/sepalith/latent-memory/`. Its deterministic generator
and content hash are committed. The environment is isolated at
`/home/m0hawk/.venvs/sepalith-latent-gate310`, with a full requirements lock in this
directory. No product/core dependencies were installed or changed.

The last recorded resource observation, at 15:21 CEST, found X5-S1 evaluation PID
1117314 under chain PID 1005134 and the S1 benchmark PID 831052 still active. The
RTX 5090 reported 15,272 MiB in use and 38% utilization. The shared GPU ledger
still ended with X5-S1's claim. The benchmark owner has S1/S2/V1c work, and the
runner-migration coordinator is arranging a drain boundary. Completion requires
both owners' releases and a reserved 90-minute window; a drop in utilization is
not that release. No dispatcher or background retry has been armed.

The [recipe and instructions](README.md) freeze the gate and record deviations
before training: a two-layer width-128 byte encoder, 32 learned slots, single-module
random-default QA, exact retrieval as a control, explicit mixer-only adapters,
PyTorch reference kernels, and no hybrid-state reuse. This small screen matches
target exposure rather than FLOPs. It cannot establish whole-workspace compression,
serving economics, superiority over trained retrieval, or editing utility.

When the resource window is released, the authorized next action is to pin the
weight bytes, pass the real decoder contract, and execute this recipe once. A
scientific failure or compute-cap interruption is a result to retain, not
permission to expand the experiment.
