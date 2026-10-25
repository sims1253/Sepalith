# Exec-env: execution-grounded tasks and reward

Exact match is the wrong target for long completions: most of a realistic
function's remainder is not determined by its context, so a single gold
under-rewards correct alternatives, and the C15 GRPO pilot saturated its
2,086-prompt exact-match pool (train reward 0.67 -> 0.94, DEV flat). Exec-env
rewards any **functionally correct** completion: real TRAIN-split R functions
get behavioural tests, are cut into finish_block (and rename_propagation)
tasks rendered exactly like TRAIN rows, and a completion is rewarded by
splicing it back and running the tests in a sandbox.

Code: `packages/sepalith/src/sepalith/exec_env/`. Data (not in git):
`/mnt/e/sepalith/campaign-20260915/data-work/EXEC-ENV-v1/`. Test authors
follow [AUTHOR-BRIEF.md](AUTHOR-BRIEF.md). Numbers: [RESULTS.md](RESULTS.md)
and `receipts/EXEC-ENV-*.json`.

## Pipeline

| Step | Module | Output |
| --- | --- | --- |
| 1. TRAIN R files from the trained CPT corpus | `pool extract` | decoded package `R/` files (scratch) |
| 2. Static analysis of every top-level function | `pool analyze` (`r_pool.R`, codetools) | globals, calls, shape |
| 3. Selection, helpers, air, dedup, exposure | `pool select` | `pool.jsonl`, `pool-summary.json` |
| 4. Frozen hold-out, then author slices | `slices` | `slices.json`, `authors/<slice>/<fid>/` |
| 5. Tests by the author fleet, gated | `gate` (`mutants.py`) | `test.R`, `gate.json`, `frozen.R` |
| 6. Cuts, rendering, per-cut gate | `tasks` | `tasks.jsonl` |
| 7. Reward | `reward.ExecReward` (`harness.py`, `r_worker.R`, `testlib.R`, `sandbox.sh`) | records |

### Split and sealed data

- Source: `CPT-prefix-extension-v1/combined-cpt-train-ctx16384.jsonl` (the CPT
  cohort actually trained, sha256 pinned), which carries per-document package
  and identity-group provenance. A document is admitted only if its group is
  `train_group` in `DAT-02-global-split-v2.json` (sha256 pinned) **and** its
  normalised package name occurs only in `train_group` groups (the C10 rule).
  Every path goes through the synth build's `guard_path` (sealed-final
  prefixes and any `final`/`sealed` component refused).
- Functions are CPT-exposed by construction; SFT TRAIN-row and DEV-panel
  token containment is recorded per function, and functions with DEV
  containment above 0.3 are dropped.
- The hold-out (`H` slices) is reserved from whole TRAIN packages before any
  cut exists; training-side near-duplicates of held-out functions are dropped.

### Harness

Persistent `Rscript` workers (JSON header + length-prefixed payloads in,
one JSON line out). Each worker runs in `unshare -rnm` (no network; every
mount remounted read-only; a private tmpfs is the only writable path) under
`prlimit --as`; each job is a fresh forked child with an elapsed-time limit
and a hard kill. Structure check (top-level assignments exactly the expected
helpers + function), source, then the full suite in an environment whose
parents (assertion library, global environment) are locked. Assertions are
counted by the helpers in the child; `quit()`, crashes and hangs end only the
child and score 0.

### Reward (`sepalith.reward.exec.v1`)

Protocol checks as reward v2.1 (-0.5 for invalid output or cap); no edit 0;
region-only splice; tamper-capable constructs not used by the original
refused (0); parse/structure/source failures 0; otherwise
`0.5 x behaviour-weighted pass fraction + 0.5 x all-pass` (per-category pass
shares averaged; assertions that never ran count as failed), halved when the
completion has more than max(2 x, +6) the gold's non-blank lines (dead code
under the suite), exact bonus 0 by default and exact match logged separately. `ExecReward` has the GRPO trainer's
reward signature; `RoutedReward` sends exec tasks to it and everything else
(legacy TRAIN replay) to reward v2.1. Nothing in the trainer, its default
reward or any bound recipe changes.

## Known limits

- In-process isolation is not a security boundary against deliberate
  stack-inspection attacks; the static screen refuses those constructs and
  the sandbox bounds the damage. Infrastructure faults raise instead of scoring.
- The inferability check (tests referencing strings or names that only the
  hidden remainder introduces) is lexical; it drops about 40% of cuts.
- Tests only see the inputs they build. Plain-list fakes let completions that
  read nonexistent fields pass; the brief requires `strict()` fakes and
  non-default argument coverage, and the overlong guard limits dead-code padding.
- The pool is a candidate pool: in the pilot 81% of sampled functions could be
  gated (most others were print-only methods).
- na_rm_propagation, pipe_rewrite, formatting and roxygen are not built:
  na.rm changes behaviour on purpose and needs NA-targeted suites; the TRAIN
  pipe family rewrites `%>%`, which needs magrittr (not installed in the
  sandbox).
