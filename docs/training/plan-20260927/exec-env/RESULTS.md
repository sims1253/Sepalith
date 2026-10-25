# Exec-env v1: pool, harness and pilot results (2026-10-25)

Receipts: [`receipts/EXEC-ENV-pool-slices.json`](../receipts/EXEC-ENV-pool-slices.json),
[`receipts/EXEC-ENV-pilot.json`](../receipts/EXEC-ENV-pilot.json). Data:
`/mnt/e/sepalith/campaign-20260915/data-work/EXEC-ENV-v1/`.

## Function pool (TRAIN split only)

- Source: 185,318 rows of the trained CPT cohort (sha256 verified); every row's
  identity group is `train_group` in DAT-02 v2 and every package name maps only
  to `train_group`, so nothing was dropped for split reasons. 176,162
  single-row package `R/` files decoded (176,161 match their source sha256);
  9,156 long multi-row documents were skipped.
- 548,846 top-level functions analysed (codetools globals, calls, shape).
  **48,414 selected from 6,797 packages** (median 4 per package, max 198).
  Main rejections: body outside 8-80 lines (237k), unresolved globals (63k),
  redefined or non-syntactic names (55k), unavailable namespaces (43k),
  fewer than 3 statements (44k), forbidden calls (42k).
- Shape: body 8-15 lines 19,004; 16-30 16,819; 31-50 8,295; 51-80 4,296.
  With in-file helpers 8,403; loops 13,076; `if` 37,837; rename candidates
  40,972; `na.rm`-style aggregations 13,898; nested call chains 6,899.
- Exposure: all CPT-exposed; 12,492 overlap the SFT TRAIN rows at all and
  1,091 with token containment above 0.5; 40 dropped for DEV containment
  above 0.3; 740 exact and 1,464 near duplicates removed.

## Hold-out and slices

- Hold-out (frozen exec-eval set, reserved before any cut): 800 functions
  from 206 whole packages (8 slices `H000`-`H007`); 7 training-side near
  duplicates dropped.
- Training: 47,607 functions, 6,590 packages, **472 slices** (`S000`-`S471`):
  466 hold 100-124 functions, 5 hold 150-198 (single big packages), 1 holds
  74. Packages are never split across slices.

## Harness

8 sandboxed workers (`unshare -rnm`, read-only remount, tmpfs, `prlimit`),
fork per job: **458 rewards/s**; a GRPO-sized batch of 32 rewards takes
**0.14 s p50, 0.21 s p95**; one job 15 ms p50, 46 ms p95 (4,976 text
completions). Verified: `quit()` ends only the child, hangs are killed,
escaping the function is a structure violation, writes outside the tmpfs
and network access fail, planting globals cannot hijack assertions.

## Pilot

- 48 functions from 48 packages (train slices, plain-input heuristics), tests
  by three Opus subagents following AUTHOR-BRIEF.md (a trial of the brief),
  revised once after the first model samples. **39 pass the gate**, 9 skipped
  (7 print/summary methods with output only, 1 needs data.table, 1 auto-skip).
- Gate: median 23 assertions (min 12); stub passes at most 10%; mutation
  kill rate median 1.0, min 0.85, pooled 98.6% (1,732 / 1,757 valid mutants;
  73 declared equivalent). Counting declared equivalents as survivors: median
  0.97, min 0.67. All 11 operator classes at least 97%.
- Tasks: 779 cuts, **438 tasks** (378 finish_block: 134 short, 163 medium,
  81 long; 60 rename_propagation), about 11 per function. Dropped: 314 because
  the tests reference strings or names only the hidden remainder introduces
  (inferability), 13 unobservable (truncated remainder passes), 10 low
  cut-local kill rate, 4 render failures. No gold failures.

### Reward on 5,191 completions

| Kind | n | Mean reward | Reward 1.0 |
| --- | --- | --- | --- |
| gold (text and token paths) | 438 + 438 | 1.0 | all |
| alt: rewritten-correct, not exact (`=`, renamed local, comment, `return()`) | 949 | 1.0 | 949 |
| broken: cut-local mutants | 1,120 | 0.18 | 21 (survivors/equivalents) |
| broken: truncated remainder | 378 | 0.04 | 0 |
| broken: syntax error | 399 | 0.04 | 15 (the removed `)` sat in a string or comment) |
| broken: `quit()` / escape / no-op / protocol-invalid | 438 each | 0 / 0 / 0 / -0.5 | 0 |
| arm A greedy | 43 | 0.45 | 17 |
| arm A T = 1.0 | 172 | 0.37 | 52 |

**The key number.** Of 215 arm A samples (43 tasks, Q8_0 on CPU), reward v2.1
counts 33 as exact; the execution reward accepts 69 (all 33 plus 36 more).
Of the 179 protocol-valid, non-exact edits, **36 (20%) are accepted**. A
manual review found all 36 behaviourally correct: 27 differ in `return(x)`
versus `x`, 6 in blank lines or comments, 3 are substantive alternatives
(an equivalent `rgb()` construction, two redundant-but-harmless column
copies). No v2.1-exact sample failed the tests.

In the first test round, which used plain-list fakes, 2 of the first 50
samples were false accepts: one copied a dozen fields the fake object lacked
into attributes (all `NULL`), one added a dead branch under an argument the
tests never set. The brief now requires `strict()` fakes and non-default
argument coverage, and the reward halves completions much longer than gold.
After the revision, both samples score 0.06 to 0.08.

## Lessons folded into the code and brief

- Message-text mutants (strings and `call.` flags inside `stop()`,
  `warning()`, ...), symmetric argument swaps and trailing-echo deletions are
  not generated; output-only methods are skipped; mutant ids bind the edit
  text; stale or killed equivalent declarations are reported.
- The global environment is locked against new bindings only, so `set.seed()`
  works in tests.
- Inferability drops about 40% of cuts on authored suites. Value-level
  assertions keep more cuts than exact-message assertions.

## Scale

At about 11 tasks per gated function and 81% of pilot functions gating,
12k-30k tasks need roughly 1,350-3,400 gated functions, about 17-42 slices
(sibling cuts of one function are correlated, so prefer more slices with
fewer cuts each over many cuts per function).
