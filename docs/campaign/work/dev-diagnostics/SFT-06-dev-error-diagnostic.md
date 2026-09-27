# SFT-06 DEV error diagnostic

Observed at `2026-09-12T14:19:44+00:00`. This is a bounded diagnostic of the
current primary SFT checkpoint. It uses the exact 75-row DAT-07 DEV panel and
the complete step-500 per-case archive. It does not score the final set, alter
labels, compare raw loss across tokenizers, launch a model, or create a second
SFT arm.

The input identities are:

| Input | Path | SHA-256 | Rows/status |
| --- | --- | --- | --- |
| DEV panel | `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl` | `b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21` | 75 |
| Step-500 per-case archive | `/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/SFT-primary-3000-a/evaluations/cases-step-500.json` | `3e06b8990ab7808dc0d5683ba108425d450391114e76f35399cbf1377a2c012a` | 75, `complete` |

The step-1000 per-case file was still `partial` at the diagnostic observation
(`73/75` results, SHA-256
`32b4cca16e061adbb9175bc902dad20c924883da22b0db02b6091d2b23767737`), so no
step-1000 row was admitted to this step-500 snapshot. It completed immediately
afterward; the handoff state is now 75/75 with cases SHA-256
`1eddac3b78251e27bce4465b86637a021c4ff34569df51a297bac52403aea4ea` and
summary SHA-256
`23aba0137126d34c8ddaa7fd9788c60887bf8d8f17f8bad3bcf2bb68e3704695`. The
completed step-1000 state is recorded for handoff by the adapter, but this
bounded report remains explicitly step-500 evidence.

The review scope is exactly the six `finish_block` and eight
`roxygen_drafting` rows. The generated JSON evidence is
[`sft06-dev-error-diagnostic.json`](./sft06-dev-error-diagnostic.json), created
by [`sft06_dev_error_diagnostic.py`](./sft06_dev_error_diagnostic.py).

The labels below are diagnostic annotations separate from evaluator fields:

* **Strict textual mismatch** means `exact_region` is false. It is counted
  independently of semantic quality.
* **Malformed or truncated** means the stored generation is not protocol
  valid or hit the 512-token cap. A **repeated block** is three or more
  contiguous copies of an identical non-blank block.
* **Unsupported behavior** means a protocol-valid `finish_block` output parses
  after the fixture close is supplied, but its algorithm, shape, or input
  contract materially differs from the target.
* **Valid alternative docs** means protocol-valid roxygen that parses and whose
  central claims fit the visible function, although the wording and coverage
  differ from the target.
* **Documentary semantic mismatch** means protocol-valid roxygen whose central
  claim conflicts with the visible function (vector versus matrix, or
  restricted-cubic versus B-spline).

All 14 rows were strict textual mismatches; 10 were protocol-valid and 4 were
capped/missing-terminal. The primary categories are:

| Family | Rows | Exact | Strict mismatch | Protocol valid | Cap hit | Valid alternative docs | Documentary semantic mismatch | Unsupported behavior | Malformed/truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `finish_block` | 6 | 0 | 6 | 3 | 3 | 0 | 0 | 3 | 3 |
| `roxygen_drafting` | 8 | 0 | 8 | 7 | 1 | 5 | 2 | 0 | 1 |
| **Total** | **14** | **0** | **14** | **10** | **4** | **5** | **2** | **3** | **4** |

The four malformed/truncated rows are also the only capped rows. Three have
repeated blocks: fplyr repeats `@rdname flply`/`@export`, rlecuyer repeats the
two `.lec.Random.seed* <- NULL` assignments, and SemNetDictionaries repeats a
commented title guard. The regmed row grows distinct `fit.edges` comment lines
until the cap and has no repeated identical block.

The row-level evidence is bounded below. `target → generated` excerpts are
short descriptions of the visible target and decoded output; the full target
and generated replacement remain in their locked source archives.

| Row ID | Package/path | Primary result | Evaluator | Target → generated evidence |
| --- | --- | --- | --- | --- |
| `dat07-existing-c54ed09efae0ca38cff22981` | `bigsimr` / `R/bigsimr.R` | Valid alternative docs | valid, 75 tokens | Bigsimr.jl setup description → Bigsimr setup with `pkg_check`, `...`, and list return; shorter but compatible. |
| `dat07-existing-152d62f7a57472f8abb30ca9` | `corHMM` / `R/convertPhangorn.R` | Documentary semantic mismatch | valid, 79 tokens | Conversion to a vector and `best` behavior → title/return claim a matrix; central result shape is wrong. |
| `dat07-existing-2838b904f1e44b7440e6da9f` | `fplyr` / `R/fdply.R` | Malformed/truncated | invalid, capped, 512 tokens | Full chunk-reader documentation → 39 repeated `@rdname flply`/`@export` pairs; no terminal. |
| `dat07-existing-06847c759fd32bdf16f2e598` | `kvh` / `R/RcppExports.R` | Valid alternative docs | valid, 20 tokens | Detailed KVH parser contract → minimal `@rdname kvh_read`/`@export` wrapper docs; compatible but under-specified. |
| `dat07-existing-81ddc4c6ddd226b04dcf0910` | `peRiodiCS` / `R/b_rcs.R` | Documentary semantic mismatch | valid, 97 tokens | Restricted cubic spline basis → B-spline basis claim; algorithm name conflicts with visible formula. |
| `dat07-existing-5b5e844d3a8795c52149f848` | `rmarchingcubes` / `R/contour3d.R` | Valid alternative docs | valid, 104 tokens | Isosurface/marching-cubes parameters and returned geometry → concise contour/marching-cubes wrapper docs; compatible. |
| `dat07-existing-06097c12b0328d475be857e1` | `standrecon` / `R/standrecon-internal-helpers.R` | Valid alternative docs | valid, 78 tokens | Conclass vector from status/decay vectors → class for one observation; shorter and omits `@noRd`, but compatible with the helper. |
| `dat07-existing-d38a792699b4cccded445dc0` | `TCpRepDesigns` / `R/TCpRep1.R` | Valid alternative docs | valid, 107 tokens | Method-I p-rep design family → concise TCpRep1 design description; compatible but omits detailed constraints. |
| `e623a61b5a4c066358a477f2` | `netassoc` / `R/generate_nul_resample.R` | Unsupported behavior | valid, 93 tokens | Matrix resampling by column with `sum(obs[,i])`, probabilities `nul[,i]`, and `tabulate` → `lapply(seq_along(nul))`, `sample(nul[i], obs[i])`, and a list. |
| `4f08633513b5c525240d2540` | `rlecuyer` / `R/rlecuyer.R` | Malformed/truncated | invalid, capped, 512 tokens | Remove/assign seed table, initialize stream, `.Call`, return 1 → alternating seed assignments repeated 26 times; no terminal. |
| `d11581e9cfa4e3971aa1466e` | `SemNetDictionaries` / `R/utils-SemNetDictionaries.R` | Malformed/truncated | invalid, capped, 512 tokens | Nested yes/no parser and `readline` loop → repeated commented `title == NULL` guard; no terminal. |
| `157517ba47dbab157f7c361a` | `partitions` / `R/partitions.R` | Unsupported behavior | valid, 52 tokens | Big-z factorial product over each `n` in `x` → `x$fac` lookup and base `factorial(x)`, changing input and return behavior. |
| `f43de3e77f2d92ed7b223464` | `AutoStrataK` / `R/autostrata.R` | Unsupported behavior | valid, 381 tokens | Deparse `target`, call `clustering_strata`, attach class → validation/factor assignment implementation that requires character target and does not call the visible helper. |
| `04834fef4fe59742f13677a9` | `regmed` / `R/mvregmed.graph.attributes.R` | Malformed/truncated | invalid, capped, 512 tokens | Graph construction and returned graph attributes → growing commented `fit.edges` column-selection guesses until cap; no terminal. |

For syntax inspection, the adapter projected `context.prefix + decoded output +
context.suffix_lines`. The `finish_block` fixture intentionally stops before
the outer function close, so the adapter appended one synthetic `}` to both
target and generated projections solely for tree-sitter inspection. With that
documented closure, the canonical R tree-sitter parser in
`/home/m0hawk/Documents/Sepalith/.venv` reported no error nodes for all 14
targets and all 14 generated projections. This establishes parse shape only;
it does not establish runtime or package-level semantic correctness. The
roxygen rows had complete suffixes and needed no synthetic close.

The useful continuation signal is concentrated in the two families. Every
finish row missed the exact body; the three protocol-valid completions were all
semantically unsupported, while the other three exhausted the generation cap.
Roxygen had five concise compatible alternatives, two central documentation
claims that contradict the visible target, and one capped repetition. The
step-500 SFT-05 decision to continue to the scheduled step-1000 readout remains
the training decision; this packet is an error diagnostic and does not promote
or open another SFT arm.
