# SYNTH-v1: synthetic, verified editing cases for the weak families

SYNTH-v1 adds original, mechanically verified PRM03 editing cases for the
families where arm A is capped by data:

| Family | Arm A TRAIN census (k=8, T=1.0): mixed / all-correct / all-wrong |
| --- | --- |
| finish_block | 12.8% / 1.2% / 86.0% |
| roxygen_drafting | 2.6% / 0.0% / 97.4% |
| format_propagation | 46.6% / 45.0% / 8.4% |
| pipe, rename, na_rm, no_op | at least 96% all-correct (no learning signal, not generated) |

The output is drop-in: rows render through
`sepalith.campaign_protocol.build_training_row` (renderer `zeta2-prm03-v1`,
pinned tokenizer), and the sidecar uses the DAT10 v4 context schema. Data
lives on E: (not in git); this directory holds the generator inputs, the
manifest and the audit pack.

- Data: `/mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v1/`
- Manifest (committed copy): `docs/training/plan-20260927/receipts/SYNTH-v1-manifest.json`
- Audit pack: [audit/](audit/)
- Code: `packages/sepalith/src/sepalith/data/synth/`, tests in
  `packages/sepalith/tests/test_synth.py`

## Method

1. **Seeds.** 24 batches of original R source (`seeds/b01-*` ... `seeds/b24-*`),
   written by authoring agents to [AUTHORING-BRIEF.md](AUTHORING-BRIEF.md):
   fictional packages, scripts, Shiny apps and test files across tidyverse,
   base R, data.table, Shiny, R6/S4/S3, testthat, Rcpp glue, plotting,
   modelling, databases, geospatial, text, ML and more. Every file parses. The
   brief asks for sibling functions that mirror each other and for endings
   that are predictable from what precedes them, so that cut-off code has a
   context-determined continuation.
2. **Normalisation** (`normalize.py`). Each file gets one of two
   deterministic blank-line styles inside top-level expressions (`dense`: none;
   `sectioned`: one blank line before each full-line comment), then `air
   format`. Rationale: in the C12 census, 434 of 477 whitespace-only
   finish_block near misses differ from gold only in blank-line placement,
   which air keeps. Gold whitespace is therefore air-canonical with a
   predictable blank-line rule, and the final brace sits at the header column
   (the v4-braceindent convention).
3. **Cases.** One generator per family:
   - `finish_block.py`: cuts a braced top-level expression (function,
     `test_that`, `setMethod`, ...) at a line boundary: a statement start at
     any depth, the top of a comment run above a statement, or a nested
     closing line (`fb_nested_close`; the target starts with `}`). The
     packaging matches TRAIN: roxygen kept (`plain`) or stripped (`strip`),
     the TRAIN `outline` or `diagnostics (static)` header above it,
     or the preceding file content (`file`), with an empty region and an empty
     suffix; for full-body cuts, sometimes the signature line is the region
     with the cursor at its end. Per unit it picks up to 2 short (2-8 lines),
     1 medium (9-20) and at most 1 long cut, so short completions dominate.
     Cuts whose target introduces too many tokens absent from the prompt are
     dropped (`novelty` limits 4 / 8 / 14 by length bucket); the novel tokens
     are recorded per case.
   - `roxygen.py`: TRAIN geometry (<= 30 prefix lines, empty region, the
     definition as suffix, the block as target, `roxy_full`) and a tags-only
     geometry where the title and description are already typed
     (`roxy_tags`). A block is admitted only if its tags are valid, its
     `@param` names match the signature in order (or it inherits them), and
     most target lines are skeleton lines or appear verbatim in the prompt
     (`determined_share` >= 0.65 full, >= 0.7 tags).
   - `format_propagation.py`: the user reformats a file statement by
     statement. A multi-line statement is collapsed into an older layout
     (`one_line`, or `aligned` arguments under the opening parenthesis); a
     collapse is used only if air turns it back into the source byte for byte.
     The history shows the previous statement reformatted; the region is the
     next collapsed statement, and the target is its air layout. Later
     statements stay in the old layout.
4. **Verification** (`verify.py`), per case, recorded in
   `verification-ledger.jsonl` and in each `cases.jsonl` record:
   - the case renders through `build_training_row` and `validate_training_row`;
   - prompt <= 3,072 tokens and completion (with EOS) <= 1,024 (the v4
     length policy);
   - the edited window parses in R. The pre-edit window parses (roxygen,
     format) or is exactly incomplete (`unexpected end of input`,
     finish_block);
   - air leaves the edited window unchanged (format_propagation: the
     byte-exact air round trip proven at generation);
   - reward v2.1 scores the gold tokens `exact`, reward 1.0, with no reference
     flags (format_propagation: only `formatting_only_reference`, as for its
     TRAIN rows);
   - family checks: finish_block ends on a closing line at the header column;
     roxygen tags and `@param` names match the signature; the
     format_propagation region differs from its target and its history edit
     parses.
5. **Dedup and decontamination** (`dedup.py`, `cpt_check.py`):
   - exact duplicates, then MinHash/LSH near duplicates (Jaccard >= 0.8 on 5-token
     shingles of the 3 lines before the cut plus region and target);
   - TRAIN: drop if a target shares a 4-line verbatim run with a TRAIN row or
     >= 60% of its shingles are in TRAIN;
   - DEV: hashes of DEV75, DEV250, the C05 DEV construction pools and the
     DAT-07 DEV sources (no file with `final` in its path is opened, enforced
     by `guard_path`); drop on any shared 4-line run in the window, or >= 50%
     target-shingle containment. Only aggregate counts are reported;
   - CPT: every CPT row is decoded and its line 4-grams are matched against
     the seed sources; windows containing a hit are dropped;
   - package names: no synthetic package name occurs in a DEV or final split
     group (DAT-02 identity forms; metadata only).
6. **Family caps.** roxygen_drafting <= 15% and format_propagation <= 20% of
   the set.

## Rebuild

```bash
PYTHONPATH=packages/sepalith/src nice -n 10 /home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python \
  -m sepalith.data.synth.build --seeds docs/training/plan-20260927/synth/seeds \
  --out /mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v1 --cpt-hits <cpt-hits.json>
PYTHONPATH=packages/sepalith/src nice -n 10 .../python -m sepalith.data.synth.cpt_check \
  --sources /mnt/e/.../SYNTH-v1/sources --out /mnt/e/.../SYNTH-v1/cpt-hits.json
PYTHONPATH=packages/sepalith/src .../python -m sepalith.data.synth.audit \
  --data /mnt/e/.../SYNTH-v1 --out docs/training/plan-20260927/synth/audit/audit-sample-60.md
PYTHONPATH=packages/sepalith/src .../python docs/training/plan-20260927/synth/make_cohort.py \
  --synth /mnt/e/.../SYNTH-v1 --out /mnt/e/.../SYNTH-v1/cohort-proposal
```

The build is deterministic given the seeds (seeded per unit), R 4.6.1 and air
0.12.0. The CPT hit file comes from a first build's `sources/`, since
normalisation does not depend on it.

## Output files (`SYNTH-v1/`)

| File | Content |
| --- | --- |
| `train-token-rows.jsonl` | Drop-in TRAIN rows (`ROW_KEYS` schema, `split` = `train`) |
| `context-sidecar.jsonl` | DAT10-style sidecar: `context` (PromptContext), `selection_geometry`, `source_identity` |
| `cases.jsonl` | Raw case records: context, `region_new`, generator metadata, verification |
| `verification-ledger.jsonl` | Verification record of every candidate, passed or not |
| `normalisation-ledger.jsonl` | Seed -> normalised source hashes and styles |
| `build-summary.json` | Counts, rejections by reason, decontamination counts, distributions |
| `sources/` | The normalised seed files the cases are cut from |
| `cpt-hits.json` | CPT line 4-gram hits (hashes and counts only) |
| `cohort-proposal/` | TRAIN v4 + SYNTH cohort and its schedule_v2 schedule (proposal, unbound) |
| `manifest.json` | sha256 of every artifact, generator id/version, git head |
