# SYNTH: synthetic, verified editing cases for the weak families

SYNTH adds original, mechanically verified PRM03 editing cases for the
families where arm A is capped by data (C12 census, arm A, k=8, T=1.0):

| Family | mixed / all-correct / all-wrong |
| --- | --- |
| finish_block | 12.8% / 1.2% / 86.0% |
| roxygen_drafting | 2.6% / 0.0% / 97.4% |
| format_propagation, pipe, rename, na_rm, no_op | mostly or entirely solved; not generated in v2 |

Rows are drop-in: they render through `campaign_protocol.build_training_row`
(renderer `zeta2-prm03-v1`, pinned tokenizer) with the DAT10 v4 sidecar
schema. Data lives on E: (not in git). This directory holds the generator
inputs, the cohort and probe scripts and the audit pack.

| Version | Data | Seeds | Status |
| --- | --- | --- | --- |
| v1 | `/mnt/e/.../data-work/SYNTH-v1/` (kept for comparison) | `seeds/` | superseded after review round 1 (whitespace/comment style, overstated determinacy, bypassable filters) |
| **v2** | `/mnt/e/.../data-work/SYNTH-v2/` | `seeds-v2/` | current; numbers in [audit/STATS.md](audit/STATS.md) |

v1 is reproducible from commit `77232ef` (generator 1.0.0); this tree builds
v2 (generator 2.0.0; `--blank-policy v1 --families finish_block,roxygen_drafting,format_propagation`
approximates v1's generation but not its selection).

## Method (v2)

1. **Seeds.** `seeds-v2/` are the v1 seeds (`seeds/`, 336 original R files in
   24 domain batches, [AUTHORING-BRIEF.md](AUTHORING-BRIEF.md)) revised by
   [REVISION-BRIEF-v2.md](REVISION-BRIEF-v2.md):
   - TRAIN-like interior blank lines and full-line comments (TRAIN finish
     targets: 7.3% blank lines, 4.6% comment lines; both rise with length);
   - a correctness review that fixed the reviewers' findings and many more
     bugs (NA handling, bytes vs characters, empty inputs);
   - base-R behavioural check scripts (`seeds-v2/<batch>/checks/`, 219 files)
     with `stopifnot()` assertions;
   - roxygen tags moved toward TRAIN's frequencies (`@family` almost gone,
     `@noRd`/`@keywords internal`/`@rdname`/`@title`/`@description` varied).
2. **Normalisation** (`normalize.py`): authored blank lines and comments are
   kept; `air format` canonicalises everything else (air collapses blank
   runs to one). Blank-line placement matters for exact match: C12 records
   309 finish_block near misses whose air-canonical region equals gold but
   whose file differs, e.g. by an extra blank line
   (`receipts/C12-census.json`, `observations.near_miss_with_canonical_region_equal`).
   A looser supplementary count (477 samples equal to gold after deleting all
   whitespace, 434 of them blank lines only) is reproduced by
   `census_whitespace.py` -> `receipts/SYNTH-census-whitespace.json`.
3. **Cases.**
   - `finish_block.py`: cuts at statement starts (any depth), comment-run
     tops and nested closing lines; final-close-only and closers-only targets
     are not generated. No target starts with a blank line (TRAIN has ~9%
     such targets; DEV golds never do). Full-body cuts use TRAIN's
     signature-line region 90% of the time.
     **Determinacy** (`common.novelty`, `common.structural_support`):
     `novelty` counts target identifiers, words inside string literals and
     numbers > 2 that the prompt does not contain (comments are split off
     first, so roxygen `'` no longer opens a string). A cut is admitted only
     as `constrained_short` (<= 8 lines, <= 2 novel items) or `sibling` (the
     prompt shows >= 50% of the target's call-preserving masked 8-grams, e.g.
     a sibling function or mirrored branch, and <= 3 novel items). Everything
     else is `author_choice` and rejected; if the preferred packaging fails,
     the file-context packaging (which shows siblings) is tried.
   - `roxygen.py`: TRAIN geometry only (v1's tags-only geometry is off);
     blocks of <= 2 lines are dropped; tags must be valid and `@param` must
     match the signature; >= 2/3 of target lines must be skeleton or appear
     verbatim in the prompt.
   - format_propagation is not built in v2 (token-identical rebreaking only,
     and arm A already solves 22/25 in the v1 probe).
4. **Verification, reported in three separate layers** (`verify.py`):
   - *mechanical / protocol*: render + `validate_training_row`, v4 length
     limits, R parse (edited window parses; finish_block's pre-edit window is
     exactly incomplete), air idempotence, reward v2.1 gold = exact with
     reward 1.0, family checks. This is a mechanical-check pass rate, not a
     correctness claim;
   - *determinacy*: the generator classes above;
   - *semantic*: `behaviour` per case: `checked` (the seed file's base-R check
     script passed and calls this unit), `file_checked_unit_unexercised`,
     `no_check` (needs add-on packages). A seed file whose check fails drops
     all its cases.
5. **Dedup and decontamination** (`dedup.py`, `cpt_check.py`):
   - formatting-invariant token fingerprints: code tokens (no whitespace or
     layout) plus comment words, 12-grams over the *edit span* (the last 40
     context tokens plus the target, keeping only grams that touch the
     target, so 1-2 line targets are covered). Per reference: containment
     and longest verbatim token run. Drop rules: DEV containment >= 0.3 or a
     run >= 24 tokens; TRAIN >= 0.5 or >= 40; CPT >= 0.5 or >= 40;
   - line 4-grams (v1 measure) kept as an extra filter;
   - identifier-masked structural containment against TRAIN and DEV,
     reported as an aggregate (it flags shared idioms and renamed copies;
     not a filter);
   - CPT: every CPT row is decoded; hits are stored as hashes in
     `cpt-hits.json`, bound to the sha256 of the source tree, the CPT corpus
     (pinned from `receipts/C05-cpt-overlap.json`) and the tokenizer plus the
     fingerprint parameters. `build.py` refuses a hit file whose binding
     differs, and a build without one is marked `admitted: false`;
   - sealed data: `build.guard_path` checks the string and the resolved path
     (symlinks followed) against the sealed locations in CONTEXT.md and any
     `final`/`sealed` path component. v2 no longer reads
     `DAT-02-global-split-v2.json` (v1 read it to compare package names with
     final groups; that read was not sanctioned by C00 and is removed);
   - near duplicates within the set: MinHash/LSH, Jaccard >= 0.8.
6. **Distribution shaping** (`build.shape_finish`, `build.shape_roxygen`):
   at most 2 cases per unit (distinct length buckets); finish_block cases
   are added greedily, longest buckets first, only while (a) the TRAIN +
   SYNTH finish_block length distribution stays within 3 percentage points
   of TRAIN's in every bucket and (b) SYNTH alone has at most twice TRAIN's
   share in any bucket; within a bucket, candidates follow TRAIN's
   (has blank line, has comment) mix; full-body cuts <= 6%, test-file cuts
   <= 1%. Roxygen is <= 10% of the set, chosen greedily toward TRAIN's tag
   frequencies with no tag order above 20%.

## Residual risk (stated plainly)

All seeds were written by LLM agents. The filters catch verbatim and
near-verbatim overlap with TRAIN, DEV and CPT, reformatted copies and (as a
reported aggregate) identifier-renamed copies. They cannot catch an
LLM-memorised *paraphrase* of existing code (same logic, different names and
structure). Seeds are fictional packages written to a brief, and CPT hits are
rare (see STATS.md), but this risk is not zero. Seeds share one author family
and one brief, so a residual house style remains.

## Rebuild

```bash
PY=/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python; export PYTHONPATH=packages/sepalith/src
S=/mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v2; SEEDS=docs/training/plan-20260927/synth/seeds-v2
nice -n 10 $PY -m sepalith.data.synth.build --seeds $SEEDS --out $S --normalise-only
nice -n 10 $PY -m sepalith.data.synth.cpt_check --sources $S/sources --out $S/cpt-hits.json
nice -n 10 $PY -m sepalith.data.synth.build --seeds $SEEDS --out $S --cpt-hits $S/cpt-hits.json
nice -n 10 $PY -m sepalith.data.synth.selfcheck --data $S --out docs/training/plan-20260927/receipts/SYNTH-v2-selfcheck.json
nice -n 10 $PY docs/training/plan-20260927/synth/make_cohort.py --synth $S --out $S/cohort-proposal --compare-13-3
nice -n 10 $PY -m sepalith.data.synth.audit --data $S --out docs/training/plan-20260927/synth/audit/audit-sample-60.md \
  --ids-out docs/training/plan-20260927/synth/audit/audit-sample-60-ids.json
$PY docs/training/plan-20260927/synth/probe_arm_a.py --data $S --exclude-ids docs/training/plan-20260927/synth/audit/audit-sample-60-ids.json \
  --out docs/training/plan-20260927/receipts/SYNTH-v2-arm-a-probe.json
$PY -m sepalith.data.synth.profile --cases /mnt/e/.../SYNTH-v1/cases.jsonl --cases $S/cases.jsonl --out $S/profile.json
$PY -m sepalith.data.synth.report --data $S --profile $S/profile.json \
  --probe docs/training/plan-20260927/receipts/SYNTH-v2-arm-a-probe.json --out docs/training/plan-20260927/synth/audit/STATS.md
```

The build is deterministic given the seeds, R 4.6.1 and air 0.12.0.

## Output files (`SYNTH-v2/`)

| File | Content |
| --- | --- |
| `train-token-rows.jsonl` | Drop-in TRAIN rows |
| `context-sidecar.jsonl` | DAT10-style sidecar |
| `cases.jsonl` | Raw cases: context, `region_new`, generator metadata (determinacy, behaviour, decontamination measures), verification |
| `verification-ledger.jsonl` | Mechanical verification of every candidate |
| `behaviour-ledger.jsonl` | Behavioural check result per seed file |
| `normalisation-ledger.jsonl` | Seed -> normalised source hashes |
| `build-summary.json` | Counts, rejections, decontamination, shaping, distributions |
| `dev-inputs.json` | Paths and sha256 of the DEV files hashed (no DEV content) |
| `cpt-hits.json` | Bound CPT hit hashes and counts (no CPT text) |
| `profile.json` | TRAIN vs v1 vs v2 distribution profile (`profile.py`) |
| `cohort-proposal/` | TRAIN v4 + SYNTH cohort, length profile, 14+2 schedule (default) and 13+3 comparison |
| `manifest.json` | sha256 of every artifact, generator id/version, git head, `admitted` |
