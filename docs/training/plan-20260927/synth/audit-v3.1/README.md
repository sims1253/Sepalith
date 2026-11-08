# SYNTH-v3.1 audit pack

v3.1 answers review round 1 on v3 (PR #87). Branch `data/synth-v3`;
generator 3.1.0. Method: [../README.md](../README.md) (v3.1 and v3
sections). **All counts are in [STATS.md](STATS.md)**, generated from the
build summary, manifest, profile, cohort receipts (1x and 2x), the rule
ablation and the probe receipts.

| Item | Path |
| --- | --- |
| Generated statistics | [STATS.md](STATS.md) |
| Seed fixes and bug-class sweep | [SEED-SWEEP.md](SEED-SWEEP.md) |
| Stratified sample of 60 v3.1 cases (seed 20261022) | [audit-sample-60.md](audit-sample-60.md), ids [audit-sample-60-ids.json](audit-sample-60-ids.json) |
| Data (not in git) | `/mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v3.1/` |
| Held-out manipulation-check set (not training data) | `SYNTH-v3.1/heldout-manipulation/` (manifest copy: `../../receipts/SYNTH-v3.1-heldout-manifest.json`) |
| Receipts | `../../receipts/SYNTH-v3.1-{manifest,build-summary,selfcheck,profile,cohort-proposal,cohort-proposal-x2,rule-ablation,arm-a-probe,heldout-arm-a-probe,heldout-manifest}.json` |

## Counts

| | v3 | v3.1 |
| --- | --- | --- |
| rows | 1,834 | 1,515 |
| finish_block | 1,703 | 1,450 |
| of which medium (9-20 lines) | 291 | 216 |
| of which long (21-45 lines) | 119 | 81 |
| `sibling` / `constrained_short` | 412 / 1,291 | 299 / 1,151 |
| roxygen_drafting | 131 | 65 |
| test-file finish rows | 17 | 0 |
| self-check | pass | pass |

## Review findings: finding -> change -> evidence

| Finding | Change | Evidence |
| --- | --- | --- |
| `kpi_targets(conversion = 2)` accepted; `check.params` read raw SQL | Both fixed, with boundary and literal/comment counterexample checks; all 272 v3 seeds swept for the two bug classes (one more bug: `samples.consume` accepted a negative amount) | [SEED-SWEEP.md](SEED-SWEEP.md); all three were `entered` in v3, so "entered" is far from "correct" |
| (a) digits inside strings; test-file datetimes | `unspecified_literal_number` (string digits must occur in the prompt); test files excluded | Ablation on v3: 12 rows; 17 test-file rows (8 not caught by another rule) |
| (b) constants, regex policies, output schemas need visible evidence (637b6a80 0.01, 55ee1d6e/b0e32218 regex, 3c695e57 `list(scaled=, norms=)`) | `unspecified_constant` (code numbers other than 0-2), `unspecified_regex` (regex-like literal not verbatim in the prompt), `unspecified_schema` (names in `list()`/`c()`/`data.frame()`/`structure()`/... must be prompt elements or doc words), `unspecified_literal_identifier` (compound names in strings). A cut that fails is rejected under every packaging; when stripping the prompt removes the spec, the cut survives only under a packaging that shows it | Ablation on v3: constant 68 rows, regex 27, schema 159, literal identifier 62 |
| (c) `$` element names and invented defaults count as novel | `elt:<name>` and `default:<name>` novel items, inside the class budgets | Ablation: 99 v3 rows over budget |
| Drop the 31 strict-recount rows and 17 test-file rows unless justified | The 3.1 rules remove 25 of the 31; the 6 kept rows are justified one by one in `SYNTH-v3.1-rule-ablation.json` (2 names visible as quoted strings and counted twice; 4 sentence-final words such as `invalid.` read as identifiers, all visible in the prompt). Test files: excluded | Receipt `strict_recount_kept_justification`; STATS ablation table |
| Report how many rows each rule removes | `rule_ablation.py` | 387 of 1,703 v3 finish rows removed in total; per rule (rows hit / removed by that rule alone): schema 159/138, elt/default budget 99/77, constant 68/61, literal identifier 62/49, regex 27/21, literal number 12/6, comment 1/1 |
| Dose: token shares next to draw shares; 2-exposure variant | `make_cohort.py` dose block, `--exposures 2` | STATS Exposure table (below) |
| "every surviving medium/long cut is used" was false | Per-bucket binding cap reported | STATS shaping table: 9-12 120/223 and 13-20 96/156 are limited by the prompt prefix-bucket cap; 21-30 and 31-45 by supply; 2, 3-4 by the 2x length ratio; 5-8 by the +-3 pp cohort rule |
| Quantify a medium-only relaxation of the prefix cap | Counterfactual run, not applied | +163 medium rows (9-12: 223, 13-20: 156, i.e. every surviving medium cut); 21-30 drops 65 -> 37 because the medium rows use up prompt-length headroom; median prefix lines 20 vs 29 applied; `<=10` share 22% vs 18%, `41-60` 11% vs 9% (TRAIN 47%, 4%) |
| One denominator for the cohort shift (+2.93 vs +2.99) | Shift = combined share - TRAIN share, combined = TRAIN finish rows in buckets 2-45 (7,759) + selected SYNTH finish rows | 5-8: +3.00 pp (the binding bucket) |
| Caps saturate at the short end | Stated | 2: 2.00x, 3-4: 1.99x, 5-8: 1.98x and +3.00 pp; available short cuts (6,002) far exceed what the caps admit (1,153) |
| `call. = FALSE` 36.8% vs 4.5% | Thinned per seed file (kept in 12% of files: 50 kept, 414 thinned, 144 had none) | Rows with it: 4.2% (TRAIN 4.3%); per `stop()`/`warning()` call 5.2% (TRAIN 15.0%) |
| Roxygen tags (`@export` 77% vs 44%, `@keywords` 14% vs 38%) | Greedy selection cut at the size with the smallest tag distance (at least half the greedy pick) | `@export` 60.0%, `@keywords` 26.2%, `@noRd` 20.0% (TRAIN 43.6/38.4/21.6); 65 rows (v3: 131); tag distance 1.23 -> 0.52. The pool has `@export` 79% and `@keywords` 12%, so a closer match needs new seeds |
| 53 short rows at masked-DEV >= 0.8 (27 at 1.0) | Sentence added to STATS | v3.1: 54 short rows >= 0.8 (23 at 1.0), kept. Short templated lines match many unrelated files after identifier masking, the exact-token DEV filters pass, and the masked filter applies only from 9 target lines |
| Probe: per-case records; explain 9/50 vs 16/50 | Per-case records (generation, score, variant, prefix lines/tokens, target lines/tokens, class, seed wave); paired diagnostic | Below |
| Manipulation-check set | `heldout-manipulation/` with manifest; rows carry `split=heldout_probe` (the SFT row loader refuses them) | 102 cases (target ~150; see limitations) |

## Exposure (dose not decided)

Both schedules are 14 + 2 with at most 3 no-op exposures (`assert_noop_cap`).

| SYNTH exposures | updates | SYNTH / finish draws | SYNTH / finish target tokens | SYNTH / all supervised tokens | SYNTH / finish prompt tokens | medium+long share of finish target tokens (cohort; SYNTH alone) | SYNTH / medium+long finish target tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1x | 1,102 | 15.7% | 5.7% | 5.0% | 18.7% | 85.9%; 51.8% | 3.4% |
| 2x | 1,211 | 27.1% | 10.7% | 9.4% | 31.6% | 84.1%; 51.8% | 6.6% |

SYNTH's draw share is about three times its target-token share: SYNTH
targets are short (median 5 lines vs TRAIN's 12), and its prompts are longer
than TRAIN's. In the 2x schedule the second copy of each row has the id
`<id>~x2`. The builder's final-update replays add 3 more SYNTH draws
(3,033 = 2 x 1,515 + 3).

## Probe (arm A Q8_0, greedy, CPU)

A fresh v3.1 sample (seed 3407, audit ids excluded), exact matches: short 14/60,
medium 10/50, long 1/10, roxygen 1/30. Medium rows are exact 4/11 with `file`
packaging and 6/39 with function-only packagings.

**Diagnostic.** The same 40 medium `sibling` cuts were each rebuilt as
`plain` (roxygen + function head; median 14 prefix lines, 263 tokens) and
as `file` (up to 60 preceding lines; median 48.5 lines, 566 tokens). Model,
greedy decoding and target were held fixed.

- `file` was exact on 10/40 and `plain` on 2/40. Nine pairs were exact only
  with `file` and one only with `plain` (mean reward 0.53 vs 0.24).
- 24 of the 40 cuts qualify only as `file`: their sibling evidence sits
  outside the function. Among those 24, `file` was exact on 6 and `plain`
  on none.

So the medium rate follows the packaging mix. v2 had 41% `file` prompts,
while v3 and v3.1 keep function-only prompts (as in TRAIN) wherever the
proxies allow. In v3.1, 25% of medium rows are `file` (53/216). That
explains 9-10/50 against v2's 16/50 better than a difference in seeds does.
The held-out set is about half `file` (49/102), and arm A is exact on 15/51
of its medium cases.

## Known limitations

1. **Determinacy classes remain proxies.** The 3.1 rules close the holes
   the reviewers named; recombined visible words and small novelty budgets
   (base-R calls, element names) are still allowed.
2. **Volume dropped** (1,703 -> 1,450 finish rows; medium 291 -> 216, long
   119 -> 81). Medium rows are capped by prompt geometry and long rows by
   supply. The relaxation above would add 163 medium rows at the cost of
   more 41-60-line prompts.
3. **Held-out set is short of target**: 102 cases (50 short, 51 medium,
   1 long). Almost every long cut that passes the filters is used in
   training, and 37 candidates were refused because a training window
   shows a line 4-gram of their target. A long held-out stratum would have
   to reserve units out of the training rows (each reserved long cut costs
   one long training row).
4. **Roxygen**: 65 rows; `@export` and `@keywords` are closer to TRAIN but
   not matched (pool limits). Prose stays free text.
5. **Behaviour**: "entered" is far from "correct" (see SEED-SWEEP.md).
   Only the two reviewed bug classes were swept.
6. **One author family**, pattern-heavy by design.
7. **Residual contamination risk**: LLM-memorised paraphrase is not
   detectable by the filters.
