# SYNTH-v3.1 seed fixes and bug-class sweep

Review round 1 on v3 found two semantic bugs in v3 seeds. Both are fixed,
and all 272 v3 seed files were swept for the same two bug classes by four
agents (brief: [../SWEEP-BRIEF-v3.1.md](../SWEEP-BRIEF-v3.1.md); each agent
read every file of its six batches in full).

## Fixes

| File | Function | Class | What was wrong | Change | Counterexample checks |
| --- | --- | --- | --- | --- | --- |
| `b36-shiny/dashkit/R/mod_valueboxes.R` | `kpi_targets` | one-sided range | `conversion` is documented as a share but only `< 0` was rejected, so `conversion = 2` passed | `conversion < 0 \|\| conversion > 1` -> "must be a share between 0 and 1." | 0 and 1 valid; 2 and -0.1 errors; exact message for -1; the negative-message loop no longer covers `conversion` |
| `b45-sql/rowcast/R/validate.R` | `check.params` | raw instead of cleaned text | the colon and at branches matched `:name`/`@name` in the raw SQL, so placeholders inside string literals and comments counted | `strip.sql.noise(sql, keep.case = TRUE)` (new documented argument; the default still upper-cases) feeds both branches | `':ghost'` literal, `-- :ghost` comment, `'@ghost'` literal each accepted with only the real parameter |
| `b36-shiny/benchlog/R/module.samples.R` | `samples.consume` | one-sided range (found by the sweep) | only `amount > volume` was rejected; a negative amount increased the volume | `amount < 0` -> "cannot give a negative amount." (lower bound from the name, the upper-bound check and the UI's `numericInput(min = 0)`) | 0 and the whole volume valid; -1 and 100.5 errors; exact message for -0.5 |

All three check scripts pass on the normalised sources in the v3.1 build
(behaviour ledger). No other file in b25-b48 has either bug class.

## Close calls left unchanged

The brief forbids inventing constraints that docs or names do not imply, so
these were reported, not changed:

- **Documented range, no check on either side** (not one-sided):
  `ci.rate`/`boot.diff` `conf.level`, `outlier.fences` `k`, `outlierFlags`
  `threshold`, `businessDaysInMonth`/`nthWeekday` `month`, `label_priority`
  `score`, `iter.split` `omega`, `flagOutliers` `probs`, `tax.rate`,
  bootstrap/summary `level`, `layer_defaults` `alpha`, `comfort_index`
  `humidity`, `rating_decay` `rate`, `screen.responses` `max.missing`,
  `score_wellbeing` `min_items`, `policy.parameters` `service`.
- **Untrimmed text without a cleaned copy** (no raw/cleaned mix-up):
  `validateAddress`, `env.get`, `coerce_flag.flag_choice`,
  `build_search_query` (a whitespace-only `q` passes the non-empty check and
  is stored as `""`).
- `parseSamLine` checks `flag >= 0` only, while sibling `samFlags` requires
  0-4095; the parser treats all its integer fields alike and no doc sets an
  upper limit.

## What the behaviour labels mean

"entered" is far from "correct": it says only that the seed file's base-R
check passed and the unit's body ran during that check. All three buggy
functions were `entered` in v3 (6 v3 rows: 2 each for `kpi_targets`,
`check.params` and `samples.consume`); their checks did not test the faulty
boundary or input. The sweep and the new counterexamples cover these two
bug classes only; other semantic errors may remain.
