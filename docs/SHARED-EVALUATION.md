# Shared paired evaluation

> **2026-09-06 retrospective correction:** The completed [saved-result reanalysis](reanalysis/README.md) separates doc_sync meaning and preservation from canonical reconstruction, reconstructs B8b paired selection, and reports LOC1 any-gold hit, multi-gold recall and parser defects separately. Historical scores and claims below are retained as recorded.

`sepalith.evaluation` owns the numeric part of the historical
`experiments/eval/paired_significance.py` audit. The historical script retains
its result-file readers and audit command, and re-exports the numeric functions
from the checkout's shared package. Direct invocation still works without a
package installation. Importing the shared module performs no file reads or
model calls.

The interface takes aligned binary outcomes. `audit_pair(name, a, b)` reports
exact McNemar probability, rates, discordant counts and a deterministic paired
bootstrap interval for A minus B. `higher_better=False` reverses winner labels
for error rates without changing the reported rate difference. Callers must
establish row identity before passing lists; equal list lengths alone do not
prove alignment.

The extraction preserves the default seed, resampling count, percentile
convention and verdict labels. Two corrections accompany it:

- `alpha` now controls both the probability threshold and the bootstrap interval.
  Previously a non-default threshold still received the default 95% interval.
- Exact binomial tails use an integer denominator, avoiding floating-point
  exponent overflow when there are more than 1,023 discordant pairs. Very small
  probabilities can still underflow to zero in the final floating-point result.

Invalid counts, empty or nonfinite deltas, invalid confidence levels and
nonpositive bootstrap counts fail explicitly. Default historical checks run
through the compatibility script as part of `python3 scripts/check_core.py`.

These intervals resample individual paired rows. Correlated trajectory points
need a separately specified cluster analysis. The preserved label
`TIE-UNDERPOWERED` means the legacy adoption rule did not find a winner; it does
not establish equivalence or diagnose statistical power. No stored experimental
results were recomputed by this extraction.
