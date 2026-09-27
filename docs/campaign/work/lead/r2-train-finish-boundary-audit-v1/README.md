# TRAIN finish-boundary audit

This read-only audit rehashed and joined the exact 15,006-row TRAIN pool, its
15,006 contexts, the admitted reward-buffer evidence, and the 3,503 new finish
rows with their source provenance. It did not read DEV or final examples.

The TRAIN pool has 7,785 finish rows. The incumbent 4,282 all end with a source
closing brace and all 4,282 applied gold documents passed the existing raw
parser check. The new 3,503 all have empty immutable suffixes and a replacement
range ending at represented-document EOF. Their source provenance says that the
corpus target ends before the outer brace and that the outer brace is absent
from the label. Of these, 3,088 are zero-width insertions and 415 replace the
last nonempty region. All 3,503 raw gold applications failed the existing parse
check; all 3,503 passed only after the reward-buffer builder appended its
diagnostic `\n}`.

The diagnostic suffix does not repair the model's output contract. PRM03
applies and scores the generated replacement body without that suffix. The
prompt requests the complete replacement region and carries no marker that the
model should return an intentionally partial function body. Therefore the
3,503 fragments are authentic source slices but incomplete applied edits under
their represented contexts.

Two source-supported repair geometries exist:

1. Append the one proven outer source brace to each target. Keep the current
   prompt and range, retokenize the new complete replacement body, and rerun
   source replay plus applied-document parsing. This is the smallest geometry
   change and makes exact reward match serving application.
2. Reconstruct the pre-edit document with the proven outer brace and place that
   brace in `suffix_lines`, moving the replacement range immediately before it.
   Keep the corpus target fragment byte-exact. This changes every prompt,
   context, replacement-range, and token identity but makes the partial boundary
   visible to inference.

Appending a parser-invented brace without per-row source replay is unsupported.
The framed parse result establishes a syntax diagnostic only; it does not prove
that the original edit is complete. Until one geometry is rebuilt and admitted,
the 3,503 rows should not be used for SFT or exact-region reward.

Run the audit from the plan root with:

```sh
nice -n 10 env PYTHONDONTWRITEBYTECODE=1 python3 \
  docs/campaign/work/lead/r2-train-finish-boundary-audit-v1/audit_train_finish_boundaries.py
```
