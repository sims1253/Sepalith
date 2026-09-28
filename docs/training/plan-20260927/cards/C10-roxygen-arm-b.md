# C10. Roxygen admission and SFT arm B

- **Where:** PC during the day for data work (CPU; Kaggle CPU for big
  batches); PC nights for training
- **Needs:** C04 for the recipe machinery. Training nights come only after
  C09 finishes.
- **Produces:** an admitted roxygen cohort, SFT arm B, and an R3 decision
- **Effort:** two to four agent-days of data work, one or two nights

## Goal

The all-eligible-data policy says reviewed data should not be silently
dropped. 10,017 roxygen candidates were reviewed. 3,903 are recommended for
context admission and 6,114 are held for insufficient evidence or format
mismatch. Admit what passes the remaining reviews, train arm B on
15,006 plus the admitted rows, and keep it only if R3 says so.

## Read first

- `docs/campaign/receipts/DAT-10-roxy-context-admission-review.json`
- `docs/campaign/receipts/DAT-10-roxy10017-root-context-budget-census.json`.
  With full-file context, 4,989 rows exceed 4,096 tokens.
- `docs/campaign/TRAINING-DATA-POLICY.md`
- The queue data at
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-supported-context-v1/`

## Steps

1. For the 3,903 recommended rows:
   - Check licenses against the ledger; the policy is permissive only.
   - Deduplicate against the 15,006 TRAIN rows and within the set.
   - Check disjointness from DEV and the final set by identity group.
   - Run a semantic spot-check of 60 rows, done the same way as C06.

   Record the admitted count.
2. Build the cohort: the 15,006 TRAIN rows plus the admitted rows. Bucket by
   actual length; do not truncate targets. Build schedule v2's analogue with
   the same composition rules: every edit once, no-op share about 18.75%, and
   no-op exposure at most 3. Build it for the new totals, with gates at the
   quarters.
3. Estimate the night cost from C07's timing, scaled by the new token count.
   Most added rows are longer. If the arm needs more than two nights, say so
   in the status file. That is fine, because the arm resumes across nights.
4. Train arm B from parent 11,586 at the LR that R2 selected, with automatic
   gates.
5. Compare arms A and B with R3. The DEV roxygen subset is the roxygen cases
   of DEV75 and DEV250 combined. Record the decision. The winner is the SFT
   model for C12.

## Stop and ask if

- Fewer than 1,000 rows survive admission. Arm B is then probably not worth a
  night; ask.
- R3 is borderline.
