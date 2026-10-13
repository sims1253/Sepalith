# C05. DEV250, CPT-overlap check, portable evaluator

- **Where:** PC, daytime, CPU only
- **Needs:** C01
- **Produces:** the DEV250 panel, an overlap receipt, an evaluator with
  selectable dtype, and a docs fix
- **Effort:** one to two agent-days

## Goal

Decisions need a larger, TRAIN-like panel. DEV75 has only 6 finish_block cases
and 43% no-ops, and it is weak statistically. The evaluator must also run on
Kaggle.

## Read first

- AUDIT.md, section DEV evaluation gate
- `docs/campaign/work/lead/corrected-dev75-v1/manifest.json` and
  `docs/campaign/work/corrected-dev75-independent-review/`. These show how
  DEV75 was built and corrected.
- `docs/campaign/receipts/DAT-02-global-split-v2.json`. The `dev_group` split
  has 169 groups and 2,846 candidate rows.
- The DEV-only source files under `/mnt/e/sepalith/campaign-20260915/data-work/`
  named `DAT-07-dev-*`. Never read `DAT-07-final-*`.

## Steps

1. **DEV250.** Build 250 cases from the DEV split only, disjoint from DEV75,
   with the same PRM03 render and target construction that DEV75 used.
   - Composition: 50 no-op cases (20%, close to the SFT schedule's 18.75%).
     Split the 200 edits in proportion to the TRAIN edit families
     (finish_block about 56%, pipe_rewrite about 14%, format and rename about
     12% each, roxygen about 6%, na_rm about 1%), with at least 4 per family.
   - Sample by identity group so that no group contributes more than 3 cases.
   - Apply DEV75's finish-brace correction logic where it applies.
   - Record hashes, per-family counts and the target token-length
     distribution.
2. **Overlap check.** Intersect the package and document identities of DEV75
   and DEV250 with the CPT cohort manifest
   (`/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-extension-v1/cohort-manifest.json`
   and its row provenance). Report overlaps. If any case's source file was in
   CPT training, flag it and exclude it from DEV250. Record it for DEV75
   without changing DEV75.
3. **Evaluator.** Parameterize DEV generation by dtype (bf16 default; fp16 and
   fp32 for Kaggle) and device selection, so it can pin one GPU on a two-GPU
   host. Keep the PC path byte-for-byte equivalent in behaviour. Add a flag
   that evaluates both panels in one model load.
4. **Docs.** In `docs/PROMPT-CONTRACT.md`, add a short note naming
   `zeta2-prm03-v1` (`sepalith.campaign_protocol`) as the render of record for
   editing SFT and evaluation.
5. Upload DEV250 and its manifest to the HF dataset at
   `campaign-20260915/dev/dev250-v1/`.

## Acceptance

- DEV250 has 250 cases and is disjoint from DEV75 and from TRAIN (checked
  against the split receipt).
- The overlap receipt exists.
- Evaluator tests pass for all three dtypes on a tiny CPU model.
- DEV250 is on HF.

## Stop and ask if

- The DEV split cannot supply the composition. Report what it can supply.
- Building targets would need any file whose name contains `final`.
