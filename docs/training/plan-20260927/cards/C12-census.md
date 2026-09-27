# C12. Pass-rate census on Kaggle

- **Where:** Kaggle T4 for generation; PC during the day or Kaggle CPU for
  scoring
- **Needs:** C03 (parity passed), C11 (reward v2.1), and the selected SFT
  model (C09, or C10 if R3 adopted arm B), published on HF
- **Produces:** per-prompt pass rates, RFT candidates, DPO pair candidates,
  and the R4 inputs
- **Effort:** one agent-day, plus about 3 to 8 Kaggle GPU-hours; use C03's
  projection

## Goal

Measure where the SFT model is always right, always wrong, or mixed. This
decides whether online RL has any signal: the last census, on an older model,
had 92.5% of groups with no signal. It also produces the raw material for RFT
and DPO.

## Steps

1. Build the prompt file: all 15,006 TRAIN prompts as token ids, with id,
   family and target, all from the public TRAIN rows. No DEV or final data.
2. Run C03's harness with k = 8 samples per prompt, temperature 1.0,
   top_p 1.0 and `max_new_tokens` 1,024, in the dtype that passed parity.
   Shard across both T4s and across sessions. Keep total use within R7's
   20-hour weekly guard.
3. Score every sample with reward v2.1: `exact`, graded reward, parse status
   and no-op outcome.
4. Report, overall and per family:
   - the exact pass-rate distribution;
   - `M`, the share of prompts with 0 < pass < 1;
   - all-correct and all-wrong shares;
   - the no-op false-suggestion rate;
   - cap hits;
   - mean completion length.
5. Emit two files:
   - **RFT candidates:** verified-correct samples, deduplicated per prompt,
     at most 2 per prompt, with families balanced to the TRAIN mix.
   - **DPO pair candidates:** chosen is a verified-correct sample of the same
     prompt, or the gold target if the model never succeeded. Rejected is an
     incorrect sample that parses, or any edit suggested on a no-op prompt.
     Prefer model-generated chosen samples, and at most 2 pairs per prompt.
6. Upload the samples, scores and both files to
   `campaign-20260915/C12/<model-id>/`. Apply R4 step 1 and record the
   outcome.

## Stop and ask if

- Parity fails for the new model. Re-run C03's parity check on DEV75 first.
- R4 says stop-for-user.
