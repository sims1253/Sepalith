# C13. RFT and DPO

- **Where:** PC during the day for code and data (CPU); PC nights for
  reference log-probabilities and training
- **Needs:** C12, with R4 step 1 satisfied
- **Produces:** up to two adopted offline steps, and R5 decisions
- **Effort:** two to three agent-days, plus two or three nights

## Goal

Get most of RL's benefit without online RL. RFT reinforces verified behaviour
cheaply. DPO adds the negative signal that online GRPO's all-correct groups
never provide, especially wrong edits on no-op prompts. This is also what the
published next-edit systems do after SFT: Zed's Zeta and JetBrains Mellum both
use DPO.

## Steps

1. **RFT data.** Convert C12's RFT candidates into TRAIN-shaped rows with the
   same PRM03 render, masking and termination as SFT. Keep no-op share and
   exposure caps as in schedule v2. Mix in 30% of the original TRAIN rows, a
   uniform sample, to limit drift. Build a schedule with gates at halves.
2. **RFT training.** One night, from the selected SFT model. Use LR equal to a
   third of the SFT LR, with the same optimizer and warmup rules and automatic
   gates. Apply R5 against the SFT model.
3. **DPO code (new).** Add a DPO trainer path to the consolidated runtime.
   Standard sigmoid DPO, β = 0.1. Reference log-probabilities are precomputed
   once and stored, so no reference model is resident. Use the policy's
   masking conventions, and sum over completion tokens only. Keep the
   full-weight aurora_mix optimizer and bf16 stochastic rounding. Test it with
   a tiny model on CPU: the loss decreases, reference values load correctly,
   and chosen and rejected masking is right.
4. **Reference pass.** One night job computes reference log-probabilities of
   all pairs under the starting model, meaning the RFT model if adopted and
   otherwise the SFT model, in bf16 on the PC. Store and hash them.
5. **DPO training.** One night, one pass over the pairs, LR a tenth of the SFT
   LR, with automatic gates on DEV75 and DEV250. Apply R5.
6. Publish each adopted model to `scholzmx/sepalith-2b-edit-sft` under a new
   folder: `rft-1/` or `dpo-1/`. Include weights, GGUF Q8_0 and a card. Record
   R5 outcomes in `status/C13.json`.
7. Run R4 step 2: re-run C12's census on the latest adopted model, reusing
   the harness, then apply R4 step 3.

## Stop and ask if

- DPO shows the known failure mode: chosen log-probability falling steadily
  while `score250` does not improve. Stop and report.
- Either step lands in R5's stop-for-user band.
