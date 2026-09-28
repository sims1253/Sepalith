# C07. Night 1: baseline, parity references, SFT smoke

- **Where:** PC nightly window, through the C02 queue
- **Needs:** C02 (the window works), C04 (recipe v2), C05 (DEV250 and
  evaluator). C06 must not have stopped for the user.
- **Produces:** step-0 scores, bf16 reference generations for C03, and
  measured SFT speed and memory
- **Effort:** a few hours of agent time the day before, one night, and a
  morning read-out

## Goal

Before any real training night, prove on the PC that the evaluator and trainer
fit and behave as expected. Also produce the reference numbers every later
rule compares against.

## Jobs to enqueue, in order

1. **Baseline evaluation.** Run DEV75 and DEV250 generation on parent 11,586
   in bf16, which is the standalone evaluator resource proof. Store the
   counts R1 needs (`valid`, `cap`, `edit`, `nofp`, `score250`) as the step-0
   baseline.
2. **Parity references.** Produce bf16 greedy generations for DEV75 and 200
   TRAIN prompts. Use a fixed seed-3407 sample stratified by family, and store
   the prompt ids. Upload them to the HF dataset at
   `campaign-20260915/C07/parity-references/` for C03.
3. **SFT smoke.** Run the LR 1e-5 arm for 20 updates with the pre-update
   masking gate on:
   - Set a fake deadline so the trainer stops after update 10.
   - Resume it in the same job.
   - Continue to update 20.
   - Save full state once, and time the save.

   Record seconds per update, peak GPU memory, host memory, checkpoint save
   and publish time, and resume time. Keep the checkpoint for inspection;
   nothing uses it later.

## Morning read-out

- Write `status/C07.json` with all measurements and the baseline counts.
- Compare seconds per update with the planning estimate of 10 to 15.
  Recompute how many updates fit in one night, allowing for about 25 minutes
  of evaluation per gate and 5 minutes per full save, then confirm or adjust
  the C08 and C09 plans.
- Stop-for-user if any of these hold:
  - peak GPU memory is above 30.5 GB;
  - the resume was not exact (loss and gradient-norm continuity broken);
  - the masking gate fired;
  - one pilot arm to update 268, including its gate, would take more than 2
    hours.
