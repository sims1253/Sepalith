# C08. Night 2: learning-rate pilot

- **Where:** PC nightly window, through the C02 queue
- **Needs:** C07 finished without stopping for the user
- **Produces:** three arms at the first gate (update 268) and an R2 selection
- **Effort:** half an agent-day across the day before and the morning after,
  plus one night

## Goal

Pick the editing-SFT learning rate on evidence, cheaply. The binder allows up
to 3e-5. CPT used 3e-6. The template suggested 1e-5 without justification.

## Jobs to enqueue, in order

1. The arm at LR 1e-5, to gate 268, with its automatic gate evaluation.
2. The arm at LR 3e-6, to gate 268, with its evaluation.
3. The arm at LR 3e-5, to gate 268, with its evaluation.

All three start from parent 11,586 with the identical schedule v2 and seed.
The middle LR runs first so the most likely winner is done even if the night
runs short. If C07's timing says three arms do not fit, the runner defers the
third arm to the next night. That is expected, not an error.

## Morning read-out

1. Apply R1's first-gate floor to each arm, then R2 to select.
2. Record every input and outcome in `status/C08.json`.
3. If R2 selects an arm, write its gate-268 checkpoint path and hash as the
   input for C09, and enqueue C09's job for the next night. That job resumes
   the winning arm.
4. If R2 says stop-for-user, write the question with all three arms' numbers
   and your recommendation.

Do not delete losing arms. Deleting needs the user; see R7.
