# C15. GRPO pilot

- **Where:** PC nights
- **Needs:** C14 go
- **Produces:** a GRPO-tuned candidate, or a documented stop
- **Effort:** half a day of agent time per morning, over several nights

## Recipe (pre-registered)

- Start from the latest adopted offline model (C13), or the SFT model if none
  was adopted.
- Prompts: C14's pool.
- Group size: as C14 fitted, preferably 8. Temperature 1.0, top_p 1.0.
- Dynamic sampling: draw 1.5 times the needed prompts, drop groups whose
  rewards are all equal, and refill before each update.
- Loss: token-normalized, like the DAPO loss (`loss_type="dapo"` if the pinned
  TRL supports it, otherwise the equivalent in the driver).
- Rewards are not divided by the group standard deviation.
- Clip range 0.2 with a higher upper clip of 0.28.
- Mask truncated completions out of the loss. β = 0, so no reference model.
- LR a tenth of the SFT LR, with the same optimizer family and a 10-update
  warmup.
- Save full state at every update boundary that ends a rollout buffer.
  Partial rollout state is not resumable. The trainer stops at the night
  deadline on such a boundary and resumes the next night.

## Per night

- Evaluations and stop conditions follow R6.
- Log per update:
  - the share of zero-variance groups before and after filtering;
  - mean absolute advantage;
  - reward and `exact` rate;
  - mean completion length;
  - clip fraction.

## After each night

- Write the R6 outcomes to `status/C15.json`.
- If R6 says continue, re-enqueue.
- Publish a candidate after every 300 updates that pass R1 against the
  starting model, to `scholzmx/sepalith-2b-edit-sft/grpo-<n>/`.
- Stop the pilot when R6 stops it, or after 1,500 updates. Then summarize the
  results against the starting model on DEV75 and DEV250.

KLPO is out of scope for this pilot. Both research reviews advise against it
at small group sizes, because its token-level Monte Carlo estimate is too
noisy.
