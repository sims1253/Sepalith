# C18 results: GRPO on the execution-reward environment — verdict ADOPT (2026-10-29)

Pre-registered final evaluation of card
[C18](../cards/C18-grpo-exec-env.md), completed 2026-10-29. Status:
[`status/C18.json`](../status/C18.json) (rule decision "final verdict").

## Pilot end

`C18-grpo-exec-pilot-1` ended by user stop at update 100: the CPT 2K NLL rise
was 0.0238 nats, inside the 0.015-0.03 stop-for-user band (the standing
nofp/cap auto-continue rule did not apply). Checkpoint 100 was selected by
that user decision, not the pre-registered best-validation or plateau rule.
Recorded in
`/mnt/e/sepalith/campaign-20260915/checkpoints/C18-grpo-exec-pilot-1/pilot-summary.json`
(schema `sepalith.c18.grpo-exec-pilot-summary.v1`).

## Final hold-out evaluation

`C18-final-holdout`, one job, run 2026-10-29T10:54-11:54+0100: frozen hold-out
`holdout-freeze-v1` (tasks sha256 `4d9a1a36`), 3,803 tasks from 630 functions,
gate pinned at `c631141` (fingerprint `7f8e073fbedb4b9d`), greedy decoding,
exec reward, 803's pinned completions reused, differences paired by task with
a function-clustered bootstrap (2,000 resamples).

| Model | Mean reward (CI95) | All-pass (CI95) | Exact (CI95) | Protocol valid |
| --- | --- | --- | --- | --- |
| arm-a-803 | 0.379 [0.359, 0.400] | 0.310 [0.291, 0.329] | 0.228 [0.212, 0.243] | 0.973 |
| c18-selected (checkpoint 100) | 0.436 [0.417, 0.455] | 0.355 [0.336, 0.374] | 0.276 [0.260, 0.291] | 0.990 |

Receipts:
`/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates/C18-grpo-exec-pilot-1/final/receipt.json`
and `receipt.md` (same directory).

## Verdict

`python -m sepalith.training.grpo_exec.jobs verdict` (tool merged in PR #93):
**adopt = true**.

- **Primary (met):** hold-out all-pass difference +0.0455, CI95 [0.038, 0.0528].
- **Secondaries (all pass):** mean reward difference +0.0568 [0.0486, 0.065];
  exact match difference +0.0481 [0.0411, 0.0554]; DEV250 score250 105 >= 105
  (start 109, bar = 109 minus 4); CPT 2K rise 0.0238 <= 0.03 nats.

## Consequence

C18 checkpoint 100
(`/mnt/e/sepalith/campaign-20260915/checkpoints/C18-grpo-exec-pilot-1/full/checkpoint-100`)
is **adopted over arm A 803** as the campaign's best model.

Follow-up C18b (CPT-text replay against the measured forgetting) is
pre-registered in PR #94, not launched.
