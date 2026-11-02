# C18b results: CPT-text replay — verdict ADOPT (2026-10-30)

Pre-registered final evaluation of the
[C18b addendum](../cards/C18-grpo-exec-env.md#addendum-c18b-cpt-text-replay-pre-registered-2026-10-29-not-launched)
of card [C18](../cards/C18-grpo-exec-env.md), completed 2026-10-30. Status:
[`status/C18.json`](../status/C18.json) (rule decision "final verdict C18b").

## Pilot end

`C18b-grpo-exec-pilot-1` (recipe sha256 `2cef84e4`, from arm A 803 — the same
start as C18, a matched control) ended by user stop at update 200: R6x stopped
for the user on nofp_rise 4 (DEV75 nofp 9 → 13, +2 in the last 50 updates), the
rollback band (≥ 4); the standing nofp/cap auto-continue rule covers only the
stop band. The gates at updates 50/100/150 all continue, and the CPT 2K NLL was
below the start value at every gate (start 1.0033; 0.9892 / 0.9820 / 0.9798):
the CPT-text replay fully controlled the forgetting C18 was stopped for (C18
rise +0.0238 at update 100; C18b −0.0272 at 200). Checkpoint 200 was selected
by that user decision (`user-stop-step-200.json`), not the pre-registered
best-validation or plateau rule; the pilot stays resumable. Recorded in
`/mnt/e/sepalith/campaign-20260915/checkpoints/C18b-grpo-exec-pilot-1/pilot-summary.json`
(schema `sepalith.c18.grpo-exec-pilot-summary.v1`).

## Final hold-out evaluation

`C18b-final-holdout`, one job, run 2026-10-30T10:54-12:41+0100: frozen hold-out
`holdout-freeze-v1` (tasks sha256 `4d9a1a36`), 3,803 tasks from 630 functions,
gate pinned at `c631141` (fingerprint `7f8e073fbedb4b9d`), greedy decoding,
exec reward, 803's pinned completions reused, differences paired by task with a
function-clustered bootstrap (2,000 resamples).

| Model | Mean reward (CI95) | All-pass (CI95) | Exact (CI95) | Protocol valid |
| --- | --- | --- | --- | --- |
| arm-a-803 | 0.379 [0.359, 0.400] | 0.310 [0.291, 0.329] | 0.228 [0.212, 0.243] | 0.973 |
| c18b-selected (checkpoint 200) | 0.444 [0.424, 0.465] | 0.381 [0.361, 0.401] | 0.300 [0.285, 0.317] | 0.963 |

(The verdict tool keeps the C18-era candidate label `c18-selected`; here that
model is the C18b pilot's checkpoint 200.)

Receipts:
`/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates/C18b-grpo-exec-pilot-1/final/receipt.json`
and `receipt.md` (same directory).

DEV250 for checkpoint 200, job `C18b-final-dev250-step-200` (entry module
`grpo_exec/c18b_final`, PR #96): score250 **110** (803: 109, bar ≥ 105).

## Verdict

`python -m sepalith.training.grpo_exec.jobs verdict`: **adopt = true**.

- **Primary (met):** hold-out all-pass difference +0.0707, CI95 [0.0625, 0.0794]
  (paired by task, function-clustered bootstrap, 2,000 resamples).
- **Secondaries (all pass):** mean reward difference +0.0648 [0.0564, 0.0739];
  exact match difference +0.0726 [0.0643, 0.0817]; DEV250 score250 110 >= 105
  (start 109, bar = 109 minus 4); CPT 2K rise −0.0272 <= 0.03 nats (below the
  start value).

## Consequence

C18b checkpoint 200
(`/mnt/e/sepalith/campaign-20260915/checkpoints/C18b-grpo-exec-pilot-1/full/checkpoint-200`)
is **adopted as the campaign's best model**, superseding C18 checkpoint 100
(which superseded arm A 803). Ladder on the same frozen hold-out, all-pass:

| Model | Hold-out all-pass |
| --- | --- |
| arm A 803 | 0.310 |
| C18 checkpoint 100 | 0.355 |
| C18b checkpoint 200 | 0.381 |

## Open signal

DEV75 nofp (false edits on no-op cases) rose 9 → 13 over 200 updates (rollback
band at 200) while everything else improved; the CPT replay held the CPT NLL
below start throughout. Next-step design is not part of this note.

## Update 2026-11-02: checkpoint 300 supersedes checkpoint 200

By user decision the pilot resumed from update 200. It ended on the plateau rule at update 500, with
every gate from 250 to 500 at R6x continue. Validation all-pass selected checkpoint 300 (0.3251).

Job `C18b-final-holdout-step-300` ran in a fresh work dir, because `final/` still held checkpoint
200's completions. The checkpoint 200 results in `final/` and `final-step-200/` are unchanged.

| Model | Hold-out all-pass | Exact | Mean reward | Protocol valid |
| --- | --- | --- | --- | --- |
| arm A 803 | 0.310 | 0.228 | 0.379 | 0.973 |
| C18 checkpoint 100 | 0.355 | | | |
| C18b checkpoint 200 | 0.381 | 0.300 | 0.444 | 0.963 |
| C18b checkpoint 300 | 0.392 | 0.311 | 0.463 | 0.973 |

**Verdict: adopt = true.** All-pass difference vs 803 +0.0818, CI95 [0.0726, 0.0913]; mean reward
+0.0836 [0.0744, 0.0933]; exact +0.0836 [0.0749, 0.0926]; DEV250 111 >= 105; CPT 2K rise -0.0314.

Checkpoint 300 vs 200, paired and function-clustered (descriptive, not pre-registered): all-pass
+0.011 [0.0058, 0.0163], reward +0.0187 [0.0126, 0.0244], protocol valid +0.0105 [0.0051, 0.0162].

**Quant gate** (`C18b-quant-step-300`, CUDA build, summary in
`receipts/C18b-quant-gate-summary.json`): Q6_K (2.20 GB) is the only tier that passes. Q8_0 fails
exact (-1.2 pp) and nofp (+0.8 pp). The arm A pick mixA-e4 fails exact, nofp and valid.

**Release tiers (user decision 2026-11-02):** Q6_K as default, and Q4_K_M+ (1.78 GB) as a user
exception for its single miss, nofp +1.2 pp vs the 1.0 limit. Publishing still needs the user's
go-ahead.

Next-step design remains out of scope for this note.
