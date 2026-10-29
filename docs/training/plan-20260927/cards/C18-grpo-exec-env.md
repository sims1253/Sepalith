# C18. GRPO on the execution-reward environment (pre-registration)

- **Where:** PC nights (00:00 to 16:40)
- **Needs:** user sign-off of this card; the smoke passing
- **Produces:** a GRPO checkpoint from arm A 803 that beats 803 on the frozen exec hold-out, or a documented stop
- **Start:** arm A checkpoint 803 (user go-ahead 2026-10-28)
- **Code:** `packages/sepalith/src/sepalith/training/grpo_exec/` (outside `exec_env/`, gate fingerprint stays `7f8e073fbedb4b9d`)

## Goal

Raise functional correctness (all hidden tests pass) of edits on R code, measured on
the frozen exec hold-out, without losing protocol validity, DEV250 or CPT knowledge.

## Evidence

- Hold-out exec eval (3,803 tasks, 630 functions, gate c631141, greedy): arm A 803
  all-pass 0.310 vs 536 0.257 (+5.3 points [4.4, 6.2], also on functions without SFT
  overlap); 1071 +2.3, 268 +1.7, arm B 684 −0.2. 803 is the best start.
- Census on the training pool (arm A 536, k = 8, T = 1.0, 1,201 tasks): the graded
  reward varies within 71% of groups; all-pass varies in 22% (short 38%, medium 4%,
  long 0.6%, rename 41%). Graded reward gives GRPO signal at every cut length.
- C15 (β = 0, 2,086 v2.1 prompts) saturated by about 450 updates (train reward 0.94),
  DEV stayed flat, CPT 2K NLL rose 0.115 nats against arm A (803 itself: +0.009 against
  536). Update 600 lost 1.1 points of exec all-pass against 536; its gain was protocol
  validity only. Lessons: a larger, harder pool, an anti-forgetting term, and gates
  that measure the target.

## Recipe

Unchanged from C15 unless noted:

- G = 8, 4 groups (32 rows) per update, 2 updates per rollout buffer, 16-row rollout
  calls, chunked log-probabilities, micro-batch 1, optimizer resident (the C14 fit).
- Sampling T = 1.0, top_p 1.0, top_k off, cap 1,024 tokens.
- Dynamic sampling: draw 1.5x, drop zero-variance groups, refill (4 rounds, 4x at most).
- DAPO token-normalised loss, clip 0.2 / 0.28, truncated completions masked, rewards not
  divided by the group std, β = 0.
- LR a tenth of SFT (hidden 1e-6, side 1e-7), aurora_mix with fresh state, 10-update
  warmup, then constant.
- **New, prompts:** the C18 exec training split. Every fifth drawn prompt comes from the C14
  TRAIN pool (legacy replay, 20%) and is scored with reward v2.1, so format, region and
  no-op behaviour keep a live reward. Advantages stay within a prompt, so the two reward
  scales never mix.
- **New, reward:** `sepalith.reward.exec.v1` at gate c631141: protocol −0.5; no edit, a
  parse failure, a broken structure or a forbidden construct 0; otherwise 0.5 x behaviour
  fraction + 0.5 x all-pass, halved for overlong edits. `exact` is all-pass.
- **New, anti-forgetting:** SFT cross-entropy replay. Each optimizer update adds
  0.1 x the token-mean NLL of 8 TRAIN rows' targets (a fixed seeded sample of 8,000 v4
  TRAIN rows, which arm A was trained on) to the DAPO loss before the step.
- **Why not KL:** a KL penalty needs a frozen 803 reference on the GPU. The bf16 reference
  takes 4,800 MiB (the trainer's measured footprint after load). The C14 probe peak plus
  the reference is 23,110 + 4,800 = 27,910 MiB. The fit limit is 24,899 MiB (32,579 total,
  6,144 desktop budget, 1,536 margin), so the reference misses it by 3,011 MiB. The C15
  pilot's measured peak plus the reference is worse: 27,452 + 4,800 = 32,252 MiB allocated,
  above the 30,669 MiB allocator cap. The replay term needs no extra memory, because its
  rows are no longer than GRPO rows. It is SFT-shaped, and SFT kept CPT NLL flat
  (+0.009 from 536 to 803), while β = 0 GRPO drifted (+0.115).
- Saves, the deadline stop, graceful stops and resume are C15's: full state at
  buffer boundaries, retain-all (R7: 150 GB floor).

## Pool

Built from the gated training functions of S000 to S017. A function is used only if its
`gate.json` passed at c631141 (fingerprint 7f8e073fbedb4b9d), is newer than its inputs,
has no stale equivalents and no SKIP, and is neither a hold-out function nor from a
hold-out package. 1,466 functions qualified (278 SKIP, 44 gated at another version,
10 failed, 2 stale). The task builder is the exec-eval one: finish_block cuts at every
depth with TRAIN prompt variants, TRAIN-geometry renames, and per-cut observability,
local-mutant and inferability gates. Cuts touching a branch that a function's
`equivalent.txt` declares untested (SCOPED/BUGFIX: 238 declarations in 83 functions)
are dropped. Validation takes whole packages, in seeded hash order, until it holds
10% of functions. Per function: at most 8 training tasks and 3 validation tasks,
round-robin over (family, cut length).

Built: 15,659 usable tasks from 1,416 functions (50 functions yielded no task; 12,004
cuts failed inferability, 419 touched an untested branch). Manifest:
`c18/C18-exec-pool.json` (sha256 `8cc19940`).

| Split | Tasks | Functions | Packages | finish_block short / medium / long | rename |
| --- | --- | --- | --- | --- | --- |
| Train | 7,752 | 1,274 | 141 | 3,264 / 2,111 / 623 (functions 1,112 / 810 / 272) | 1,754 (1,052 functions) |
| Validation | 403 | 142 | 16 | 163 / 115 / 66 (functions 133 / 105 / 62) | 59 (55 functions) |

Training prompts average 424 tokens (max 1,943) and gold completions 74 (max 913).
Legacy replay uses the C14 pool: 2,086 TRAIN prompts, reward v2.1. SFT replay draws on
8,000 TRAIN rows (target 104 tokens on average, prompt 809).

Zero-variance tasks are left to dynamic sampling. A k = 8 census of the whole pool
would cost about 22 GPU-hours, and the existing census covers only 11% (887 tasks; 31% of those had zero reward variance at 536) of the pool.

## In-run gates (R6x)

Every 50 updates: DEV75, the validation split (greedy, exec reward) and the CPT 2K
probe (`c09` panel and evaluator, plain bf16 load as the trainer). DEV250 runs too
every 150 updates. Once, before training, the driver measures 803's validation result
and CPT NLL with the same evaluators. The worst row wins. Missing inputs or an
evaluator crash mean stop-for-user.

| Row (reference) | Continue | Stop for user | Rollback |
| --- | --- | --- | --- |
| DEV75 edit drop / nofp rise / valid drop / cap (previous accepted, as R6) | 0-2 / 0-1 / 0-1 / 0-1 | 3-7 / 2-3 / 2-4 / 2-4 | ≥ 8 / ≥ 4 / ≥ 5 or valid < 68 / ≥ 5 |
| Validation all-pass drop (803) | < 2 points | 2 to < 4 | ≥ 4 |
| Validation mean exec reward drop (803) | < 0.02 | 0.02 to < 0.04 | ≥ 0.04 |
| CPT 2K NLL rise (803, same harness) | ≤ 0.015 nats | to 0.03 | > 0.03 |
| DEV250 score250 drop (803's 109; every 150) | ≤ 4 | 5-9 | ≥ 10 |
| Zero-variance share after filtering > 50% at 3 evaluations in a row | no | yes | |
| Mean completion length beyond x2 or /2 of updates 1-10 | no | yes | |

An evaluation that continues (by rule or by override) is *accepted*. The frozen hold-out
is never used in-run.

## Stop rules

- Rollback ends the pilot. Stop-for-user exits 3 (`override-step-<n>.json`, then
  `night_queue retry`). R7 disk is stop-for-user.
- The pilot is complete at update 1,000. It also ends at a plateau: none of the last
  6 accepted evaluations (300 updates) beats the earlier best validation all-pass,
  803 included, by 1 point.
- At the end, the driver selects the accepted checkpoint with the highest validation
  all-pass (ties: higher validation reward, then the earlier step). It runs DEV250 on
  that checkpoint if it has none, then writes `pilot-summary.json` and
  `final-holdout-job.json`.

## Final evaluation (pre-registered)

The selected checkpoint against 803 on the frozen hold-out (`holdout-freeze-v1`, tasks
sha256 `4d9a1a36`), run as one job:

- greedy decoding with the DEV gate settings, scored with the exec reward at c631141;
- 803's pinned completions are reused;
- differences are paired by task, with a function-clustered bootstrap (2,000 resamples).

- **Primary:** the all-pass difference has a 95% CI lower bound > 0.
- **Secondary:** the differences in mean exec reward and exact match, protocol validity,
  DEV250 score250 ≥ 105 (803's 109 minus 4), and a CPT 2K rise ≤ 0.03 nats.
- **Adopt** only if the primary endpoint holds, DEV250 is non-inferior and the CPT rise
  is within the limit. Verdict: `python -m sepalith.training.grpo_exec.jobs verdict`.

## Jobs and cost

- `C18-grpo-exec-smoke` (about 40 minutes, not resumable): updates 1-4, save, then
  DEV75 + 96 validation tasks + CPT on checkpoint 4 and the 803 CPT reference, then
  resume to 6 and save. The receipt has seconds per update, projected updates per
  night, peak memory and headroom, the replay NLL and both gradient norms.
- `C18-grpo-exec-pilot-1` (resumable, requires the smoke).
- Cost per night: the whole 16.7-hour GPU window. The projection is about 100 s per
  update: exec completions average 107 sampled tokens against C15's 40, and rollouts
  are about 2.5x C15's 62 s per buffer. Gates take about 22 minutes per 50 updates
  (save 4, DEV75 4, validation about 9, CPT 2, restart 3), plus DEV250 every 150.
  That gives **about 430 to 480 updates per night**. 1,000 updates take about 2.2
  nights; a plateau would end the pilot sooner. The smoke replaces this projection with
  measured numbers.
- Disk: 17.7 GB per save. 20 evaluation saves plus about 3 deadline saves come to about
  410 GB, against 456 GB usable on /mnt/e (606 GB free, 150 GB floor).

## Risks

- **Reward hacking through partial credit.** The overlong scaling and forbidden
  constructs limit it. Selection and the gates use validation all-pass, not training
  reward, and the hold-out decides.
- **Validation noise.** About 403 tasks; a paired difference has an SE of about
  1.5 to 2 points, so the bands sit at 2 and 4 points. Validation packages are training
  slices, so the hold-out (separate packages) is the unbiased number.
- **The replay coefficient** may be too weak, which the CPT row catches, or too strong,
  which shows as small gains. The smoke logs both gradient norms.
- **Memory.** C15 peaked at 28.4 GB reserved, and exec rows are longer. The allocator cap
  turns an overflow into an OOM, not a sysmem fallback, and the smoke reports headroom.
- **Disk.** Retain-all leaves little slack at 1,000 updates (see above).
- The legacy pool saturated under C15 and its groups may turn zero-variance. They are then
  dropped and only act as a guard. That is intended.

## Addendum C18b: CPT-text replay (pre-registered 2026-10-29, not launched)

**C18 outcome.** At update 100, R6x stopped for the user on the CPT 2K rise. The rise was
+0.0117 at update 50 and +0.0238 at 100, and the forgetting was uniform: 485 of 499 rows and all
50 packages got worse. Validation all-pass went 0.283 → 0.298 → 0.305, validation reward
0.396 → 0.424, and DEV75 did not change (edit 28, valid 75, cap 0). The user ended the pilot with
checkpoint 100 as the C18 result. That checkpoint gets the pre-registered hold-out evaluation and
DEV250 (`grpo_exec/final.py`). The selection was the user's, not the plateau rule.

**Why not a stronger SFT replay.** The SFT replay NLL did not rise: 0.215 at updates 1-10, 0.214,
0.263, 0.203, then 0.176 at updates 76-100. The term held its own distribution while raw R code
drifted. It already contributed a sizeable gradient: the norm went from 0.99 for GRPO alone to 1.17
with replay. SFT itself moved CPT NLL +0.052 from the CPT parent to arm A 536. Raising its weight
to 0.3 or 0.5 would scale a gradient that points at the wrong distribution.

**Change (one variable).** Everything in C18 stays the same, including SFT replay 0.1 x 8 rows,
LR 1e-6 and the R6x bands. C18b adds this term before the optimizer step:

    0.2 x token-mean NLL of 4 CPT training chunks per update

The chunks come from a seeded sample of 8,000 rows drawn from 3,418 packages of the `cpt_train`
union. Each chunk is at most 2,048 tokens, with 1,382 supervised tokens on average. Hold-out-slice
packages and the 50 CPT 2K validation packages are excluded, so the probe still measures held-out
code.

- **Coefficient.** 0.2 gives the CPT term an effective learning rate of 2e-7, a fifteenth of the CPT
  run's 3e-6. Its gradient should be comparable to the GRPO gradient: SFT replay at 0.1 added a
  component of about 0.6 against GRPO's 1.0.
- **Cost.** About 5,500 tokens of extra forward and backward per update, roughly +7 s on C18's 70 s,
  so about 400-440 updates per night.
- **Memory.** C18 peaked at 23.2 GB reserved, and the chunks are shorter than the 4,096-token GRPO
  bound.

**Start: arm A 803 again, not C18 checkpoint 100.**
- Checkpoint 100 has already used 0.024 of the 0.03 CPT budget.
- It is the C18 result under its own final evaluation.
- Restarting from 803 makes C18 a matched control at updates 50 and 100.

**Bands and endpoints.** Identical to C18: R6x, the stop rules, selection and the final hold-out
evaluation against 803. A descriptive expectation is also pre-registered: a CPT rise of at most 0.006
at update 50 (C18: 0.0117), with the validation all-pass at update 100 within 1 point of C18's 0.305.
If the rise at update 50 is still above 0.009, the term is too weak. The next step would then be the
coefficient (0.4), not the SFT weight.

**Code.** `grpo_exec/c18b.py` (recipe module, CPT rows, `install()`), `c18b_trainer.py`
(`CptReplayTrainer`), and entry modules `c18b_{night,val_eval,cpt_probe,dev_generation}.py`. No C18
file changed, so the C18 recipes still verify. Recipes are in `c18b/bound/` and jobs in
`c18b/jobs/` (the smoke, about 45 minutes, then the pilot).
