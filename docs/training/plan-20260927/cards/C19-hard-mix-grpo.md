# C19: GRPO on a hard-task-weighted exec mix

User go-ahead, 2026-11-02: "sounds good. Prepare that for tonight".
Prepared for 2026-11-03 00:00 Europe/Berlin. Jobs are written, never enqueued
by this card's preparation. No GPU smoke was run.

## Why

The 1,201-task paired census from SFT 536 to C18b 300 improved mean pass
0.187 to 0.331 and all-right 7.7% to 26.6%, but all-wrong only fell from
70.0% to 61.0%. Pass-signal share fell from 22.3% to 12.5%.
rename_propagation pass rose from 0.38 to 0.95.

Across 4,725 tasks so far, finish_block all-wrong shares by target lines are
16% at 2, 38% at 3-4, 64% at 5-8, 89% at 9-16 and 98% at 17+.
Among all-wrong finish_block tasks, 70-79% still have reward variance
across samples from behaviour partial credit. Of tasks all-wrong at 536,
11.7% of short tasks pass at least one of eight at 300, versus 2.3% medium
and 0% long. Mean behaviour rose .28 to .35 short, .19 to .23 medium,
and .10 to .13 long. Checkpoint 300 hold-out all-pass is 0.392.

## Frozen training mix

The [manifest](../c19/mix-manifest.json) pins both partial census inputs at
freeze time. The paused pool census may append after midnight; the frozen
mix never reads the live census during training or verification.

Only task ids in C18b's bound C18 training file are eligible. Exactly eight
samples, numbered 0 through 7, are required. Exclude frozen hold-out functions
and packages and validation functions using the pool identity helpers.
357 census tasks outside the training pool were excluded.

- HARD: finish_block, zero of eight pass, target lines 3 through 16.
- MIXED: one through seven pass, any family, target lines at most 16.
- SOLVED: eight pass, any family, subject to the global exclusion of 17+ lines.

Keep every HARD task. Draw round(N * 25/60) MIXED and round(N * 15/60)
SOLVED with seed 3407 over sorted task ids. Fail if a stratum is short.
The frozen result is 1,375 HARD, 573 MIXED and 344 SOLVED, 2,292 tasks
from 868 functions. Original training-file row bytes are retained. ExecReward
loads all rows. MixedSampler walks uniformly per epoch. The 1-in-5 legacy
pattern stays on top; at 32 rows per update and group size eight, about
3.2 exec tasks are drawn per update before dynamic filtering/refill.

## Design and references

Everything follows [C18b](C18-grpo-exec-env.md#addendum-c18b-cpt-text-replay-pre-registered-2026-10-29-not-launched)
except the training mix and start. Start from
`/mnt/e/sepalith/campaign-20260915/checkpoints/C18b-grpo-exec-pilot-1/full/checkpoint-300`,
weights only, with a fresh optimizer and sampler. The checkpoint's files,
DEV75, DEV250, validation and CPT results at step 300 are SHA256 pinned in
both C19 recipes. No 803 reference or stale step-200 final data is used.
The C19 driver reuses validation and CPT results directly and the inherited
DEV reader checks the recorded counts. Smoke independently measures the
start CPT with the same evaluator, as C18b smoke does.

Exec reward remains at c631141, fingerprint 7f8e073fbedb4b9d. SFT replay
remains 0.1 times eight rows; CPT replay remains 0.2 times four chunks.
LR stays 1e-6. Replay rows, legacy pool, validation and CPT panel are unchanged.

## Gates, selection and stop conditions

R6x bands and worst-row precedence remain exactly C18b's. DEV75, validation
and CPT run every 50 updates; DEV250 every 150. All start-relative rows now
reference checkpoint 300's own metrics. R6x length_factor compares against
the first ten updates of C19, on this same mix, so the longer mixture does
not itself trip a comparison against C18b's old completion lengths.

Rollback ends the pilot. Stop-for-user exits 3. Missing inputs, wrong hashes,
evaluator crashes, non-finite losses and R7 disk failures stop. Complete at
1,000 updates or the unchanged plateau rule: none of the last six accepted
evaluations beats the earlier best validation all-pass by one point.
Select accepted checkpoints by validation all-pass, then reward, then earliest.
The hold-out never enters training, selection or in-run gates.

The standing C18 nofp/cap auto-continue rule, recorded in status/C18.json,
applies only to stop-for-user with only nofp_rise and/or cap in the stop
band, no rollback, and all other rows continuing. This card does not widen
that rule or implement a broader automatic override.

Every validation evaluation writes a C19 evidence sidecar with all-pass and
mean reward by short/medium/long bucket, family and target-line bins
1, 2, 3-4, 5-8, 9-16 and 17+. These reports never affect selection or gates.
The selected validation breakdown is included in the closeout summary.

## Final endpoints

Run the selected C19 checkpoint against checkpoint 300 on the unchanged
frozen hold-out tasks and SHA256. Job id `C19-final-holdout`; labels
`c18b-300` and `c19-selected`. Reuse pinned checkpoint-300 completions from
`C18b-grpo-exec-pilot-1/final-step-300/completions-c18-selected.jsonl`.
Never recompute those completions and never use the stale `final/` directory.
Each selection uses `final-step-<selected>/`, with model inputs checked on
resume. Differences are paired by task and function-clustered, 2,000 bootstrap
resamples. Primary: all-pass difference CI95 lower bound > 0. Secondaries:
mean reward and exact-match differences, protocol validity, DEV250 score250
at least 300's score minus four, and CPT NLL rise from 300 at most 0.03.
Adopt only when primary, DEV250 non-inferiority and CPT all pass.
Verdict uses `python -m sepalith.training.grpo_exec.c19_jobs verdict`.

## Deployment and resources

The PR is unmerged. Bind repository_root, cwd and PYTHONPATH to this absolute
worktree, with interpreter `/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python`.
C19 chains install() over C18b; no C18 or C18b source or bound file changes.
The source manifest pins source bytes from the code commit; recipes pin its
hash, data files, references and checkpoint files. A later source edit requires
rebinding. The follow-up binding commit contains only generated artifacts.

`C19-grpo-exec-smoke`: priority 40, not resumable, about 45 minutes;
updates 1-4, save, gates, resume to 6. `C19-grpo-exec-pilot-1`: priority 45,
resumable, requires smoke, about 960 minutes. Archives and evidence are distinct.

Memory uses C18b's fit receipt, same model, buffers and token caps:
prompt 3,072, completion 1,024. This is a subset of the already admitted pool;
no prompt cap is expanded. HARD tasks are longer and mean completion length
is expected to rise; actual generated lengths and peak memory remain unmeasured
until the required smoke. The receipt is a fit justification, not a new smoke pass.

At the stated 522 GB free, 20 saves of 17.7 GB consume 354 GB, leaving
168 GB; 21 leave 150.3 GB. 22 saves leave 132.6 GB and cannot fit above the
150 GB floor. The common 20 evaluation saves plus three deadline saves do
not fit that conservative budget. Actual free space is recorded in the readiness
receipt; no cleanup or floor change is authorized. R7 may stop a long run.

R binds `/opt/R/4.6.1/bin/Rscript`, parser identity `R 4.6.1 (2026-06-24)`.
Jobs prepend `/opt/R/4.6.1/bin` to PATH for ExecHarness and all children.
Every C19 entry checks the exact Rscript version line before launching.
The rig installation is absent at preparation time and requires the user's sudo
password. Enqueueing waits on that installation and a successful strict verify.
No fallback to /usr/bin and no installation attempt is made here.
The exec fingerprint hashes source bytes only, not the R binary or build.
Parser identity uses a version string; this does not establish behavioural
parity between pacman and rig builds. That uncertainty remains for the smoke.

## Risks

Partial-credit hacking may increase reward on HARD tasks while few pass.
R6x all-pass and final endpoints constrain that risk. Rename and short-task
forgetting can hide under overall R6x metrics; subgroup reports are reporting
only. Selecting all-wrong tasks invites regression to the mean. Validation
noise has SE about 1.5-2 points. Longer completions can raise rollout cost
and memory use; the mandatory smoke must measure these before pilot admission.
