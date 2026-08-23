# The coding simulator — design (user idea, 2026-08-23)

> "Rather than actually using vscode to gather data, we can build a state
> machine-ish thing that can produce the same output that the extension
> would get from vscode in a realistic way (typing, pausing, typos,
> clicking into another line/function, waiting to think, accepting or
> rejecting ... a complete work stream). Rather than looking at one
> block, we could take edit pairs from commits and simulate someone
> developing that commit and getting multiple suggestions throughout."

## Why this is the right next environment

1. **The product metric is per-workstream, not per-cursor-position.**
   The FP-annoyance number that matters is "false suggestions per
   developed commit/hour", which no isolated-block eval can measure.
2. **The v8 lesson**: SFT families with contradictory targets on the
   same contexts fight each other (propose-vs-stop at statement
   boundaries, 20:1 mass). In a work stream the discrimination is
   SEQUENTIAL — the same cursor position after 10 s of typing means
   something different than after 90 s of idle. Episode-level training
   can express what mixture data cannot.
3. **It generates the missing distributions natively**: pauses,
   mid-typing cursors, navigation clicks, delete-and-retype sequences —
   exactly the geometries where v7/v8 propose 100%.

## The state machine

Discrete-event timeline; states emit the same events the extension sees
(document changes, selection changes, quiesce points).

```
Typing(line, col) ──key──▶ Typing ──quiesce(debounce)──▶ [SUGGESTION POINT]
   │ typo+backspace (rate τ)      │
   │ line done                    ▼
   ▼                          Pausing(d ~ lognormal)
LineDone ──next line──▶ Typing   │ short → keep typing
   │ EOL + think                  │ long → Navigating / Thinking
   ▼                              ▼
Navigating(row) ──click──▶ [NO-OP CURSOR GEOMETRY]  (interior line ends,
   mid-identifier, blank lines, refactorable lines — labeled by construction)
Reviewing(diff) ──accept/reject prior suggestion──▶ Typing
```

### Trajectory generator (per commit)

1. Input: an edit pair (before, after) — sources already in hand:
   CRAN Archive version diffs (~0.15B diff tokens, the edit-diff stratum),
   edit_pairs_v1, removed_block corpus (delete-and-retype events).
2. Reconstruct a plausible typing path for each hunk: type new lines
   left-to-right; typo rate τ (insert/transpose/backspace corrections);
   between-hunk behavior drawn from the state mix (navigate-to-context,
   pause-to-think, re-read-old-code — these produce the no-op geometries
   WITH their true labels: nothing was warranted there, by construction).
3. At every quiesce point (the extension's debounce), the simulator
   records the exact prompt the extension would build (byte-faithful
   port — eval_noop_fp.py already carries it) + ground truth = the next
   chunk of the reconstructed path.
4. Label per suggestion point, by construction: ACCEPTABLE (the next
   typed chunk), COMPETING (they typed something else — model output
   scored against the actual next tokens), NO-OP (nothing typed within
   the action window).

### The acceptance policy (the one modeled component)

Someone decides accept/reject. Stage 1: a simple stochastic rule —
accept iff the suggestion prefix-matches the next k≥1 typed tokens
(exact product semantics: ghost text is accepted word-by-word);
dismiss otherwise; long-pause implies ignore. Stage 2 (only if stage 1
shows signal): a small learned acceptance model. The sim-to-real risk
lives here AND in the timing distributions — mitigations: parameter
sensitivity analysis on τ/state-mix, calibration against the extension's
opt-in telemetry counters (shown/accepted already instrumented), and
always reporting RELATIVE comparisons (v7 vs candidate on the same
simulator, same seed).

### Judged acceptance + summary-network RL (user synthesis, 2026-08-23)

> "That rl thing would also work well to RL the summary network — we'd
> work a lot in full packages/projects starting from the parent commit
> and have a judge accept or reject proposals based on the shape/goal of
> the commit, rather than perfect literal match."

This upgrades the acceptance policy from prefix-match to JUDGED
acceptance, and makes the simulator the RL environment for the
workspace-level summary network (the Q2 lane):

- **Trajectory scope**: full packages from the PARENT COMMIT (not single
  hunks) — the normalized tree's version transitions give before/after
  states for thousands of packages; the simulated developer works
  through the whole transition.
- **Acceptance = judge vs the goal card**: each transition gets a GOAL
  CARD (what the developer was trying to achieve — corpus being mined
  now via gpt-5.6-sol, `mine_commit_goals.py`); a judge scores each
  proposal by "does this advance the goal / match the change shape",
  NOT literal diff equality. This finally rewards the multimodality the
  posterior view identified: many different spans can be correct.
- **Summary network RL**: the workspace state (parent-commit sources
  across files) is exactly the input the summary network must compress;
  the judged-accept/accept-reject signal over the work stream is its
  reward. Train it jointly with the trunk (the SBI lesson) inside the
  simulator loop — cheap encoder, cross-attention adapter, gradients
  from episode return.
- Literal-match reward stays as a CALIBRATION anchor (the exact/valid
  metrics) so the judge can't drift unboundedly: judge scores are
  periodically re-anchored on transitions where ground truth is known.

Sequencing: goal-card corpus (today) → stage-1 simulator with
prefix-match acceptance → judge upgrade → summary-network arm last
(it needs the simulator + judge both working first).

## RL integration — two stages

- **Stage 1 (offline trajectories, works with the current GRPO
  trainer)**: the simulator emits prompt/reward streams; rewards =
  accepted-suggestion value − false-suggestion penalty (NO-OP points
  reward only the empty span — this is the no-op arm with the correct
  sequential context) − repetition penalty. No trainer surgery.
- **Stage 2 (closed loop)**: model outputs influence the simulated
  developer (accepted text enters the document; the path branches).
  Needs a custom generation loop — build only if stage 1 moves the
  FP metric.

## Episode metrics (the new battery member)

Per simulated hour: suggestions shown, acceptance rate, false
suggestions (NO-OP points where the model proposed), keystrokes saved
(accepted tokens / total typed), interruption score (proposals during
Thinking/Navigating). Report alongside eval_noop_fp — the simulator is
the workstream-level view of the same discipline.

## Sizing

A 20-hunk commit ≈ 300-800 suggestion points ≈ 15-40 min of sim time;
1k commits ≈ the eval battery scale in an afternoon of CPU+one model
server. The trajectory generator is pure CPU; only the suggestion
scoring needs the model.

## Status

Design (this doc). Implementation order after RL-run-2 lands: trajectory
generator + label export → offline dataset build → episode metrics for
v7 (baseline) → stage-1 RL arm.
