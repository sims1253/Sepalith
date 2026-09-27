# Post-CPT plan: editing SFT, offline preference training, optional RL

Written September 27, 2026. This plan replaces the SFT-11, RL-11 and RL-12
preparations where they conflict. The [audit](AUDIT.md) records what was kept,
what was corrected and why.

The work is split into cards. Each card is a self-contained hand-off for one
Claude Code agent. No single session has to stay open across days. Decisions
between steps follow [pre-registered rules](RULES.md), so agents can continue
without waiting for the user. Borderline results stop and ask.

## Decisions already made by the user

- CPT checkpoint 11,586 is the parent for editing SFT.
- Compute: free resources plus the PC's nightly window only. The window is
  01:00 to 09:00 Europe/Berlin, and no paid compute is allowed. Prefer free
  Kaggle or Colab resources where they work.
- SFT gates continue automatically when pre-registered rules pass. Borderline
  results stop for the user.
- All code is tracked in the public GitHub repository. Data and weights go to
  public Hugging Face repositories. The sealed final set is never uploaded.
- Claude Code agents execute the cards.

## What runs where

| Work | Where | Why |
| --- | --- | --- |
| Full-weight training: SFT arms, RFT, DPO, GRPO | PC, nightly window | Needs bf16 and about 22 GB of static state. Kaggle T4s cannot run the reviewed recipe. |
| Gate evaluations during training (DEV75, DEV250) | PC, same night | Stays bf16 and authoritative, and lets gates continue without a day's delay. |
| Pass-rate census, sample generation, extra evaluations | Kaggle T4 (free, 30 GPU-h/week) | Inference only. Needs a one-time fp16 parity check against the PC. |
| Data preparation, reward code, tests, consolidation | PC during the day at low priority; Kaggle CPU for big batches | CPU only. |
| GGUF latency checks | Notebook | Close to the analyst laptops the product targets. |

Colab's free tier offers the same T4 class as Kaggle with less predictable
limits. Use it only as a fallback for Kaggle inference jobs.

## Order of work

```
C00 secure artifacts ─┐
C01 consolidate ──────┼─> C04 SFT recipe v2 ─┐
C02 night window ─────┤                      ├─> C07 night 1 ─> C08 night 2 ─> C09 night 3 ─┐
C05 evaluation ───────┘                      │   baseline       LR pilot        arm A done   │
C06 finish check ────────────────────────────┘                                               │
C03 Kaggle harness ─────────────────────────────────────────────────────────────────────────┤
C11 reward v2.1 ────────────────────────────────────────────────────────────────────────────┤
C10 roxygen arm B (data now, nights after C09) ──> R3 picks the SFT model ──────────────────┤
                                                                                             v
                                   C12 census (Kaggle) ─> C13 RFT and DPO ─> C14 GRPO gate ─> C15 GRPO pilot
C16 HF dataset re-sync (independent)            C17 release preparation (user-gated, later)
```

Cards on the same line can run in parallel. Example calendar, assuming about
one agent-day per preparation card:

| When | PC night | Day work |
| --- | --- | --- |
| Sep 28 to 29 | none | C01, C02, C03, C05, C06, C11, C16 in parallel; C10 starts |
| Sep 29 to 30 | none, or the C02 dry run | C04 |
| Night to Sep 30 | C07: baseline, parity references, SFT smoke | C03 parity check |
| Night to Oct 1 | C08: learning-rate pilot | Selection by R2 |
| Night to Oct 2 | C09: arm A to completion | Export, publish, notebook latency |
| Oct 2 to 3 | none | C12 census on Kaggle; C13 data build |
| Nights from Oct 3 | C10 arm B, C13 RFT and DPO | Evaluations and decisions by R3 to R5 |
| From about Oct 7 | C14, C15 only if R4 allows | |

## Cards

| Card | Title | Needs | GPU |
| --- | --- | --- | --- |
| [C00](cards/C00-secure-artifacts.md) | Secure code, data and weights | none | no |
| [C01](cards/C01-consolidate-runtime.md) | Consolidate the full-weight runtime into `packages/sepalith` | C00 | no |
| [C02](cards/C02-night-window.md) | Nightly GPU window: queue, runner, scheduler | C00 | dry run only |
| [C03](cards/C03-kaggle-inference.md) | Kaggle inference harness and fp16 parity | C01, C07 for parity | Kaggle |
| [C04](cards/C04-sft-recipe-v2.md) | Editing-SFT recipe v2: schedule, gates, deadline stop | C01 | no |
| [C05](cards/C05-evaluation.md) | DEV250, CPT-overlap check, portable evaluator | C01 | no |
| [C06](cards/C06-finish-repair-check.md) | Spot-check the repaired finish_block rows | none | no |
| [C07](cards/C07-night1-baseline.md) | Night 1: baseline, parity references, SFT smoke | C02, C04, C05 | PC night |
| [C08](cards/C08-night2-lr-pilot.md) | Night 2: learning-rate pilot | C07 | PC night |
| [C09](cards/C09-night3-arm-a.md) | Night 3: finish arm A, export, publish | C08 | PC night |
| [C10](cards/C10-roxygen-arm-b.md) | Roxygen admission and SFT arm B | C04; its nights after C09 | PC nights |
| [C11](cards/C11-reward-v21.md) | Reward v2.1 and RL length policy | C01 | no |
| [C12](cards/C12-census.md) | Pass-rate census on Kaggle | C03, C09 or C10, C11 | Kaggle |
| [C13](cards/C13-rft-dpo.md) | RFT and DPO | C12 | PC nights |
| [C14](cards/C14-grpo-gate.md) | GRPO gate: re-census and memory probe | C13 | PC night, Kaggle |
| [C15](cards/C15-grpo-pilot.md) | GRPO pilot | C14 | PC nights |
| [C16](cards/C16-hf-dataset-resync.md) | Re-sync the public HF dataset | C00 | no |
| [C17](cards/C17-release-prep.md) | Release preparation (user-gated) | the post-training decision | later |

## Operating rules for every agent

1. **Read first.** Read [CONTEXT.md](CONTEXT.md), [RULES.md](RULES.md) and
   your card. Do what the card says. If the card is wrong or blocked, stop
   and record it rather than improvising a different plan.
2. **Git.** Work in a fresh worktree from `origin/main`, on the branch
   `plan/<card>-<slug>`. Your card authorizes you to commit, push and open a
   pull request for its changes. Do not merge. The user merges. Keep each pull
   request to one card.
3. **Status.** Create or update `docs/training/plan-20260927/status/<card>.json`
   in your pull request, following [status/README.md](status/README.md). Use
   `needs_user` with a single clear question when you stop.
4. **GPU.** Never start GPU work on the PC outside the nightly window. After
   C02 lands, enqueue GPU jobs in the night queue instead of running them.
   Kaggle GPU jobs are allowed at any time within the quota limit in R7.
5. **Sealed final set.** Follow [CONTEXT.md](CONTEXT.md). Never touch it, and
   never tune on DEV beyond the decisions these rules define.
6. **Uploads.** Upload only TRAIN- or DEV-derived data and weights, through
   explicit allow-lists, to the public repositories named in CONTEXT.md.
   Record what you uploaded.
7. **Secrets.** Read secrets from the environment only.
8. **Resources.** Daytime CPU work on the PC uses at most 8 threads at
   `nice 10`. Keep one CUDA workload at a time. Never delete checkpoints,
   datasets or published artifacts without the user.
9. **Evidence.** Put small JSON receipts in your status file or in
   `docs/training/plan-20260927/receipts/`. Large artifacts go to the HF dataset
   under `campaign-20260915/<card>/`.

## Standard hand-off prompt

Paste this into a fresh Claude Code thread, replacing `CXX`:

> You are executing card CXX of the Sepalith post-CPT plan. Work in a fresh
> worktree of github.com/sims1253/Sepalith from origin/main. Read
> docs/training/plan-20260927/README.md (operating rules), CONTEXT.md,
> RULES.md, then cards/CXX-*.md, and do exactly that card. You may commit,
> push and open a pull request for this card; do not merge. If something in
> the card is wrong or blocked, stop and record `needs_user` in
> docs/training/plan-20260927/status/CXX.json with one clear question.
