# comms.md — inter-agent communication protocol (Sepalith)

Multiple agents work this repo and share one machine (WSL2, RTX 5090 32GB,
`/mnt/h` NAS). We coordinate through files — no agent owns the whole
machine, and no agent may assume another is watching. This file is the
protocol; it is the second thing any new agent reads (after `SYSTEMS.md`).

## Layout

- `comms.md`      — this protocol + the agent registry (you are here)
- `comms/board.md`  — append-only shared log: messages, decisions, handoffs
- `comms/gpu.md`    — ledger for the GPU, the one contended resource
- `docs/EXPERIMENT-QUEUE.md` — the single, central experiment queue (status only; verdicts live in RESULTS/docs and on the board)

## Registering

Append a row to the registry below before doing work:

| id | role | owner session | status |
|----|------|---------------|--------|

Ids are `<kind>-<slug>`, e.g. `zcode-pvf-poc`. Update your status cell
(`active` / `idle` / `done <date>`); never delete another agent's row.

## Messages — `comms/board.md`

Append-only; never edit, reorder, or delete another agent's entries.
One message per block, newest last:

```
## [2026-08-26T22:40+02] FROM zcode-pvf-poc TO ALL — replay done
Body. @zcode-<slug> for direct mentions. Keep decisions crisp; link
file paths, not prose descriptions.
```

- `TO ALL` for broadcasts; direct messages address one id.
- Reply within one poll cycle (see Defaults) or say nothing — silence is
  not consent, it is silence.
- Destructive or irreversible actions need an explicit `ACK` from the
  addressed agent (or a human) on the board first.

## GPU ledger — `comms/gpu.md`

Claim **before** creating any CUDA context; release when done:

```
[2026-08-26T22:40+02] zcode-pvf-poc CLAIM pvf critic training ~26GB ETA 20m
[2026-08-26T22:58+02] zcode-pvf-poc RELEASE pvf critic training
```

House rules (mirrors `SYSTEMS.md` §discipline):
- One trainer at a time unless the active claim says it co-exists and
  states its memory fraction.
- A claim older than 6h with no heartbeat may be reaped — append a reap
  note before taking the GPU, never silently.
- Never kill a PID you didn't log. Unknown GPU squatter → note it on the
  board with the PID and wait.
- CPU llama-server builds are a fallback only; the GPU is the target
  (user directive 2026-08-26). `experiments/bin/llama/llama-b10453`
  is CPU-only — use `/tmp/llama.cpp-cuda/build/bin/llama-server` (b10453,
  CUDA 13, sm_120) or rebuild from the same commit.

## Long jobs

Any job >30min appends a heartbeat to the board every ~30 min:
`HEARTBEAT <task> pid <pid> <log path> <one-line state>`. A heartbeat
>1h stale means the job is probably dead — say so on the board before
cleaning up its artifacts.

## Defaults

- Read order: `SYSTEMS.md` → `comms.md` → tail of `comms/board.md` →
  tail of `comms/gpu.md`.
- Poll the board at the start of every work unit and before every GPU
  claim.
- Questions about repo state → `git log`/`git status` + the latest
  `docs/research/` night notes; don't ask on the board what git knows.
- Git discipline (RFC 1, adopted 2026-08-26): commit only paths you own
  (your claim). The shared `comms/` files may be committed by anyone — the
  append-only board carries attribution inline; note in the commit message
  when you're committing another agent's entries. Never `git add` a
  claimed dir that isn't yours.
- Protocol changes: post an `RFC` message, wait one poll cycle for
  objections, then apply.

## Watchers

Scheduled/automated polling is RETIRED (scheduler reliability issues,
user directive 2026-08-27). The standard mechanism is session-bound:
run `comms/watch.sh [interval_s] [max_s] [mention_egrep]` as a
background task — it fingerprints comms.md/gpu.md + git HEAD and exits
on any change; the task-completion wake-up is the notification. With
`mention_egrep` set (noise trim, e.g. `'zcode-pvf-poc|pvf'`),
board.md changes only wake on a NEW line matching the pattern —
routine heartbeats addressed to no one don't wake you; gpu.md/comms.md/
git changes always do. On wake: read the board tail, act per protocol,
relaunch. Session-bound means exactly that — a down session watches
nothing; the append-only board remains the source of truth between
sessions. (Pattern credit: zcode-ddot-poc's 2026-08-27T00:06
correction.)

## Registry

| id | role | owner session | status |
|----|------|---------------|--------|
| zcode-pvf-poc | PVF/TETHER POC (critic bake-off → RL-run-4) | sess_6e6e815d | done 2026-08-27 (run-4 readout posted; on comms-watch duty) |
| zcode-ddot-poc | POC-DDOT (OT position coupling, `poc-ddot-ot-coupling-plan-2026-08-26.md`) | sess_unknown (2026-08-26) | done 2026-08-29 — VERDICT: KILLED (negative, family E closed) |
| zcode-pocdiff | POC-DIFF masked-diffusion NSE twin vs AR-FIM twin (`poc-diff-twin-plan-2026-08-26.md`) | sess_unknown-2 (2026-08-26) | done 2026-08-27 (VALIDATED, fbea1c7) |
| zcode-ddot-graph | POC-DDOT full-step CUDA-graph capture (7bf3df3) | sess_unknown-3 (2026-08-27) | done 2026-08-28 (full-graph verdict negative, kept behind --full-graph) |
| zcode-main | main session — user-directed recon + adoption planning (slime v0.3.2, Miles flash recipes, `2026-08-28-slime-miles-adoption-plan.md`; decay/CMA+MuonH POC plan `2026-08-28-decay-cma-muonh-poc-plan.md`, executed to completion by main session) | this session | done 2026-08-30 (plan closed; so_r_qa dose-response open thread) |
| zcode-stabtok | papers-recon POCs: Muon-hygiene A/B + 4×-LR stress gate + LR refit + tokenizer compression/forced-R-pattern sweep (`docs/research/2026-08-29-papers-recon-poc-plan.md`; P0 CPU prep done 2026-08-29; GPU queued behind flash E2/E3, unowned) | prior session | done 2026-09-01 (P1 chain released 08-29; P2/P3 parked queue §2 Q5; superseded by zcode-queue-mgr) |
| zcode-cma-poc | decay/CMA + MuonH POC (`2026-08-28-decay-cma-muonh-poc-plan.md`) | dead sessions (usage limits; work inherited + finished by zcode-main, commit fe401b9) | done 2026-08-30 (all tasks; verdict posted) |
| zcode-base-bakeoff | external-base bake-off × param-floor ladder scoping (sub-1B supplement `model-survey-sub1b-supplement-2026-08-31.md`; queue B-series) | prior session | done 2026-09-01 (B-series prepped + parked §2b; execution → zcode-queue-mgr) |
| zcode-micro-probe | M-series micro-specialist design (mdlARC-derived M1: ~75M MD twin, curated-vs-scale; plan `docs/research/2026-09-01-micro-specialist-probe-plan.md`, queued §3) | prior session | done 2026-09-01 (design landed §3; execution → zcode-queue-mgr) |
| zcode-queue-mgr | primary researcher — owns `docs/EXPERIMENT-QUEUE.md`; B-series chain execution + eval-strategy v2 (user activation 2026-09-01T23:4x) | prior session | done 2026-09-04 22:3x (gates B-α/B-β landed, D-grid closed, production plan v0 open; handoff /tmp/sepalith-queue-mgr-handoff-2026-09-04.md) |
| zcode-o3-telemetry | O3-S0 suffix-entropy + length telemetry on banked RL runs (queue §3 O3) | this session | done 2026-09-05 (verdict NOT LAND; O3-S1 closed; readout O3_S0_RESULTS.md) |
| zcode-queue-mgr-2 | primary researcher — owns docs/EXPERIMENT-QUEUE.md; round-1 working set (P12/TU1/O3-S0/S0/E1/O1/B8-patch) | this session | active 2026-09-05 |
| zcode-p12-roofline | P12 roofline bench (build+run, quiet window) | subagent of zcode-queue-mgr-2 | active |
| zcode-tu1-sufficiency | TU1 judge-sufficiency validation (glm-5.3, CPU/API) | subagent of zcode-queue-mgr-2 | done 2026-09-05 — VERDICT: DEAD (pooled gap +56.6pp but direction 1/5; TU2 arm (d) cancelled; doc_sync suff 0.267); TU1_RESULTS.md |
| zcode-o3-telemetry | O3-S0 suffix-entropy/advantage telemetry on banked RL runs | subagent of zcode-queue-mgr-2 | active |
| zcode-s0-traces | S0 spec-decode trace freeze (500-2000 frozen continuations) | subagent of zcode-queue-mgr-2 | active |
| zcode-e1-build | E1 EL-scheduler sampler patch (build only, no GPU run) | subagent of zcode-queue-mgr-2 | active |
| zcode-o1-build | O1 diverse-16 prompt-selection script (build only) | subagent of zcode-queue-mgr-2 | active |
| zcode-b8-patch | B8 prerequisite: train_sft.py completion-only masking + packing (build only) | subagent of zcode-queue-mgr-2 | done 2026-09-05 (commit b7dd226 + train_sft.py hunk; 31 tests; NOT fired) |
| zcode-e1-build | E1 EL-scheduler retrofit on rl_smoke.py (ordered difficulty, queue row E1; build-only 2026-09-04) | this session | done 2026-09-04 (patch + 26 tests + dry-run landed; GPU run NOT fired — queue manager arms it) |
| zcode-s0-traces | S-series S0: spec-decode trace freeze (Q7/RT-2 trace set; miner experiments/data-mining/freeze_spec_traces.py, data /mnt/h/sepalith/datasets/spec_traces/) | this session | done 2026-09-05 (1100 traces, validation clean, board posts 2026-09-04T23:2x + 2026-09-05T02:5x) |
