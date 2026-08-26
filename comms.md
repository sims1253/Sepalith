# comms.md — inter-agent communication protocol (Sepalith)

Multiple agents work this repo and share one machine (WSL2, RTX 5090 32GB,
`/mnt/h` NAS). We coordinate through files — no agent owns the whole
machine, and no agent may assume another is watching. This file is the
protocol; it is the second thing any new agent reads (after `SYSTEMS.md`).

## Layout

- `comms.md`      — this protocol + the agent registry (you are here)
- `comms/board.md`  — append-only shared log: messages, decisions, handoffs
- `comms/gpu.md`    — ledger for the GPU, the one contended resource

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

## Registry

| id | role | owner session | status |
|----|------|---------------|--------|
| zcode-pvf-poc | PVF/TETHER POC (critic bake-off → RL-run-4) | sess_6e6e815d | active |
| zcode-ddot-poc | POC-DDOT (OT position coupling, `poc-ddot-ot-coupling-plan-2026-08-26.md`) | sess_unknown (2026-08-26) | active |
| zcode-pocdiff | POC-DIFF masked-diffusion NSE twin vs AR-FIM twin (`poc-diff-twin-plan-2026-08-26.md`) | sess_unknown-2 (2026-08-26) | active |
