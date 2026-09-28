# Lead agent: Tuesday model campaign

> Active user amendment: use all eligible TRAIN data across CPT, SFT and RL, with the appropriate objective for each stage. Time alone is not an exclusion reason, and 30 hours was not a training limit. This supersedes both the original calendar and the later one-day extension; no replacement hard deadline has been set. Freeze follows training and development selection. Keep final evaluation sealed until that freeze, preserve the tested rollback models, and retain the $60 Anyscale / €100 Azure ceilings. Before applying any historical time, quota or length cutoff below, read the [all-eligible-data policy](receipts/LEAD-all-eligible-data-policy-20260914.json). Use the latest campaign state and [handoff](receipts/LEAD-HANDOFF.json) for active full-weight training and resource ownership.

This is the execution runbook for one lead experiment agent. Deliver a useful
R next-edit model, matched prompt and practical runtime. Generic language-model
scores are diagnostics; editing quality is the objective. Calendar allocations
below describe the original campaign and are subject to the active amendment.

## Load the truth before taking a task

Read the [72-hour plan](../72-HOUR-MODEL-PLAN.md) and
[prompt/training contract](../research/72h-prompt-training-contract.md). Load
`tasks.json` as canonical task specifications and `state.json` as canonical
local status from the paths reported by the campaign tool. Do not infer status
from chat, filenames, or subprocesses.

Use these distinct source roles in manifests:

```text
plan worktree  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
canonical data /home/m0hawk/Documents/Sepalith
owner source   /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912
previous owner /home/m0hawk/.t3/worktrees/Sepalith/t3code-e6aed5ff
CV references  /home/m0hawk/.t3/worktrees/Sepalith/t3code-3d83226e
```

Before launch, make an immutable snapshot containing required reviewed
uncommitted changes and every file hash; environment secrets stay outside it.
A plain `git archive HEAD` is valid only when it contains that reviewed source.

The local UI reads and updates shared `state.json` through the local server.
PostPlan is an uploaded static snapshot with no shared JavaScript,
`localStorage`, or progress backend. File-opened `board.html` can store browser
progress and export JSON, which the lead explicitly imports. Conflicting
branches of progress need reconciliation; a hosted page never updates the repo.

From the plan worktree, start with:

```sh
python3 docs/campaign/campaign.py validate
python3 docs/campaign/campaign.py list --ready
python3 docs/campaign/campaign.py brief PRE-01
python3 docs/campaign/campaign.py serve --port 8766
```

Open `http://127.0.0.1:8766`. `brief` supplies the selected task's actual steps,
inputs, prerequisites and acceptance criteria. Use the ID shown by `list`.
Record ownership and results with:

```sh
python3 docs/campaign/campaign.py update TASK-ID --status in_progress --owner lead --note 'Starting the assigned work'
python3 docs/campaign/campaign.py update TASK-ID --status done --owner lead --note 'Acceptance checked; see receipt' --receipt docs/campaign/receipts/TASK-ID.md
python3 docs/campaign/campaign.py import-state /path/to/exported-progress.json
python3 docs/campaign/build.py
```

The last command rebuilds the standalone tracker and published snapshot. It
does not publish or launch work. Replace `TASK-ID` and receipt text with the
actual task and verified result. Use `--revision` for an optimistic update when
working from an earlier state snapshot. The CLI rejects incomplete dependencies,
missing completion evidence and stale revision updates.

## Non-negotiable campaign shape

- **One primary lineage:** `openbmb/MiniCPM5-2B-Midtrain` at the plan's pinned
  revision; b4 Q8 with its legacy renderer is rollback, other bases reserves.
- Freeze prompt/context before SFT. Proposed 2048/4096 SFT buckets and
  2048+192 RL limits stay provisional until measured. Keep the clean final set
  sealed until Monday's freeze.
- The lead owns the single CUDA lease, launches, resource changes, and
  promotion. Workers can read/build/test/CPU-evaluate or own an allocated
  serving subtree; architecture-level serving can run in parallel, with GPU
  work scheduled through the lead.
- Cloud branches are conditional production capacity. Start from exact
  `theta0`, tokenizer, renderer, and split; proposed all-in ceilings are **$60
  Anyscale / €100 Azure**, including underlying infrastructure, with live
  credit, capacity, termination, and artifact evidence.
- Mark a task complete only with its acceptance artifact and receipt. A
  subprocess exit does not prove a job, checkpoint, or training stage started;
  verify external identity and first-step evidence.

## Board and gates

| Gate | Work and dependency | Acceptance / owner |
|---|---|---|
| L0 intake | Validate/list/brief the campaign; inspect `state.json`, active O2 control, storage, and leases. | One current status record; no second CUDA owner. Lead. |
| L1 prompt | Build the pre-SFT renderer contract, shared Python/TypeScript fixtures, token accounting, and missing-evidence policy. | One selected layout, output semantics, and versioned fixtures pass round-trip/context tests. Lead plus bounded worker. |
| L2 data | Build provenance/package-disjoint train, development, selector, and final manifests. Reject contradictory prompts, unsupported targets, and unsafe truncation. | Training registry is hashed; final manifest is sealed but unopened. Lead. |
| L3 SFT smoke | Train the pinned MiniCPM parent with audited projections, LoRA r32/alpha64, BF16, effective batch 16, and the measured length profile. | 50-step smoke has finite loss, expected attachment, target visibility, export, and timing receipt. Lead owns CUDA. |
| L4 common parent | Run the 3000-update SFT readout; extend toward 6000 only if the smoke and development evidence justify it. Add durable adapter/full-state cadence and resume test. | Select one development winner and merge once into hashed `theta0` with tokenizer, renderer, and policy manifest. Lead. |
| L5 rollout gate | Validate evidence use, no-op behavior, parser/stop parity, length buckets, reward, and policy log-probability path on `theta0`. | Rollout contract is frozen; unresolved limits become explicit exclusions, never silent truncation. Lead. |
| L6 RL | Run local LoRA RL from `theta0` (initial r16/alpha16 recipe) through the 26-hour ceiling, stopping Monday 00:00. | Checkpoints/optimizer/RNG/sampler state and 500-update development receipts support a selected checkpoint. Lead owns CUDA. |
| L7 cloud | After `theta0`, prepare Anyscale first and Azure only after live gates. Use breadth SFT for Anyscale and restraint SFT for Azure when their data contracts pass. | Exact source/hash manifest, provider identity, all-in quote, cleanup, private adapter, and branch receipt by Monday 00:00. Lead launches. |
| L8 serving | In parallel, profile Q8 serving, `-b 256`, CPU/SSH placement, and optional target-matched DSpark training/evaluation. | Add DSpark only after Midtrain compatibility is established; recheck draft acceptance after final weights change. Runtime worker owns serving files; lead schedules GPU. |
| L9 freeze | Compare unmerged branches, then at most three predeclared blends or a short consolidation. | Monday 12:00 freezes weights, harness, tokenizer, and runtime; every final input remains traceable. Lead. |
| L10 delivery | Run locked final evaluation, Q8 export/hash checks, target-host editor smoke, rollback package, and installation check. | Candidate package and b4 rollback are ready by Tuesday 09:00. Lead. |

Keep the active O2 control as the sole local CUDA job until its receipt closes;
use its matched battery to separate filtering from step count. Cut optional
new draft/head training and runtime forks first, then extra harness search,
blends/consolidation, the second cloud provider and SFT extension. Keep final
serving validation, the prompt gate, valid parent, RL checkpoint integrity,
final evaluation and b4 rollback.

## Lead loop

1. Read `state.json`; choose the highest-priority unblocked `tasks.json` card,
   with dependencies and lease.
2. Reserve the lease, do only that bounded work, and place acceptance beside
   its receipt.
3. Update the task through the CLI above. Rebuild the static snapshot with
   `build.py` when useful; keep `state.json` as the shared status source.
4. Reconcile external identities; record failures, timeouts, invalid judges,
   and cuts. Incomplete evidence is not `done`.

At a context limit, write task ID, accepted artifacts, unresolved dependency,
next action, lease, and deadline before stopping; the successor resumes from
state and receipts.
