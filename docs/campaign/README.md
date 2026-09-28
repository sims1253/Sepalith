# Sepalith campaign board

An executable checklist for the Tuesday, September 15, 2026 delivery. It
packages the [scientific plan](../72-HOUR-MODEL-PLAN.md) into bounded tasks with
dependencies, owners, steps, completion criteria and receipts.

**Published checklist:** https://vy9o07xkka3l.postplan.dev

## Open the working board

From the repository root:

```sh
python3 docs/campaign/campaign.py validate
python3 docs/campaign/build.py
python3 docs/campaign/campaign.py serve --port 8766
```

Open **http://127.0.0.1:8766**. The local board and CLI share
`docs/campaign/state.json`. The server binds to the local machine. It does not
start experiments, allocate resources, or publish files.

For a VS Code Remote SSH workspace, forward port 8766 to the notebook. The
server and shared state stay beside the repository. The board itself is already
built; PRE-09 verifies and adopts it during campaign intake.

Search by task, family, method or output. Filter by phase, readiness, required
scope, workstream or resource. Select a task to see its full brief, assign an
owner, record a blocker, or save a result. Completion requires a result note,
receipt reference and completed prerequisites. The lead still verifies the
receipt's contents; the tool checks structure and dependencies, not scientific
truth.

The **Copy agent brief** button produces a self-contained task packet. The
**Agent guide** button opens the lead and delegation runbooks. Use **Export
progress** to save a JSON snapshot. Imports reject conflicting existing work;
reconcile conflicts task by task through normal updates with revision checks.

## Start a lead experiment agent

Give a new agent the following prompt, from this worktree:

```text
Act as the lead experiment agent for the Sepalith Tuesday campaign.
Read docs/campaign/LEAD-AGENT.md, then validate the campaign and list ready tasks.
Read docs/72-HOUR-MODEL-PLAN.md and the prompt/training contract it links.
Use docs/campaign/tasks.json for task specifications and the campaign CLI for status.
Begin with current source, process, storage and time-budget verification.
Keep one primary model lineage and one CUDA owner. Preserve the existing control.
Use the user's authorized local resources and cloud ceilings under the plan's
live capacity, credit and lifecycle checks. Recompute available time at intake.
Delegate independent ready tasks with the packets in docs/campaign/DELEGATION.md.
The user may own the serving workstream while you own the training lane.
Record evidence and accept worker results before marking tasks done.
Keep the final set closed until the weight/harness freeze and protect delivery.
At each handoff, write the next action, current resource owner and receipt paths.
```

Agents with access only to the published page need a task packet and the
specific approved sources for their assignment. The hosted HTML does not grant
access to the private wiki, repository, datasets, GPU, or provider accounts.

## Agent commands

```sh
python3 docs/campaign/campaign.py list --ready
python3 docs/campaign/campaign.py brief PRE-01
python3 docs/campaign/campaign.py update PRE-01 --status in_progress --owner lead --note 'Inspecting current source roots'
```

When the real work is complete, use `update ID --status done --note RESULT
--receipt PATH`. Record a reason for `blocked` or `deferred`; required tasks
cannot be deferred. A deferred dependency does not count as complete. Use
`--revision N` when updating from a previous snapshot. Run `--help` on any
command for its arguments.

## Files and authority

| File | Role |
|---|---|
| `tasks.json` | Canonical task definitions and dependency graph |
| `state.json` | Shared progress, owners, notes and receipt references |
| `LEAD-AGENT.md` | Lead loop, launch gates, cut rules and handoff |
| `DELEGATION.md` | Local and external task packets and return receipt |
| `RESOURCE-LEASES.md` | Lead-maintained resource ownership ledger |
| `SPECULATIVE-PATHS.md` | Target/draft options, training feasibility and final-pair validation |
| `receipts/` | Evidence produced by the actual experiment tasks |
| `campaign.py` | Standard-library CLI, validation and local state API |
| `board.template.html` / `build.py` | Editable UI source and artifact builder |
| `board.html` | Self-contained interactive tracker |
| `postplan.html` | Script-free publishing snapshot |

Task changes belong in `tasks.json`; UI changes belong in the template. Run the
builder after changing either or after accepting new progress for publication.
State updates are atomic and locked. They are not process supervision or a
GPU scheduler. Reconcile real process identities with `RESOURCE-LEASES.md`.

## Local file and PostPlan behavior

Opening `board.html` directly uses browser storage instead of the local state
API. That state belongs to that browser. Export it, then use
`campaign.py import-state FILE` to submit it to the lead's shared state.

PostPlan documents a `script-src 'none'` policy for hosted drafts. Accordingly,
`postplan.html` provides the full expandable checklist, phase navigation and
resource rules without script-dependent controls. It is a readable snapshot;
it cannot synchronize task progress. The publishing artifact excludes embedded
guide/source payloads, owner notes, and absolute home-directory paths.

To update the same hosted draft after reviewing the rebuilt artifact:

```sh
python3 docs/campaign/build.py
npm exec --yes --package=postplan@0.0.4 -- postplan upload docs/campaign/postplan.html --draft vy9o07xkka3l --description 'Sepalith Tuesday campaign'
```

PostPlan mappings are managed by its CLI. Uploads are public by default. Publish
only the intended checklist artifact; keep private datasets, logs, credentials
and agent receipts in their authorized locations.

## Verification

```sh
python3 -m unittest discover -s docs/campaign -p 'test_*.py' -v
python3 docs/campaign/campaign.py validate
```

The tests cover state/dependency rules, receipts, revision conflicts and import
behavior. UI verification uses an isolated temporary campaign state, so test
clicks do not create fake experiment progress. The fonts are bundled locally
under their included SIL Open Font Licenses. No JavaScript packages or external
CDN are needed to run the board.
