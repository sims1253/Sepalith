# Campaign delegation and return protocol

`tasks.json` owns specifications, dependencies and acceptance; `state.json` owns
task status, ownership, notes and receipt references. The lead records active
resource leases in [RESOURCE-LEASES.md](RESOURCE-LEASES.md). The tracker does
not launch jobs or enforce a hardware scheduler. This file supplies packets
and return shapes without duplicating the plan:

```text
python3 docs/campaign/campaign.py brief TASK-ID
python3 docs/campaign/campaign.py update TASK-ID --status in_progress --owner worker-name --note 'Assigned scope and next action'
```

Replace `TASK-ID` with an actual ready task. The local UI renders shared state.
PostPlan is a static snapshot with no shared JavaScript backend or browser
`localStorage`. The local board exports progress JSON; the CLI imports it with
conflict checks. See [LEAD-AGENT.md](LEAD-AGENT.md) for exact commands.

## Assignment rules

The lead assigns a bounded card with one owner, dependencies, lease, timebox,
and checkable acceptance.

- Local workers may inspect, build, test, CPU-evaluate, prepare manifests, or
  edit packet-named files; serving workers may own their subtree. The lead
  retains CUDA, training, cloud, launch, and resource-change actions.
- External agents receive minimum source pointers and no credentials, private
  data, or private artifacts unless that input is necessary and authorized.
  Prefer redacted fixtures.
- Conditional cards remain to do until the lead records the decision and
  prerequisite receipt. Defer one with a reason when it is cut from this
  campaign. Workers do not expand scope.
- A command exit proves only that command exited. Verify job/checkpoint identity
  and first-step or artifact evidence before completing launch/training work.

## Local worker packet

Copy this block into the worker message and replace every bracketed field.

```text
TASK: [stable task ID from tasks.json]
OBJECTIVE: [one observable result]
OWNER: [agent]
DEPENDENCIES: [task IDs and required receipt paths]
SOURCE: [exact worktree/files; distinguish plan, canonical assets, owner source, CV refs]
INPUTS: [hashes, fixtures, or redacted examples]
SCOPE: [files/subtree the worker may read or change]
LEASE: [CPU | serving subtree | none; CUDA/cloud are lead-owned]
TIMEBOX: [wall-clock limit and campaign deadline]
ACCEPTANCE: [tests, counts, hashes, or decision rule that proves done]
RETURN: [receipt path plus changed-file list]
```

For prompt work require token/boundary fixtures and renderer version; for data,
provenance/split/contradiction checks and hashes; for serving, model/revision,
backend, host, and p50/p95 cycle time.

## External-agent packet

Use this smaller shape for a reviewer, coding-plan agent, or external model.
It keeps private context out of the request while preserving a useful return.

```text
TASK: [stable task ID]
QUESTION: [single decision or bounded implementation question]
KNOWN FACTS: [short, non-sensitive facts only]
RELEVANT POINTERS: [public/approved paths or excerpts]
REDACTED FIXTURE: [small input/output example, if needed]
CONSTRAINTS: [timebox, no launch/resource changes, required format]
RETURN: [recommendation, uncertainty, tests/reproduction, and artifact schema]
```

The external agent returns a patch or analysis, never a claim that an unseen job
ran. The lead reviews it locally and records acceptance.

## Return receipt

Every worker returns this structure; failure is a valid receipt status.

```text
TASK: [ID]
STATUS: verified | partial | blocked | failed
OWNER: [agent]
STARTED: [timestamp]
ENDED: [timestamp]
DEPENDENCIES_CHECKED: [IDs and receipt paths]
ACTION: [what was actually done]
COMMANDS_OR_METHOD: [exact commands, API-free reproduction, or notebook cell]
RESULT: [metrics with denominators, hashes, paths, and model/backend identity]
ACCEPTANCE: pass | fail | inconclusive — [why]
CHANGED_FILES: [paths; say none]
ARTIFACTS: [receipt, logs, manifests, screenshots, adapters, or patch]
UNRESOLVED: [specific blocker or uncertainty]
NEXT: [one concrete lead action]
LEASE_RELEASED: yes | no — [reason]
```

For cloud, include job/resource identity, allocation start, first-step and
termination evidence, final cost/credit coverage, and artifact identity. For a
checkpoint, include parent, tokenizer, renderer, data/source hashes,
optimizer/RNG state, and save location.

## Handoff at context limits

Near a context or time limit, return the receipt before continuing. Add
`CURRENT_STATE`, `LAST_ACCEPTED_ARTIFACT`, `OPEN_DEPENDENCY`, `LEASE`,
`DEADLINE`, and `NEXT_COMMAND`. The successor reads it and `state.json`, then
resumes or marks the card blocked; it does not infer completion from a partial
log.

The lead changes a card to `done` only when the acceptance evidence is
present in the receipt and the dependency graph is satisfied. Keep partial,
failed, timeout, and deferred records visible so the Monday freeze can cut
them deliberately rather than counting missing evidence as progress.
