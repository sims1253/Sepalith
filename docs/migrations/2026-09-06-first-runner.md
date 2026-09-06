First runner migration — prepared, cutover pending

Owner: `codex-runner-migration`. Branch: `migration/runner-first-real-20260906`.
Worktree: `/home/m0hawk/Documents/Sepalith-runner-migration`.
Base commit: `83b43b2`. Read this record with current `comms/board.md`,
`comms/gpu.md` and `docs/EXPERIMENT-QUEUE.md`; observations below are dated
investigation evidence, never permission to reuse a PID.

No real experiment has run through the new runner. No legacy dispatcher has
been stopped or changed. The queue manager has not yet acknowledged the drain
or reserved the candidate. Large model/runtime input capture remains deferred
to the coordinated resource window. The source is captured and the complete
candidate workflow has passed fake-job tests.

The current blocker is specific: X5-S1 must finish evaluation, residual replay,
verdict and NAS mirroring; the benchmark owner must finish or hand off the
required S1/S2 work and release its resource window; the queue manager must
confirm that no legacy/session/cloud follow-on will launch. V1c's exact scope
also needs resolution before it is an eligible complete migration experiment.
The candidate below does not silently waive its calibration requirements.

The protocol requests are in the **shared checkout's** board, at
2026-09-06T15:02:49+02:00 and 15:04:37+02:00. They address
`zcode-queue-mgr-3` and `zcode-quietwindow`. No response had arrived at the
15:11 process audit. The historical `/tmp/sepalith-queue-mgr-handoff-2026-09-06.md`
was read and preserved. Its old PIDs, ETAs and broad authorization summary do
not override the current queue or parked user decisions.

Observed dispatchers and resource owners, 2026-09-06 15:02–15:11 +02:00:

| Owner / process | Evidence and automatic continuation | Required drain action |
|---|---|---|
| `zcode-x5-s1`, shell 1005134, group/session 1005134 | Live child 1117314 runs `x5_s1_eval run`; GPU reports this PID. `run_x5_s1_chain.sh` proceeds to residual replay, verdict, rsync, then exits. No next experiment in that file. | Owner confirms all four stages complete and mirrored artifacts; inspect descendants even after shell exits. |
| `zcode-quietwindow`, supervisor 753199 | `/tmp/s1_run3_auto.sh` launched detached spec process 831052, then waits. No S2/V1c launch in this file. | Let S1 complete; confirm separately that the owner's session does not launch S2/V1c outside the agreed boundary. Never edit the executing wrapper. |
| S1 process 831052; server 1005713 in its own group/session | Current command and logs show ngram-simple@2, 8k, CPU t8. `spec_bench` automatically runs the remaining model-draft arm. The server's separate session demonstrates why losing a supervisor is not a drain. | Complete required measurement/readout/archive; investigate both process groups and ports. |
| Dashboard supervisor 251940 | `run_loop.sh` repeats refresh/build/upload every 1800s. Actual stop path is `experiments/dashboard/STOP_DASHBOARD`, unlike the historical handoff's path. Refresh subprocess sites were inspected: dashboard build and upload, not experiment dispatch. | Remains with owner. A quiet measurement window must account for its CPU/I/O activity; arrange a cycle-boundary pause if needed. |
| Load logger 860265 and S1 waiter 833845 | `/tmp/load_logger.sh` samples every 5s; waiter only waits for 831052. Neither dispatches experiments. | Preserve load evidence; owner handles lifecycle. |
| Preexisting toy 1099539 | Live Python waits for `/tmp/tmpbcndi63o/release`; group/session 1099539. It predates this task and was not launched by this migration. | Reported to board; no kill or unknown-launch resolution performed. |
| Other project processes | Live apin C++ builds / dune tests were observed independently of Sepalith. | Do not kill them. Coordinate a genuinely quiet window; core affinity does not isolate SMT siblings. |
| Queue-manager/session launchers | Registry names queue manager 3 active. Board describes possible S2/V1c and cloud LR follow-ons; no current FIRE/hold ACK obtained. | Owner must enumerate and hold session/cloud follow-ons. Absence of a local process cannot establish cloud quiescence. |
| Scheduled launchers | User crontab has trace backup only; user systemd timer list is empty. | Recheck before cutover. This is not proof about remote scheduling or other users. |

The current queue and wrappers are preserved outside Git under
`/home/m0hawk/.local/state/sepalith/migration-20260906/legacy/`.
`legacy-manifest.json` records capture time, HEAD, paths, lengths and SHA256.
The queue bytes had SHA256
`872f68133d8babe6b466208a2203be72be4cd3fca921505fc2fe6ead11fc02f6`.
The archive includes the live queue/board/GPU ledger/protocol, X5 and dashboard
wrappers, benchmark supervisor, load logger and historical handoff.
Do not restore this old queue wholesale over subsequent owner edits.

Candidate: V1c's existing v7 serving column, pending owner reservation. It is
modest relative to training: no model updates, API spend or CUDA context; 100
serial streaming requests and 140 concurrent/superseded requests on one frozen
GGUF and the S0 trace set. It requires a quiet CPU window. Runtime is not
estimated from a stale dashboard; request timeouts alone allow a long run.

The current untracked `latency_load.py` implements v7 only, levels 1/2/4.
The design in `docs/research/2026-09-02-eval-strategy-v2.md` also requires
qwen0.5b/minicpm5 ordering, v8_2/base parity calibration and levels 1/4/8.
The owner must either confirm that this separately bounded v7 column is the
eligible experiment, supply the complete calibration specification, or select
another already-authorized modest experiment. Nothing in this record declares
full V1c complete, changes parked authorizations or invents a parity tolerance.

The [recipe template](v1c-v7.recipe.json.in) describes all four steps:

1. Validate the frozen trace inventory and persist the exact deterministic
   serial/sweep selections plus Python/platform information.
2. Measure using explicit `--model`, `--server`, `--traces`, `--out` and
   `--foreground`. Servers remain in the recorded step group and are reaped
   before the command returns. Startup/readiness failure also invokes cleanup.
3. Validate exact row coverage and trace identity, recompute summaries, and
   write `evaluation.json`, `verdict.json` and `VERDICT.md`.
4. Copy closed artifacts, captured source and available runtime/recipe records
   to an attempt-specific NAS archive, rehash both sides and write `archive.json`.
   The final open archive log and final runner receipt remain in local state;
   mirror those after the dispatcher returns and verify that final copy too.

Scientific acceptance for this column is explicit: all 100 serial and 140 sweep
rows align with the frozen selections; TTFT is finite/nonnegative; there are no
request errors or starvation, and requests complete or follow the intended
abort path. Malformed/missing rows or inconsistent summaries fail execution.
Validly recorded negative observations produce `INCONCLUSIVE` and still archive
successfully. A complete measurement says `MEASUREMENT-COMPLETE`, with adoption
`NOT-ASSESSED`. Quiet-window review and full instrument calibration remain
necessary before comparative claims. Abort-waste counts are observed content
chunks, not independent server-side token accounting.

Reviewed source capture:

| Original dependency | Captured working bytes before adaptation | Review |
|---|---|---|
| Untracked `experiments/eval/latency_load.py` | `b7fb8ecc42a84760c523fd705c41468dd756f05ebdc7390b01ce5425df484449` | Standard-library imports plus `spec_bench`; traced sampling, outputs, server lifecycle and summary flow. Added explicit roots and foreground mode only in migration worktree. |
| Dirty `experiments/eval/spec_bench.py` | `2722aeacb065b86be3fe894bbe6fab172446fba5b999152bc69a1c5a0db45b54` | Preserved reviewed ngram-depth correction and persisted greedy text. Added optional server path/foreground mode; legacy default remains detached. |
| Dirty `test_spec_bench.py` | Working-tree correction retained | Updated two existing flag tests to use tiny model-path fixtures instead of installed weights. |
| New migration helpers and patched core | Task worktree snapshot | Standard library only. No source imports resolve to the shared development checkout. |

Source snapshot:
`324236bc82a82a3a8fb2bd76bfb42155d63153b2fcc3f66bda5069666f556b6a`.
It is in `migration-20260906/prepared-state/snapshots/`; its manifest is also
under `evidence/source-manifest.json`. The preparation queue is paused and has
no jobs. Snapshot capture includes the needed uncommitted source and the core
runner package; it is not a worktree-from-HEAD substitute.

`prepare_v1c.py` freezes model, traces, llama executable and runtime libraries
into a content-addressed external bundle. It dereferences library symlinks into
regular files, checks source/copy hashes, preserves executable mode and sets
captured files read-only. It resolves the actual loader dependencies with `ldd`
after relocation, rejects non-system dependencies outside the bundle, and
records system-library and interpreter hashes as runner inputs. There is no
`uv run` or environment sync during execution. The intended interpreter is
`/usr/bin/python3.10`; all experiment imports use the captured source.
The host's standard-library/OS installation remains an external environment,
not a container image. Keep it stable throughout the run.

Large input capture was intentionally not run during the live benchmark.
Therefore the template's `/REQUIRES-FROZEN-INPUTS` paths are not executable
identities; do not enqueue the template. This is a remaining resource-dependent
step, not a completed immutable-input claim.

After owner ACK and a verified resource window, bind the recipe without launching:

```sh
/usr/bin/python3.10 scripts/migration/prepare_v1c.py \
  --state /home/m0hawk/.local/state/sepalith/runner \
  --assets /home/m0hawk/.local/share/sepalith/runner-inputs \
  --archive /mnt/h/sepalith/runs/runner-migration \
  --model /home/m0hawk/Documents/Sepalith/experiments/models/sft_v7_minicpm5-Q8_0.gguf \
  --traces /mnt/h/sepalith/datasets/spec_traces/traces.jsonl \
  --server-dir /home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453 \
  --output /home/m0hawk/.local/state/sepalith/migration-20260906/v1c.bound.json
```

These mutable paths are capture origins only. The generated experiment argv
references the frozen bundle and `{source}`/`{run}`. Review the generated
recipe and all loader paths before enqueue. The authoritative state root above
is proposed; it still needs the queue manager's confirmation.

Cutover procedure, after resolving candidate scope:

1. Re-read live canonical records and obtain owner ACKs for completion,
   evaluation/verdict/archive evidence, and hold of all legacy follow-ons.
2. Investigate live processes from current `/proc` start times, parents,
   groups/sessions, cwd, command lines, logs and listening ports. Check escaped
   servers and remote jobs separately. Do not infer exit from old PIDs or a
   vanished supervisor. Record the evidence and the owners' resource release.
3. Preserve a second, boundary-time queue/wrapper snapshot. Keep the first
   historical snapshot. Freeze inputs in the agreed I/O window and inspect
   the bound recipe, scope, interpreter and hashes. Verify relocated loader
   resolution does not point into any development checkout.
4. Run the dispatcher from the immutable captured package: set `PYTHONPATH`
   to the selected snapshot's `source/packages/sepalith/src`, use
   `/usr/bin/python3.10 -m sepalith.runner --state <authoritative-root>`.
   Enqueue only the reviewed bound recipe, inspect `plan`, then `resume` and
   `run-next`. Do not use an automatic `run` loop for this first cutover.
   Exactly one authoritative dispatcher may launch real experiments.
5. Pause immediately after `run-next` returns. Investigate unfinished attempts
   before any retry. Verify declared artifacts, NAS archive hashes, source
   identity, actual argv and interpreter/runtime record. Mirror final closed
   receipts to the NAS and verify them. Keep operational success separate from
   the scientific verdict.
6. Post attempt ID, artifact references, provenance, operational status and
   scientific verdict to the queue manager. The manager reconciles the central
   queue. Do not mark all V1c complete based on the v7 column.

Rollback: pause the runner, inspect its plan and audit all attempt processes.
A failed supervisor is not proof that its children stopped. Use `recover` only
when the recorded group has no live work. Use `resolve-unknown` only for a real
unknown launch after a fresh process investigation; preserve the full audit.
Retries are explicit, use new attempt directories, and preserve failed outputs.
After every new-runner process is accounted for and no dispatcher can launch,
the owner may restore the next uncompleted legacy command from the preserved
records. Do not rerun a completed experiment or overwrite the current queue.
The legacy and new dispatchers must never be active together.

Validation evidence is outside Git at
`/home/m0hawk/.local/state/sepalith/migration-20260906/evidence/`:

| Evidence | Result |
|---|---|
| `runner-red.log` | Reproduced missing-include omission and overwritten/deleted previously verified artifacts: three failing cases. |
| `core-tests.log` | 53 tests pass, including source changes, input mismatch, dependency blocking, pause/drain through later steps, missing output, surviving child recovery, explicit retry and audited unknown-launch rejection/transaction tests. |
| `migration-tests.log` | 8 tests pass: complete fake recipe/archive, scientific inconclusive vs command success, command failure/retry, missing rows, frozen input/library bytes, foreground launch, readiness cleanup, frozen recipe generation without dispatch. |
| `spec-tests.log` | 25 tests pass with no model/runtime assets required. |
| `fake-plan.json` | Persistent fake attempt `657ad5da5ce84515b3134e8cb68f96b6`, succeeded; validation state paused. No real model loaded. |
| `process-audit.json`, `process-table.txt`, `x5-tail.txt`, `s1-tail.txt` | Fresh process/start/group evidence and current experiment telemetry at 15:11. |

Fake artifact directory:
`migration-20260906/validation-state/attempts/657ad5da5ce84515b3134e8cb68f96b6`.
Fake archive:
`migration-20260906/fake-archive/657ad5da5ce84515b3134e8cb68f96b6`.
These are validation receipts, not the requested completed real experiment.
There are no real-run artifact references yet.
