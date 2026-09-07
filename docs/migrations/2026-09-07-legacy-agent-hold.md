# Legacy agent investigation and reversible hold

The user authorized inspection and, if needed, termination of the old experiment
agents on 7 September 2026. This record covers the agent hold, not runner cutover.

The four Zcode worker processes belonged to different workspaces. PID 90896 was
Sepalith, with REPL child 90965. The other workers belonged to `ry`, `apin/stan`
and the plugin workspace. The shared server (90793) serves all four and was left
running. Process identities were checked against `/proc` start ticks and working
directories; signals used pidfds rather than an unchecked PID or name pattern.

Read-only SQLite inspection found the queue-manager conversation
`sess_18d91974-03c0-43b1-bb23-2256c1bfca68` and its benchmark child
`sess_subagent_agent_64f09570-b3aa-4377-8534-eca97717ec4f`. The child failed with a
provider usage-limit error while awaiting a separately launched V1c process.
The parent task index records an error. These records explain the missing agent
handoff; they do not imply that the experiment child stopped. Independent artifact
inspection established S1 completion at 04:37, S2 summary at 05:01 and V1c summary
at 06:18 on 7 September. Verdict and archive reconciliation remains separate.
No Sepalith automation or off-peak task rows were found in the task-index database.

Before sending signals, the parent and descendant conversation records (35
sessions) and queue/comms files were preserved under
`~/.local/state/sepalith/legacy-agent-stop-20260907/`, with a SHA-256 manifest.
This private recovery snapshot is about 48.5 MB and remains outside Git. Raw session
records may contain sensitive material; they were not uploaded or published.

SIGTERM ended 90896 and 90965. The server then created replacement Sepalith worker
3007450 and REPL 3007996. Both replacements received SIGSTOP after identity checks.
Verification at 12:36 Berlin time found both in state `T` (stopped), while the
three unrelated workspace workers remained in state `S`. The old PIDs were gone.
The dashboard and load logger were left untouched; neither is the identified
experiment-dispatch agent. No experiment process was terminated by this action.

## Hold limits and rollback

This is a reversible process hold, not durable server-side retirement. The two
stopped processes cannot execute commands, but a server restart or a new workspace
session could create another dispatcher. The shared server must not be terminated
to solve a Sepalith-only problem. Before runner activation, recheck all workspace
workers, detached experiment jobs, follow-ons and cloud jobs, then retire or
otherwise control the server's workspace restart path. Do not infer exclusion
from the runner's own state-directory lock.

Rollback requires an explicit dispatch decision and a drained/paused new runner.
Read `suspended.json` from the recovery snapshot and revalidate each PID's start
ticks, working directory and parent relationship before sending SIGCONT via a
pidfd. Never send signals to the historical PID numbers without identity checks.
Resuming may execute pending agent work; review queued conversation inputs first.
Do not restart the old agent merely to read its history: the snapshot and live
read-only session database provide that history without execution.

V1c-v7 was already executed by legacy dispatch and must not be duplicated as the
first migrated experiment. Choose an eligible unrun scope only after closing the
existing result records. The runner remains paused; migration has not succeeded.
