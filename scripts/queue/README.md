# Sepalith queue heartbeat

**Disabled after user clarification:** the intended heartbeat should wake the
original conversation. This implementation starts a separate agent and does
not meet that requirement. Its timer and service have been stopped; retain
the code as an unused implementation, not an active default.

The user requested a check every 30 minutes on 8 September 2026. The systemd
user timer invokes `supervisor.py --once` at minute 00 and 30. A check reads
runner state and actual worker liveness. Healthy experiments continue without
an agent call. An idle queue, missing worker or exceeded recorded deadline
wakes one Codex agent to review, recover, prepare or launch authorized work.

This uses the supported [non-interactive Codex workflow](https://learn.chatgpt.com/docs/non-interactive-mode)
and the existing local ChatGPT login. It removes inherited OpenAI API-key
variables from the agent environment. It does not spend Anyscale/Azure credit.
The existing Codex model/configuration is retained.

The oneshot service cannot overlap itself; a lock also excludes duplicate
supervisors. An existing batch-controller lock excludes a competing owner.
Each agent invocation is bounded to 45 minutes. A timer tick during an active
invocation does not start another. Failed calls back off; they do not retry in
a tight loop. Persistent timer state permits a catch-up check after downtime
when the WSL user manager starts. Nothing runs while the machine is off.

Installed service: `sepalith-queue-heartbeat.service`.
Installed timer: `sepalith-queue-heartbeat.timer`.
Private state: `~/.local/state/sepalith/queue-supervisor/`.

- `status.json`: latest actual heartbeat/agent state.
- `SUMMARY.md`: short user-facing update maintained by the queue agent.
- `HANDOFF.md`: next concrete action and active resource identities.
- `USER-INSTRUCTIONS.md`: user steering; queue agents must not edit it.
- `reviews/`: private invocation logs and structured results.
- `installed/`: the installed code/prompt/schema and source hashes.

Check: `systemctl --user status sepalith-queue-heartbeat.timer`.
Run a check now: `systemctl --user start sepalith-queue-heartbeat.service`.
Pause future checks: create `~/.local/state/sepalith/queue-supervisor/HOLD`.
Stop the timer: `systemctl --user stop sepalith-queue-heartbeat.timer`.
Stopping the timer does not stop an already active check. To take over the
worktree manually, stop the heartbeat service too and inspect experiment
units separately. Already launched experiments should have their own bounded
systemd units and remain owned by the runner.

The heartbeat automates selection, not authorization: parked proposals and
cloud-spending constraints remain intact. Idle hardware alone is not a reason
to repeat an experiment. If nothing useful is currently authorized, the agent
records specific blockers and the heartbeat remains scheduled.

Validation: `pytest -q scripts/queue/test_supervisor.py`.
The suite includes two real fake-agent invocations across a new completion,
missed-worker recovery, PID reuse, deadline review, duplicate-owner exclusion,
user hold and failure backoff. No experiment or paid API is used by the tests.
