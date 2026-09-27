# C02. Nightly GPU window: queue, runner, scheduler

- **Where:** PC. Code during the day. One short dry run with the user's OK.
- **Needs:** C00
- **Produces:** a night queue, a runner, a scheduler setup, and a morning report
- **Effort:** one agent-day, plus one user action for the Windows scheduled task

## Goal

GPU jobs run unattended between 01:00 and 09:00 Europe/Berlin, stop cleanly
before 09:00, and resume the next night. The user's PC is otherwise free.

## Design

- **Queue:** job files in `~/.local/state/sepalith/night-queue/pending/*.json`.
  Each job has an id, a command, the working directory, and an env allow-list
  with names only, never values. It also records a priority, whether it is
  resumable, and an estimated duration. The runner moves jobs to `running/`,
  then to `done/`, `failed/` or `deferred/`.
- **Runner:** `packages/sepalith/src/sepalith/ops/night_runner.py`. It starts
  work only when all of these hold:
  - the local time is inside the window;
  - `nvidia-smi` reports no compute processes and less than 2 GB used;
  - host memory passes the existing guard thresholds;
  - `/mnt/e` has at least 150 GB free.

  The runner exports `SEPALITH_DEADLINE_EPOCH` for 08:40. Jobs must save and
  stop at the next safe boundary once that deadline passes. The existing
  trainer supervisors already support deadline stops; see the extension48
  receipt. At 08:50 the runner sends a graceful-stop request. At 09:00 it
  kills the job's process group. Jobs are sequential, never parallel.
- **Report:** `~/.local/state/sepalith/night-reports/YYYY-MM-DD.md` and a JSON
  file listing jobs run, exit codes, checkpoints written, evaluation
  summaries, and any rule outcome that needs the user.
- **Scheduler:** a systemd user service and timer inside WSL (`systemd=true`
  is already set). A Windows Task Scheduler task wakes the PC and starts WSL
  at 00:55 with `wsl.exe -d <distro> -u m0hawk -- true`, so the timer fires.
  A second task at 09:05 checks that nothing is still running and logs it.
  Generate the task XML and a PowerShell import script. The user imports them
  once with administrator rights. Document the power settings that must be
  on: wake timers allowed, no sleep on AC during the window.

## Steps

1. Implement the queue, runner and report, with unit tests using fake jobs
   and a fake clock.
2. Add a CLI to enqueue, list and cancel jobs (`python -m sepalith.ops.night_queue ...`).
3. Write the systemd units, the Windows task XML, the import script, and
   `docs/training/plan-20260927/NIGHT-WINDOW.md` with setup and recovery steps.
4. Dry run. Ask the user for a 20-minute daytime slot. Use a compressed test
   window: start now, deadline in 10 minutes, hard stop in 15. Run a
   synthetic CUDA job that honours the deadline, then a second one that
   ignores it, to prove the hard stop. Record both results.
5. After the user imports the Windows tasks, confirm that WSL and the timer
   come up by themselves on the first real night. Run a no-op job and check
   the morning report.

## Acceptance

- The unit tests pass.
- In the dry run, the honouring job stopped cleanly and the ignoring job was
  killed on time.
- One real night produced a morning report with no user action.

## Stop and ask if

- The Windows side needs anything beyond importing the task and allowing wake
  timers.
- The PC was found off or asleep and did not wake.
