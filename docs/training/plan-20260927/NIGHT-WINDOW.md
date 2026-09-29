# Nightly GPU window

GPU jobs run unattended from 01:00 to 09:00 Europe/Berlin, one at a time,
and stop before 09:00. Resumable jobs continue the next night. Card
[C02](cards/C02-night-window.md) built this. The code is in
`packages/sepalith/src/sepalith/ops/` and `scripts/night/`.

## Timeline of a night

| Time | What happens |
| --- | --- |
| 00:55 | Windows task `\Sepalith\NightWake` wakes the PC, starts WSL and keeps both awake |
| 01:00 | systemd user timer starts `sepalith-night-runner.service` |
| 01:00 to 08:30 | The runner starts queued jobs one after another once the preflight checks pass |
| 08:40 | `SEPALITH_DEADLINE_EPOCH`: jobs save at the next safe boundary and exit |
| 08:50 | Graceful-stop request: the stop file is written and SIGTERM goes to the job's leader process |
| 09:00 | Hard stop: SIGKILL to every process the job started, including ones in their own session |
| 09:05 | Windows task `\Sepalith\NightCheck` logs whether anything is still running |

The runner starts no job within 10 minutes of the deadline. It starts a
one-shot (non-resumable) job only if its estimated duration fits before 08:40.

Before each job, all preflight checks must pass. If one fails, the runner
tries again every 5 minutes until the deadline.

| Check | Limit |
| --- | --- |
| GPU | no compute processes, and less than 2,048 MiB used (`--max-gpu-used-mib`) |
| Host memory | Windows: at least 16,384 MiB available and commit at most 90%. Linux: at least 6 GiB `MemAvailable`. These are the thresholds of the campaign guard. |
| Disk | at least 150 GB free on `/mnt/e` (R7) |
| CUDA lock | `~/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock` is not held |

## Writing a job

A job is any command. The runner passes it a small base environment:
`PATH`, `HOME`, `USER`, locale, `TZ` and the WSL interop variables. It adds
the variables named in the job's `env` allow-list, plus these:

| Variable | Meaning |
| --- | --- |
| `SEPALITH_DEADLINE_EPOCH` | 08:40 as Unix seconds. After it, save and stop at the next safe boundary. |
| `SEPALITH_STOP_REQUEST_FILE` | Written at 08:50, together with SIGTERM to the leader: save and stop now |
| `SEPALITH_GRACEFUL_STOP_EPOCH`, `SEPALITH_HARD_STOP_EPOCH` | 08:50 and 09:00 |
| `SEPALITH_JOB_RESULT` | Optional JSON the job writes for the report |
| `SEPALITH_NIGHT_ATTEMPT_DIR` | Directory for this attempt; holds `output.log` |

Exit codes: `0` means finished (`done/`). `75` means the job saved at a safe
boundary and has work left. A resumable job then goes to `deferred/` and runs
again the next night, and it must pick up its own latest checkpoint. Any other
exit code, or a hard kill, goes to `failed/` and is flagged for the user.

The result JSON may contain `checkpoints` (list), `evaluations`,
`rule_decisions`, `summary` and `needs_user`. All of them appear in the
morning report.

The campaign trainers already stop at a deadline, and their supervisors
translate SIGTERM into the trainer's `graceful_stop` file (see
`docs/campaign/state-snapshot/resume-20260921/supervise_extension48.py`).
A night job should therefore wrap training in a supervisor that:

- sets its own deadline from `SEPALITH_DEADLINE_EPOCH`;
- treats SIGTERM or the stop-request file as a request to save and stop;
- runs the GPU work under `cuda_host_guard.py`, which holds the CUDA lock and
  watches host memory during the job;
- exits 75 when training is not finished.

Signal only the leader: the runner never sends SIGTERM to the whole group,
so a trainer is not killed before it can save.

## Queue commands

```bash
export PYTHONPATH=/home/m0hawk/Documents/Sepalith/packages/sepalith/src
python3 -m sepalith.ops.night_queue enqueue --id C07-baseline --cwd /home/m0hawk/Documents/Sepalith \
    --env HF_TOKEN --priority 10 --resumable --estimated-minutes 90 -- /path/to/job.sh arg
python3 -m sepalith.ops.night_queue enqueue --file job.json
python3 -m sepalith.ops.night_queue list
python3 -m sepalith.ops.night_queue show C07-baseline
python3 -m sepalith.ops.night_queue cancel C07-baseline   # pending or deferred -> failed
python3 -m sepalith.ops.night_queue retry C07-baseline    # failed or deferred -> pending
```

Job files live in `~/.local/state/sepalith/night-queue/{pending,running,done,failed,deferred}/`.
Priority 0 runs first. Among jobs of equal priority, deferred jobs run first,
then jobs in order of creation. `--requires ID` holds a job until `ID` is in
`done/`. The runner skips it if `ID` failed. `env` takes names only. The
enqueue step refuses a job that contains the value of any secret-looking
variable (`*TOKEN*`, `*KEY*`, `*SECRET*`, `*PASSWORD*`).

## Reports

Each night writes `~/.local/state/sepalith/night-reports/YYYY-MM-DD.md` and
`.json`, named by the morning date. The runner updates them after every job,
so a crash still leaves a partial report. Each report lists:

- the preflight results;
- each job's outcome, exit code, start and end, and the graceful and hard stop
  times;
- the job's log, checkpoints, evaluations and rule decisions;
- the jobs that did not start, and why;
- the queue afterwards;
- a **Needs the user** list.

The 09:05 check adds a `post_window_check` section and appends a line to
`night-reports/checks.jsonl`. The Windows tasks log to
`%LOCALAPPDATA%\Sepalith\night\night.log`.

## One-time setup

1. **Linux (after this change is merged).** From a stable checkout on `main`
   (normally `/home/m0hawk/Documents/Sepalith`, not a `~/.t3` worktree):

   ```bash
   scripts/night/install_systemd_units.sh --enable
   systemctl --user list-timers sepalith-night-runner.timer
   ```

   Lingering is already enabled for `m0hawk`, and `/etc/wsl.conf` has
   `systemd=true`. The units call `scripts/night/night-runner.sh`. That script
   loads `HF_TOKEN`, `ZAI_API_KEY` and `KAGGLE_API_TOKEN` from `~/.zshrc` into
   the runner's environment only.

2. **Windows (administrator PowerShell, once).**

   ```powershell
   powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu-22.04\home\m0hawk\Documents\Sepalith\scripts\night\windows\import-night-tasks.ps1 -EnableWakeTimers
   ```

   This copies `night-wake.ps1` and `night-check.ps1` to
   `%LOCALAPPDATA%\Sepalith\night\` and registers `\Sepalith\NightWake`
   (daily at 00:55, wakes the computer) and `\Sepalith\NightCheck` (daily at
   09:05). `-EnableWakeTimers` sets "Allow wake timers" to Enable on AC power
   in the current plan. Leave it out to set it by hand.

3. **Page-cache trimmer (Linux, sudo once).**

   ```bash
   sudo scripts/night/install_cache_trim.sh
   ```

   Windows counts the WSL VM's Linux page cache as used memory. On
   2026-09-28 the cache held 38 to 41 GB while Linux had 43 GB available, so
   Windows free memory fell to 7 to 10 GB. That is below the runner's 16 GB
   start threshold and the guard's 8 GB soft floor, and it blocked the CUDA
   dry run's phase B.

   `autoMemoryReclaim` cannot fix this under load. WSL 2.7.11 reclaims only
   when user CPU stays below 0.5% of all cores for 10 minutes (`dropCache`)
   or 3 minutes (`gradual`). The agent sessions measured 16 times that, and
   a training job keeps the CPU busy all night.

   `sepalith-cache-trim` is a root system service that does what `gradual`
   does, without waiting for idle:
   - every 10 seconds it caps the file cache at 8 GiB through the root
     cgroup's `memory.reclaim`, evicting the coldest pages first;
   - after a large reclaim it compacts memory, so free-page reporting returns
     the freed blocks to Windows.

   The install copies the script to `/usr/local/sbin`. Change the cap in
   `/etc/systemd/system/sepalith-cache-trim.service`. Remove it with
   `sudo scripts/night/install_cache_trim.sh --uninstall`. Check it with
   `journalctl -u sepalith-cache-trim` and `sepalith-cache-trim --status`.

4. **Power settings that must be on.**
   - Control Panel > Power Options > Change plan settings > Change advanced
     power settings > Sleep > Allow wake timers: **Enable** (on AC). Check
     with `powercfg /waketimers` after 01:00 or `powercfg /q SCHEME_CURRENT SUB_SLEEP RTCWAKE`.
   - The PC must be asleep or hibernated, not shut down, and the user must
     stay signed in. Locking the screen is fine.
     The tasks run with the signed-in user's token, so no password is stored.
   - No sleep on AC during the window: `night-wake.ps1` holds a Windows
     "system required" request until the runner finishes, so idle sleep
     cannot interrupt it. For extra safety, set "Sleep after" on AC to
     **Never** (`powercfg /change standby-timeout-ac 0`).

### Why the wake task holds WSL instead of running `true`

The card suggested `wsl.exe -d <distro> -u m0hawk -- true`. That starts WSL,
but WSL shuts an idle distribution down seconds after its last `wsl.exe`
client exits. The runner would then die mid-night. Instead, `night-wake.ps1`
runs `night-runner.sh hold`. It keeps a `wsl.exe` client open from 00:55
until the runner has finished and nothing is in `running/`, or until 09:05 at
the latest. It also holds the no-sleep request. The Windows side still needs
only the import and wake timers.

## Dry run

`scripts/night/dry_run.sh cuda|cpu [runner options]` uses its own state root,
`~/.local/state/sepalith/night-dryrun-*`, and runs two compressed windows:

- **Phase A** (deadline 4 min, graceful 5, hard 6): a synthetic CUDA job
  holds about 2 GiB and keeps the GPU busy. It must save at the deadline and
  exit 75.
- **Phase B** (deadline 10 min, graceful 12.5, hard 15): the same job
  ignores the deadline and SIGTERM. It must be killed at 15 minutes, and GPU
  memory must drop back to where it started.

The CUDA variant is GPU work, so run it only with the user's OK or inside
the window. A CPU-only run of the mechanics passed on 2026-09-28, and the CUDA run passed on
2026-09-28/29. See [receipts/C02-cpu-dry-run.json](receipts/C02-cpu-dry-run.json)
and [receipts/C02-cuda-dry-run.json](receipts/C02-cuda-dry-run.json).

## Recovery

| Situation | What to do |
| --- | --- |
| A job is in `failed/` with `hard_stop_killed` | Read the log in `night-queue/attempts/<night>/<id>-<n>/`. Check that its last checkpoint is complete, then run `night_queue retry <id>`. |
| A job is in `failed/` with `runner_lost` | The runner died, for example because the PC lost power. Check the checkpoint and retry. |
| The runner reports a stale `running/` job with live processes | Inspect the processes (`ps -o pid,pgid,cmd -g <pgid>`) and stop them by hand. The runner starts nothing until they are gone. |
| Preflight never passed | The report lists the failing check. For `host_memory`, check that `sepalith-cache-trim` is running. GPU memory held by Windows apps counts as used, so close GPU-heavy apps before bed or raise `--max-gpu-used-mib` in the service's `ExecStart`. |
| The report is missing in the morning | Look at `%LOCALAPPDATA%\Sepalith\night\night.log` to see whether WSL woke, then `systemctl --user status sepalith-night-runner` and `journalctl --user -u sepalith-night-runner --since yesterday`. |
| Stop tonight's run now | `systemctl --user stop sepalith-night-runner`. The runner asks the job to save, waits 45 seconds, then kills it and writes the report. |
| Pause the nights | `systemctl --user disable --now sepalith-night-runner.timer`, and disable the two Windows tasks in Task Scheduler. |
