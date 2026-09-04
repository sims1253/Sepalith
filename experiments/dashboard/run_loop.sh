#!/bin/zsh
# run_loop.sh — SELF-UPDATING dashboard daemon (detached, wall-clock-paced).
#
# One cycle every DASHBOARD_LOOP_S (default 1800s):
#   gather repo digest -> muse-spark editorial patch on state_v2.json
#   (dynamic sections only: stand / live-now / timeline / next; validated,
#   fail-soft) -> deterministic queue parse from EXPERIMENT-QUEUE.md ->
#   rebuild index.html via build_dashboard_v2.py (static ground truth +
#   mtime-cached inventories refresh even when spark flakes) ->
#   npx postplan upload. Cycle logic: refresh_cycle_v2.py (read it for the
#   validation/hard-rule details). Design/copy provenance: design_v2.json +
#   copy_v2.json (muse-spark, designer of record — see spark_design_v2.py).
#
# STOP convention (matches the tournament's results/ideation_tournament
# pattern): create
#   experiments/dashboard/STOP_DASHBOARD
# to stop after the current cycle; delete it and relaunch to resume.
#
# Single instance: /tmp/dashboard_loop.lock (flock). No GPU anywhere —
# CUDA_VISIBLE_DEVICES is emptied defensively; all heavy steps run niced.
# Upload auth/quota: the loop self-stops after 5 consecutive upload
# failures (creates the STOP file itself and logs why) — auth problems
# are a blocker to report, not spin on.
#
# Launch (detached, reaper-proof — the harness reaps long-lived tracked
# tasks, so this MUST NOT be a tracked task):
#   nohup setsid zsh /home/m0hawk/Documents/Sepalith/experiments/\
# dashboard/run_loop.sh >/dev/null 2>&1 & disown
# Log: experiments/dashboard/results/loop.log
# Counters: experiments/dashboard/results/loop_state.json
eval "$(grep -E '^export (POSTPLAN_API_KEY|OPENCODE_API_KEY)=' ~/.zshrc)"
export CUDA_VISIBLE_DEVICES=""
cd /home/m0hawk/Documents/Sepalith/experiments/dashboard
D=results
LOG=$D/loop.log
LOOP_S=${DASHBOARD_LOOP_S:-1800}
STOP=STOP_DASHBOARD
mkdir -p "$D"
# resolve npx through any fnm multishell symlink NOW (multishell dirs die
# with the launching shell; the resolved binary path survives)
export NPX_BIN=$(readlink -f "$(command -v npx)")
[ -x "$NPX_BIN" ] || { echo "[loop] FATAL: npx not resolvable" >> "$LOG"; exit 1; }

# single-instance lock
exec 9>/tmp/dashboard_loop.lock
flock -n 9 || { echo "[loop] another instance holds /tmp/dashboard_loop.lock; exit $(date)" >> "$LOG"; exit 0; }

echo "[loop] start pid=$$ loop=${LOOP_S}s npx=$NPX_BIN $(date)" >> "$LOG"
upload_fails=0
while true; do
  [ -f "$STOP" ] && break
  echo "[loop] cycle start $(date)" >> "$LOG"
  nice -n 19 python3 -u refresh_cycle_v2.py >> "$LOG" 2>&1
  rc=$?
  echo "[loop] cycle rc=$rc $(date)" >> "$LOG"
  [ -f "$STOP" ] && break
  # upload failed (auth/quota/network)? rc==3 from refresh_cycle.py
  if (( rc == 3 )); then
    upload_fails=$((upload_fails + 1))
    if (( upload_fails >= 5 )); then
      touch "$STOP"
      echo "[loop] SELF-STOP: $upload_fails consecutive upload failures (auth/quota?) — created $STOP; fix POSTPLAN_API_KEY/quota, remove the flag, relaunch $(date)" >> "$LOG"
      break
    fi
  else
    upload_fails=0
  fi
  # chunked sleep so a STOP flag is honored within a minute
  waited=0
  while (( waited < LOOP_S )); do
    [ -f "$STOP" ] && break
    sleep 60
    waited=$((waited + 60))
  done
  [ -f "$STOP" ] && break
done
echo "[loop] exit (STOP file) $(date)" >> "$LOG"
