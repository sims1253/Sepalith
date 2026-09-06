#!/usr/bin/env bash
# X5-S1 chain (queue §3 X5, S1 arm — zcode-x5-s1).
# Stage A (two-pass SC continuation from banked md_final.pt) smoke->full
# -> Stage B (FPF) smoke->full -> 216-row eval legs -> residual replay ->
# verdict. DETACHED per B13/B8 pattern (setsid nohup); the owning session
# runs a short tracked watcher + heartbeats q30min. One CUDA workload at
# a time (W37); trainer taskset 16-23; memfrac 0.42 (anchor class).
# Pre-registered budget: A 400 + B 260 steps (~3.3h train) + ~1h eval;
# readouts + verdict bars in x5_s1_train.py / x5_s1_eval.py headers.
set -u
cd /home/m0hawk/Documents/Sepalith
RUN=/mnt/h/sepalith/runs/x5_s1
LOG=$RUN/chain.log
mkdir -p "$RUN"
exec >>"$LOG" 2>&1
echo "=== X5-S1 chain start $(date -Is) pid $$ ==="
export POC_MEM_FRACTION=0.42

PY() { taskset -c 16-23 /home/m0hawk/.local/bin/uv run python -m "$@"; }

# gate <tag> <jsonl> <start_line> <min_step> — a done event at >= min_step
# appeared AFTER start_line, losses finite, carry channel moved off zero.
gate() {
  local tag=$1 j=$2 start=$3 minstep=$4
  python3 - "$j" "$start" "$minstep" <<'EOF'
import json, sys
lines = open(sys.argv[1]).read().splitlines()[int(sys.argv[2]):]
recs = [json.loads(l) for l in lines if l.strip()]
dones = [r for r in recs if r.get("event") == "done"
         and r.get("step", 0) >= int(sys.argv[3])]
assert dones, f"no done event at step>={sys.argv[3]} since line {sys.argv[2]}"
losses = [r["loss"] for r in recs if "loss" in r]
assert losses and all(v == v and abs(v) < 1e4 for v in losses), losses[-3:]
assert any(r.get("carry_w_absmax", 0) > 0 for r in recs), \
    "carry channel never moved"
print(f"gate ok: done step {dones[-1]['step']}, loss tail {losses[-1]:.4f}")
EOF
  [ $? -eq 0 ] || { echo "GATE FAIL $tag — chain aborting"; exit 1; }
}

nlines() { wc -l < "$1" 2>/dev/null || echo 0; }
LA=experiments/training/poc_diff/logs_x5_s1_a.jsonl
LB=experiments/training/poc_diff/logs_x5_s1_b.jsonl

echo "--- stage A smoke $(date -Is)"
N=$(nlines $LA)
PY experiments.training.poc_diff.x5_s1_train --stage A \
  --resume /mnt/h/sepalith/runs/poc_diff/md_final.pt --smoke --no-gate
gate A-smoke $LA "$N" 10

echo "--- stage A full (400 steps) $(date -Is)"
N=$(nlines $LA)
PY experiments.training.poc_diff.x5_s1_train --stage A --steps 400 --compile \
  --resume /mnt/h/sepalith/runs/poc_diff/md_final.pt
gate A-full $LA "$N" 400

echo "--- stage B smoke $(date -Is)"
N=$(nlines $LB)
PY experiments.training.poc_diff.x5_s1_train --stage B \
  --resume /tmp/poc_diff/ckpt/x5_s1_a_final.pt --smoke --no-gate
gate B-smoke $LB "$N" 10

echo "--- stage B full (260 steps) $(date -Is)"
N=$(nlines $LB)
PY experiments.training.poc_diff.x5_s1_train --stage B --steps 260 --compile \
  --resume /tmp/poc_diff/ckpt/x5_s1_a_final.pt
gate B-full $LB "$N" 260

echo "--- eval legs $(date -Is)"
PY experiments.training.poc_diff.x5_s1_eval run \
  --ckpt /tmp/poc_diff/ckpt/x5_s1_b_final.pt \
  --out experiments/training/poc_diff/results_x5_s1
[ -f experiments/training/poc_diff/results_x5_s1/eval_x5_s1.json ] \
  || { echo "GATE FAIL eval — chain aborting"; exit 1; }

echo "--- residual replay $(date -Is)"
PY experiments.training.poc_diff.x5_s1_eval resid --resume \
  --ckpt /tmp/poc_diff/ckpt/x5_s1_b_final.pt \
  --out experiments/training/poc_diff/results_x5_s1
[ -f experiments/training/poc_diff/results_x5_s1/analysis_x5_s1.json ] \
  || { echo "GATE FAIL resid — chain aborting"; exit 1; }

echo "--- verdict $(date -Is)"
PY experiments.training.poc_diff.x5_s1_eval verdict \
  --out experiments/training/poc_diff/results_x5_s1

# mirror per-row data to the NAS (repo jsonl is gitignored; B8b convention)
rsync -a experiments/training/poc_diff/results_x5_s1/ "$RUN/results_x5_s1/"
echo "=== X5-S1 chain end $(date -Is) ==="
