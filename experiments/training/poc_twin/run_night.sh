#!/bin/bash
# Twin POC orchestrator: queues EVERYTHING behind the 16GB-free launch gate.
# Phases: LR probes -> LR pick -> step-budget pick -> Muon arm -> AdamW arm
# -> paired eval -> final ckpt copies. All GPU pythons run nice -n 10 with
# the in-process 0.42 memory cap + watchdog (see train.py).
set -u
POC=/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
LOG=$POC/logs/orchestration.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_gate() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge 16384 ]; then log "gate OPEN (${free}MiB free)"; return 0; fi
    log "gate closed (${free}MiB free); sleeping 600s"
    sleep 600
  done
}

cd $POC
log "orchestrator start"

# ---------- phase 1: LR probes ----------
HOUR=$(date +%H)
if [ "$HOUR" -lt 5 ]; then PSTEPS=40; elif [ "$HOUR" -lt 7 ]; then PSTEPS=30; else PSTEPS=25; fi
wait_gate
for cfg in "adamw 0.001" "adamw 0.002" "adamw 0.004" "muon 0.01" "muon 0.02" "muon 0.04"; do
  set -- $cfg
  log "probe arm=$1 lr=$2 steps=$PSTEPS"
  nice -n 10 $PY train.py --arm $1 --lr $2 --probe --steps $PSTEPS \
    --tokens-per-step 262144 --compile --tag probe_$1_$2 --log-every 10 \
    >> $POC/logs/probe_stdout.log 2>&1
  tail -1 $POC/logs/probe_$1_$2.jsonl | tee -a "$LOG"
done

# ---------- phase 2: pick LRs + budget ----------
$PY pick.py || exit 1
LR_ADAMW=$($PY -c "import json;print(json.load(open('logs/plan.json'))['lr_adamw'])")
LR_MUON=$($PY -c "import json;print(json.load(open('logs/plan.json'))['lr_muon'])")
STEPS=$($PY -c "import json;print(json.load(open('logs/plan.json'))['steps'])")
log "plan: lr_adamw=$LR_ADAMW lr_muon=$LR_MUON steps=$STEPS (both arms identical)"

# ---------- phase 3: arms ----------
wait_gate
log "arm MUON start"
nice -n 10 $PY train.py --arm muon --lr $LR_MUON --lr-embed $LR_ADAMW \
  --steps $STEPS --tokens-per-step 524288 --compile --tag muon \
  >> $POC/logs/muon_stdout.log 2>&1
log "arm MUON done (rc=$?)"

wait_gate
log "arm ADAMW start"
nice -n 10 $PY train.py --arm adamw --lr $LR_ADAMW \
  --steps $STEPS --tokens-per-step 524288 --compile --tag adamw \
  >> $POC/logs/adamw_stdout.log 2>&1
log "arm ADAMW done (rc=$?)"

# ---------- phase 4: eval + final copies ----------
nice -n 10 $PY evaluate.py --ckptA /tmp/poc_twin/ckpt_muon/final.pt \
  --ckptB /tmp/poc_twin/ckpt_adamw/final.pt --armA muon --armB adamw \
  >> $POC/logs/eval_stdout.log 2>&1
log "eval done (rc=$?)"
cp /tmp/poc_twin/ckpt_muon/final.pt $POC/checkpoints/muon_final.pt 2>/dev/null
cp /tmp/poc_twin/ckpt_adamw/final.pt $POC/checkpoints/adamw_final.pt 2>/dev/null
cp /tmp/poc_twin/ckpt_muon/mid.pt $POC/checkpoints/muon_mid.pt 2>/dev/null
cp /tmp/poc_twin/ckpt_adamw/mid.pt $POC/checkpoints/adamw_mid.pt 2>/dev/null
log "orchestrator complete"
