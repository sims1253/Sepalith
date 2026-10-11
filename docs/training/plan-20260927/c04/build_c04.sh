#!/usr/bin/env bash
# Rebuild the card C04 artifacts: schedule v2, recipe template, arm admissions,
# source manifest, bound recipes and night-queue job files.
#
#   docs/training/plan-20260927/c04/build_c04.sh [all|bind]
#
# "all" rebuilds everything (the schedule, template and admissions are
# write-once: move old ones aside first). "bind" only rebuilds the source
# manifest and the bound recipes, which is what a code change needs.
# A corrected TRAIN row file (for example after the C06 brace fix) needs no code
# change: point ROWS/ROWS_SHA256 at it, add a data-queue file that accepts it,
# generate its length profile with `recipe_v2 length-profile`, then run "all".
set -euo pipefail

LOCAL_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
DEPLOY_ROOT=${DEPLOY_ROOT:-/home/m0hawk/sandbox/Sepalith}
PYTHON=${PYTHON:-/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python}
OUT=$LOCAL_ROOT/docs/training/plan-20260927/c04
WORK=${WORK:-/mnt/e/sepalith/campaign-20260915/data-work/C04-sft-recipe-v2}
ROWS=${ROWS:-/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/train-token-rows.jsonl}
ROWS_SHA256=${ROWS_SHA256:-3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e}
MATERIALIZATION=${MATERIALIZATION:-/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/materialization.json}
LEAD=$LOCAL_ROOT/docs/campaign/work/lead
LENGTH_PROFILE=${LENGTH_PROFILE:-$LEAD/r2-full-weight-edit-sft-eval-gate-v1/length-profile.json}
DATA_QUEUE=${DATA_QUEUE:-$LEAD/r2-full-weight-edit-sft-eval-gate-v1/data-queue.json}
DISPATCH=$LEAD/r2-full-weight-hybrid-smoke-root-v3/expected-aurora-mix-manifest.json
DEV75=/mnt/e/sepalith/campaign-20260915/campaign-overlay/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl
DEV250=/mnt/e/sepalith/campaign-20260915/data-work/C05-dev250-v1r3/dev250-v1.jsonl
DEV250_MANIFEST=/mnt/e/sepalith/campaign-20260915/data-work/C05-dev250-v1r3/manifest.json
PARENT=/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586
DECIDED="user decisions of 2026-09-27 in docs/training/plan-20260927/README.md (parent 11,586; pilot LRs 3e-6, 1e-5, 3e-5)"

export PYTHONPATH=$LOCAL_ROOT/packages/sepalith/src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
recipe() { nice -n 10 "$PYTHON" -m sepalith.training.sft.recipe_v2 --deploy-root "$DEPLOY_ROOT" --local-root "$LOCAL_ROOT" "$@"; }
ARMS=(lr3e-6:3e-6 lr1e-5:1e-5 lr3e-5:3e-5 smoke-lr1e-5:1e-5)

if [[ ${1:-all} == all ]]; then
  mkdir -p "$WORK" "$OUT/admissions" "$OUT/bound" "$OUT/jobs"
  nice -n 10 "$PYTHON" -m sepalith.training.sft.schedule_v2 --rows "$ROWS" --rows-sha256 "$ROWS_SHA256" \
    --output "$WORK/draw-schedule-v2.json" > "$WORK/draw-schedule-v2.summary.json"
  recipe template --rows "$ROWS" --rows-sha256 "$ROWS_SHA256" --schedule "$WORK/draw-schedule-v2.json" \
    --materialization "$MATERIALIZATION" --length-profile "$LENGTH_PROFILE" --data-queue "$DATA_QUEUE" \
    --optimizer-dispatch "$DISPATCH" --dev75 "$DEV75" --dev250 "$DEV250" --dev250-manifest "$DEV250_MANIFEST" \
    --output "$OUT/sft-v2-recipe.template.json"
  recipe admission-template --template "$OUT/sft-v2-recipe.template.json" --output "$OUT/admissions/arm-admission.template.json"
  for arm in "${ARMS[@]}"; do
    name=${arm%%:*}; lr=${arm##*:}
    purpose="C08 learning-rate pilot arm (stops at the first gate; the R2 winner resumes in C09)"
    [[ $name == smoke-* ]] && purpose="C07 SFT smoke: 10 updates, save, resume, 10 more; output kept for inspection only"
    recipe admission --template "$OUT/sft-v2-recipe.template.json" --parent "$PARENT" --parent-step 11586 \
      --learning-rate "$lr" --arm "$name" --purpose "$purpose" --decided-by "$DECIDED" --output "$OUT/admissions/$name.json"
  done
fi

rm -f "$OUT/bound/source-manifest.json" "$OUT"/bound/sft-v2-*.json
recipe source-manifest --output "$OUT/bound/source-manifest.json"
for arm in "${ARMS[@]}"; do
  name=${arm%%:*}
  recipe bind --template "$OUT/sft-v2-recipe.template.json" --admission "$OUT/admissions/$name.json" \
    --source-manifest "$OUT/bound/source-manifest.json" --output "$OUT/bound/sft-v2-$name.json"
done
nice -n 10 "$PYTHON" -m sepalith.training.sft.c04_jobs --deploy-root "$DEPLOY_ROOT" --output "$OUT/jobs"
