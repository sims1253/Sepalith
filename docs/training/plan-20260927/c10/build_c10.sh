#!/usr/bin/env bash
# Card C10 data build: roxygen admission, the arm-B cohort, its length profile
# and the proposed arm-B draw schedule. CPU only; writes under data-work/C10-*.
#
#   docs/training/plan-20260927/c10/build_c10.sh [admit|schedule|all|v4]
#
# "v4" (user decision 2026-10-12) rebuilds the cohort on the brace-reindented
# DAT10-finish-source-repair-v4-braceindent TRAIN rows in a new directory
# (arm-b-v4/), copying the admitted roxygen rows byte for byte from the v3-based
# cohort, then builds its length profile and the option (a) schedule.
#
# "admit" runs the admission checks and writes the cohort (the hand
# classifications in semantic-classifications.json must cover the drawn
# sample). "schedule" builds the length profile and the arm-B schedule with the
# C04 schedule v2 code (PR #45) unchanged. Outputs are write-once.
#
# Schedule: option (a), user decision 2026-10-12. 13 edit + 3 no-op per update
# (18.75%); with 1,094 no-op rows and 17,773 edits schedule v2 needs a fourth
# no-op pass, so up to 4 exposures per no-op row are accepted.
set -euo pipefail

LOCAL_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
PYTHON=${PYTHON:-/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python}
WORK=${WORK:-/mnt/e/sepalith/campaign-20260915/data-work/C10-roxygen-admission-v1}
HERE=$LOCAL_ROOT/docs/training/plan-20260927/c10

export PYTHONPATH=$LOCAL_ROOT/packages/sepalith/src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

step=${1:-all}
if [[ $step == admit || $step == all ]]; then
  nice -n 10 "$PYTHON" "$HERE/admit_roxygen.py" check --work-dir "$WORK/work"
  nice -n 10 "$PYTHON" "$HERE/admit_roxygen.py" finalize --work-dir "$WORK/work" --output-dir "$WORK/admission"
fi
if [[ $step == schedule || $step == all ]]; then
  ROWS=$WORK/admission/arm-b-cohort-rows.jsonl
  ROWS_SHA256=$(sha256sum "$ROWS" | cut -d' ' -f1)
  mkdir -p "$WORK/arm-b"
  nice -n 10 "$PYTHON" -m sepalith.training.sft.recipe_v2 length-profile --rows "$ROWS" --rows-sha256 "$ROWS_SHA256" \
    --output "$WORK/arm-b/length-profile.json"
  nice -n 10 "$PYTHON" -m sepalith.training.sft.schedule_v2 --rows "$ROWS" --rows-sha256 "$ROWS_SHA256" \
    --output "$WORK/arm-b/draw-schedule-v2-armb-proposed.json" > "$WORK/arm-b/draw-schedule-v2-armb-proposed.summary.json"
fi
if [[ $step == v4 ]]; then
  V4=$WORK/arm-b-v4
  nice -n 10 "$PYTHON" "$HERE/admit_roxygen.py" rebase-cohort \
    --base-cohort "$WORK/admission/arm-b-cohort-rows.jsonl" \
    --base-cohort-sha256 a43d775b950d1cdd1a598a6c585589dc294e3eb8dacf4a54becc4bb7b0190949 \
    --train-rows /mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl \
    --train-rows-sha256 ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef \
    --admission-dir "$WORK/admission" --output-dir "$V4"
  ROWS=$V4/arm-b-cohort-rows.jsonl
  ROWS_SHA256=$(sha256sum "$ROWS" | cut -d' ' -f1)
  nice -n 10 "$PYTHON" -m sepalith.training.sft.recipe_v2 length-profile --rows "$ROWS" --rows-sha256 "$ROWS_SHA256" \
    --output "$V4/length-profile.json"
  nice -n 10 "$PYTHON" -m sepalith.training.sft.schedule_v2 --rows "$ROWS" --rows-sha256 "$ROWS_SHA256" \
    --output "$V4/draw-schedule-v2-armb.json" > "$V4/draw-schedule-v2-armb.summary.json"
fi
