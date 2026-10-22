#!/usr/bin/env bash
# Card C13: data, templates, bound recipes and night jobs for the offline steps.
# CPU only (nice 10, at most 8 threads); outputs are write-once.
#
#   docs/training/plan-20260927/c13/build_c13.sh data|templates|bind|jobs [args]
#
#   data       RFT cohort + DPO pairs from one C12 census (MODEL_ID, CENSUS_DIR,
#              CENSUS_SUMMARY_SHA256); writes data-work/C13-offline-v1/<model id>/
#   templates  parent-free rft-<ITER> and dpo-<ITER> templates on that data, plus the
#              source manifest of the code the recipes bind (an existing manifest is
#              kept only if the tree still matches it)
#   bind KIND PARENT_TAG PARENT DECISION PARENT_RECIPE
#              bind a template to a starting model: KIND rft or dpo; PARENT the
#              checkpoint directory; DECISION its gate decision record; PARENT_RECIPE
#              the recipe that decision binds (an SFT v2 arm or a C13 RFT recipe)
#   jobs RECIPE
#              write the night-queue job files of a bound recipe (not enqueued)
#
# Arm A (the parent until R3 says otherwise, user decision 2026-10-17):
#   build_c13.sh bind rft arm-a-536 \
#     /mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536 \
#     ~/.local/state/sepalith/campaign-20260915/evaluation-gates/SFT-v2-r4-lr1e-5/decisions/decision-step-536-1.json \
#     docs/training/plan-20260927/c04/bound/sft-v2-lr1e-5.json
# If R3 adopts arm B (gate 684), R4 needs a new census on B first: rerun C12, then
# `data` with MODEL_ID=sepalith-2b-edit-sft-arm-b, `templates` with TAG=arm-b, and bind
# with the arm-B checkpoint, decision and docs/training/plan-20260927/c10/bound/sft-v2-arm-b-lr1e-5.json.
# DPO after an adopted RFT step: bind dpo with the RFT result checkpoint, its final
# gate decision (evaluation-gates/C13-rft-1-<tag>/decisions/) and the RFT recipe.
#
# Second offline iteration at temperature 1.2 (R4 step 4, user decision 2026-10-22):
# the same rules on the census of arm A 536 at T = 1.2; only ITER, the census and the
# names change (status/C13.json, iteration_2):
#   ITER=2 MODEL_ID=sepalith-2b-edit-sft-arm-a-t12 TAG=arm-a-t12 \
#     CENSUS_DIR=~/.local/state/sepalith/c12/census-arm-a-t12 CENSUS_SUMMARY_SHA256=<sha> build_c13.sh data
#   ITER=2 MODEL_ID=... TAG=arm-a-t12 build_c13.sh templates
#   ITER=2 TAG=arm-a-t12 DECIDED_BY=... build_c13.sh bind rft arm-a-536-t12 <checkpoint-536> <decision-536> <sft recipe>
set -euo pipefail

LOCAL_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
PYTHON=${PYTHON:-/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python}
MODEL_ID=${MODEL_ID:-sepalith-2b-edit-sft-arm-a}
TAG=${TAG:-arm-a}
ITER=${ITER:-1}
CENSUS_DIR=${CENSUS_DIR:-$HOME/.local/state/sepalith/c12/census-arm-a}
CENSUS_SUMMARY_SHA256=${CENSUS_SUMMARY_SHA256:-eaeadf7b9640ddd069cff093d92fedb8831042ca53f79415b0e0187bf3269171}
DATA=/mnt/e/sepalith/campaign-20260915/data-work/C13-offline-v1/$MODEL_ID
HERE=$LOCAL_ROOT/docs/training/plan-20260927/c13

export PYTHONPATH=$LOCAL_ROOT/packages/sepalith/src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

step=${1:?usage: build_c13.sh data|templates|bind|jobs}
shift
case $step in
  data)
    nice -n 10 "$PYTHON" -m sepalith.training.offline.data build --census-dir "$CENSUS_DIR" \
      --census-summary-sha256 "$CENSUS_SUMMARY_SHA256" --model-id "$MODEL_ID" --output "$DATA" \
      ${EXPECTED_RFT_TOTAL:+--expected-rft-total "$EXPECTED_RFT_TOTAL"}
    ;;
  templates)
    if [[ -e $HERE/bound/source-manifest.json ]]; then
      fresh=$(mktemp -d)
      nice -n 10 "$PYTHON" -m sepalith.training.offline.recipe source-manifest --output "$fresh/source-manifest.json" >/dev/null
      cmp -s "$fresh/source-manifest.json" "$HERE/bound/source-manifest.json" \
        || { echo "bound/source-manifest.json no longer matches the tree; write a new one" >&2; exit 1; }
      rm -r "$fresh"
    else
      nice -n 10 "$PYTHON" -m sepalith.training.offline.recipe source-manifest --output "$HERE/bound/source-manifest.json"
    fi
    for kind in rft dpo; do
      nice -n 10 "$PYTHON" -m sepalith.training.offline.recipe template --kind "$kind" --step-name "$kind-$ITER" \
        --data-manifest "$DATA/manifest.json" --output "$HERE/templates/$kind-$ITER-$TAG-census.template.json"
    done
    ;;
  bind)
    kind=$1 parent_tag=$2 parent=$3 decision=$4 parent_recipe=$5
    nice -n 10 "$PYTHON" -m sepalith.training.offline.recipe bind \
      --template "$HERE/templates/$kind-$ITER-${TEMPLATE_TAG:-$TAG}-census.template.json" --parent "$parent" \
      --parent-decision "$decision" --parent-recipe "$parent_recipe" --parent-tag "$parent_tag" \
      --decided-by "${DECIDED_BY:?set DECIDED_BY to who decided the starting model and where it is recorded}" \
      --source-manifest "$HERE/bound/source-manifest.json" --output "$HERE/bound/$kind-$ITER-$parent_tag.json"
    ;;
  jobs)
    nice -n 10 "$PYTHON" -m sepalith.training.offline.jobs --recipe "$1" --output "$HERE/jobs"
    ;;
  *)
    echo "unknown step: $step" >&2
    exit 2
    ;;
esac
