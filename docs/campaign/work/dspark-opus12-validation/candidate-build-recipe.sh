#!/usr/bin/env bash
set -euo pipefail

# Root-owned, separate staging recipe.  It never edits the pinned checkout.
# Usage: candidate-build-recipe.sh [staging-directory]
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PINNED=/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453
PIN=3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70
EVIDENCE="$PLAN/docs/campaign/work/dspark-opus12-validation"
RUN_DIR=${1:-/tmp/run06-dspark-opus12-candidate}

test "$(git -C "$PINNED" rev-parse HEAD)" = "$PIN"
mkdir -p "$RUN_DIR/src/common" "$RUN_DIR/src/tests" "$RUN_DIR/build"
git -C "$PINNED" archive "$PIN" | tar -x -C "$RUN_DIR/src"
cp "$EVIDENCE/new-header/speculative-dspark.h" "$RUN_DIR/src/common/speculative-dspark.h"
cp "$EVIDENCE/tests/test-dspark-prefix.cpp" "$RUN_DIR/src/tests/test-dspark-prefix.cpp"
(cd "$RUN_DIR/src" && patch --batch --forward --fuzz=0 -p1 < "$EVIDENCE/candidate.patch")
(cd "$RUN_DIR/src" && patch --batch --forward --fuzz=0 -p1 < "$EVIDENCE/audit/candidate-safety-amendment.patch")

# The safety amendment is mandatory for runtime admission: return success from
# seq_rm is followed by an aggregate max-position check before another write.
grep -Fq 'ctx_dft cleanup left a row at/after p0' "$RUN_DIR/src/common/speculative.cpp"

env PYTHONDONTWRITEBYTECODE=1 g++ -std=c++17 -O1 -Wall -Wextra \
    -I"$RUN_DIR/src/common" "$RUN_DIR/src/tests/test-dspark-prefix.cpp" \
    -o "$RUN_DIR/build/test-dspark-prefix"
env PYTHONDONTWRITEBYTECODE=1 "$RUN_DIR/build/test-dspark-prefix"

# Root performs the project build and model run after reviewing the identities.
# Required runtime profile: DSpark, n_max=3, n_seq=1, n_batch/n_ubatch >= 7,
# LLAMA_EXP_DSPARK_FULL_BLOCK=1; compare with the environment unset.
echo "staged candidate at $RUN_DIR/src"
