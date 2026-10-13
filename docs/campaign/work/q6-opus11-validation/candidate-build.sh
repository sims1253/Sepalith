#!/usr/bin/env bash
set -euo pipefail

# RUN-05 bounded recipe. Root runs this in a temporary area; this worker did not run it.
# The recipe performs only local archive/apply/configure/build/hash work. It never starts
# a model or server and never enables GGML_VK_Q6K_EXACT_DIV_POISON.

PLAN=${PLAN:-/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb}
SRC=${SRC:-/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453}
COMMIT=3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70
GLSLC=${GLSLC:-/usr/bin/glslc}
GEN=${GENERATOR:-Ninja}
PATCH_FILE="$PLAN/docs/campaign/work/q6-opus11-validation/candidate.patch"

if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$COMMIT" ]]; then
    printf 'source commit mismatch\n' >&2
    exit 2
fi
if [[ ! -x "$GLSLC" ]]; then
    printf 'missing glslc: %s\n' "$GLSLC" >&2
    exit 2
fi
if [[ ! -s "$PATCH_FILE" ]]; then
    printf 'missing candidate patch: %s\n' "$PATCH_FILE" >&2
    exit 2
fi

RUN=$(mktemp -d "${TMPDIR:-/tmp}/run05-opus11-q6k.XXXXXX")
mkdir -p "$RUN/src" "$RUN/baseline-build" "$RUN/candidate-build"
# A local git archive keeps the candidate source separate from the pinned source tree.
git -C "$SRC" archive "$COMMIT" | tar -x -C "$RUN/src"
patch -p1 -d "$RUN/src" < "$PATCH_FILE"
Q6="$RUN/src/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp"
python3 - "$Q6" <<'PY2'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
marker = '#define GGML_VK_Q6K_EXACT_DIV 1'
if marker not in s:
    prefix = '#version 450\n'
    if not s.startswith(prefix):
        raise SystemExit('unexpected q6 shader header')
    s = prefix + marker + '\n' + s[len(prefix):]
    p.write_text(s)
PY2

COMMON=(
  -G "$GEN"
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_VULKAN=ON
  -DGGML_VULKAN_DEBUG=ON
  -DGGML_NATIVE=OFF
  -DGGML_AVX2=ON
  -DVulkan_GLSLC_EXECUTABLE="$GLSLC"
)
cmake -S "$SRC" -B "$RUN/baseline-build" "${COMMON[@]}"
cmake --build "$RUN/baseline-build" --target ggml-vulkan --parallel 1
cmake -S "$RUN/src" -B "$RUN/candidate-build" "${COMMON[@]}"
cmake --build "$RUN/candidate-build" --target ggml-vulkan --parallel 1

python3 - "$RUN" <<'PY2'
from pathlib import Path
import hashlib, json, sys
run = Path(sys.argv[1])
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
record = {}
for label in ('baseline-build', 'candidate-build'):
    base = run / label
    files = {}
    for p in sorted(base.rglob('*')):
        if p.is_file() and (p.suffix in ('.spv', '.cpp', '.hpp') or p.name == 'CMakeCache.txt'):
            files[str(p.relative_to(base))] = digest(p)
    record[label] = files
(run / 'q6-generated-hashes.json').write_text(json.dumps(record, indent=2) + '\n')
PY2

printf 'temporary result directory: %s\n' "$RUN"
printf 'Next root-owned action: run the exact Q6 parity/dispatch trace with GGML_VULKAN_DEBUG output; no POISON build.\n'
