#!/usr/bin/env bash
# Build the pinned llama.cpp (tag b10453, commit 3cb7ffb) outside the repo.
#
#   scripts/packaging/build_llama_cpp.sh [src|cpu|cuda|vulkan|convert|all]
#
# Layout under SEPALITH_LLAMA_CPP (default ~/.local/opt/llama.cpp-b10453):
#   src/          full (non-shallow) clone; a shallow clone reports "build: 1"
#   build-cpu/    CPU tree (always)
#   build-cuda/   -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120, nvcc from CUDA_HOME
#   build-vulkan/ only with `vulkan` and only if Vulkan headers and glslc exist
#   venv-convert/ uv venv with llama.cpp's converter requirements
# Build only: nothing here touches the GPU. Builds run under nice with at most
# SEPALITH_BUILD_JOBS (default 8) jobs.
set -euo pipefail
PIN=3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70
TAG=b10453
OPT=${SEPALITH_LLAMA_CPP:-$HOME/.local/opt/llama.cpp-$TAG}
SRC=$OPT/src
JOBS=${SEPALITH_BUILD_JOBS:-8}
CUDA_HOME=${CUDA_HOME:-/opt/cuda}
what=${1:-all}
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)

fetch_src() {
  if [[ ! -d $SRC/.git ]]; then
    mkdir -p "$OPT"
    git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$SRC"
  fi
  if [[ $(git -C "$SRC" rev-parse --is-shallow-repository) == true ]]; then
    git -C "$SRC" fetch --unshallow origin
  fi
  git -C "$SRC" fetch origin "refs/tags/$TAG:refs/tags/$TAG"
  test "$(git -C "$SRC" rev-parse "$TAG^{commit}")" = "$PIN"
  git -C "$SRC" -c advice.detachedHead=false checkout --quiet "$PIN"
}

check_pin() {
  test "$(git -C "$SRC" rev-parse HEAD)" = "$PIN" || { echo "src is not at $PIN" >&2; exit 1; }
  test "$(git -C "$SRC" rev-parse --is-shallow-repository)" = false || { echo "src is shallow; run: $0 src" >&2; exit 1; }
}

build_cpu() {
  check_pin
  nice -n 10 cmake -S "$SRC" -B "$OPT/build-cpu" "${common[@]}"
  nice -n 10 cmake --build "$OPT/build-cpu" -j "$JOBS"
}

build_cuda() {
  check_pin
  PATH=$CUDA_HOME/bin:$PATH nice -n 10 cmake -S "$SRC" -B "$OPT/build-cuda" "${common[@]}" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME"
  PATH=$CUDA_HOME/bin:$PATH nice -n 10 cmake --build "$OPT/build-cuda" -j "$JOBS"
}

build_vulkan() {
  check_pin
  if [[ ! -f /usr/include/vulkan/vulkan.h ]] || ! command -v glslc >/dev/null; then
    echo "skipping Vulkan: needs Vulkan headers and glslc (packages vulkan-headers, shaderc)" >&2
    return 0
  fi
  nice -n 10 cmake -S "$SRC" -B "$OPT/build-vulkan" "${common[@]}" -DGGML_VULKAN=ON
  nice -n 10 cmake --build "$OPT/build-vulkan" -j "$JOBS"
}

build_convert() {
  check_pin
  [[ -x $OPT/venv-convert/bin/python ]] || uv venv --python 3.12 "$OPT/venv-convert"
  nice -n 10 uv pip install --python "$OPT/venv-convert/bin/python" \
    -r "$SRC/requirements/requirements-convert_hf_to_gguf.txt"
  # Sepalith checkpoints are saved by transformers 5.5.0 (tokenizer_class
  # "TokenizersBackend"), which the pinned 4.57.6 cannot load. The campaign's
  # own b10453 conversions ran under 5.5.0 as well.
  nice -n 10 uv pip install --python "$OPT/venv-convert/bin/python" \
    "transformers==${SEPALITH_CONVERT_TRANSFORMERS:-5.5.0}"
}

case $what in
  src) fetch_src ;;
  cpu) build_cpu ;;
  cuda) build_cuda ;;
  vulkan) build_vulkan ;;
  convert) build_convert ;;
  all) fetch_src; build_cpu; build_cuda; build_vulkan; build_convert ;;
  *) echo "usage: $0 [src|cpu|cuda|vulkan|convert|all]" >&2; exit 2 ;;
esac
