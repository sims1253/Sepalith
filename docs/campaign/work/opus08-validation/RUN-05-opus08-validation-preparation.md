# RUN-05 OPUS08 q6_K validation preparation

Prepared 2026-09-12 for root review.  This packet is CPU-only preparation.  It does not change the pinned source, invoke Vulkan, load a model, start a server, or touch the local RL run.

## Pinned source and contract

The source root is `/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453`, at commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`.  The current q6_K shader is unchanged at this checkout.  The source files used by the CPU reference and later generator are hash-pinned as follows:

| file | SHA-256 |
| --- | --- |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp` | `bc785a2457aa5b04d416e18f5ae2304da267e35cea7b54ac67b5d20987e450af` |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl` | `3d493e6ee65459a44b2b81645f34f4aa5f3bca4a22ca18cb07706535032d9d8c` |
| `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` | `8b65027ce3cbd48d7a6a926717cfe78bc27ea8156ba8e2bb3fbd6d07142e710a` |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl` | `fc7bb5cc47bd9761f8a3484466e10b61d7866df41a150cdd47cfa1f389110323` |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | `c27978351492b0cdffac906f10662a979e5e7a0810e0b73b59853e7230a231d9` |
| `ggml/src/ggml-vulkan/CMakeLists.txt` | `95b77526cbede60b669abf1bc13657956284d0b20835096e64f24682a5a1a0e2` |
| `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt` | `0cb2c1fd49730793094cbaca4a46d0e0fc56c8f279aa9712a4e7a53535400d64` |
| `ggml/src/ggml-common.h` | `af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0` |
| `ggml/src/ggml-quants.c` | `07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d` |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | `c73e8f7980cd416f7fd30860c43f85e4ecacb7db84c0bf057cbb505aaf90142f` |

`block_q6_K` is 210 bytes: `ql[128]`, `qh[64]`, `scales[16]`, and binary16 `d`.  The canonical CPU function is `dequantize_row_q6_K` in `ggml-quants.c:1939`; it consumes two 128-value halves, with four 32-wide groups and scale offsets `is + 0, +2, +4, +6`.  The Vulkan lane map is in `mul_mat_vec_q6_k.comp:83-100`: 16 lanes per 256-value block, `ql_offset = 64*v_im + l0`, `qh_offset = 32*v_im + l0`, and `y_offset = 128*v_im + l0`.  The generator selects `main` at `vulkan-shaders-gen.cpp:739-746` and emits the q6 shader variants from `mul_mat_vec_q6_k.comp`.

The OPUS08 reference fixes subgroup and workgroup size to 64.  Thus `it_size = gl_WorkGroupSize.x / 16 = 4`, and one workgroup covers four q6_K blocks.  The reference also runs a three-block tail workgroup to exercise the shader's `all_threads == false` path.  This is a lane/index check, not a claim that a future specialized kernel may skip the generic runtime's shape checks.

## CPU reference and executed coverage

`q6k_index_coverage.py` creates deterministic in-memory q6_K blocks and compares the shader lane map with the canonical C dequantization formula.  It records every ql, qh, scale, packed16, and output index.  It fails on duplicate or missing scalar writes, out-of-range byte indices, or a numeric/quantized-term mismatch.  The test includes an intentional one-byte ql-index slip; it is detected by decoded-value parity even though the same ql byte is also used for the other nibble.

The focused test file is `test_q6k_index_coverage.py`.  The executed command was:

```sh
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHONPATH="$PLAN/docs/campaign/work/opus08-validation" \
  python3 "$PLAN/docs/campaign/work/opus08-validation/test_q6k_index_coverage.py"
```

Result: 5 tests passed.  Direct source-pinned execution was also run:

```sh
python3 /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/opus08-validation/q6k_index_coverage.py \
  --source-root /home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453
```

Result: `status=cpu_reference_pass`, source commit and all ten source hashes matched.  The two target cases were width 2048 (8 blocks, 2048 exact scalar writes) and width 6144 (24 blocks, 6144 exact scalar writes), both with exact ql/qh/scale storage coverage.  Width 768 (three blocks) passed the partial-workgroup reference.  Width 2050 (partial q6_K block), subgroup 32, and workgroup 32 select explicit `fallback` plans.  No performance or GPU result is inferred from these checks.

## Pinned notebook compile recipe

The local `command -v glslc` probe returned no path, so compilation was not claimed here.  Root owns the notebook compiler invocation.  On the notebook, after confirming the existing `/usr/bin/glslc` and SPIR-V headers, use the pinned source and an isolated build directory:

```sh
SRC=/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453
BUILD=/tmp/opus08-b10453-vulkan
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX2=ON \
  -DGGML_VULKAN_SHADER_DEBUG_INFO=OFF \
  -DVulkan_GLSLC_EXECUTABLE=/usr/bin/glslc
cmake --build "$BUILD" --target ggml-vulkan --parallel 2
```

The CMake generator places the installed host generator at `$BUILD/Release/vulkan-shaders-gen` and invokes it with `--glslc`, `--source`, `--output-dir`, `--target-hpp`, and `--target-cpp` (`ggml/src/ggml-vulkan/CMakeLists.txt:197-243`).  To regenerate only the q6 source family after that build, use a new output directory:

```sh
SHADER_DIR="$SRC/ggml/src/ggml-vulkan/vulkan-shaders"
OUT="$BUILD/opus08-q6-shaders"
mkdir -p "$OUT"
"$BUILD/Release/vulkan-shaders-gen" \
  --glslc /usr/bin/glslc \
  --source "$SHADER_DIR/mul_mat_vec_q6_k.comp" \
  --output-dir "$OUT" \
  --target-hpp "$OUT/ggml-vulkan-q6-k.hpp" \
  --target-cpp "$OUT/mul_mat_vec_q6_k.comp.cpp"
```

The generator filters by source basename and will emit the standard q6 files, including `mul_mat_vec_q6_k_f32_f32.spv`, `mul_mat_vec_q6_k_f16_f32.spv`, their subgroup and subgroup-no-shmem variants, and any q8_1 variants only when CMake's integer-dot feature probe enabled `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT`.  The generator's compiler command is `glslc -fshader-stage=compute --target-env=vulkan1.2 <source> -o <spv> -O`, with generated `-D` values.  The entrypoint is `main`.

For review of generated command lines, the q6 standard variants derive these exact define sets from `matmul_shaders(..., MatMulIdType::NONE, ...)`:

| variant | required defines; subgroup suffix adds the final define shown |
| --- | --- |
| `f32_f32` | `DATA_A_Q6_K=1`, `FLOAT_TYPE=float`, `FLOAT_TYPEV2=vec2`, `FLOAT_TYPEV4=vec4`, `FLOAT_TYPEV8=mat2x4`, `B_TYPE=vec4`, `B_TYPE_SCALAR=float`, `B_TYPEV4=vec4`, `D_TYPE=float`, `LOAD_VEC_A=2`, `LOAD_VEC_B=4`, `ACC_TYPE=float`, `ACC_TYPEV2=vec2`; add `USE_SUBGROUP_ADD=1` or `USE_SUBGROUP_ADD_NO_SHMEM=1` for the corresponding files |
| `f16_f32` | `DATA_A_Q6_K=1`, `FLOAT16=1`, `FLOAT_TYPE=float16_t`, `FLOAT_TYPEV2=f16vec2`, `FLOAT_TYPEV4=f16vec4`, `FLOAT_TYPEV8=f16mat2x4`, `B_TYPE=mat2x4`, `B_TYPE_SCALAR=float`, `B_TYPEV4=vec4`, `D_TYPE=float`, `LOAD_VEC_A=2`, `LOAD_VEC_B=8`, `ACC_TYPE=float`, `ACC_TYPEV2=vec2`; add `USE_SUBGROUP_ADD=1` or `USE_SUBGROUP_ADD_NO_SHMEM=1` for the corresponding files |

These are generator-derived defines, not a request to hand-edit or bypass the generator.  If a direct compiler control is needed, use the generator output and its recorded command line; do not assume a manually assembled SPIR-V binary is equivalent.

## Later root acceptance

Root should record the notebook `glslc --version`, generator output inventory and hashes, and `spirv-val` results before any runtime comparison.  The smallest useful GPU arm is the generated q6 standard pipeline at subgroup 64 with the existing model shape; compare against the unchanged b10453 CPU/native path on identical inputs and exact output/protocol checks.  A candidate kernel is acceptable only after SPIR-V compilation, numerical parity over the 2048 and 6144 q6 paths, exact model output/quality checks, and a measured latency result on the intended backend.  Any unsupported subgroup, partial q6_K block, missing extension, compile failure, or output mismatch must retain the existing fallback.  This packet contains no claim that the candidate is compiled, accelerated, serving-ready, or faster.

The sibling Opus response remained an empty 0-byte `response.json` with empty `stderr.log` at the preparation check; no patch was copied or applied.  If a response appears later, review it against the hashes above in an isolated copy and rerun this CPU suite before root considers a compile.
