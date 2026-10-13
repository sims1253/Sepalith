# RUN-05 Q6 exact-division candidate review

This packet is a CPU/read-only review of the supplied Opus11 Q6 candidate. It does not build, launch a model, invoke Vulkan/CUDA, or claim GPU behavior.

## Pinned closure

The local source at `/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453` is commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`; its tracked `ggml-vulkan.cpp` SHA-256 is the root-pinned `c73e8f7980cd416f7fd30860c43f85e4ecacb7db84c0bf057cbb505aaf90142f`. The relevant source files are copied under `candidate-source/`; their hashes are in `source-hashes.json`.

The submitted diff was extracted verbatim from `docs/campaign/work/serving-hillclimb/opus11-q6-shader-v2-run/result.md` into `candidate.patch`. It applied cleanly to the pristine Q6 shader copy. `baseline-source/.../mul_mat_vec_q6_k.comp` remains the exact pinned shader; the patched isolated copy has the macro-guarded direct path.

## Review result

- **Candidate patch: accept as an isolated opt-in compile/parity candidate.** It changes only `mul_mat_vec_q6_k.comp`, puts the new function and branch behind `GGML_VK_Q6K_EXACT_DIV`, and leaves the original loop/tail path text intact for the macro-off source. The branch is uniform in the push constant and specialization workgroup size. The direct path corrects the scale stride to `s_offset + 2*g` and keeps the baseline FMA arithmetic order.
- **CPU mirror: pass.** `q6k_exact_div_index_test.py` ran with `PYTHONDONTWRITEBYTECODE=1` and printed `OK q6k exact-div mirror: 31 configurations + negative control`. It covers 2048 and 6144 widths, block sizes 16/32/64/128, row counts 1/2/4, a 256-block-size fallback, additional widths 256/1024/1280/1536/2304, unaligned fallback, full term coverage, and an intentionally wrong scale stride that must fail. This proves the bounded index/coverage mirror only; it proves no shader compilation or GPU numerics.
- **POISON build method: reject.** The supplied `GGML_VK_Q6K_EXACT_DIV_POISON` switch is retained in the isolated review copy for provenance, but it is not part of the recommended recipe. A debug pipeline dispatch trace and exact output comparison provide a bounded observation without deliberately producing invalid output.
- **Performance/quality: pending.** No build, SPIR-V validation, GPU parity, latency, or model-quality result was run in this worker.

## Dispatch and feature findings

1. `vulkan-shaders-gen.cpp:734-746` chooses `mul_mat_vec_q6_k.comp` for the ordinary Q6_K f32/f16 DMMV families and maps the three reduction forms to the base, `_subgroup`, and `_subgroup_no_shmem` variants. The integer-dot family is separately generated from `mul_mat_vecq.comp` at lines 766-777 and is compile-gated.
2. `ggml-vulkan.cpp:7280-7282` resolves `integer_dot_product` only when the extension property `integerDotProduct4x8BitPackedSignedAccelerated` and `shaderIntegerDotProduct` are both true. The existing Renoir observation (`docs/campaign/work/q8-ordinary-path-capsule/runtime-observation.json`) records shader integer-dot true, packed signed acceleration false, and resolved ggml integer-dot false. Therefore Q6 f32 input does not take the Q8_1 integer-dot route on that runtime; the ordinary f32 Q6 DMMV route is the relevant one.
3. `ggml_vk_should_use_mmvq` at `ggml-vulkan.cpp:9414-9425` returns false for Q6_K on non-Intel devices unless `GGML_VK_FORCE_MMVQ` overrides it. `mmvq_mode` defaults to zero at lines 7075-7079. The trace arm must leave `GGML_VK_FORCE_MMVQ` unset and record the effective mode; `GGML_VK_DISABLE_MMVQ=1` is a separate isolation check, not a promotion setting.
4. The Q6 ordinary pipelines are registered at line 5252 with `wg_size_subgroup16`, `rm_kq`, and `NUM_ROWS=i+1`. For AMD GCN, lines 5192-5197 set `rm_kq=4`; `wg_size_subgroup16` is the selected subgroup size (or four times it for the large DMMV family) at lines 5224-5234. `ggml_vk_get_dequantize_mul_mat_vec` lines 7816-7840 selects the subgroup family by default on AMD; the actual selected pipeline and workgroup still need a debug trace on the exact binary.
5. `mul_mat_vec_base.glsl:89-91` defines `BLOCK_SIZE`, `NUM_ROWS`, and `NUM_COLS` as specialization constants. The direct branch removes per-superblock cache barriers, but `reduce_result` still retains the synchronization required by the selected reduction variant (`:93-128` subgroup-no-shmem, `:129-228` shared/hybrid). The result's broad “no barriers” wording therefore means no direct-path cache/tail barriers; it must not be read as removing all reduction synchronization for every generated variant.
6. Cooperative matrix is not a prerequisite for this ordinary Q6 DMMV candidate. The prior runtime observation marks `coopmat_support` and `coopmat2` unobserved; this packet makes no value inference. The exact candidate branch has no cooperative-matrix predicate. The pinned source has a `shader_core_count` field used by other matmul/split-K heuristics, but no fixed compute-unit prerequisite in the Q6 shader or its DMMV selection; the packet assumes no CU count.

The direct condition is `nbr_par_th == 0 && (p.ncols % QUANT_K) == 0`, with `QUANT_K=256`. For each selected workgroup this means `num_blocks_per_row` must be divisible by `gl_WorkGroupSize.x/16`; 2048 (8 blocks) and 6144 (24 blocks) are exact for the 64-thread subgroup case, while a larger 256-thread workgroup falls back for both. The root trace must record the actual specialization/workgroup and whether the branch's shape predicate is reached.

## Root-owned bounded recipe

Use `candidate-build.sh` from this directory. It archives the pinned commit into a temporary isolated source tree, applies `candidate.patch`, inserts the opt-in define only in that temporary tree, configures baseline and candidate builds with `GGML_VULKAN_DEBUG=ON`, and builds `ggml-vulkan` with one build worker. It does not touch the pinned source. It stops after recording generated Q6 artifacts; it does not launch a server or model.

After both builds, the root trace should correlate the candidate and baseline on the same synthetic Q6 f32 matvec geometries and capture:

- source commit, binary and generated shader identities;
- `mul_mat_vec_q6_k_f32_f32` (or the selected subgroup suffix) pipeline name, workgroup size, specialization `NUM_ROWS`, `MMVQ` mode, and `integer_dot_product` value;
- exact output/token parity for 2048 and 6144 widths, with generic fallback cases included;
- native debug logs from `GGML_VULKAN_DEBUG`.

The trace is the dispatch proof. Stop the candidate arm on compile failure, missing ordinary Q6 pipeline, a changed macro-off generated artifact, output divergence, or fallback behavior outside the shape guard. Do not use the POISON define.
