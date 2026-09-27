# RUN-06 Advisory: Kernel and Mixed-Quant Audit

This advisory uses only the packet. I executed nothing.

Tags used below:
- **[O]** observed in the packet.
- **[D]** arithmetic derived from packet numbers.
- **[H]** hypothesis that needs measurement.
- **[V]** a name or signature that is not in the packet and must be checked by grep against pinned b10453.

## 0. Bottom line

- **Cold requests are mostly prefill; warm replay is mostly decode.** Prefill is 68–89% of cold wall time [D]. Warm identical replay is almost pure decode [D]. Work that only speeds up decode cannot fix the cold 5 s misses.
- **No per-kernel timing exists on any backend.** The first spend should be attribution: on the notebook, now, with no source change.
- **Existing CUDA fusions probably match this graph, but firing is unverified.** Several fusion paths in the excerpt appear to match this exact Llama graph. Vulkan fusions must be listed from the perf logger, not assumed.
- **Persistent megakernel: NO-GO for Tuesday (2026-09-15).** Reasons and the revisit gate are in §6.
- **Please confirm the serving target.** If production is the RTX 5090, tuning Vulkan kernels on Vega/GCN does not transfer. The notebook's value is then attribution method, quant quality on DEV, the shape test harness, and orchestration findings.

## 1. Packet facts and derivations

| Item | Value | Tag |
|---|---|---|
| Parameter check | 42 × (4.19M·2 + 0.52M·2 + 12.58M·3 + 4096 norms) + 2 × 267.39M embed/head + 2048 = **2,516,756,480**. This matches exactly. 9 × 42 + 3 = **381 tensors**, which also matches. So there are no biases, no q/k-norm, and no rope-freq tensor. | [D] |
| Decode weight bytes per token at Q8_0 (all except token_embd) | 2.2494e9 × 34/32 ≈ **2.39 GB** | [D] |
| Logits head bytes per token (Q8_0) | 284 MB = **11.9%** of decode weight bytes. Rises to about 15–20% once the layers go to Q6_K–Q4_K and the head stays Q8_0. This is a byte share, not a time share. | [D] |
| KV cache, F16, ctx 4096 | 42 × 2 × 256 × 4096 × 2 B = **168 MiB**. At position 1903, decode reads about 82 MB of KV, roughly 3.4% of weight bytes. **Quantizing KV is not worth its numerics risk.** | [D] |
| Warm Vulkan ms per generated token | 56.9 / 57.5 / 58.0 / 59.2. This is consistent with 17 tok/s decode alone, so warm identical replay is effectively a prompt-cache hit plus decode. | [D] |
| Cold Vulkan prefill (cold minus warm) | 229 / 238 / 212 / 202 tok/s. Prefill share of cold wall: **69 / 68 / 83 / 89%**. | [D] |
| Cold CPU prefill; CPU decode | About 55–67 tok/s; about 13.3–15.2 tok/s | [D] |
| Vulkan vs CPU speedup | About 3.4–3.7× on prefill, but only **1.16–1.27× on decode** | [D] |
| Effective weight streaming | Vulkan decode about 40–42 GB/s; CPU decode about 32–36 GB/s. Both share one DRAM. | [D] |
| Interpretation | Decode is bound by DRAM streaming on both CPU and iGPU. **Not established:** no bandwidth ceiling was measured. | [H] |
| Effective prefill arithmetic rate | About 4.0–4.3 GFLOP/token → about 0.9 TFLOP/s effective. This is not a ceiling. | [D] |
| Prefill needed for a 5 s cold deadline (decode unchanged, zero other overhead) | F2 ≥ 276 tok/s (1.16×), F3 ≥ 317 tok/s (1.5×), F4 ≥ 499 tok/s (**2.5×**). These are requirements, not predictions. | [D] |
| Prompt ubatch tails at ub=256 | 90 / 119 / 193 / 111 tokens. Tails may take different GEMM tile paths. | [D] |
| 5 s experiment inconsistency | If the 8 requests were 4 cold + 4 identical-warm, the 60 s timings imply CPU 4/8 and Vulkan 5/8 would pass. Observed was 1/8 and 4/8. So the requests differ from identical replay, or there is unaccounted overhead (load, harness, queueing). **Instrument request-level timestamps before attributing anything to kernels.** | [D] |

## 2. Audit of existing fusion and graph paths (source only)

### CUDA, ggml-cuda.cu L3750–3997

These are the patterns visible in the excerpt. Whether each applies to this model is marked.

- **{MUL_MAT, MUL_MAT, GLU} → one mmvq/mmvf launch (L3753–3789).** This matches the gate/up/SwiGLU pattern at decode (ncols=1), 42 times per token [H]. It only fires if `ggml_cuda_should_fuse_mul_mat_vec_q/f` returns true; those conditions are not in the packet [V].
- **{MUL_MAT, ADD} with same-shape operand (L3888–3946).**
  - The residual add after `wo` and after `ffn_down` has shape [2048, 1] on both sides. So it structurally qualifies as `x_bias` fusion at decode [H].
  - If intermediate nodes (e.g. out_ids `get_rows`) sit between the matvec and the add, it would not fire there.
  - It is exact only if the bias is added once after full reduction. Verify with a bitwise logits comparison.
- **RMS_NORM+MUL fusion (L3964).** Applies twice per layer plus the final norm.
- **Patterns that do not apply to this model** (it has no q/k-norm, biases, scales, SSM, or softcap): RMS_NORM+MUL+ROPE(+SET_ROWS), RMS_NORM+MUL+ADD, MUL_MAT+scale, SSM_CONV, softcap, and UNARY+MUL (if the GLU op is used). Lines before L3750 are not in the packet.
- **`GGML_CUDA_GRAPH_OPT=1`** (off by default; read once per process through a function-static lambda):
  - It needs CUDA graphs active and exactly one device.
  - It only counts fan-out for nodes with `ggml_nrows <= 1`, so it affects **decode only**.
  - The root must be named `attn_norm` with fan-out exactly 3. That matches Q/K/V here, but only if the name comes from the standard llama builder [H].
  - Stream concurrency should not change arithmetic, so this is an exact-parity candidate.
- **mmvq.cu `calc_nwarps` tables.** The GCN, RDNA3, RDNA4 and Turing tables affect CUDA/HIP builds only. **They have no effect on the notebook's Vulkan build.** Which table SM120 selects is decided outside the excerpt [V].

### Vulkan, ggml-vulkan.cpp

The knobs visible in the excerpt are `GGML_VK_DISABLE_FUSION`, `GGML_VK_DISABLE_MMVQ`, `GGML_VK_FORCE_MMVQ`, `GGML_VK_SERIALIZE_SUBMISSIONS` (debug) and `GGML_VK_ASYNC_USE_TRANSFER_QUEUE`.

- **Fusion.** `add_rms_fusion` requires subgroup arithmetic and a non-Intel device. It is expected ON on RADV, but record it.
- **Transfer queue.** `prefers_transfer_queue` is false on this device, because it is GCN and UMA. The transfer-queue knob is not worth an arm for decode.
- **MMVQ numerics.** `mul_mat_vecq.comp` quantizes activations to Q8_1 (`block_q8_1_x4`). It needs `GL_EXT_integer_dot_product`. So **MMVQ vs the float matvec path is a precision change**, not a free optimization.
- **Perf logger (L2200–2315).** It logs per-op names including `_VEC`, type, m/n/k and fusion names, plus a "Total time" line. That is exactly the attribution tool needed; the env var name is [V] (upstream uses `GGML_VK_PERF_LOGGER`).
- **Enumerate all knobs.** Run `grep -n 'getenv("GGML_' ggml/src/ggml-vulkan/ggml-vulkan.cpp ggml/src/ggml-cuda/*.cu` to get the full list for the pinned revision.

### Q6_K matvec shader, mul_mat_vec_q6_k.comp L108–113

- For every Q6_K shape in this model, `num_blocks_per_row` is 8 or 24. That is a multiple of `it_size` for any BLOCK_SIZE in {32, 64, 128}.
- So `nbr_par_th == 0`, and the unconditional tail call does no arithmetic. It still executes `num_rows` barriers per workgroup.
- Skipping the tail is bit-exact (see §5b). The gain is not predicted and may be negligible if one workgroup is one wavefront.
- This only matters if Q6_K takes the float path rather than MMVQ.

## 3. Cost decomposition

| Component | What we know | Instrument | Decision it drives |
|---|---|---|---|
| Prefill | 202–238 tok/s [D]; dominates cold latency | Perf logger: MUL_MAT n=256/tail rows vs attention ops (FLASH_ATTN_EXT or KQ/softmax) at F4 | Is the prefill lever GEMM or attention? Can flag-level changes reach the E-rates in §1? |
| Decode | 57–59 ms/token [D]; streaming-bound [H] | Logger: Σ MUL_MAT_VEC time and per-shape GB/s relative to the best shape in the same run | Only bytes (quant) matter, or specific shapes lag |
| Launch/barrier/submit | Unmeasured. "GPU busy 99" cannot see intra-command-buffer bubbles | Gap = wall per token (logger OFF) − Σop "Total time" (logger ON) | Whether fusion or launch reduction can matter at all |
| Logits head | Byte share 11.9% [D]; time unmeasured | Logger line m=130560, k=2048, n=1. Also confirm prefill emits n=1 for the head, not per-token logits. | Whether the head is less efficient than the layer matvecs |
| CPU orchestration | Unmeasured | llama perf output (load / prompt eval / eval / **sampling time**), `graph splits`, request-level harness timestamps | Whether greedy argmax over 130,560 logits, detokenization, or harness costs are material. Any argmax rewrite must keep identical tie-breaking. |

## 4. Ranked experiments

### Parity classes (used by every experiment)

- **P-exact.** For arithmetic-equivalent changes. Requires:
  - identical token IDs, PRM03 text, and generated lengths (12/31/20/20);
  - prompt token counts asserted at 346/887/1217/1903 in every run (this catches BOS/tokenizer drift);
  - **bitwise-identical full logits** at every generated position.
  - Logits come from a ≤40-line dump harness using `llama_get_logits_ith(ctx, -1)` that writes 130,560 f32 values per position, built on CPU now.
- **P-neartie.** For changed reductions (fusion off, MMVQ override, FA or ubatch changes). Accept a divergence only if the reference top-1 minus top-2 margin at the first divergent position is ≤ 2 × the max |Δlogit| observed at the earlier, agreeing positions. The change must also be DEV non-inferior. Otherwise reject.
- **Quant.** Judged on DEV (edits / no-ops / format). Fixture parity is informational only.
- **Noise band.** At least 3 repeats in ABAB order. A "win" must beat the baseline's max–min spread on both cold and warm. Laptop hygiene: AC power, fixed power profile, record clocks and temperatures if available.

### E1 (rank 1): Attribution and existing-fusion audit on notebook Vulkan and CPU

**When and cost.** Now. Notebook only. About 20 min wall. No source change.

**Arms.** Q8_0, current weights, existing flags `-t 6 -b 256 -ub 256 -c 4096 -np 1 -ngl 99`:

| Arm | Setup | Fixtures | Runs |
|---|---|---|---|
| A0 | Logger OFF, fusion ON | F1 and F4, cold + warm | 3 each (wall baseline and noise band) |
| A1 | Logger ON [V], fusion ON | F4, cold + warm | 1 |
| A2 | Logger ON, `GGML_VK_DISABLE_FUSION=1` | F4, cold + warm | 1 (the diff of op tables against A1 lists which fusions fire) |
| A3 | Logger OFF, `GGML_VK_DISABLE_FUSION=1` | All 4, cold | 1 (P-neartie against A0) |

**Also record:**
- the startup device line (uma, fp16, int dot, warp size, matrix cores);
- FA on/off, KV type, `graph splits`, and llama perf breakdown;
- process-cold (load plus pipeline creation) separately from request-cold.

**CPU-only sub-step (zero GPU).** Tokenize TRAIN editor-transition pairs and compute re-prefill = prompt_len − common_prefix_len. This decides whether real traffic looks like cold or like warm.

**Decision rules (proposed thresholds; lead may reset):**
- If Σ matvec ≥ 85% of decode wall per token, **reject** fusion, launch and megakernel work for Vulkan decode. Only bytes and per-shape efficiency remain as levers.
- If the gap is ≥ 15%, look at orchestration first (sampling time, splits, submit granularity) before any kernel work.
- A shape whose GB/s is < 70% of the best shape in the same run is a specialization candidate. K/V at m=256 is the suspect [H].
- If the head's time share is > 1.5× its byte share, the head is a dispatch candidate.
- If F4 attention is > 25% of prefill, FA and attention become the prefill arm. Otherwise GEMM is.
- If A3 diverges from A0, existing fusion is not bit-identical. All future fusion edits then need P-neartie.
- If the head appears with n>1 during prefill, flag it to the lead. Do not change the protocol.

**Fail-fast.** If the logger does not exist under that name, or perturbs wall time by more than 2×, use its per-op output only for relative attribution. If the 5 s request timestamps show more than 1 s of non-inference overhead, fix the harness finding before any kernel work.

### E2 (rank 2): Mixed-quant build, exact-shape kernel harness, matvec-path A/B

**When and cost.** Now. CPU plus notebook. About 1–2 h wall, mostly quantization and test runs.

1. **Build the three files from verified F16.** Commands are in §5. Verify each file:
   - 381 tensors;
   - `output.weight` and `token_embd.weight` are Q8_0;
   - no "fallback quantization" lines in the log (all K dimensions, 2048 and 6144, are multiples of 256, so none are expected [D]);
   - a saved per-layer type table (the `_M` recipes mix types on attn_v/ffn_down by layer index and GQA [V]);
   - decode bytes per token = tensor bytes − token_embd bytes.
2. **Run test-backend-ops (diff in §5a), eval mode, Vulkan0 vs CPU**, for {q8_0, q6_K, q5_K, q4_K} × model shapes × n ∈ {1, 90, 111, 119, 193, 256}. Use perf mode for GB/s (n=1) and GFLOP/s (n=256).
3. **Matvec-path A/B per type:** default vs `GGML_VK_FORCE_MMVQ=1` vs `GGML_VK_DISABLE_MMVQ=1`. Each arm is a separate process, since the knobs are read at device init. Measure with test-backend-ops perf, then warm F2 plus cold F4 end to end.
4. **DEV evaluation per quant file** on the default path. Optionally add KL divergence vs F16 logits on DEV text [V: `llama-perplexity --kl-divergence*`].

**Rejection criteria:**
- Any eval FAIL for a (type, shape, backend) stops that type on that backend.
- If the device reports no integer dot product, drop the FORCE_MMVQ arms.
- A path override is adopted only if it beats the noise band **and** passes P-neartie against the default **and** is DEV non-inferior. It must then be recorded as an explicit precision setting.
- A quant type is rejected on DEV non-inferiority against Q8_0, never on fixture parity. It must be re-run on final weights; tensor type layout is shape and index driven, so guards stay valid, but errors do not.

### E3 (rank 3, queued): CUDA SM120 support evidence and graph/fusion verification

**When and cost.** The first GPU window after the training gates. At most 45 min. Nothing else on the GPU.

**Prep, CPU-only, now:**
- a script;
- build with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` (needs CUDA ≥ 12.8);
- record nvcc and driver versions.

**Evidence gate:**
- The startup line shows compute capability 12.0.
- `cuobjdump --list-elf libggml-cuda.so | grep sm_120` is non-empty.
- `test-backend-ops -b CUDA0` eval passes on:
  - the §5a shapes and types;
  - FLASH_ATTN_EXT with head size 128, GQA 8, F16 KV, kv up to 4096 [V signature];
  - RMS_NORM, GLU, ROPE, SET_ROWS, GET_ROWS, ADD.
- **Any FAIL or missing sm_120 ELF ends the window. No serving benchmark.**

**Attribution.** Run nsys (CUDA trace plus graph node tracing [V flag]) on F4 cold and warm:
- kernel count per token;
- Σkernel / wall;
- whether graph launches are present;
- whether fused mmvq GLU/bias kernels appear, or standalone add/GLU kernels.

**Arm.** Default vs `GGML_CUDA_GRAPH_OPT=1`. P-exact is required.

**Rejection criteria:**
- Any bit difference under GRAPH_OPT means reject (a hazard, not noise). No speedup beyond noise means leave it off.
- If decode Σkernel / wall ≥ 0.9 with graphs on, reject all launch-overhead work on CUDA, including the megakernel.
- If standalone residual-add or GLU kernels appear, read `should_fuse_mul_mat_vec_q` in the pinned source to find out why before writing any kernel.

## 5. Smallest diffs

### 5a. Recommended: test-only diff, no runtime risk

Add to `tests/test-backend-ops.cpp` in `make_test_cases_perf()`, and a subset to eval. The signature below follows upstream (type_a, type_b, m, n, k, bs, nr) [V]; confirm against the pinned file.

```cpp
// RUN-06 model shapes {m = out rows, k = in dim}
{
    const int64_t mk[][2] = { {2048,2048}, {256,2048}, {6144,2048}, {2048,6144} };
    const int64_t ns[]    = { 1, 90, 111, 119, 193, 256 };
    for (ggml_type ta : { GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K })
        for (const auto & s : mk)
            for (int64_t n : ns)
                test_cases.emplace_back(new test_mul_mat(ta, GGML_TYPE_F32, s[0], n, s[1], {1,1}, {1,1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 130560, 1, 2048, {1,1}, {1,1})); // head
}
```

### 5b. Optional, bit-exact, gated on E1/E2 showing a slow Q6_K float path

In `mul_mat_vec_q6_k.comp`:

```glsl
-    calc_superblock(a_offset, b_offset, itid, ix, ql_offset, qh_offset, s_offset, y_offset, i0 + ix, num_blocks_per_row, first_row, num_rows, false);
+    if (nbr_par_th != 0) {   // workgroup-uniform (push constant / WG size); tail has no work otherwise
+        calc_superblock(a_offset, b_offset, itid, ix, ql_offset, qh_offset, s_offset, y_offset, i0 + ix, num_blocks_per_row, first_row, num_rows, false);
+    }
```

**Checks:**
- Confirm `reduce_result` in `mul_mat_vec_base.glsl` does not alias `sccache` [V].
- test-backend-ops eval.
- P-exact via bitwise logit dumps, old vs new, on a Q6_K file.

**Reject** on any bit difference.

### 5c. Guard template for any future shape-specialized kernel

Opt-in, default OFF, falls back to the existing path:

```cpp
// read at device init beside GGML_VK_DISABLE_MMVQ (L7075); name is new
device->run06_spec = getenv("GGML_VK_RUN06_SPEC") != nullptr;

static bool run06_guard(const vk_device & dev, const ggml_tensor * d) {
    const ggml_tensor * a = d->src[0], * b = d->src[1];
    if (!dev->run06_spec || d->op != GGML_OP_MUL_MAT) return false;
    if (b->type != GGML_TYPE_F32 || b->ne[1] != 1 || b->ne[2] != 1 || b->ne[3] != 1) return false;
    if (a->ne[2] != 1 || a->ne[3] != 1 || !ggml_is_contiguous(a) || !ggml_is_contiguous(b)) return false;
    const bool shp = (a->ne[0] == 2048 && (a->ne[1] == 256 || a->ne[1] == 2048 || a->ne[1] == 6144)) ||
                     (a->ne[0] == 6144 && a->ne[1] == 2048);
    const bool typ = a->type == GGML_TYPE_Q8_0 || a->type == GGML_TYPE_Q6_K ||
                     a->type == GGML_TYPE_Q5_K || a->type == GGML_TYPE_Q4_K;
    return shp && typ;
}
```

Log a per-graph hit counter. It must equal the expected count for each decode graph (7 × 42 = 294, minus any matvecs already fused). A mismatch is a fail-fast.

### Quantization commands

Flags are [V] (`--help` on b10453). Do not pass `--allow-requantize` or `--imatrix` until a TRAIN-only imatrix with a recorded hash exists.

```sh
sha256sum model-F16.gguf    # must equal the verified F16 hash on record
for T in Q6_K Q5_K_M Q4_K_M; do
  ./llama-quantize --output-tensor-type q8_0 --token-embedding-type q8_0 \
      model-F16.gguf model-$T.gguf $T 6 2>&1 | tee quant-$T.log
done
```

## 6. Persistent megakernel by Tuesday: NO-GO

1. **It cannot be sized.** There is zero per-kernel timing on any backend.
2. **The notebook evidence points elsewhere.** Decode looks bytes-bound [H, D]: the iGPU is only about 1.2× the CPU on shared DRAM. Cold latency is 68–89% prefill. A decode megakernel reduces neither bytes nor prefill.
3. **Vulkan offers no forward-progress guarantee** for persistent workgroups or inter-workgroup barriers. On an iGPU shared with the display, that means deadlock or TDR risk.
4. **CUDA cannot be tested before Tuesday.** The GPU is blocked by training, SM120 support is unproven, and CUDA graphs plus existing fusions may already remove most launch cost (E3 verifies this).
5. **The correctness surface is large.** It spans 42 layers × mixed per-tensor quant types × FA × the full-vocab head, and all of it must be re-validated on final weights.

**Revisit gate** (after E3, not before). Open a design spike only if, on CUDA with graphs ON, both of the following hold:
- decode (wall − Σkernel) / wall is ≥ 25%, **or** small kernels plus gaps are ≥ 30%;
- confirmed existing fusions plus GRAPH_OPT fail to close that gap.

Otherwise the topic is closed. Proposed Tuesday deliverable: E1 and E2 results, the E3 script ready, and a decision memo.

## 7. Not verifiable from the packet

- CUDA fusion lines before L3750.
- The `should_fuse_*` conditions.
- CUDA graph enable/disable rules and their env var.
- SM120 table selection.
- The Vulkan fusion list, `mul_mat_vec_max_cols`, and the MMVQ default heuristic.
- The perf logger env var name.
- FA and KV settings actually in use.
- The test-backend-ops signatures.
- The composition of the 5 s experiment.
- The GGUF arch string. It must be `llama`; also confirm that any MiniCPM μP scalings were folded in, or the kernel audit is moot.