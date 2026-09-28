## RUN-05-OPUS08: opt-in exact-division Q6_K matvec path

### 1. Dispatch audit: what the supplied source proves and what it does not

**Proven by the supplied source (`mul_mat_vec_q6_k.comp`)**
- For both model input widths, the baseline shader always makes the tail call `calc_superblock(..., all_threads=false)`.
  - ncols 2048 gives `num_blocks_per_row` 8. ncols 6144 gives 24.
  - With `it_size = BLOCK_SIZE/16` equal to 1, 2, 4 or 8, `nbr_par_th` is 0 in every case. So the tail runs with `i = nb + ix >= nb` for every thread.
  - Each tail row iteration does no loads, but still executes `barrier()` and `continue`. That is `num_rows` barriers of pure overhead per workgroup.
- Every non-tail pass does a `sccache` store, then `barrier()`, then 4 LDS reads per row per superblock. This exists only to share 16 int8 scales per superblock.

**Not proven (the source was not supplied)**
- `mul_mat_vecq_funcs.glsl` contains a `DATA_A_Q6_K` integer-dot implementation. Whether RADV RENOIR runs `mul_mat_vec_q6_k.comp` or `mul_mat_vecq.comp` for Q6_K is decided in `ggml-vulkan.cpp`, which I don't have.
- `BLOCK_SIZE` and `NUM_ROWS` specialization values and `reduce_result` are also unseen.
- Gate G2 below settles dispatch empirically before any performance run.

**The bounded change**
- Under the compile switch `GGML_VK_Q6K_EXACT_DIV`, and only when `nbr_par_th == 0 && ncols % 256 == 0`, run a direct path:
  - no tail pass;
  - no shared memory and no barriers;
  - each thread reads its 4 scale bytes (`s_offset + 0/2/4/6`) directly from the block.
- The arithmetic order is copied from the `all_threads` branch.
- Any other shape takes the original code, unchanged.
- The branch condition depends only on a push constant and the workgroup size, so it is uniform and fallback barriers stay in uniform control flow.
- With the switch undefined, the preprocessed shader is token-identical to the original.

**Performance expectation (hypothesis, not measured)**
- Cost: 4 global byte loads replace 1 byte load plus LDS store, barrier and 4 LDS reads. On GCN this could be neutral or even a regression, hence the A/B.
- Scope: this affects only matvec, i.e. decode, not prefill (mul_mm).
- In the stress diagnostics, server prompt time was 4.2–7.5 s against decode 1.08–1.20 s.
- Even a 10% decode gain (~110 ms) cannot bring the 5.3–8.6 s changing-prompt walls under 5 s. This candidate cannot fix the deadline misses on its own.

### 2. Patch (`ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp`)

```diff
--- a/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp
+++ b/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp
@@ -11,6 +11,60 @@
 FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
 uint csel = 0;
 
+#ifdef GGML_VK_Q6K_EXACT_DIV
+// Experimental, default off. Precondition, checked uniformly in compute_outputs:
+// num_blocks_per_row % it_size == 0, so every thread's i < num_blocks_per_row in every pass.
+// Scales are read directly (no sccache, no barrier). Arithmetic order equals calc_superblock.
+void calc_superblock_direct(const uint a_offset, const uint b_offset, const uint ql_offset, const uint qh_offset, const uint s_offset, const uint y_offset, const uint i, const uint num_blocks_per_row, const uint first_row, const uint num_rows) {
+    const uint y_idx = i * QUANT_K + y_offset;
+
+    [[unroll]] for (uint n = 0; n < num_rows; ++n) {
+        const uint ib0 = a_offset + (first_row+n)*num_blocks_per_row;
+
+        const uint32_t ql0_u32 =  uint32_t(data_a_packed16[ib0 + i].ql[ql_offset / 2]) | (uint32_t(data_a_packed16[ib0 + i].ql[ql_offset / 2 + 1]) << 16);
+        const uint32_t ql32_u32 = uint32_t(data_a_packed16[ib0 + i].ql[ql_offset / 2 + 16]) | (uint32_t(data_a_packed16[ib0 + i].ql[ql_offset / 2 + 17]) << 16);
+
+        const uint32_t ql0_u32_lo4 = ql0_u32 & 0x0F0F0F0F;
+        const uint32_t ql0_u32_hi4 = (ql0_u32 >> 4) & 0x0F0F0F0F;
+        const uint32_t ql32_u32_lo4 = ql32_u32 & 0x0F0F0F0F;
+        const uint32_t ql32_u32_hi4 = (ql32_u32 >> 4) & 0x0F0F0F0F;
+
+        const uint32_t qh_u32 = uint32_t(data_a_packed16[ib0 + i].qh[qh_offset / 2]) | (uint32_t(data_a_packed16[ib0 + i].qh[qh_offset / 2 + 1]) << 16);
+        const uint32_t qh0_u32 = (qh_u32 & 0x03030303) << 4;
+        const uint32_t qh2_u32 = (qh_u32 & 0x0C0C0C0C) << 2;
+        const uint32_t qh4_u32 = (qh_u32 & 0x30303030);
+        const uint32_t qh6_u32 = (qh_u32 & 0xC0C0C0C0) >> 2;
+
+        const vec4 q0 = vec4(unpack8(ql0_u32_lo4  | qh0_u32)) - 32;
+        const vec4 q1 = vec4(unpack8(ql32_u32_lo4 | qh2_u32)) - 32;
+        const vec4 q2 = vec4(unpack8(ql0_u32_hi4  | qh4_u32)) - 32;
+        const vec4 q3 = vec4(unpack8(ql32_u32_hi4 | qh6_u32)) - 32;
+
+        const FLOAT_TYPE sc0 = FLOAT_TYPE(data_a[ib0 + i].scales[s_offset    ]);
+        const FLOAT_TYPE sc2 = FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 2]);
+        const FLOAT_TYPE sc4 = FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 4]);
+        const FLOAT_TYPE sc6 = FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 6]);
+        const FLOAT_TYPE d = FLOAT_TYPE(data_a[ib0 + i].d);
+
+        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
+            vec4 by0  = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4     ]);
+            vec4 by32 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 +  8]);
+            vec4 by64 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 16]);
+            vec4 by96 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 24]);
+
+            FLOAT_TYPE sum[4] = {0, 0, 0, 0};
+            [[unroll]] for (uint l = 0; l < 4; ++l) {
+                sum[0] = fma(FLOAT_TYPE(by0[l]), q0[l], sum[0]);
+                sum[1] = fma(FLOAT_TYPE(by32[l]), q1[l], sum[1]);
+                sum[2] = fma(FLOAT_TYPE(by64[l]), q2[l], sum[2]);
+                sum[3] = fma(FLOAT_TYPE(by96[l]), q3[l], sum[3]);
+            }
+            temp[j][n] = fma(fma(sum[0], sc0, fma(sum[1], sc2, fma(sum[2], sc4, sum[3] * sc6))), d, temp[j][n]);
+        }
+    }
+}
+#endif
+
 void calc_superblock(const uint a_offset, const uint b_offset, const uint itid, const uint ix, const uint ql_offset, const uint qh_offset, const uint s_offset, const uint y_offset, const uint i, const uint num_blocks_per_row, const uint first_row, const uint num_rows, const bool all_threads) {
     const uint y_idx = i * QUANT_K + y_offset;
 
@@ -107,6 +161,17 @@
 
     const uint nbr_par_th = num_blocks_per_row%it_size;
     const uint nbr_all_th = num_blocks_per_row - nbr_par_th;
+#ifdef GGML_VK_Q6K_EXACT_DIV
+    // Uniform condition (push constant + workgroup size): fallback barriers stay in uniform control flow.
+    if (nbr_par_th == 0 && (p.ncols % QUANT_K) == 0) {
+#ifndef GGML_VK_Q6K_EXACT_DIV_POISON
+        [[unroll]] for (uint k0 = 0; k0 < num_blocks_per_row; k0 += it_size)
+            calc_superblock_direct(a_offset, b_offset, ql_offset, qh_offset, s_offset, y_offset, k0 + ix, num_blocks_per_row, first_row, num_rows);
+#endif
+        reduce_result(temp, d_offset, first_row, num_rows, tid);
+        return;
+    }
+#endif
     uint i0 = 0;
     [[unroll]] for (; i0 < nbr_all_th; i0 += it_size)
         calc_superblock(a_offset, b_offset, itid, ix, ql_offset, qh_offset, s_offset, y_offset, i0 + ix, num_blocks_per_row, first_row, num_rows, true);
```

**Switches**
- **Experiment build only:** `sed -i '1a #define GGML_VK_Q6K_EXACT_DIV 1' <shader>` in a separate worktree, then rebuild. I don't have `vulkan-shaders-gen.cpp`, so I cannot wire this as a named generated variant.
- **Dispatch-proof build only:** additionally insert `#define GGML_VK_Q6K_EXACT_DIV_POISON 1`. The fast path then writes zeros.

### 3. Test: `q6k_exact_div_index_test.py` (stdlib only; `python3 q6k_exact_div_index_test.py`)

```python
#!/usr/bin/env python3
"""Index/coverage mirror of mul_mat_vec_q6_k.comp (baseline + GGML_VK_Q6K_EXACT_DIV).
Integer stand-ins for d/scales/y make comparisons exact: checks index mapping, full
coverage, sccache slot provenance, bounds, path selection and the (assumed) sum
reduction of the MIRROR. GLSL==mirror is by review; this proves nothing about GPU numerics."""
import random

QK = 256
M32 = 0xFFFFFFFF


def req(c, m):
    if not c:
        raise AssertionError(m)


def u16(buf, k):  # data_a_packed16[..].ql/qh[k], little-endian, bounds-checked
    req(0 <= k and 2 * k + 1 < len(buf), "OOB packed16 %d/%d" % (k, len(buf)))
    return buf[2 * k] | (buf[2 * k + 1] << 8)


def make_block(rng):
    return {"ql": bytes(rng.randrange(256) for _ in range(128)),
            "qh": bytes(rng.randrange(256) for _ in range(64)),
            "sc": [rng.randrange(-128, 128) for _ in range(16)],
            "d": rng.randrange(1, 9)}


def ref_dequant(b):  # ggml dequantize_row_q6_K layout -> (q[256], scale_index[256])
    ql, qh = b["ql"], b["qh"]
    q, s = [None] * QK, [None] * QK
    for h in range(2):
        base, lo, ho, so = 128 * h, 64 * h, 32 * h, 8 * h
        for l in range(32):
            q[base + l] = ((ql[lo + l] & 0xF) | ((qh[ho + l] & 3) << 4)) - 32
            q[base + l + 32] = ((ql[lo + l + 32] & 0xF) | (((qh[ho + l] >> 2) & 3) << 4)) - 32
            q[base + l + 64] = ((ql[lo + l] >> 4) | (((qh[ho + l] >> 4) & 3) << 4)) - 32
            q[base + l + 96] = ((ql[lo + l + 32] >> 4) | (((qh[ho + l] >> 6) & 3) << 4)) - 32
            for g in range(4):
                s[base + l + 32 * g] = so + l // 16 + 2 * g
    return q, s


def select_exact(ncols, bs):  # nbr_par_th == 0 && ncols % QUANT_K == 0
    return ncols % QK == 0 and (ncols // QK) % (bs // 16) == 0


def thread_quants(b, ql_off, qh_off):
    ql0 = u16(b["ql"], ql_off // 2) | (u16(b["ql"], ql_off // 2 + 1) << 16)
    ql32 = u16(b["ql"], ql_off // 2 + 16) | (u16(b["ql"], ql_off // 2 + 17) << 16)
    qhw = u16(b["qh"], qh_off // 2) | (u16(b["qh"], qh_off // 2 + 1) << 16)
    words = [(ql0 & 0x0F0F0F0F) | (((qhw & 0x03030303) << 4) & M32),
             (ql32 & 0x0F0F0F0F) | (((qhw & 0x0C0C0C0C) << 2) & M32),
             ((ql0 >> 4) & 0x0F0F0F0F) | (qhw & 0x30303030),
             ((ql32 >> 4) & 0x0F0F0F0F) | ((qhw & 0xC0C0C0C0) >> 2)]
    return [[((w >> (8 * l)) & 0xFF) - 32 for l in range(4)] for w in words]


def run(A, Y, ncols, nrows, bs, nr, experimental, bug=False):
    req(bs % 16 == 0 and bs >= 16, "BLOCK_SIZE must be a multiple of 16")
    nb, it = ncols // QK, bs // 16
    exact = experimental and select_exact(ncols, bs)
    refs = [[ref_dequant(blk) for blk in r] for r in A]
    out = [[None] * nrows for _ in Y]
    touch, barriers = {}, 0
    for wg in range(-(-nrows // nr) + 1):  # +1 workgroup exercises early return
        first = nr * wg
        if first >= nrows:
            continue
        num_rows = min(nr, nrows - first)
        temp = [[[0] * num_rows for _ in Y] for _ in range(bs)]
        if exact:
            passes = [(i0, "direct") for i0 in range(0, nb, it)]
        else:
            full = nb - nb % it
            passes = [(i0, "all") for i0 in range(0, full, it)] + [(full, "tail")]
        for i0, kind in passes:
            for n in range(num_rows):
                row, lds = first + n, {}
                if kind != "direct":  # sccache write phase, then barrier()
                    for tid in range(bs):
                        ix, itid = divmod(tid, 16)
                        i = i0 + ix
                        if kind == "all" or i < nb:
                            req(i < nb, "OOB block in all_threads pass")
                            lds[(ix, itid)] = (row, i, A[row][i]["sc"][itid])
                    barriers += 1
                for tid in range(bs):
                    ix, itid = divmod(tid, 16)
                    i = i0 + ix
                    if kind == "tail" and i >= nb:
                        continue
                    req(i < nb, "OOB block %d in %s pass" % (i, kind))
                    v_im, v_in = divmod(itid, 8)
                    l0 = 4 * v_in
                    ql_off, qh_off = 64 * v_im + l0, 32 * v_im + l0
                    s_off, y_off = 8 * v_im + v_in // 4, 128 * v_im + l0
                    b = A[row][i]
                    qs = thread_quants(b, ql_off, qh_off)
                    sidx = [s_off + (g if bug else 2 * g) for g in range(4)]
                    req(max(sidx) < 16, "OOB scale index")
                    if kind == "direct":
                        scs = [b["sc"][k] for k in sidx]
                    else:
                        scs = []
                        for k in sidx:
                            src = lds.get((ix, k))
                            req(src is not None and src[:2] == (row, i), "sccache slot not written")
                            scs.append(src[2])
                    rq, rs = refs[row][i]
                    for j, y in enumerate(Y):
                        acc = 0
                        for g in range(4):
                            sg = 0
                            for l in range(4):
                                e = y_off + 32 * g + l
                                col = i * QK + e
                                req(0 <= col < len(y), "OOB B index")
                                req(qs[g][l] == rq[e] and sidx[g] == rs[e],
                                    "row %d blk %d elem %d q/scale mismatch" % (row, i, e))
                                if j == 0:
                                    touch[(row, i, e)] = touch.get((row, i, e), 0) + 1
                                sg += y[col] * qs[g][l]
                            acc += sg * scs[g]
                        temp[tid][j][n] += acc * b["d"]
        for j in range(len(Y)):  # reduce_result ASSUMED = sum over all BLOCK_SIZE threads
            for n in range(num_rows):
                req(out[j][first + n] is None, "row written twice")
                out[j][first + n] = sum(temp[t][j][n] for t in range(bs))
    return out, touch, {"exact": exact, "barriers": barriers}


def ref_out(A, Y, nb):
    res = []
    for y in Y:
        col = []
        for r in A:
            tot = 0
            for i in range(nb):
                q, s = ref_dequant(r[i])
                tot += r[i]["d"] * sum(r[i]["sc"][s[e]] * q[e] * y[i * QK + e] for e in range(QK))
            col.append(tot)
        res.append(col)
    return res


def check(ncols, nrows, bs, nr, experimental, expect_exact, bug=False):
    rng = random.Random(ncols * 31 + bs * 7 + nr * 3 + int(experimental))
    nb = ncols // QK
    A = [[make_block(rng) for _ in range(nb)] for _ in range(nrows)]
    Y = [[rng.randrange(-64, 65) for _ in range(ncols)] for _ in range(2)]  # NUM_COLS=2
    out, touch, st = run(A, Y, ncols, nrows, bs, nr, experimental, bug)
    req(st["exact"] == expect_exact, "path selection %r" % ((ncols, bs, st),))
    req(not st["exact"] or st["barriers"] == 0, "direct path must have 0 barriers")
    req(len(touch) == nrows * nb * QK and set(touch.values()) == {1}, "coverage")
    req(out == ref_out(A, Y, nb), "output mismatch")


def main():
    n = 0
    for ncols in (2048, 6144):  # 8 and 24 superblocks
        for bs in (16, 32, 64, 128):  # 64 = assumed RADV wave64 workgroup
            for nr in ((1, 2, 4) if bs == 64 else (2,)):  # 5 rows -> partial last workgroup
                for exp in (False, True):
                    check(ncols, 5, bs, nr, exp, expect_exact=exp)
                    n += 1
        check(ncols, 3, 256, 2, True, expect_exact=False)  # it_size 16 divides neither 8 nor 24
        n += 1
    for ncols, ex in ((256, False), (1024, True), (1280, False), (1536, False), (2304, False)):
        check(ncols, 3, 64, 2, True, expect_exact=ex)
        n += 1
    req(not select_exact(2000, 64) and not select_exact(2176, 64), "unaligned must fall back")
    try:
        check(2048, 2, 64, 2, True, True, bug=True)
    except AssertionError:
        pass
    else:
        raise SystemExit("FAIL: negative control (wrong scale stride) not detected")
    print("OK q6k exact-div mirror: %d configurations + negative control" % n)


if __name__ == "__main__":
    main()
```

**Mirror assumptions**, since `mul_mat_vec_base.glsl` was not supplied:
- `reduce_result` sums `temp` over all threads.
- `b_offset` and `batch_stride_b` are simplified to separate columns.
- No subgroup operations appear in either path of this file.

### 4. Compile (root)

The defines below are from memory. Only `DATA_A_Q6_K` is visible in the supplied `types.glsl`. If a macro is reported missing, take the define set from the build.

```sh
SH=ggml/src/ggml-vulkan/vulkan-shaders
DEFS="-DDATA_A_Q6_K=1 -DB_TYPE=float -DB_TYPE_VEC2=vec2 -DB_TYPE_VEC4=vec4 -DD_TYPE=float -DFLOAT_TYPE=float -DFLOAT_TYPEV2=vec2"
G="glslc -fshader-stage=compute --target-env=vulkan1.2 -O -I$SH $DEFS"
git show HEAD:$SH/mul_mat_vec_q6_k.comp > /tmp/q6k_orig.comp
$G /tmp/q6k_orig.comp -o /tmp/orig.spv
$G $SH/mul_mat_vec_q6_k.comp -o /tmp/off.spv
$G -DGGML_VK_Q6K_EXACT_DIV=1 $SH/mul_mat_vec_q6_k.comp -o /tmp/on.spv
$G -DGGML_VK_Q6K_EXACT_DIV=1 -DGGML_VK_Q6K_EXACT_DIV_POISON=1 $SH/mul_mat_vec_q6_k.comp -o /tmp/poison.spv
cmp /tmp/orig.spv /tmp/off.spv && spirv-val /tmp/on.spv
```

### 5. Root matrix and acceptance / rejection

| Gate | Action | Accept | Reject |
|---|---|---|---|
| G0 | Compile off / on / poison | off `.spv` byte-identical to orig (or `spirv-dis` diff empty apart from source metadata); on and poison pass `spirv-val` | any compile error or semantic diff in the off build |
| G1 | Python test | prints `OK` | any failure; return the traceback |
| G2 | Poison build, `test-backend-ops -o MUL_MAT -b Vulkan0` | q6_K × f32 matvec cases (n=1) **fail** | no failures: this shader is not dispatched on RADV. Stop, send back the dispatch source, retarget `mul_mat_vecq` Q6_K |
| G3 | Experiment build, same op test | all pass | any fail |
| G4 | Q6 SFT500 greedy: 4 TRAIN mechanics + 5 changing stress prompts | token IDs identical to baseline Q6 build (expected bit-identical, hypothesis) | any divergence (return the first divergent step) |
| G5 | `llama-bench -p 0 -n 64 -r 10 -ngl 99`, baseline vs experiment, interleaved; Vulkan profiler Q6_K matvec time | tg median ≥ +3% and greater than 2× same-session spread; Q6_K matvec kernel time down | < +2% or any regression |
| G6 | Server (ctx4096, b/ub256, np1, t6): 5 changing prompts pristine cold (restart each), 4 TRAIN cold, identical-replay warm; ABBA, ≥3 repeats per arm; Q8 model as unchanged control | decode ms median ≥ −3%; cold wall aggregate not worse than +1%; prompt ms unchanged within ±2%; Q8 control unchanged | greedy mismatch, cold regression > 1%, or decode gain < 2% |

**Promotion.**
- Promotion over the Q8 baseline requires Q6 plus this switch to beat Q8 end-to-end on cold changing prompts.
- That is unlikely: Q6 is 9.84% slower cold and the gap is prefill-dominated.
- If G0–G6 pass without beating Q8, keep it off by default as a shelf candidate.
- Any adoption needs final-weight re-confirmation after RL.

### 6. Narrow source request for iteration 2

1. From `ggml-vulkan.cpp`: the Q6_K matvec pipeline selection (mmvq decision) and the `BLOCK_SIZE`/`NUM_ROWS` specialization constants for AMD GCN.
2. `mul_mat_vec_base.glsl`, for `reduce_result`, `get_offsets` and the `data_b_v4` declaration.
3. The matvec section of `vulkan-shaders-gen.cpp`, so the switch can become a named variant instead of the `sed` activation.