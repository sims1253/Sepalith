line` dividing by `wg_denoms`; the `vk_device_architecture::AMD_GCN` spelling; RENOIR having about 7 CUs. |
| Measured (supplied receipt, arithmetic only) | r1 processed 958 tokens in 7502 ms (7.83 ms/token). r2 processed 992 in 5205 ms (5.25 ms/token). cursor-r2 processed 795 in 4240 ms (5.33 ms/token). My hypothesis is that r1 carries first-use warm-up, such as lazy pipeline compilation, so cold-arm denominators must stay separate. |
| Bound, from supplied numbers | Assume RUN-09's 0.80 matrix share transfers to r2, and give the arm a 20% matmul speedup. Prompt time falls to about 4370 ms; adding decode of about 1144 ms gives about 5.5 s. That still exceeds the 5 s deadline, so this arm alone cannot clear the long-panel timeouts. |

## 2. Per-dispatch N (root correction applied)

This assumes the server submits the unprocessed suffix as contiguous 256-token ubatches starting at the cache boundary. That is a hypothesis; the trace in §3 is authoritative.

| Event | Total prompt tokens | Processed | ubatch N sequence | Ubatches with N≥128 |
|---|---|---|---|---|
| long-baseline-r1 | 1941 | 958 | 256, 256, 256, 190 | 4/4 |
| long-typing-r2 | 1975 | 992 | 256, 256, 256, 224 | 4/4 |
| long-cursor-r2 | 1927 | 795 | 256, 256, 256, 27 | 3/4 (27 stays on the baseline `s`) |
| cold r1 / r2 / cursor | 1941 / 1975 / 1927 | all | 7×256 + 149 / 183 / 135 | 8/8 |

The notebook-relevant N values are 256 plus tails in [128, 255]. The tests therefore cover N ∈ {128, 129, 255, 256} and the observed tails; 1900 and 1975 are generality controls only.

Eligible shapes per layer, assuming separate unfused Q/K/V/O/gate/up/down tensors (to be confirmed by the trace):

- Eligible: Q (2048, 2048), O (2048, 2048), gate and up (6144, 2048), down (2048, 6144).
- Excluded: K and V (256, 2048).
- That is about 97.8% of projection MACs per token.
- Decode (N=1) and the last-token output head go through mat-vec and never reach `ggml_vk_mul_mat_q_f16`.

## 3. Gates for root (in order; stop at the first no-go)

### G0: device capability (zero code, no build)

- Check the existing llama startup line for `int dot: 0|1`.
- Check `vulkaninfo | grep -E 'shaderIntegerDotProduct|integerDotProduct4x8BitPackedSignedAccelerated'`.
- If `int dot: 0`, return a **bounded no-go**. The MMQ branch cannot run, the patch is inert, and there is nothing to measure.
  - Do not force the feature.
  - The analogous ordinary-path (`mul_mat_m[Q8_0]`, `mul_mm.comp`) selection is untouched here and would need a new packet.

### G1: baseline instrumentation (patch built, switch off)

- Run a diagnostic with `GGML_VK_Q8_PREFILL_M=0 GGML_VK_Q8_PREFILL_M_TRACE=1`.
- Require, on the eligible shapes with n≥128, all of the following in the trace: `src1_eff=q8_1`, `int_dot=1`, `base=…q8_0_q8_1_l[128x128]`, `split_k_base=1`, `m_pipe=1`, `would_apply=1`.
- Return a no-go if any of these hold:
  - `base` is already `_m` (that is, `mm_l_int=0`, reported as `reasons=…base_is_m`).
  - `src1_eff` is not `q8_1`.
  - `would_apply=0` everywhere.

### G2: identity (switch on)

- Run with `GGML_VK_Q8_PREFILL_M=1 GGML_VK_Q8_PREFILL_M_TRACE=1`.
- Require `chosen=…_m` exactly on the lines where G1 showed `would_apply=1`, and `chosen==base` on every other line.
- Require `split_k_cand=1`.
- Require output IDs and text hashes identical to the G1 run.

### Paired latency (root's command contract)

- Set `GGML_VK_Q8_PREFILL_M_TRACE=0` explicitly in both arms.
- Treat the first request after a fresh start as a cold denominator: the `_m` pipeline may compile on first use.

## 4. Files

Labels used below:

- **[NEW]** is complete code authored here.
- **[ANCHOR]** is a line copied from the capsule excerpt; root must match its literal text in the real file.
- **[LOCATE]** is a statement that does not appear in the capsule and must be found by search.

### 4a. [NEW] `ggml/src/ggml-vulkan/ggml-vulkan-q8-prefill-m.h`

```cpp
// Q8-PREFILL-M: opt-in host-side pipeline-selection experiment for Vulkan Q8_0 x Q8_1 (MMQ, quantize_y) matmuls.
// No shader, format, quantizer, layout, split-k or default-selector change. When every predicate holds, the already
// generated, already availability/shmem-checked unaligned medium MMQ pipeline (mmp->m) replaces the baseline choice.
//
//   GGML_VK_Q8_PREFILL_M=1           enable (exact string "1"; unset or anything else = off)
//   GGML_VK_Q8_PREFILL_M_TRACE=0|1|2 0 silent, 1 first occurrence of each distinct decision line, 2 every dispatch.
//                                    Default: 1 if enabled, else 0. TRACE=1 with the switch off instruments the
//                                    baseline selection without changing it.
//
// Pure C++17; no Vulkan/ggml includes, so evaluate()/parse_config() are CPU-testable.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

namespace ggml_vk_q8pm {

enum reason : uint32_t {
    R_NONE          = 0,
    R_DISABLED      = 1u << 0,
    R_NOT_AMD_GCN   = 1u << 1,
    R_NO_INT_DOT    = 1u << 2,
    R_COOPMAT2      = 1u << 3,
    R_SRC0_NOT_Q8_0 = 1u << 4,
    R_SRC1_NOT_Q8_1 = 1u << 5,  // effective src1 type after the quantize_y decision
    R_ALIGNED       = 1u << 6,  // quantize_y forces aligned=false; true means not the MMQ route
    R_M_NOT_IN_SET  = 1u << 7,
    R_K_NOT_IN_SET  = 1u << 8,
    R_N_BELOW_MIN   = 1u << 9,
    R_BATCH_NOT_1   = 1u << 10,
    R_NO_M_PIPELINE = 1u << 11, // mul_mat_m_int false or mmp->m null (do not force)
    R_SPLIT_K_BASE  = 1u << 12, // normal split_k != 1
    R_SPLIT_K_CAND  = 1u << 13, // split_k recomputed for mmp->m != 1 (candidate would alter split-k)
    R_BASE_IS_M     = 1u << 14, // baseline already medium: substitution would be a no-op
};

constexpr uint64_t N_MIN = 128;

inline bool m_in_set(uint64_t m) { return m == 2048 || m == 6144; }
inline bool k_in_set(uint64_t k) { return k == 2048 || k == 6144; }

struct inputs {
    bool     enabled             = false;
    bool     arch_amd_gcn        = false;
    bool     integer_dot_product = false;
    bool     coopmat2            = false;
    bool     src0_is_q8_0        = false;
    bool     src1_eff_is_q8_1    = false;
    bool     aligned             = false;
    uint64_t m = 0, n = 0, k = 0;
    uint64_t batch               = 0;     // src1 ne2*ne3
    uint64_t src0_batch          = 0;     // src0 ne2*ne3
    bool     mul_mat_m_int       = false; // device->mul_mat_m_int[src0 type]
    bool     m_pipeline_present  = false; // mmp->m != nullptr
    bool     base_is_m           = false;
    uint32_t split_k_base        = 0;
    uint32_t split_k_cand        = 0;     // 0 = not computed
};

// 0 => substitute mmp->m. Otherwise a mask of ALL failed predicates (no short-circuit, for tracing).
inline uint32_t evaluate(const inputs & in) {
    uint32_t r = R_NONE;
    if (!in.enabled)                                 r |= R_DISABLED;
    if (!in.arch_amd_gcn)                            r |= R_NOT_AMD_GCN;
    if (!in.integer_dot_product)                     r |= R_NO_INT_DOT;
    if (in.coopmat2)                                 r |= R_COOPMAT2;
    if (!in.src0_is_q8_0)                            r |= R_SRC0_NOT_Q8_0;
    if (!in.src1_eff_is_q8_1)                        r |= R_SRC1_NOT_Q8_1;
    if (in.aligned)                                  r |= R_ALIGNED;
    if (!m_in_set(in.m))                             r |= R_M_NOT_IN_SET;
    if (!k_in_set(in.k))                             r |= R_K_NOT_IN_SET;
    if (in.n < N_MIN)                                r |= R_N_BELOW_MIN;
    if (in.batch != 1 || in.src0_batch != 1)         r |= R_BATCH_NOT_1;
    if (!in.mul_mat_m_int || !in.m_pipeline_present) r |= R_NO_M_PIPELINE;
    if (in.split_k_base != 1)                        r |= R_SPLIT_K_BASE;
    if (in.split_k_cand != 1)                        r |= R_SPLIT_K_CAND;
    if (in.base_is_m)                                r |= R_BASE_IS_M;
    return r;
}

inline bool would_apply_if_enabled(uint32_t r) { return (r & ~uint32_t(R_DISABLED)) == 0; }

inline std::string reasons_to_string(uint32_t r) {
    static const struct { uint32_t bit; const char * name; } names[] = {
        {R_DISABLED, "disabled"},           {R_NOT_AMD_GCN, "not_amd_gcn"},   {R_NO_INT_DOT, "no_int_dot"},
        {R_COOPMAT2, "coopmat2"},           {R_SRC0_NOT_Q8_0, "src0_not_q8_0"}, {R_SRC1_NOT_Q8_1, "src1_eff_not_q8_1"},
        {R_ALIGNED, "aligned"},             {R_M_NOT_IN_SET, "m_not_in_set"}, {R_K_NOT_IN_SET, "k_not_in_set"},
        {R_N_BELOW_MIN, "n_below_min"},     {R_BATCH_NOT_1, "batch_not_1"},   {R_NO_M_PIPELINE, "no_m_pipeline"},
        {R_SPLIT_K_BASE, "split_k_base"},   {R_SPLIT_K_CAND, "split_k_cand"}, {R_BASE_IS_M, "base_is_m"},
    };
    if (r == 0) {
        return "none";
    }
    std::string s;
    for (const auto & e : names) {
        if (r & e.bit) {
            if (!s.empty()) {
                s += '|';
            }
            s += e.name;
        }
    }
    return s;
}

struct config {
    bool enabled = false;
    int  trace   = 0;
};

inline config parse_config(const char * sw, const char * tr) {
    config c;
    c.enabled = sw != nullptr && std::strcmp(sw, "1") == 0;
    if (tr != nullptr && tr[0] != '\0') {
        const int t = std::atoi(tr);
        c.trace = t < 0 ? 0 : (t > 2 ? 2 : t);
    } else {
        c.trace = c.enabled ? 1 : 0;
    }
    return c;
}

// Read once per process so the decision is deterministic across repeated recording passes.
inline const config & get_config() {
    static const config c = parse_config(std::getenv("GGML_VK_Q8_PREFILL_M"), std::getenv("GGML_VK_Q8_PREFILL_M_TRACE"));
    return c;
}

inline bool first_seen(const std::string & line) {
    static std::mutex            mtx;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex>  lock(mtx);
    return seen.insert(line).second;
}

} // namespace ggml_vk_q8pm
```

### 4b. Edits to `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

These are three anchored insertions, not an exact diff: the full file is not in the capsule.

**Insertion 1 [NEW].** Add after the last existing `#include` line near the top of the file:

```cpp
#include "ggml-vulkan-q8-prefill-m.h"
```

**Insertion 2 [NEW].** Add immediately before `static void ggml_vk_mul_mat_q_f16(` (capsule anchor about line 9145). That position is after `ggml_vk_guess_split_k` (about 8715) and `ggml_vk_guess_matmul_pipeline` (about 8757), so both are already declared.

```cpp
// Q8-PREFILL-M (experimental, opt-in; see ggml-vulkan-q8-prefill-m.h).
// Called right after the baseline ggml_vk_guess_matmul_pipeline() and before any other use of `pipeline`, so
// split_k, buffer sizing, descriptor-set requests / lazy compilation and the dispatch all see one pipeline object.
static vk_pipeline ggml_vk_q8_prefill_m_select(ggml_backend_vk_context * ctx, const vk_matmul_pipeline & mmp,
                                               const vk_pipeline & base, const ggml_tensor * src0,
                                               const ggml_tensor * src1, ggml_type effective_src1_type,
                                               bool aligned, bool disable_split_k) {
    const ggml_vk_q8pm::config & cfg = ggml_vk_q8pm::get_config();
    if (!cfg.enabled && cfg.trace == 0) {
        return base; // default: unchanged, no extra work
    }
    const vk_device & dev = ctx->device;

    vk_pipeline cand;
    if (mmp) {
        cand = mmp->m;
    }

    ggml_vk_q8pm::inputs in;
    in.enabled             = cfg.enabled;
    in.arch_amd_gcn        = dev->architecture == vk_device_architecture::AMD_GCN;
    in.integer_dot_product = dev->integer_dot_product;
    in.coopmat2            = dev->coopmat2;
    in.src0_is_q8_0        = src0->type == GGML_TYPE_Q8_0;
    in.src1_eff_is_q8_1    = effective_src1_type == GGML_TYPE_Q8_1;
    in.aligned             = aligned;
    in.m                   = (uint64_t) src0->ne[1];
    in.n                   = (uint64_t) src1->ne[1];
    in.k                   = (uint64_t) src1->ne[0];
    in.batch               = (uint64_t) (src1->ne[2] * src1->ne[3]);
    in.src0_batch          = (uint64_t) (src0->ne[2] * src0->ne[3]);
    in.mul_mat_m_int       = dev->mul_mat_m_int[src0->type];
    in.m_pipeline_present  = cand != nullptr;
    in.base_is_m           = mmp && (base == mmp->m || base == mmp->a_m);
    in.split_k_base        = ggml_vk_guess_split_k(ctx, (uint32_t) in.m, (uint32_t) in.n, (uint32_t) in.k, disable_split_k, base);
    in.split_k_cand        = cand ? ggml_vk_guess_split_k(ctx, (uint32_t) in.m, (uint32_t) in.n, (uint32_t) in.k, disable_split_k, cand) : 0u;

    const uint32_t    reasons = ggml_vk_q8pm::evaluate(in);
    const vk_pipeline chosen  = reasons == 0 ? cand : base;

    if (cfg.trace > 0) {
        char line[1024];
        snprintf(line, sizeof(line),
            "ggml_vk_q8_prefill_m: enabled=%d m=%llu n=%llu k=%llu batch=%llu src0_batch=%llu src0=%s src1_eff=%s "
            "aligned=%d arch_amd_gcn=%d int_dot=%d coopmat2=%d mm_l_int=%d mm_m_int=%d mm_s_int=%d m_pipe=%d "
            "shader_cores=%u split_k_base=%u split_k_cand=%u would_apply=%d decision=%s reasons=%s "
            "base=%s[%ux%u] cand=%s chosen=%s[%ux%u]\n",
            (int) in.enabled, (unsigned long long) in.m, (unsigned long long) in.n, (unsigned long long) in.k,
            (unsigned long long) in.batch, (unsigned long long) in.src0_batch,
            ggml_type_name(src0->type), ggml_type_name(effective_src1_type),
            (int) in.aligned, (int) in.arch_amd_gcn, (int) in.integer_dot_product, (int) in.coopmat2,
            (int) dev->mul_mat_l_int[src0->type], (int) dev->mul_mat_m_int[src0->type], (int) dev->mul_mat_s_int[src0->type],
            (int) in.m_pipeline_present, (unsigned) dev->shader_core_count, in.split_k_base, in.split_k_cand,
            (int) ggml_vk_q8pm::would_apply_if_enabled(reasons), reasons == 0 ? "use_m" : "keep_base",
            ggml_vk_q8pm::reasons_to_string(reasons).c_str(),
            base->name.c_str(), (unsigned) base->wg_denoms[0], (unsigned) base->wg_denoms[1],
            cand ? cand->name.c_str() : "null",
            chosen->name.c_str(), (unsigned) chosen->wg_denoms[0], (unsigned) chosen->wg_denoms[1]);
        if (cfg.trace >= 2 || ggml_vk_q8pm::first_seen(line)) {
            fputs(line, stderr);
        }
    }
    return chosen;
}
```

**Insertion 3.** Inside `ggml_vk_mul_mat_q_f16`.

[ANCHOR] Keep this existing line as-is; it is copied from the capsule excerpt:

```cpp
    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline(ctx, mmp, ne01, ne11, aligned, qx_needs_dequant ? f16_type : src0->type, effective_src1_type);
```

[NEW] Add these two lines immediately after it:

```cpp
    const vk_pipeline q8pm_base_pipeline = pipeline;
    pipeline = ggml_vk_q8_prefill_m_select(ctx, mmp, pipeline, src0, src1, effective_src1_type, aligned, disable_split_k);
```

[LOCATE] Find the existing statement in this function that assigns `split_k` from `ggml_vk_guess_split_k(...)`. [NEW] Add this line immediately after it:

```cpp
    GGML_ASSERT(pipeline == q8pm_base_pipeline || split_k == 1);
```

Placement rationale:

- **Why not immediately before `ggml_vk_matmul` (the brief's placement):** upstream, as I recall it, computes `padded_n`, requests descriptor sets and may lazily compile the pipeline before that call. Swapping late would dispatch a pipeline that was never requested or compiled.
- **Why this is still the "normal" split-K decision:** placing the swap right after the anchor means every later use sees one object. The existing `split_k` line recomputes with the chosen pipeline, and that equals `split_k_cand`, which must be 1 before any swap. The normal decision is evaluated first, as `split_k_base`.

No CMake change is needed: the header sits in the same directory. No shader variant is added, so the generated SPIR-V set is unchanged.

### 4c. Reviewer checklist (symbols outside the capsule)

- **R1:** `ggml_vk_guess_split_k`'s first parameter is `ggml_backend_vk_context * ctx`; the capsule elides it as `...`. The body must be side-effect free.
- **R2:** the spelling of `ctx->device->architecture` and `vk_device_architecture::AMD_GCN`. A compile error will reveal a mismatch.
- **R3:** `vk_pipeline_struct::name` (a `std::string`), `wg_denoms[]`, and the `vk_matmul_pipeline` fields `m` and `a_m`.
- **R4:** if the pinned build still has a dry-run/record double pass, the decision stays deterministic because the configuration is cached; `TRACE=1` de-duplicates lines.
- **R5:** no other selector call site changes. `mul_mat_id`, mat-vec and `mul_mm.comp` are untouched.

## 5. [NEW] `tests/test-vk-q8-prefill-m.cpp`, part 1 of 3 (harness and selector)

```cpp
// Standalone CPU test for Q8-PREFILL-M (ggml-vulkan-q8-prefill-m.h). Synthetic data only; no Vulkan/ggml/model.
// Build (llama.cpp root):
//   c++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -I ggml/src/ggml-vulkan \
//       tests/test-vk-q8-prefill-m.cpp -o test-vk-q8-prefill-m
// Run:  ./test-vk-q8-prefill-m          selector + geometry exhaustive; Q8 reference on tile-edge rows, all N, all K
//       ./test-vk-q8-prefill-m --full   Q8 reference on every M*N output of every shape (slow, minutes)
// -ffp-contract=off is required: the float shader-mirror must not be FMA-contracted by the host compiler.
#include "ggml-vulkan-q8-prefill-m.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace q = ggml_vk_q8pm;
typedef unsigned long long ull;

static long g_checks = 0;
static long g_fail   = 0;

#define CHECK(cond, ...)                                                           \
    do {                                                                           \
        ++g_checks;                                                                \
        if (!(cond)) {                                                             \
            ++g_fail;                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s :: ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                                     \
            std::fprintf(stderr, "\n");                                            \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------- Part A: selector
enum class tile { S, M, L };
static const char * tn(tile t) { return t == tile::S ? "s" : t == tile::M ? "m" : "l"; }

// Mirror of the non-coopmat2 branch of ggml_vk_guess_matmul_pipeline (capsule 8757-8801).
static tile baseline_tile(bool mm_s, bool mm_m, bool mm_l, uint64_t m, uint64_t n) {
    if ((mm_s && (m <= 32 || n <= 32)) || (!mm_m && !mm_l)) return tile::S;
    if ((mm_m && (m <= 64 || n <= 64)) || !mm_l) return tile::M;
    return tile::L;
}

struct device_flags { bool mm_s = true, mm_m = true, mm_l = true; };

static q::inputs intended(uint64_t m, uint64_t n, uint64_t k) {
    q::inputs in;
    in.enabled = true; in.arch_amd_gcn = true; in.integer_dot_product = true; in.coopmat2 = false;
    in.src0_is_q8_0 = true; in.src1_eff_is_q8_1 = true; in.aligned = false;
    in.m = m; in.n = n; in.k = k; in.batch = 1; in.src0_batch = 1;
    in.mul_mat_m_int = true; in.m_pipeline_present = true; in.base_is_m = false;
    in.split_k_base = 1; in.split_k_cand = 1;
    return in;
}

// Host decision as wired: baseline tile, then optional substitution. CREATE_MMQ makes ->m only if mul_mat_m_int.
static tile decide(q::inputs in, const device_flags & f, uint32_t * r_out) {
    const tile base = baseline_tile(f.mm_s, f.mm_m, f.mm_l, in.m, in.n);
    in.mul_mat_m_int      = in.mul_mat_m_int && f.mm_m;
    in.m_pipeline_present = in.m_pipeline_present && f.mm_m;
    in.base_is_m          = base == tile::M;
    const uint32_t r = q::evaluate(in);
    *r_out = r;
    return r == 0 ? tile::M : base;
}

static const uint64_t kMs[] = {2048, 6144};
static const uint64_t kKs[] = {2048, 6144};

static void test_intended_profile() {
    const uint64_t Ns[] = {128, 129, 255, 256};
    const device_flags f;
    for (uint64_t m : kMs) for (uint64_t k : kKs) for (uint64_t n : Ns) {
        CHECK(baseline_tile(true, true, true, m, n) == tile::L, "baseline must be l m=%llu n=%llu", (ull) m, (ull) n);
        uint32_t r = 0;
        const tile t = decide(intended(m, n, k), f, &r);
        CHECK(r == 0 && t == tile::M, "m=%llu n=%llu k=%llu -> %s reasons=%s",
              (ull) m, (ull) n, (ull) k, tn(t), q::reasons_to_string(r).c_str());
    }
}

// Generality controls only: per-dispatch N <= 256 under ubatch 256, so these are NOT notebook-relevant.
static void test_generality_controls() {
    const uint64_t Ns[] = {1900, 1975};
    const device_flags f;
    for (uint64_t m : kMs) for (uint64_t k : kKs) for (uint64_t n : Ns) {
        uint32_t r = 0;
        CHECK(decide(intended(m, n, k), f, &r) == tile::M && r == 0, "control n=%llu", (ull) n);
    }
}

static void test_small_n_keeps_baseline() {
    const uint64_t Ns[] = {1, 8, 27, 32, 33, 64, 65, 127};   // 27 = observed 795-token tail
    const device_flags f;
    for (uint64_t m : kMs) for (uint64_t k : kKs) for (uint64_t n : Ns) {
        uint32_t r = 0;
        const tile t = decide(intended(m, n, k), f, &r);
        CHECK((r & q::R_N_BELOW_MIN) && t == baseline_tile(true, true, true, m, n),
              "n=%llu -> %s reasons=%s", (ull) n, tn(t), q::reasons_to_string(r).c_str());
    }
}
```

## 6. Test part 2 of 3 (predicate failures, config, ubatch table)

```cpp
static void test_single_predicate_failures() {
    struct flip { const char * name; std::function<void(q::inputs &)> apply; uint32_t expect; };
    const flip flips[] = {
        {"switch off",        [](q::inputs & in) { in.enabled = false; },             q::R_DISABLED},
        {"non-AMD_GCN",       [](q::inputs & in) { in.arch_amd_gcn = false; },        q::R_NOT_AMD_GCN},
        {"int dot off",       [](q::inputs & in) { in.integer_dot_product = false; }, q::R_NO_INT_DOT},
        {"coopmat2",          [](q::inputs & in) { in.coopmat2 = true; },             q::R_COOPMAT2},
        {"src0 not q8_0",     [](q::inputs & in) { in.src0_is_q8_0 = false; },        q::R_SRC0_NOT_Q8_0},
        {"src1 eff not q8_1", [](q::inputs & in) { in.src1_eff_is_q8_1 = false; },    q::R_SRC1_NOT_Q8_1},
        {"aligned",           [](q::inputs & in) { in.aligned = true; },              q::R_ALIGNED},
        {"batch 2",           [](q::inputs & in) { in.batch = 2; },                   q::R_BATCH_NOT_1},
        {"src0 batch 2",      [](q::inputs & in) { in.src0_batch = 2; },              q::R_BATCH_NOT_1},
        {"m pipeline absent", [](q::inputs & in) { in.m_pipeline_present = false; },  q::R_NO_M_PIPELINE},
        {"mul_mat_m_int off", [](q::inputs & in) { in.mul_mat_m_int = false; },       q::R_NO_M_PIPELINE},
        {"split_k base 2",    [](q::inputs & in) { in.split_k_base = 2; },            q::R_SPLIT_K_BASE},
        {"split_k cand 2",    [](q::inputs & in) { in.split_k_cand = 2; },            q::R_SPLIT_K_CAND},
        {"split_k cand n/a",  [](q::inputs & in) { in.split_k_cand = 0; },            q::R_SPLIT_K_CAND},
    };
    const uint64_t Ns[] = {129, 256};
    const device_flags f;
    for (uint64_t m : kMs) for (uint64_t k : kKs) for (uint64_t n : Ns) for (const flip & fl : flips) {
        q::inputs in = intended(m, n, k);
        fl.apply(in);
        uint32_t r = 0;
        const tile t = decide(in, f, &r);
        CHECK(r == fl.expect && t == tile::L, "%s m=%llu n=%llu k=%llu -> %s reasons=%s",
              fl.name, (ull) m, (ull) n, (ull) k, tn(t), q::reasons_to_string(r).c_str());
    }
}

static void test_shape_outside_set() {
    const device_flags f;
    const uint64_t bad_m[] = {256, 2560, 4096, 131072};  // K/V proj, hypothetical fused QKV, other, vocab-like
    for (uint64_t m : bad_m) for (uint64_t n : {128ull, 256ull}) {
        uint32_t r = 0;
        const tile t = decide(intended(m, n, 2048), f, &r);
        CHECK(r == q::R_M_NOT_IN_SET && t == tile::L, "m=%llu reasons=%s", (ull) m, q::reasons_to_string(r).c_str());
    }
    const uint64_t bad_k[] = {256, 2560, 4096};
    for (uint64_t k : bad_k) {
        uint32_t r = 0;
        const tile t = decide(intended(2048, 256, k), f, &r);
        CHECK(r == q::R_K_NOT_IN_SET && t == tile::L, "k=%llu reasons=%s", (ull) k, q::reasons_to_string(r).c_str());
    }
}

static void test_device_flag_variants() {
    uint32_t r = 0;
    device_flags no_l; no_l.mm_l = false;               // baseline already medium -> no-op
    CHECK(decide(intended(2048, 256, 2048), no_l, &r) == tile::M && r == q::R_BASE_IS_M, "no_l %s", q::reasons_to_string(r).c_str());
    device_flags no_m; no_m.mm_m = false;               // m disabled by availability/shmem -> never forced
    CHECK(decide(intended(2048, 256, 2048), no_m, &r) == tile::L && r == q::R_NO_M_PIPELINE, "no_m %s", q::reasons_to_string(r).c_str());
    device_flags only_s; only_s.mm_m = false; only_s.mm_l = false;
    CHECK(decide(intended(2048, 256, 2048), only_s, &r) == tile::S && r == q::R_NO_M_PIPELINE, "only_s %s", q::reasons_to_string(r).c_str());
}

static void test_config_and_strings() {
    struct c { const char * sw; const char * tr; bool en; int trace; };
    const c cases[] = {
        {nullptr, nullptr, false, 0}, {"0", nullptr, false, 0}, {"", nullptr, false, 0}, {"true", nullptr, false, 0},
        {"1", nullptr, true, 1}, {"1", "0", true, 0}, {"1", "2", true, 2}, {"1", "7", true, 2},
        {nullptr, "1", false, 1}, {"0", "1", false, 1},
    };
    for (const c & x : cases) {
        const q::config got = q::parse_config(x.sw, x.tr);
        CHECK(got.enabled == x.en && got.trace == x.trace, "sw=%s tr=%s -> en=%d trace=%d",
              x.sw ? x.sw : "(unset)", x.tr ? x.tr : "(unset)", (int) got.enabled, got.trace);
    }
    CHECK(q::reasons_to_string(0) == "none", "none");
    CHECK(q::reasons_to_string(q::R_DISABLED | q::R_N_BELOW_MIN) == "disabled|n_below_min", "join");
    CHECK(q::would_apply_if_enabled(q::R_DISABLED) && !q::would_apply_if_enabled(q::R_DISABLED | q::R_BATCH_NOT_1), "would_apply");
}

// Hypothesis: server submits the unprocessed suffix as contiguous 256-token ubatches. The trace is authoritative.
static std::vector<uint32_t> ubatch_sizes(uint32_t tokens, uint32_t ubatch) {
    std::vector<uint32_t> v;
    while (tokens > 0) { const uint32_t t = tokens < ubatch ? tokens : ubatch; v.push_back(t); tokens -= t; }
    return v;
}

static void test_ubatch_expectation() {
    struct e { const char * name; uint32_t processed; uint32_t tail; uint32_t eligible; };
    const e evs[] = {
        {"r1 warm", 958, 190, 4}, {"r2 warm", 992, 224, 4}, {"cursor warm", 795, 27, 3},
        {"r1 cold", 1941, 149, 8}, {"r2 cold", 1975, 183, 8}, {"cursor cold", 1927, 135, 8},
    };
    for (const e & x : evs) {
        const std::vector<uint32_t> v = ubatch_sizes(x.processed, 256);
        uint32_t elig = 0;
        for (uint32_t n : v) elig += n >= q::N_MIN;
        CHECK(v.back() == x.tail && elig == x.eligible, "%s tail=%u eligible=%u", x.name, v.back(), elig);
        std::printf("ubatch-expectation %-11s processed=%4u ubatches=%zu tail=%3u eligible_ubatches=%u\n",
                    x.name, x.processed, v.size(), v.back(), elig);
    }
}
```

## 7. Test part 3 of 3 (Q8 reference, dispatch geometry, `main`)

### 7a. Q8 helpers and synthetic problem

```cpp
// ---------------------------------------------------------------- Part B: Q8 scalar reference
struct rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

static float fp16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h >> 15) << 31, e = (h >> 10) & 0x1f, man = h & 0x3ff;
    if (e == 0) {
        const float v = std::ldexp(float(man), -24);
        return sign ? -v : v;
    }
    const uint32_t bits = e == 31 ? (sign | 0x7f800000u | (man << 13)) : (sign | ((e + 112) << 23) | (man << 13));
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t rd_u16(const uint8_t * p) { return uint16_t(p[0] | (uint16_t(p[1]) << 8)); }
static inline int16_t  rd_i16(const uint8_t * p) { const uint16_t u = rd_u16(p); int16_t s; std::memcpy(&s, &u, 2); return s; }
static inline int32_t  rd_i32(const uint8_t * p) {
    const uint32_t u = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    int32_t s; std::memcpy(&s, &u, 4); return s;
}
static inline void wr_u16(uint8_t * p, uint16_t v) { p[0] = uint8_t(v & 0xff); p[1] = uint8_t(v >> 8); }

// GLSL pack32(i16vec2(lo, hi)): lo in bits 0..15.
static inline int32_t pack32_i16(int16_t lo, int16_t hi) {
    const uint32_t u = uint32_t(uint16_t(lo)) | (uint32_t(uint16_t(hi)) << 16);
    int32_t s; std::memcpy(&s, &u, 4); return s;
}
static inline int32_t sbyte(uint32_t w, int i) { const int32_t v = int32_t((w >> (8 * i)) & 0xffu); return v >= 128 ? v - 256 : v; }
// dotPacked4x8EXT (signed x signed).
static inline int32_t dot4x8(int32_t a, int32_t b) {
    int32_t s = 0;
    for (int i = 0; i < 4; ++i) s += sbyte(uint32_t(a), i) * sbyte(uint32_t(b), i);
    return s;
}

static uint16_t rand_scale_bits(rng & r) {
    const uint64_t x = r.next();
    if (x % 97 == 0) return 0;                          // occasional zero-scale block
    const uint16_t e   = uint16_t(9 + (x >> 8) % 6);    // exponent 9..14 -> [2^-6, 2^0)
    const uint16_t man = uint16_t((x >> 16) & 0x3ff);
    return uint16_t((e << 10) | man);
}

static void fill_i8(std::vector<int8_t> & v, rng & r) {
    size_t i = 0;
    while (i < v.size()) {
        const uint64_t x = r.next();
        for (int b = 0; b < 8 && i < v.size(); ++b, ++i) v[i] = int8_t(int((x >> (8 * b)) & 0xff) - 128);
    }
}

struct q8_problem {
    uint32_t M = 0, N = 0, K = 0, nbk = 0;
    std::vector<int8_t>   aq, bq;   // logical int8: A M x K row-major; B N x K (each column contiguous in K)
    std::vector<uint16_t> ad, bd;   // logical fp16 scale bits per 32-block
    std::vector<uint8_t>  a_dev;    // block_q8_0: {f16 d; int8 qs[32]} = 34 bytes, ib = row*nbk + kb
    std::vector<uint8_t>  b_dev;    // block_q8_1_x4 packed128: {f16vec2 ds[4]; ivec4 qs[8]} = 144 bytes
};

static const uint16_t kDsYSentinel = 0x7BFF;  // 65504 in ds.y; the Q8_0 dot product must never read it

static q8_problem make_problem(uint32_t M, uint32_t N, uint32_t K, uint64_t seed) {
    q8_problem p;
    p.M = M; p.N = N; p.K = K; p.nbk = K / 32;
    rng r{seed};
    p.aq.resize(size_t(M) * K); fill_i8(p.aq, r);
    p.bq.resize(size_t(N) * K); fill_i8(p.bq, r);
    p.ad.resize(size_t(M) * p.nbk); for (auto & d : p.ad) d = rand_scale_bits(r);
    p.bd.resize(size_t(N) * p.nbk); for (auto & d : p.bd) d = rand_scale_bits(r);
    for (int j = 0; j < 32; ++j) { p.aq[j] = -128; p.aq[32 + j] = 127; p.bq[j] = -128; p.bq[32 + j] = -128; }  // extremes
    if (M > 1) p.ad[size_t(1) * p.nbk + 2] = 0;

    const size_t nba = size_t(M) * p.nbk;
    p.a_dev.assign(nba * 34, 0);
    for (size_t ib = 0; ib < nba; ++ib) {
        wr_u16(&p.a_dev[ib * 34], p.ad[ib]);
        std::memcpy(&p.a_dev[ib * 34 + 2], &p.aq[ib * 32], 32);
    }
    const size_t nbb = size_t(N) * p.nbk;          // K % 128 == 0 => x4 groups never straddle columns
    p.b_dev.assign((nbb / 4) * 144, 0);
    for (size_t ib = 0; ib < nbb; ++ib) {
        uint8_t * sb = &p.b_dev[(ib / 4) * 144];
        const size_t in = ib % 4;
        wr_u16(sb + 4 * in, p.bd[ib]);             // ds[in].x
        wr_u16(sb + 4 * in + 2, kDsYSentinel);     // ds[in].y (unused for Q8_0)
        std::memcpy(sb + 16 + 32 * in, &p.bq[ib * 32], 32);
    }
    return p;
}
```

### 7b. Per-output check (shader mirror against direct reference)

```cpp
// Shader mirror: block_a_to_shmem/registers + block_b_to_shmem/registers + mmq_dot_product, K loop in
// BK*BK_STEP = 128 steps, split_k == 1 (one float accumulator). Direct reference: logical int8 + binary64.
// Each direct term |isum| <= 2^19 times two 11-bit significands needs <= 42 bits, so it is exact in binary64.
// Tolerance: mirror has 2 roundings per term + (nbk-1) additions => |err| <= gamma_{nbk+1} * T,
// T = sum |term|, u = 2^-24; plus binary64 accumulation error nbk * 2^-53 * T.
static bool check_output(const q8_problem & p, uint32_t row, uint32_t col, double * ratio, bool report) {
    const uint8_t * A = p.a_dev.data();
    const uint8_t * B = p.b_dev.data();
    float    acc = 0.0f;
    double   exact = 0.0, T = 0.0;
    uint32_t blocks = 0;
    bool     int_ok = true;
    for (uint32_t block = 0; block < p.K; block += 32 * 4) {
        for (uint32_t k_step = 0; k_step < 4; ++k_step) {
            const uint32_t kb = block / 32 + k_step;
            const size_t ib_a = size_t(row) * p.nbk + kb;
            const uint8_t * pa = A + ib_a * 34;
            int32_t aqs[8];
            for (uint32_t iqs = 0; iqs < 8; ++iqs) {
                aqs[iqs] = pack32_i16(rd_i16(pa + 2 + 2 * (iqs * 2)), rd_i16(pa + 2 + 2 * (iqs * 2 + 1)));
            }
            const float adm = fp16_to_f32(rd_u16(pa));

            const size_t ib_b = size_t(col) * p.nbk + kb, ib_outer = ib_b / 4, ib_inner = ib_b % 4;
            const uint8_t * pb = B + ib_outer * 144;
            const float bdx = fp16_to_f32(rd_u16(pb + 4 * ib_inner));   // ds[ib_inner].x
            int32_t bqs[8];
            for (uint32_t iqs = 0; iqs < 2; ++iqs) {
                const uint8_t * v = pb + 16 + (ib_inner * 2 + iqs) * 16; // qs[ib_inner*2+iqs] (ivec4)
                for (uint32_t c = 0; c < 4; ++c) bqs[iqs * 4 + c] = rd_i32(v + 4 * c);
            }
            int32_t q_sum = 0;
            for (int i = 0; i < 8; ++i) q_sum += dot4x8(aqs[i], bqs[i]);
            acc += float(q_sum) * adm * bdx;   // ACC_TYPE(float(q_sum) * float(dm) * float(ds.x))

            const int8_t * la = &p.aq[size_t(row) * p.K + size_t(kb) * 32];
            const int8_t * lb = &p.bq[size_t(col) * p.K + size_t(kb) * 32];
            int32_t isum = 0;
            for (int j = 0; j < 32; ++j) isum += int32_t(la[j]) * int32_t(lb[j]);
            if (isum != q_sum) int_ok = false;
            const double term = double(isum) * double(fp16_to_f32(p.ad[ib_a])) * double(fp16_to_f32(p.bd[ib_b]));
            exact += term;
            T += std::fabs(term);
            ++blocks;
        }
    }
    const double u   = std::ldexp(1.0, -24);
    const double tol = (double(p.nbk) + 1.0) * u * T * 1.0001 + double(p.nbk) * std::ldexp(1.0, -53) * T;
    const double err = std::fabs(double(acc) - exact);
    *ratio = tol > 0 ? err / tol : (err == 0 ? 0.0 : INFINITY);
    const bool ok = int_ok && blocks == p.nbk && err <= tol;
    if (!ok && report) {
        std::fprintf(stderr, "q8-ref mismatch M=%u K=%u row=%u col=%u int_ok=%d blocks=%u/%u mirror=%.9g exact=%.17g tol=%.3g\n",
                     p.M, p.K, row, col, (int) int_ok, blocks, p.nbk, double(acc), exact, tol);
    }
    return ok;
}
```

### 7c. Reference driver

```cpp
static void test_q8_reference(bool full) {
    const uint32_t N = 256;  // N in {128,129,255} are column prefixes; tails matter only for dispatch (Part C)
    for (uint64_t M64 : kMs) for (uint64_t K64 : kKs) {
        const uint32_t M = uint32_t(M64), K = uint32_t(K64);
        CHECK(K % 128 == 0, "K=%u must be a multiple of BK*BK_STEP", K);
        const auto t0 = std::chrono::steady_clock::now();
        const q8_problem p = make_problem(M, N, K, 0x51A8ull ^ (uint64_t(M) * 31 + K));
        std::vector<uint32_t> rows;
        for (uint32_t r = 0; r < M; ++r) {
            if (full || r % 64 == 0 || r % 64 == 63 || r == 1 || r == 2) rows.push_back(r);
        }
        long bad = 0, n_out = 0;
        double worst = 0;
        for (uint32_t r : rows) for (uint32_t c = 0; c < N; ++c) {
            double ratio = 0;
            if (!check_output(p, r, c, &ratio, bad < 5)) ++bad;
            if (ratio > worst) worst = ratio;
            ++n_out;
        }
        CHECK(bad == 0, "q8-ref M=%u K=%u bad=%ld", M, K, bad);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("q8-ref M=%u N=%u K=%u rows=%zu%s outputs=%ld blocks/output=%u worst_err/tol=%.3g (%.1fs)\n",
                    M, N, K, rows.size(), full ? "(all)" : "(tile-edge sample)", n_out, p.nbk, worst, s);
    }
}
```

### 7d. Dispatch geometry and `main`

```cpp
// ---------------------------------------------------------------- Part C: dispatch geometry
// Grid = CEIL_DIV(m, wg_denoms[0]) x CEIL_DIV(n, wg_denoms[1]) (division inside ggml_vk_dispatch_pipeline:
// outside-capsule recall). Denoms assumed from the capsule warptiles: m 64x64, l 128x128; the trace prints real ones.
static void test_dispatch_geometry() {
    struct geom { const char * name; uint32_t bm, bn; };
    const geom gs[] = {{"m", 64, 64}, {"l", 128, 128}};
    const uint32_t Ns[] = {27, 128, 129, 135, 149, 183, 190, 224, 255, 256};
    for (uint64_t M64 : kMs) for (uint32_t N : Ns) for (const geom & g : gs) {
        const uint32_t M = uint32_t(M64);
        const uint32_t gx = (M + g.bm - 1) / g.bm, gy = (N + g.bn - 1) / g.bn;
        std::vector<uint8_t> hits(size_t(M) * N, 0);
        for (uint32_t tx = 0; tx < gx; ++tx) for (uint32_t ty = 0; ty < gy; ++ty)
            for (uint32_t rr = 0; rr < g.bm; ++rr) for (uint32_t cc = 0; cc < g.bn; ++cc) {
                const uint32_t r = tx * g.bm + rr, c = ty * g.bn + cc;
                if (r < M && c < N) ++hits[size_t(r) * N + c];   // bounded write; OOB B loads are zero
            }
        bool once = true;
        for (uint8_t h : hits) once = once && h == 1;
        CHECK(once, "coverage %s M=%u N=%u", g.name, M, N);
        const double util = double(M) * N / (double(gx) * gy * g.bm * g.bn);
        std::printf("geometry %s[%ux%u] M=%u N=%3u workgroups=%4u tile_util=%.1f%%%s\n", g.name, g.bm, g.bn, M, N,
                    gx * gy, 100.0 * util, N < q::N_MIN ? " (N<128: arm inactive)" : "");
    }
}

int main(int argc, char ** argv) {
    const bool full = argc > 1 && std::strcmp(argv[1], "--full") == 0;
    test_intended_profile();
    test_generality_controls();
    test_small_n_keeps_baseline();
    test_single_predicate_failures();
    test_shape_outside_set();
    test_device_flag_variants();
    test_config_and_strings();
    test_ubatch_expectation();
    test_dispatch_geometry();
    test_q8_reference(full);
    std::printf("checks=%ld failures=%ld -> %s\n", g_checks, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
```

## 8. Commands for root (none were run here)

```sh
# CPU only (no GPU, synthetic)
c++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -I ggml/src/ggml-vulkan \
    tests/test-vk-q8-prefill-m.cpp -o /tmp/test-vk-q8-prefill-m
/tmp/test-vk-q8-prefill-m
/tmp/test-vk-q8-prefill-m --full     # optional exhaustive M*N

# Host build (no new shader variant; confirm configure reports glslc integer-dot support)
cmake --build <root-b10453-vulkan-build> --target ggml-vulkan
# optional: spirv-val on the existing generated matmul_q8_0_q8_1* SPIR-V in vulkan-shaders.spv/

# G1 (baseline instrumentation) / G2 (identity), then paired latency with TRACE=0 in both arms
GGML_VK_Q8_PREFILL_M=0 GGML_VK_Q8_PREFILL_M_TRACE=1 <root runner> ... --deadline-ms 60000 --diagnostic-first3
GGML_VK_Q8_PREFILL_M=1 GGML_VK_Q8_PREFILL_M_TRACE=1 <root runner> ... --deadline-ms 60000 --diagnostic-first3
GGML_VK_Q8_PREFILL_M=0 GGML_VK_Q8_PREFILL_M_TRACE=0 <root runner> ... --deadline-ms 5000 --fixture <synthetic-long-panel>
GGML_VK_Q8_PREFILL_M=1 GGML_VK_Q8_PREFILL_M_TRACE=0 <root runner> ... --deadline-ms 5000 --fixture <same-synthetic-long-panel>
```

## 9. What each test proves and what remains open

**What the tests would show if they pass** (none have been run):

- The selector predicates and default preservation hold, including N ∈ {128, 129, 255, 256}, the observed tails, the controls, and every single-predicate failure.
- Q8_0 packed16 → pack32 → dot4x8 index and scale semantics agree with an independent int8/binary64 reference across all K blocks, and `ds.y` is never used.
- Both tile geometries write every output exactly once for the observed tails, and the test prints the tile-utilization table.

**Hypotheses the tests do not cover:**

- *Occupancy.* `m` gives 4× the workgroups of `l` (128 vs 32 at M=2048, N=256) and better tail utilization (N=190: about 99% vs about 74%). It also moves about 2× more loaded data per MAC. Whether `m` beats `l` on RENOIR is unknown.
- *Numerics.* The tests do not prove GPU numerics, driver FMA contraction behaviour, integer-dot availability, the actual baseline pipeline or the real per-dispatch N. Only G0 and G1 on the notebook can establish those.

**If G0 shows `int dot: 0`, or G1 shows the baseline is already `_m`:** return a bounded no-go. Keep the patch out of promotion, or drop it; with both variables unset it has no behavioural effect.
