# DSpark full-block draft with 3-token prefix verification (RUN-06-OPUS09, iteration 1)

## Summary

The repair removes the reason for the output-capacity bug. It does not grow any reservation.

- **Draft decode:** in the opt-in mode, `ctx_dft` decodes the full trained 7-row block `[id_last, mask×6]`. Only rows 0..2 are marked as outputs, so the draft emits 3 outputs. The current default `n_max=3` DSpark path already marks exactly 3 outputs per draft decode. The already-running configuration therefore proves the reservation for this count.
- **Target verify:** the verify batch stays at `k+1 ≤ 4` outputs, matching `reserved = n_max+1 = 4`.
- **Why rows 3..6 still matter:** they get KV entries and take part in non-causal attention. They only skip the output gather and the output head.
- **KV cleanup:** there are three local cleanup points. The draft KV never holds noise, rejected or stale rows when the next decode or injection runs.
- **Model-load changes:** none are needed. The constructor refuses the mode, and logs why, unless `0 < n_max < block_size`, `n_seq == 1`, and `block_size ≤ n_batch/n_ubatch` of `ctx_dft`.

**Honest ceiling.** Your lead update shows the changing-prompt misses are dominated by prefill:

| Diagnostic | Server prompt ms | Decode ms |
|---|---|---|
| 1941 IDs | 7502 | 1204 |
| 1975 IDs | 5205 | 1145 |
| 1927 IDs | 4240 | 1084 |

No decode-side speculation can bring the 8.6 s and 6.2 s cases under 5 s. DSpark also adds draft encode/inject work on every prefill ubatch. This patch is a correctness repair of the requested mode, not a deadline fix. The promotion gates below are set accordingly.

---

## 1. New file `common/speculative-dspark.h`

```cpp
#pragma once
// Pure helpers for the experimental DSpark full-block / prefix-verify mode.
// Deliberately free of llama.h so the invariants are CPU-testable without a model.
#include <cstdint>

struct dspark_block_plan {
    int32_t n_block;   // tokens decoded in ctx_dft (all get KV + non-causal attention)
    int32_t n_outputs; // leading block rows marked as outputs == max drafts returned
};

// full_block=false reproduces the pre-patch layout exactly:
//   DSpark n_block = n_max, DFlash n_block = n_max + 1, every row an output.
// Experimental DSpark: decode the whole trained block, output only the first n_max rows.
inline dspark_block_plan dspark_plan_block(bool is_dspark, bool full_block, int32_t n_max, int32_t block_size) {
    if (is_dspark && full_block && n_max > 0 && n_max < block_size) {
        return { block_size, n_max };
    }
    const int32_t n_block = n_max + (is_dspark ? 0 : 1);
    return { n_block, n_block };
}

// true if ctx_dft holds rows at/after p0 that must be dropped before writing p0
inline bool dspark_needs_rm(int32_t pos_max, int32_t p0) {
    return pos_max >= p0;
}

inline bool dspark_env_on(const char * v) {
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}
```

## 2. Patch `common/speculative.cpp`

Line numbers follow the capsule. The include context at the top of the file is not in the capsule, so place the include next to the existing `#include "speculative.h"`.

```diff
--- a/common/speculative.cpp
+++ b/common/speculative.cpp
@@ (top of file, next to the existing includes) @@
 #include "speculative.h"
+#include "speculative-dspark.h"
@@ -931,6 +932,28 @@ struct common_speculative_impl_draft_dflash : public common_speculative_impl {
     // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
     std::vector<float> features_buf;
 
+    // EXPERIMENTAL opt-in (env LLAMA_EXP_DSPARK_FULL_BLOCK=1; DSpark, n_seq==1, 0<n_max<block_size):
+    // decode the full trained block in ctx_dft but mark/return only the first n_max rows, and keep
+    // ctx_dft free of noise/rejected/stale rows at every write point. Default path unaffected.
+    bool     exp_full_block = false;
+    uint64_t exp_steps = 0, exp_drafted = 0, exp_rm_pre_draft = 0, exp_rm_pre_inject = 0;
+
+    // exp only: drop ctx_dft rows at/after p0 for seq_id; true if something was removed
+    bool exp_rm_from(llama_seq_id seq_id, llama_pos p0) {
+        auto * mem = llama_get_memory(params.ctx_dft);
+        if (!dspark_needs_rm(llama_memory_seq_pos_max(mem, seq_id), p0)) {
+            return false;
+        }
+        if (!llama_memory_seq_rm(mem, seq_id, p0, -1)) {
+            GGML_ABORT("DSpark exp: llama_memory_seq_rm(ctx_dft) failed (seq %d, p0 %d)", (int) seq_id, (int) p0);
+        }
+        return true;
+    }
+
+    void exp_drop_blocks(const std::vector<llama_pos> & pos0) {
+        if (!exp_full_block) {
+            return;
+        }
+        for (llama_seq_id s = 0; s < (llama_seq_id) pos0.size(); ++s) {
+            if (pos0[s] >= 0) {
+                exp_rm_from(s, pos0[s]);
+            }
+        }
+    }
+
     common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
@@ -977,6 +1000,20 @@
             this->params.n_min = std::min(this->params.n_min, n_draft_max);
         }
 
+        if (is_dspark && dspark_env_on(std::getenv("LLAMA_EXP_DSPARK_FULL_BLOCK"))) {
+            if (n_seq != 1) {
+                LOG_WRN("%s: LLAMA_EXP_DSPARK_FULL_BLOCK ignored: needs n_seq == 1 (n_seq=%u)\n", __func__, n_seq);
+            } else if (this->params.n_max <= 0 || this->params.n_max >= block_size) {
+                LOG_WRN("%s: LLAMA_EXP_DSPARK_FULL_BLOCK ignored: needs 0 < n_max (%d) < block_size (%d)\n",
+                        __func__, this->params.n_max, block_size);
+            } else if (block_size > (int32_t) llama_n_ubatch(ctx_dft) || block_size > (int32_t) llama_n_batch(ctx_dft)) {
+                LOG_WRN("%s: LLAMA_EXP_DSPARK_FULL_BLOCK ignored: block_size %d exceeds ctx_dft n_batch/n_ubatch\n", __func__, block_size);
+            } else {
+                exp_full_block = true;
+                LOG_INF("%s: EXPERIMENTAL full-block DSpark: %d block rows decoded, %d rows output/verified\n",
+                        __func__, block_size, this->params.n_max);
+            }
+        }
+
         batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
@@ -1017,6 +1054,11 @@
     ~common_speculative_impl_draft_dflash() override {
+        if (exp_full_block) {
+            LOG_INF("%s: EXPERIMENTAL full-block DSpark: steps=%llu drafted=%llu rm_pre_draft=%llu rm_pre_inject=%llu\n", __func__,
+                    (unsigned long long) exp_steps, (unsigned long long) exp_drafted,
+                    (unsigned long long) exp_rm_pre_draft, (unsigned long long) exp_rm_pre_inject);
+        }
         auto * ctx_dft = this->params.ctx_dft;
@@ -1094,6 +1136,12 @@
             const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
 
+            if (exp_full_block) {
+                // never inject over an existing ctx_dft row: drop stale rows (noise block, rejected
+                // verify rows, previous request tail) at/after this batch's first position
+                exp_rm_pre_inject += exp_rm_from(seq_id, batch_in.pos[i_batch_beg[seq_id]]) ? 1 : 0;
+            }
+
             for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
@@ -1163,6 +1211,8 @@
         std::vector<int32_t> i_block_beg(n_seq, -1);
         std::vector<int32_t> n_block    (n_seq,  0);
+        std::vector<int32_t> n_out      (n_seq,  0);
+        std::vector<llama_pos> pos0     (n_seq, -1);
 
@@ -1174,13 +1224,24 @@
             const int32_t n = (int32_t) dp.n_past;
 
-            const int32_t n_draft = params.n_max;
-
-            const int32_t n_block_tokens = n_draft + (is_dspark ? 0 : 1);
+            // default: identical to pre-patch (n_max [+1 for DFlash] rows, all outputs)
+            const dspark_block_plan plan = dspark_plan_block(is_dspark, exp_full_block, params.n_max, block_size);
+            const int32_t n_block_tokens = plan.n_block;
+
+            if (exp_full_block) {
+                // output-capacity invariant: never more outputs than the default n_max path marks
+                GGML_ASSERT(plan.n_outputs <= params.n_max && plan.n_block == block_size);
+                // pre-draft: ctx_dft must end at n-1 so the block occupies exactly [n, n + block_size)
+                exp_rm_pre_draft += exp_rm_from(seq_id, n) ? 1 : 0;
+            }
             i_block_beg[seq_id] = batch.n_tokens;
             n_block    [seq_id] = n_block_tokens;
+            n_out      [seq_id] = plan.n_outputs;
+            pos0       [seq_id] = n;
             for (int32_t i = 0; i < n_block_tokens; ++i) {
-                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, true);
+                // rows >= n_outputs are attention-only (KV + non-causal context), no output row
+                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, i < plan.n_outputs);
             }
         }
@@ -1191,6 +1252,7 @@
         int ret = llama_decode(ctx_dft, batch);
         if (ret != 0) {
             LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
+            exp_drop_blocks(pos0);
             return;
         }
@@ -1213,7 +1275,9 @@
                 const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
 
-                for (int32_t i = 0; i < n_block_tokens; ++i) {
+                // exp mode requires n_seq==1 => beg==0, so batch row idx == output row idx for i < n_out
+                for (int32_t i = 0; i < n_out[seq_id]; ++i) {
                     const int32_t idx = beg + i;
@@ -1263,5 +1327,12 @@
             if (result.size() < (size_t) params.n_min) {
                 result.clear();
             }
+            if (exp_full_block) {
+                ++exp_steps;
+                exp_drafted += result.size();
+            }
         }
+
+        // exp: the noise block is scratch - restore ctx_dft to exactly its pre-draft state
+        exp_drop_blocks(pos0);
     }
```

**Why the default path is unchanged:**
- With `exp_full_block == false`, `plan == {n_max + (is_dspark?0:1)}` for both fields. That makes every flag `true` and the loop bound `n_block_tokens`, exactly as before. Test 1 checks this formula for `n_max` 0..16.
- Every `exp_*` call either returns immediately or is guarded.
- The only default-path additions are two small per-call vectors.

## 3. New test `tests/test-dspark-prefix.cpp`

Build and run:

```
g++ -std=c++17 -O1 -Wall -Wextra -Icommon tests/test-dspark-prefix.cpp -o /tmp/tdp && /tmp/tdp
```

Optionally, add `llama_build_and_test(test-dspark-prefix.cpp)` to `tests/CMakeLists.txt`.

```cpp
// CPU state-machine test for LLAMA_EXP_DSPARK_FULL_BLOCK (DSpark full-block draft, prefix verify).
// Plan/cleanup decisions come from the real common/speculative-dspark.h used by the patch.
// KV cache, greedy verify loop and server reuse below are a MODEL of code not in the capsule.
#include "speculative-dspark.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

using toks = std::vector<int>;
enum Tag { TOK, INJ, NOISE };
static const int EOS_A = 1, EOS_B = 130073, WRONG = 777777;
static bool is_eog(int t) { return t == EOS_A || t == EOS_B; }

struct Kv { // multimap => a duplicated position (poisoned attention) is representable
    std::multimap<int, Tag> c;
    int  pos_max() const { return c.empty() ? -1 : c.rbegin()->first; }
    void rm(int p0) { c.erase(c.lower_bound(p0), c.end()); }
    bool dense() const { int e = 0; for (const auto & kv : c) { if (kv.first != e++) return false; } return true; }
    bool has(Tag t) const { for (const auto & kv : c) { if (kv.second == t) return true; } return false; }
    // token batches must continue at pos_max+1; embedding batches may overlap (worst case)
    void add_tokens(int p0, int n, Tag t) { CHECK(p0 == pos_max() + 1); for (int i = 0; i < n; ++i) c.emplace(p0 + i, t); }
    void add_embd(int p0, int n) { CHECK(p0 <= pos_max() + 1); for (int i = 0; i < n; ++i) c.emplace(p0 + i, INJ); }
};

struct Cfg { int n_max = 3, block = 7, reserve_out = 4, cap = 192, ubatch = 256; bool generic_cleans_dft = true; };

struct Sim {
    Cfg  cfg;
    Kv   tgt, dft;
    toks cached;
    std::vector<int> acc_log;
    long rm_pre_draft = 0, rm_pre_inject = 0, dft_outputs = 0, dft_block_rows = 0;

    // mirrors patched draft(); n_good<0 => draft proposes EOS at row 0 (target must reject)
    toks draft(int n_past, const toks & gold, int G, int n_good) {
        const dspark_block_plan p = dspark_plan_block(true, true, cfg.n_max, cfg.block);
        CHECK(p.n_block == cfg.block && p.n_outputs == cfg.n_max && p.n_outputs <= cfg.reserve_out);
        if (dspark_needs_rm(dft.pos_max(), n_past)) { dft.rm(n_past); ++rm_pre_draft; }
        dft.add_tokens(n_past, p.n_block, NOISE);
        toks out;
        for (int i = 0; i < p.n_outputs; ++i) {
            if (n_good < 0 && i == 0) { out.push_back(EOS_A); continue; }
            out.push_back(i < n_good && G + i < (int) gold.size() ? gold[G + i] : WRONG);
        }
        dft_outputs += p.n_outputs; dft_block_rows += p.n_block;
        dft.rm(n_past); // post-draft drop of the scratch block
        return out;
    }
    // mirrors patched process(): pre-inject cleanup, then feature injection at target positions
    void process(int p0, int n) {
        if (dspark_needs_rm(dft.pos_max(), p0)) { dft.rm(p0); ++rm_pre_inject; }
        dft.add_embd(p0, n);
        CHECK(dft.dense() && !dft.has(NOISE));
    }
    toks run(const toks & prompt, const toks & gold, const std::vector<int> & n_good, int cancel_at = -1) {
        int L = 0; // prefix reuse, always re-evaluate >= 1 prompt token
        while (L < (int) cached.size() && L + 1 < (int) prompt.size() && cached[L] == prompt[L]) ++L;
        tgt.rm(L); cached.resize(L);
        if (cfg.generic_cleans_dft) dft.rm(L);
        for (int p = L; p < (int) prompt.size(); p += cfg.ubatch) {
            const int n = std::min(cfg.ubatch, (int) prompt.size() - p);
            tgt.add_tokens(p, n, TOK);
            process(p, n);
        }
        cached = prompt;
        toks em = { gold[0] };
        int n_past = (int) prompt.size(), G = 1, id_last = gold[0];
        if (is_eog(id_last) || G >= cfg.cap) return em;
        for (int step = 0;; ++step) {
            CHECK(tgt.dense() && tgt.pos_max() == n_past - 1 && (int) cached.size() == n_past);
            toks d = draft(n_past, gold, G, step < (int) n_good.size() ? n_good[step] : cfg.n_max);
            CHECK(dft.dense() && !dft.has(NOISE) && dft.pos_max() == n_past - 1); // pre-draft state
            if (step == cancel_at) return em;                                      // cancellation
            const int room = cfg.cap - G;                                          // verify emits k+1
            if ((int) d.size() > room - 1) d.resize(std::max(0, room - 1));
            const int k = (int) d.size();
            CHECK(k + 1 <= cfg.reserve_out);
            tgt.add_tokens(n_past, k + 1, TOK);
            process(n_past, k + 1);
            int a = 0;
            while (a < k && d[a] == gold[G + a]) ++a;
            acc_log.push_back(a);
            toks step_out(d.begin(), d.begin() + a);
            step_out.push_back(gold[G + a]);
            tgt.rm(n_past + a + 1);
            if (cfg.generic_cleans_dft) dft.rm(n_past + a + 1);
            cached.push_back(id_last);
            cached.insert(cached.end(), d.begin(), d.begin() + a);
            n_past += a + 1;
            id_last = step_out.back();
            bool done = false;
            for (int t : step_out) { em.push_back(t); ++G; if (is_eog(t)) { done = true; break; } }
            CHECK(G <= cfg.cap);
            if (done || G >= cfg.cap) return em;
        }
    }
};

static toks expect(const toks & gold, int cap) {
    toks e;
    for (int t : gold) { if ((int) e.size() == cap) break; e.push_back(t); if (is_eog(t)) break; }
    return e;
}
static toks rand_toks(std::mt19937 & rng, int n) { toks v(n); for (auto & t : v) t = 2 + (int) (rng() % 998); return v; }

int main() {
    // 1. default layout == pre-patch formula; switch never touches DFlash or n_max>=block
    for (int bs : {7, 16}) for (int n_max = 0; n_max <= 16; ++n_max) for (int ds = 0; ds < 2; ++ds) {
        const int old_n_block = n_max + (ds ? 0 : 1);
        const auto p = dspark_plan_block(ds != 0, false, n_max, bs);
        CHECK(p.n_block == old_n_block && p.n_outputs == old_n_block);
        const auto q = dspark_plan_block(false, true, n_max, bs);
        CHECK(q.n_block == n_max + 1 && q.n_outputs == n_max + 1);
        if (n_max <= 0 || n_max >= bs) { const auto r = dspark_plan_block(true, true, n_max, bs); CHECK(r.n_block == n_max && r.n_outputs == n_max); }
    }
    // 2. output capacity: exp outputs == n_max (what default DSpark n_max already marks) <= n_max+1
    for (int n_max = 1; n_max < 7; ++n_max) {
        const auto p = dspark_plan_block(true, true, n_max, 7);
        CHECK(p.n_block == 7 && p.n_outputs == n_max && p.n_outputs <= n_max + 1);
    }
    { const dspark_block_plan rejected = { 7, 7 }; CHECK(rejected.n_outputs > 3 + 1); } // prior patch violated reserve 4
    CHECK(dspark_env_on("1") && !dspark_env_on(nullptr) && !dspark_env_on("") && !dspark_env_on("0") && !dspark_env_on("10"));

    std::mt19937 rng(12345);
    for (int cleans = 0; cleans < 2; ++cleans) {
        Cfg cfg; cfg.generic_cleans_dft = cleans != 0;
        // 3. accept exactly 0..3 on every step
        for (int a = 0; a <= 3; ++a) {
            Sim s; s.cfg = cfg;
            const toks prompt = rand_toks(rng, 346), gold = rand_toks(rng, cfg.cap + 16);
            CHECK(s.run(prompt, gold, std::vector<int>(400, a)) == expect(gold, cfg.cap));
            for (size_t i = 0; i + 1 < s.acc_log.size(); ++i) CHECK(s.acc_log[i] == a);
            CHECK(s.dft_outputs * 7 == s.dft_block_rows * 3);
            if (!cleans && a < 3) CHECK(s.rm_pre_draft > 0); // rejected rows purged locally
        }
        // 4. EOS (both native EOG ids) at each verified row of a step, incl. the bonus row; and as first token
        for (int eos : {EOS_A, EOS_B}) for (int j = 0; j <= 3; ++j) {
            Sim s; s.cfg = cfg;
            toks gold = rand_toks(rng, cfg.cap + 16); gold[1 + j] = eos;
            const toks em = s.run(rand_toks(rng, 887), gold, {3});
            CHECK(em == expect(gold, cfg.cap) && (int) em.size() == 2 + j && em.back() == eos);
        }
        { Sim s; s.cfg = cfg; toks g = rand_toks(rng, 8); g[0] = EOS_A; CHECK(s.run(rand_toks(rng, 50), g, {}) == toks{EOS_A}); }
        // 5. full rejection, including a drafted EOS the target rejects
        for (int ng : {0, -1}) {
            Sim s; s.cfg = cfg;
            const toks gold = rand_toks(rng, cfg.cap + 16);
            CHECK(s.run(rand_toks(rng, 1217), gold, std::vector<int>(400, ng)) == expect(gold, cfg.cap));
            for (int x : s.acc_log) CHECK(x == 0);
        }
        // 6. repeated requests on one slot: replay, changed tail, shorter prefix, cancel, reuse
        {
            Sim s; s.cfg = cfg;
            const toks pa = rand_toks(rng, 1903), ga = rand_toks(rng, cfg.cap + 16);
            toks pb = pa; for (size_t i = 1500; i < pb.size(); ++i) pb[i] = 2 + (int) (rng() % 998);
            const toks pc(pa.begin(), pa.begin() + 900), gb = rand_toks(rng, cfg.cap + 16), gc = rand_toks(rng, cfg.cap + 16);
            std::vector<int> mix; for (int i = 0; i < 400; ++i) mix.push_back((int) (rng() % 4));
            CHECK(s.run(pa, ga, mix) == expect(ga, cfg.cap));
            CHECK(s.run(pa, ga, mix) == expect(ga, cfg.cap));
            CHECK(s.run(pb, gb, mix) == expect(gb, cfg.cap));
            CHECK(s.run(pc, gc, mix) == expect(gc, cfg.cap));
            s.run(pa, ga, mix, 5); // cancel between draft and verify; invariants checked inside
            CHECK(s.run(pa, ga, mix) == expect(ga, cfg.cap));
            if (!cleans) CHECK(s.rm_pre_inject > 0);
        }
        // 7. cap boundaries of the final verify clamp
        for (int cap : {1, 2, 3, 4, 5, 6, 7, 192}) for (int a : {0, 3}) {
            Sim s; s.cfg = cfg; s.cfg.cap = cap;
            const toks gold = rand_toks(rng, cap + 16);
            const toks em = s.run(rand_toks(rng, 346), gold, std::vector<int>(400, a));
            CHECK((int) em.size() == cap && em == expect(gold, cap));
        }
        // 8. fuzz: shared prefixes, random acceptance/EOG/cancellation on one slot
        {
            Sim s; s.cfg = cfg;
            const toks base = rand_toks(rng, 1300);
            for (int r = 0; r < 150; ++r) {
                const int len = 2 + (int) (rng() % 1298);
                toks p(base.begin(), base.begin() + len);
                if (rng() % 2) for (int i = len / 2; i < len; ++i) p[i] = 2 + (int) (rng() % 998);
                toks g = rand_toks(rng, cfg.cap + 16);
                if (rng() % 3 == 0) g[rng() % 40] = (rng() % 2) ? EOS_A : EOS_B;
                std::vector<int> ng; for (int i = 0; i < 400; ++i) ng.push_back((int) (rng() % 5) - 1);
                const int cancel = (rng() % 5 == 0) ? (int) (rng() % 10) : -1;
                const toks em = s.run(p, g, ng, cancel), ex = expect(g, cfg.cap);
                CHECK(em.size() <= ex.size() && std::equal(em.begin(), em.end(), ex.begin()));
                if (cancel < 0) CHECK(em == ex);
            }
        }
    }
    std::printf("test-dspark-prefix: OK\n");
    return 0;
}
```

---

## 4. Stage-by-stage walk-through (np1)

Notation: `n = n_past`, `k ≤ 3` drafts. Each stage is marked **[S]** proven from the capsule, **[M]** unseen code modelled in the test, or **[H]** hypothesis.

1. **Prefill (process).** Unchanged in the default path.
   - Target ubatches have ≤256 rows. Taps are the layer inputs of 2/11/21/31/40, each `[n_chunk,2048]` F32. **[S]**
   - They are concatenated to `[n_chunk,10240]`, run through `llama_encode(ctx_dft)` to `[n_chunk,n_embd_dec]`, and injected at the target positions with 0 outputs. **[S]**
   - In exp mode a pre-inject `seq_rm(p0,-1)` runs first, so injections never duplicate a position.
2. **Draft.**
   - A pre-draft `seq_rm(n,-1)` leaves the draft `pos_max = n-1`.
   - The batch is 7 tokens `[id_last, mask×6]` at positions `n..n+6`, seq 0, with logits flags `[1,1,1,0,0,0,0]`. That gives 3 output rows: logits `[3,130560]` and, if `p_min>0`, `[3,n_embd_dec]` conf rows.
   - Row `i` proposes the token for position `n+1+i`. **[S]** from line 1211: "predicts next token from position 0".
   - A post-draft `seq_rm(n,-1)` leaves the draft `pos_max = n-1` again. The noise block never survives `draft()`, including on decode failure.
3. **Verify (generic).** Identical to the current `n_max=3` path. The batch is `[id_last,d0..d(k-1)]` at `n..n+k` with `k+1 ≤ 4` outputs. Taps are `[k+1,2048]×5`. **[M]**
4. **process(verify batch).** `p0 = n = pos_max+1`, so no removal happens. Injection covers `n..n+k`.
5. **Accept `a` (0..3).**
   - The target removes rows from `n+a+1` onward. **[M]**
   - The draft's stale rows from `n+a+1` onward are removed either by the generic loop or, at the latest, by the next pre-draft or pre-inject step. The test runs both variants.
   - Then `n' = n+a+1` and `id_last` becomes the bonus token.
6. **EOS, cap, cancellation.**
   - EOS and the cap are handled by the generic loop; drafts are not truncated at EOG, same as the default. **[M]**
   - A cancel after `draft()` leaves `ctx_dft` in exactly its pre-draft state.
   - After a verify, stale rows are purged before the next write.

**Hypotheses [H]:**
- **Rows 0..2 match the 7-row draft.** Rows 0..2 of a 7-row block with 3 outputs should equal rows 0..2 when all 7 rows are outputs. Non-output rows still produce K/V and take part in non-causal attention. The Markov head is assumed to be left-causal over output order, and the output gather is a prefix. Floating-point differences from different matmul row counts could flip near-ties. Gate G3 tests this.
- **Masked embedding extraction.** I assume `masked=true` means "output rows only" for the nextn embeddings. With `n_seq==1` both readings of the conf index agree.

**Assumptions about unseen code:**
- `ctx_dft` inherits `n_batch = n_ubatch = 256`. The constructor guard refuses the mode otherwise.
- `llama_memory_seq_rm(mem, seq, p0, -1)` supports partial removal on the draft's KV cache. It aborts loudly if not.
- `batch_in.pos` is non-null; existing line 1139 already assumes this.

## 5. Root gates: accept/reject, about 5 GPU-minutes

| Gate | Run | Reject if |
|---|---|---|
| G0 | CPU test above | Any `FAIL` or any build warning in the new files. |
| G1 | Env unset, DSpark `n_max=3`, the 4 TRAIN prompts cold+warm | Any token-ID or drafted/accepted-count difference vs the current `n_max=3` run, or an `EXPERIMENTAL` log line appears. |
| G2 | `LLAMA_EXP_DSPARK_FULL_BLOCK=1`, `n_max=3`, same 8 cases plus the 3 changing-prompt cases | No `7 block rows decoded, 3 rows output` log line; any assert, decode error, protocol or cap failure; any output-ID difference vs draftless. |
| G3 | First draft step per request, exp vs existing `n_max=7` (verbose `LOG_DBG` candidates, pos 0..2; `n_predict` small) | Any mismatch in the first 3 draft IDs, unless the logged top-2 probabilities differ by less than 1e-3 (numeric tie, which is recorded). |
| G4 | Destructor counters plus per-case draft counts | Accepted per draft step (exp) ≤ current `n_max=3` default on the same cases. A non-zero `rm_pre_draft` means the generic loop leaves rejected rows; that is informative, not a failure. |
| G5 | End-to-end, changing prompts, same cache transitions as the lead's diagnostics, 3 arms: draftless, `n_max=3`, exp | Any case slower than draftless by more than 3%; decode ms not at least 10% better than `n_max=3`; DSpark server prompt ms higher than draftless by more than the decode saving. |

**Budget:** G1 and G2 together are 8+11+8 runs ≈ 3 min, G3 is 4 short runs ≈ 0.5 min, and G5 reuses the G2/G1 changing-prompt runs plus 3 draftless runs ≈ 1 min.

**Expectation (hypothesis, from the measured 7-row screen):** if the 3-row prefix keeps the 7-row block's cumulative acceptance of .542/.500/.375, the expected accepted count is about 1.42 per step with a 4-row verify. That compares with 2.29 per step with an 8-row verify in the screen. Whether this beats the current `n_max=3` depends only on G4, because the verify shapes are identical.

## 6. The capsule questions, briefly

- **Does the confidence head participate?** Only when `p_min > 0` (line 1213). It then truncates at the first row whose `conf[row*n_embd_dec] < p_min`, so it is already a variable draft length. The capsule does not show the writer of that buffer, so it is not proven that element 0 is the trained confidence head. Check the `p_min=` value in the line 967 log before interpreting G4. Exp mode keeps that gating on the 3 output rows.
- **Is a target backend-sampling arm more practical?** No. It measured under 1% and is off; no re-arm is warranted.
- **Should draftless stay default?** Yes, unless G5 passes. Given prefill-dominated misses on changing prompts, I expect draftless to remain default for this workload whatever G3 and G4 show.