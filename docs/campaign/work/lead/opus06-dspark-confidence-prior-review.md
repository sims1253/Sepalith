# RUN-06 advisory: DSpark7 on Renoir/Vulkan

## Bottom line

1. **Keep draftless as the default for now.** In the screen, DSpark was slower in aggregate: warm 5116 ms vs 4807 ms (+6.4%) and cold 26509 ms vs 25209 ms (+5.2%). On the short no-op prompt it was +79% warm. This is screening data, not a statistical result, but it contains no evidence of a win.
2. **The limit is step cost, not acceptance.** Each DSpark verify step costs about 3.4–3.65× one draftless token. At k=7 you need about 2.5 accepted tokens per step to break even. The screen delivered 2.29.
3. **The confidence head changed nothing in the screen.** All 48 drafts were exactly 7 tokens long.
4. **Stock `n_max=3` or `5` does two things at once.** It shortens the proposal, and it also shrinks the non-causal draft block the model was trained on. The cleaner arm is: decode the full 7-row block, then propose only the first k rows. A patch for this is below.
5. **Gating can reduce the no-op regression but cannot remove it.** The floor is (D+P)/t1 per token, where D is draft-decode time and P is feature-injection time.
6. **Target backend sampling is second-order.** Check host sampling time in existing logs before spending GPU time on it.

---

## 1. Observations derived from the packet

**Draft structure**
- 336 = 48 × 7. Per prompt, drafts were 42/49/42/35 tokens, i.e. 6/7/6/5 drafts of full length 7.
- Cold and warm counts were identical, so there are really only **24 unique drafts and 55 unique accepted tokens**. Per-position rates carry roughly ±0.2 uncertainty.
- Output accounting is exact: T_out = 1 + n_drafts + accepted.
  - 12 = 1+6+5
  - 31 = 1+7+23
  - 20 = 1+6+13
  - 20 = 1+5+14
- So every step drafted, and no draft was truncated or cleared.

**Timing**
- Draftless warm cost per token (t1): 56.9 / 57.5 / 58.0 / 59.2 ms. Warm is effectively decode-only.
- DSpark warm step cost, C7 = (W − t1)/n_drafts: **194 / 201 / 206 / 216 ms**, which is 3.42–3.65 × t1. It rises with prompt length.
- Expected accepted tokens A_k for a length-k prefix, from the cumulative acceptance curve: 0.54, 1.04, 1.42, 1.75, 1.96, 2.13, 2.29 (k = 1..7).

**Break-even step cost** C(k) < (1+A_k)·t1, using t1 = 57.5 ms. The second row requires a ≥10% win.

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| C(k) max, break-even (ms) | 89 | 117 | 139 | 158 | 170 | 180 | 189 |
| C(k) max, ≥10% gain (ms) | 81 | 107 | 126 | 144 | 155 | 163 | 172 |

- The measured C7 of about 204 ms predicts roughly 8% slower. The observed aggregate was 6.4% slower, so the simple cost model is consistent.

**Cold prefill overhead** (DSpark cold−warm delta minus baseline cold−warm delta)
- +327 / +123 / +163 / +378 ms.
- n=1 per cell and not monotonic in prompt length, so I treat the cause as unexplained.

## 2. What the code actually does vs. what is hypothesis

**Supported by the supplied code**
- **Confidence gate:**
  - It is read only when `p_min > 0` (line 1213), from `llama_get_embeddings_nextn(ctx_dft)`, element `[idx * n_embd_dec]`.
  - It stops the draft at the first position below threshold. It only shortens drafts; it never changes which token is chosen.
  - If the pointer is null, gating is silently skipped.
- **Token choice:**
  - Each block row is argmaxed independently: top_k=10, then `data[0]`, from a single decode.
  - Sampled tokens are never fed back into later rows. Any Markov/joint coupling must therefore happen inside the graph, which is not in the packet.
- **`n_max` controls the block itself:**
  - The decoded block is `[id_last, mask×(n_max−1)]` (lines 1176–1182).
  - With `llama_set_causal_attn(ctx_dft, false)`, rows 0..k−1 attend to fewer mask rows when `n_max < 7` than they did in training.
- **No adaptive state:** `accept()` is a no-op.
- **`n_min`:** clears any draft shorter than `n_min`, i.e. all-or-nothing.
- **DFlash-only probability gate:** the token-probability `p_min` gate (line 1253) exists only in the DFlash branch. DSpark has no token-probability gate.

**Hypotheses (unverified)**
- **H1:** Column 0 of the nextn row holds a sigmoid acceptance probability from the released head. The graph and converter are not supplied. If it is actually a hidden activation, `p_min` compares an arbitrary value against a probability.
- **H2:** The screen ran with `p_min = 0`, or the conf pointer was null. Verify from the constructor log line `- n_max=%d, n_min=%d, p_min=%.2f`.
- **H3:** Most of C7 is the 8-row target verify on this iGPU, not draft compute. Currently C − t1 ≈ 145 ms/step is unsplit.
- **H4:** A shorter block changes the logits of rows 0..k−1.
- **H5:** The framework trims ctx_dft rows at positions ≥ new n_past after verification. If it does not, stale noise or rejected rows are present.

## 3. Answers to your questions

### Does the released confidence head participate?

- **In code:** only if `p_min > 0`, and only if H1 holds. The mechanism is truncation only.
- **In the screen:** it had zero effect, because 48/48 drafts ran at full length.
- Whether column 0 really is the released head cannot be established from this packet. The logging below tests it directly.

### Can confidence- or acceptance-gated variable length avoid the short no-op regression without task leakage?

**It can reduce the regression but not avoid it.**
- The draft decode (D) runs before confidence is known.
- `process()` (P) runs on every target batch.
- So even a gate that truncates every draft to zero costs t1 + D + P per token. The no-op floor is 1 + (D+P)/t1. At the current C7, the no-op is at 1.79×.
- Removing D requires skipping the draft decode entirely, for example backing off for S steps after an m=0 draft. Even then, P remains.
- Both variants can be evaluated offline from one logging run (see Replay), so do that before patching.

**Leakage rules**
- Allowed signals: per-row confidence, draft top-1 probability, and prior-step acceptance *within the current request*, reset at `begin()`.
- Not allowed: task labels, prompt-template detection, prompt-length buckets, or any cross-request state.
- Tune τ and S on TRAIN logs only. Freeze them before any held-out evaluation.

### Is a target backend-sampling arm more practical?

**It is lower risk but low payoff, and it is not the lever.**
- It can save only host readback plus host argmax over (k+1) × 130560 float32 logits per step (4.2 MB at k=7). I expect single-digit milliseconds (hypothesis) against about 145 ms of overhead per step.
- It also speeds up the baseline, so the relative gap narrows only by the extra rows.
- Support for multi-row greedy verify with backend sampling is not visible in the packet.
- It changes the graph tail. The graph1 lesson applies: it would need a fresh 8/8 exactness check.

**Gate:** read target sampler time per call from the existing perf output (or time the host sample/accept call). Pursue only if it is ≥10 ms per verify step.

## 4. Change: full-block decode, prefix proposal, plus instrumentation

This is env-gated. With both variables unset, behaviour is byte-identical to stock. DFlash (non-DSpark) is untouched.

```diff
--- a/common/speculative.cpp   (b10453 / 3cb7ffb)
+++ b/common/speculative.cpp
@@ ~933 (members, after features_buf)
     std::vector<float> features_buf;
+
+    // RUN-06, env-gated; unset == stock
+    const bool dspark_full = std::getenv("DSPARK_FULL_BLOCK") != nullptr; // decode trained block, propose first n_max rows
+    const bool dspark_log  = std::getenv("DSPARK_LOG")        != nullptr; // per-step conf/p/ids/us
@@ ~1052 process(): wrap existing body
-    bool process(const llama_batch & batch_in) override {
+    bool process(const llama_batch & batch_in) override {
+        const int64_t t0 = ggml_time_us();
+        const bool ok = process_impl(batch_in);
+        if (dspark_log) {
+            LOG_INF("DSPARK_P n=%d us=%lld\n", batch_in.n_tokens, (long long) (ggml_time_us() - t0));
+        }
+        return ok;
+    }
+
+    bool process_impl(const llama_batch & batch_in) {   // original body unchanged
@@ ~1156 draft()
     void draft(common_speculative_draft_params_vec & dparams) override {
+        const int64_t t0 = ggml_time_us();
         auto & ctx_dft = params.ctx_dft;
@@ ~1176
             const int32_t n_draft = params.n_max;
-
-            const int32_t n_block_tokens = n_draft + (is_dspark ? 0 : 1);
+            // keep trained non-causal block geometry; only the proposal length changes
+            const bool full = is_dspark && dspark_full &&
+                              n + block_size <= (int32_t) llama_n_ctx(ctx_dft);
+            const int32_t n_block_tokens = full ? block_size : n_draft + (is_dspark ? 0 : 1);
+            if (full) {
+                auto * mem = llama_get_memory(ctx_dft);
+                const llama_pos pm = llama_memory_seq_pos_max(mem, seq_id);
+                if (dspark_log) { LOG_INF("DSPARK_KV n_past=%d pos_max=%d\n", n, (int) pm); }
+                if (pm >= n) { llama_memory_seq_rm(mem, seq_id, n, -1); } // rows >= n_past can only be stale
+            }
@@ ~1213 DSpark branch
-                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
+                const float * conf_all = (dspark_log || params.p_min > 0.0f) ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
+                const float * conf     = params.p_min > 0.0f ? conf_all : nullptr;
+                const int32_t n_prop   = std::min(n_block_tokens, params.n_max);
+                std::string   s_log;
+                if (dspark_log) {
+                    for (int32_t i = 0; i < n_block_tokens; ++i) {
+                        s_log += string_format(" c%d=%.4f", i,
+                                 conf_all ? conf_all[(size_t) (beg + i) * n_embd_dec] : -1.0f);
+                    }
+                }
-                for (int32_t i = 0; i < n_block_tokens; ++i) {
+                for (int32_t i = 0; i < n_prop; ++i) {
@@ after: const llama_token id = cur_p->data[0].id;
+                    if (dspark_log) { s_log += string_format(" d%d=%d:%.4f", i, id, cur_p->data[0].p); }
@@ after the DSpark for-loop
+                if (dspark_log) {
+                    LOG_INF("DSPARK_D n_past=%d n_block=%d n_prop=%d n_out=%zu us=%lld%s\n",
+                            (int) dp.n_past, n_block_tokens, n_prop, result.size(),
+                            (long long) (ggml_time_us() - t0), s_log.c_str());
+                }
@@ ~1269 accept()
-    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
-        // noop
+    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
+        if (dspark_log) {
+            LOG_INF("DSPARK_A seq=%d n_acc=%u other=%d\n", (int) seq_id, (unsigned) n_accepted, (int) is_other);
+        }
     }
```

**Shape, sampling, verification and cache accounting**

- **Draft batch:**
  - Always `block_size` = 7 rows `[id_last@n, mask@n+1..n+6]`, all with logits=true. This is exactly the screen's geometry, so rows 0..k−1 have the same logits as in the screen.
  - Batch capacity is 256 ≥ 7.
  - The ctx guard falls back to stock behaviour near the ctx limit (max position is about 2100+7 < 4096).
- **Draft outputs:**
  - 7 × 130560 logits, via top-k backend when enabled.
  - 7 × `n_embd_dec` nextn rows. Confidence is read at the existing column-0 convention; it is only logged, never newly interpreted.
- **Markov / joint sampling:**
  - The proposed tokens are an exact prefix of the 7-row joint greedy path, so any in-graph chaining is preserved.
  - Stock `n_max<7` changes the non-causal context (H4). Truncating after a full-block decode does not.
- **Greedy verification:**
  - The target batch is 1+k rows, with (1+k) × 130560 logits.
  - Accepted tokens = LCP(draft, target argmax). This is the unchanged stock path; stock p_min truncation already produces variable k.
  - A different k means a different Vulkan kernel/batch shape and therefore different accumulation order. Exactness must be re-proven for each arm.
- **`process()`:**
  - 5 taps × (1+k) × 2048 fp32 values are concatenated into (1+k) × 10240, then encoded and injected at (1+k) × `n_embd_dec`.
  - This part is unchanged.
- **Cache:**
  - At draft time, `id_last` at position n has not yet been processed by the target. Any ctx_dft rows at positions ≥ n are therefore stale (old noise rows or rejected injections), and `seq_rm(n, -1)` is safe and idempotent.
  - `DSPARK_KV` tests H5 directly.
- **Timers:**
  - `DSPARK_D` covers decode plus sampling. Sampling synchronises on the logits read, so D is reliable.
  - `DSPARK_P` is host wall time. The inject decode may be asynchronous, so P is a lower bound, and any remainder lands in the next target step.

## 5. Experiments (≤ 5 min total GPU; hard stop at 300 s)

### E0 — no GPU
1. Read the constructor log line for effective `n_max`, `n_min`, `p_min` in the screen and in root's current `n_max=3` run.
2. From root's `n_max=3` (and `5`) warm runs, compute C(k) = (W − t1_prompt)/n_drafts, where t1_prompt = baseline W / T_out. Compare against the table in §1.
   - Caveat: this C includes a 3- or 5-row block and a possibly different acceptance (H4).
   - If A_3,stock is below about 1.2 accepted per draft versus the prefix prediction of 1.42, that supports H4. Stop using stock short blocks in that case.
3. Read target sampler time per call from the existing perf output. This is the gate for target backend sampling (§3).

### E1 — optional, ≤ 90 s
Skip this if E0 already yields C(3) and C(5).

```
llama-bench -m <target Q8> -ngl 99 -t 6 -b 256 -ub 256 -fa <as served> -p 1,2,4,6,8 -n 0 -d 512 -r 3
```

- This gives V(N) = N / (t/s of pp-N).
- It slightly underestimates verify cost, because llama-bench computes head logits only for the last row.

### E2 — A0 control + log, ≤ 60 s
- Settings: `DSPARK_FULL_BLOCK=1 DSPARK_LOG=1`, `n_max=7`, `n_min=0`, `p_min=0`. Keep draft backend sampling as in the screen.
- Run the same 4 TRAIN prompts, cold and warm.
- Stop conditions (do not run E3 if any occurs):
  - **Reproduction:** draft counts must be 6/7/6/5 and accepted counts 5/23/13/14 per prompt, cold and warm. Output IDs must match the screen. A mismatch means the patch changed semantics.
  - **Cache hole:** any `DSPARK_KV` line with `pos_max < n_past−1` means missing features.
- Record, but do not stop on: `pos_max > n_past−1`. This means stale draft KV in stock (a finding in itself). The guard removes those rows, so the reproduction check will show whether that changes drafts.

### Replay — no GPU
For each `DSPARK_D` line j:
- Compute m_j = LCP(d_j, out[n_j+1 …]). Cross-check against `DSPARK_A` if `accept()` is called.

For each candidate arm (k, τ):
- k_j = min(k, first i with c_j[i] < τ)
- acc_j = min(m_j, k_j)
- cost_j = V(k_j+1) + D + P(k_j+1)

Where the inputs come from:
- **D and P:** measured from the logs.
- **V:** from E1, or V(8) = C7 − D − P anchored with E0's C(3).

Then:
- Predict warm ms per prompt. The trajectory approximation is about ±10%.
- Also replay the backoff variant, where a skipped step costs t1 + P(1). This is not patched now; implement it only if replay shows it cuts the no-op prompt by ≥15 points without hurting the aggregate.

**Confidence validity** (all three must pass before any `p_min` arm):
- all values in [0, 1];
- standard deviation across positions ≥ 0.02;
- AUROC(c_i vs. "position i accepted") ≥ 0.70 over the logged positions.

If any fails, record the confidence path as non-functional in this build and use no `p_min` arm.

**Precommit:** choose k* (and τ* if confidence is valid) as the argmin of predicted warm aggregate. If the best prediction is <10% faster than baseline, **stop, skip E3, and keep draftless as the default.**

### E3 — ≤ 120 s
- Arms: baseline rerun (same session, to catch drift), A1 (`FULL_BLOCK`, `n_max=k*`, `p_min=0`), and A2 (A1 + `p_min=τ*`) only if confidence is valid and budget remains.
- Hard rejection criteria (any one rejects the arm):
  - **R1:** any output token-ID mismatch versus the draftless baseline in any of the 8 runs, including length or EOS position. Zero tolerance.
  - **R2:** any cap, protocol, transport, `llama_decode`/`llama_encode` error, backend-offload fallback, or unexpected clamp warning.
  - **R3:** `pos_max < n_past−1` anywhere.
  - **R4:** measured C(k*) deviates more than 20% from the replay prediction. The model is wrong; do not extrapolate.
- Screen-level promotion to "candidate" requires all of:
  - warm aggregate ≤ 0.90 × baseline;
  - no-op prompt (346) warm ≤ 1.10 × its baseline;
  - cold aggregate ≤ 1.00 × baseline.
  - Otherwise draftless stays the default.

## 6. Decision and what to record

- **Durable, target-independent output:** C(k) and the required acceptance A_req(k) = C(k)/t1 − 1 for this Q8 shape on Renoir/Vulkan. The final target will change A_k but not C(k), as long as quantization, shapes and backend stay the same. Re-deciding on the final target then needs only one E2-style logging run.
- **Candidate result:** even if A1 passes, it is a TRAIN-prompt screen. Promotion needs a separate held-out statistical evaluation on the final target.
- **Not recommended on current evidence:**
  - graph1 (already rejected);
  - draft requantization or retraining;
  - megakernels;
  - lazy or batched injection to amortize P. Consider this only if the logs show P ≥ 10% of t1.