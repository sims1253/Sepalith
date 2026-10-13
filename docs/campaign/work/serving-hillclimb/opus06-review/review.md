# RUN-06 Opus06 static review: DSpark full-block proposal

Status: **review complete; draftless remains the default**. No source was edited, and no build, model load, GPU run, SSH action, or server launch was performed.

## Review basis

The advisory input was [opus06-dspark-confidence-run/response.json](/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/serving-hillclimb/opus06-dspark-confidence-run/response.json), SHA256 `11c516d362259636c3718084e8fe30474cf40a154f0d87a4af3681ef58b64761`. Its supplied screen scalars report DSpark 7/3/5 exactness 8/8; warm means 1279/973/1072 ms versus 1201 ms draftless; cold means 6627/6321/6471 ms versus 6302 ms; and draft counts 110/336, 98/180, and 106/260. Those are existing screen observations and do not validate the proposed full-block patch.

The pinned source is commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` at `/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453`. Relevant source files and hashes are recorded in the receipt.

## Finding 1: reject the full-block patch as written

The proposal changes a DSpark draft call from `n_max` rows to the trained `block_size` rows, which is 7 for the reviewed arm, while leaving `n_max=3` or `5` as the proposal limit. Every full-block row is added with `logits=true`.

The current bound is still derived from `n_max`: `common_base_params_to_speculative` sets `per_seq = n_max + 1` and the total to `n_parallel * per_seq` (`common/speculative.cpp:2344-2348`). The server computes the same limit before constructing the draft context (`tools/server/server-context.cpp:42-52, 954-956, 1036, 1120`). For a single sequence this gives 4 outputs at `n_max=3` or 6 at `n_max=5`, below the 7 rows decoded by the proposal.

With backend sampling, `llama_context::decode` counts every `logits=true` row and returns `-1` when a sequence exceeds `cparams.n_outputs_max_per_seq` (`src/llama-context.cpp:1670-1688`). With CPU sampling, the 7-row batch reaches `output_reserve`; the implementation asserts that the requested output count does not exceed the context limit (`src/llama-context.cpp:1779-1786, 2032-2189`). The scheduler's worst-case graph is also reserved using `min(n_tokens, cparams.n_outputs_max)`, so a larger output batch is not covered by the normal reserve (`src/llama-context.cpp:629-646`).

The smallest safe repair is to size the **draft context** for the decoded block, independently of the proposed prefix: `n_outputs_max_per_seq >= block_size` and `n_outputs_max >= n_parallel * block_size`, with `n_batch >= block_size`. Keep the target verify limit tied to `1 + proposal_k`. An initial arm may set the draft capacity/limit to 7 and carry a separate frozen `proposal_k`; using `n_max` for both cannot test full-decode/prefix-proposal semantics safely. Add an explicit pre-launch guard that rejects full-block mode when these bounds are not present.

## Finding 2: KV cleanup is conditionally valid for DSpark DFlash, not a universal claim

The reviewed DSpark model is `LLM_ARCH_DFLASH`; when its DSV4 hyperconnection metadata is present, model creation selects `llama_kv_cache_iswa` (`src/llama-model.cpp:2228-2253`). That cache removes the requested range from both its base and SWA caches and returns the combined result (`src/llama-kv-cache-iswa.cpp:113-148`). Thus removing positions `[n, +inf)` can remove rejected/full-block rows for this actual DFlash draft context.

The proposed cleanup must check the boolean return and then require `llama_memory_seq_pos_max(seq_id) == n - 1` before decoding. `seq_pos_max` only identifies the highest present position; it does not itself perform cleanup. The existing server path also has an important invariant: it determines the context removal mode with `common_context_can_seq_rm` and, for full-only contexts, restores the draft checkpoint before removing positions (`common/common.cpp:1559-1598`; `tools/server/server-context.cpp:2961-2990`). The in-draft `seq_rm` call must preserve that checkpoint flow.

This conclusion is specific to the DFlash/ISWA draft. The separate `llama_kv_cache_dsv4` implementation can reject an existing suffix removal when `n_rs_seq == 0` (the `p0 <= pos_max` path in `src/llama-kv-cache-dsv4.cpp:1405-1448`); a generic claim that the patch is safe for every DSV4 context would be false.

## Finding 3: the confidence and Markov graph are structurally real, but the screen does not establish calibration

`build_dspark_markov_head` constructs the Markov bias for each block position, feeds the previous position's `ggml_argmax` into the next position, and applies a one-output confidence projection followed by `ggml_sigmoid` (`src/models/dflash.cpp:227-319`). The confidence is therefore structurally bounded to `[0,1]`, then repeated across the hidden width for `llama_get_embeddings_nextn`. The plain DFlash path supplies the pre-output-norm decoder hidden state; the DSV4 DSpark path supplies the collapsed hyperconnection head state (`src/models/dflash.cpp:450-507, 514-684`).

`p_min` only truncates at the first low-confidence position in the DSpark draft loop. It is disabled at zero, and a null nextn buffer silently removes the gate. The supplied all-full draft counts show that the screen did not exercise a threshold crossing; they do not show whether the head is calibrated. Read the effective constructor `p_min` and log the confidence rows before any threshold arm. The advisor's `[0,1]`/variance/AUROC checks are suitable empirical gates, but must be run on frozen TRAIN logs before held-out evaluation.

The graph's Markov recurrence uses internal argmax IDs. Host top-k or stochastic sampling does not feed sampled IDs back into later graph positions. Therefore “prefix of the joint path” is exact for the graph's greedy path; it is not a claim that arbitrary sampled host IDs were the conditioning path. Exact output IDs still need to be checked for each arm.

## Finding 4: E0 cost arithmetic is a screen statistic, not target-independent capacity

The source records draft candidates, accepted candidates, and verification steps as separate quantities (`tools/server/server-context.cpp:614-634, 3879-3888`). A target verification batch contains the sampled anchor plus the draft rows; response emission contains the accepted prefix plus its replacement. Consequently `1 + n_drafts + accepted` is not the llama.cpp target-output count unless the advisory defines it as a private combined-work unit. Dividing baseline wall time by that quantity is not a valid per-token decode baseline without that definition and matched timing.

`C(k) = (W - t1) / n_drafts` is useful only as a descriptive statistic for the exact prompt lengths, cache state, target/output trajectory, quantization, and backend that produced it. It changes with context length, verify shape, backend quantization/offload, and acceptance trajectory. A full-block acceptance prefix cannot predict the stock `n_max=3/5` arm because shortening the block also changes the non-causal Markov context. Measure matched warm decode-only and cold runs per prompt, with direct draft/decode/injection/verify timers.

Prompt length is prediction-time metadata and is not intrinsically leakage. Length-based routing is admissible if the rule is frozen before evaluation and is not a proxy for task labels. Backend sampling should be pursued only after its actual host time is measured; the advisor's single-digit-millisecond estimate is unverified.

## Safe next arm and stop rules

1. Keep draftless as default. Do not launch the full-block patch until draft output capacity is fixed and checked for the actual `block_size`.
2. Run a capacity-only/preflight check, then a full 7-row, `p_min=0` arm with a separate `proposal_k`; retain checkpoint restore, check KV cleanup return/postcondition, and require exact target output IDs and lengths on every TRAIN case.
3. Log confidence values, Markov draft IDs, block size, output mapping, cache positions, and separate timing components. Use a confidence arm only if the frozen TRAIN calibration checks pass.
4. Treat E0 predictions as local to the measured target/backend. Reject any arm with a capacity error, KV postcondition failure, output-ID mismatch, protocol error, or no measured aggregate win.

The reviewed advisory patch is therefore **static-reject for launch as written**, with a narrowly defined capacity/KV repair path for a future root-owned arm. No quality or promotion claim follows from this review.
