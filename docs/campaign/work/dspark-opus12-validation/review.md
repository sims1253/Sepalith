# RUN-06 Opus12 DSpark patch review

The capsule is an actual-success response, but it is a candidate source artifact only. This review used the local pinned llama.cpp source at commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. No model, server, GPU, network, or project build was run.

The response contains one new header, one patch, and one CPU test. The original patch is preserved in `raw-candidate.patch`; its include hunk uses the prose pseudo-header `@@ (top of file, next to the existing includes) @@`. I regenerated `candidate.patch` from the pristine pinned `common/speculative.cpp`, with real unified hunk headers. It has 10 hunks and applies with `patch --fuzz=0 --forward --batch -p1`; the applied file is byte-identical to the isolated candidate (`audit/normalized-apply-diff.txt` is empty). The normalized patch changes only `common/speculative.cpp`; the header and test are staged separately.

## Component decisions

| Component | Decision | Evidence and limit |
|---|---|---|
| Pinned source and extraction | Accept | 12 source identity hashes pass. Raw patch, normalized patch, header, and test are retained. |
| Full-block layout | Accept statically | `dspark_plan_block(true, true, 3, 7)` is `{ n_block=7, n_outputs=3 }`. Rows 0..6 are decoded; only rows 0..2 have `batch.logits=true`. |
| Output reservation | Accept statically | Existing `common_base_params_to_speculative` reserves `n_max + 1 = 4` per sequence. The experimental path marks 3 draft outputs, and target verification remains `k + 1 <= 4`. `llama_context::decode` counts only flagged rows. |
| Default DFlash/DSpark behavior | Accept statically | With `exp_full_block=false`, the helper reproduces `n_max + (is_dspark ? 0 : 1)` and marks every row. `accept()` is unchanged. The experimental guard requires DSpark, `n_seq==1`, `0<n_max<block_size`, and draft `n_batch/n_ubatch >= block_size`. |
| Non-causal model path | Accept as a hypothesis for runtime | The pinned DFlash model graph has a DSpark Markov/confidence head and a non-causal decoder. Source inspection supports decoding the full block for KV context, but no numerical parity run was allowed. |
| KV cleanup | Hold pending one safety amendment | The candidate cleans before draft, before target-feature injection, after successful draft, and after a decode failure; it checks `llama_memory_seq_rm`'s boolean result. It does not check `llama_memory_seq_pos_max` after a successful removal. Add the isolated `audit/candidate-safety-amendment.patch` before runtime admission so a backend that leaves a suffix cannot silently proceed. |
| Accept/cancel/reuse flow | Static pass; runtime pending | Pinned server calls draft, then clears the draft sequence, verifies, and calls `common_speculative_accept`; the DFlash `accept()` implementation is a no-op. A cancel task releases the slot. The candidate removes its scratch block before `draft()` returns, including the `llama_decode` nonzero path. No cancellation or reuse server run was performed. |
| Multi-ubatch | Accept the guarded design | The full block is constrained to one draft ubatch. Target feature injection still walks `n_ubatch` chunks. A real project build/run must confirm the selected profile exposes at least 7 draft batch rows. |
| Numerical output/confidence parity | Pending | The CPU test models KV tags and greedy acceptance. It does not execute the DFlash graph or prove that the first three confidence/logit rows from a 7-row non-causal block equal a 3-row decode. |
| Deadline claim | Reject | The supplied diagnostics show changing-prompt work dominated by prefill (7.5/5.2/4.2 s prompt versus about 1.1/1.2 s decode). This optional decode path cannot support a 5 s deadline claim. |

## CPU evidence

`audit/source_gate.py` passes 44/44 checks against the pinned source and writes `audit/source-gate-result.json`. The standalone test compiles with `g++ -std=c++17 -O1 -Wall -Wextra` and prints `test-dspark-prefix: OK`. Its eight groups cover default formulas, output capacity, exact environment parsing, accept counts 0..3, both EOG IDs (`1`, `130073`), full rejection including drafted EOS, repeated/changed/shorter/cancelled requests, cap boundaries, and 150 fuzz requests in each cleanup configuration (300 fuzz scenarios total). Its KV cache and server loop are explicitly models, so this is a source/header gate, not model evidence.

The actual DFlash source selects an ISWA cache for the DSV4-backed path; ISWA removes from both base and SWA caches. The generic DSV4 cache can reject some partial removals when its recurrent rollback budget is zero, so a runtime receipt must identify the actual selected cache and record every cleanup result. The API documents contiguous `[pos_min,pos_max]` positions and leaves processed ubatches in memory on abort, which is why the post-removal max-position check is required.

## Separate candidate recipe

`candidate-build-recipe.sh` archives the pinned commit into a caller-supplied staging directory, installs the header and test, applies the normalized candidate and required safety amendment with zero fuzz, and runs only the standalone CPU test. Root may then use that staged tree for the project build and bounded server check. The intended run profile is DSpark, `n_max=3`, `n_seq=1`, draft `n_batch/n_ubatch >= 7`, and `LLAMA_EXP_DSPARK_FULL_BLOCK=1`; compare it with the environment unset. Record raw draft row count, flagged output count, draft `seq_pos_max` before/after each cleanup, acceptance/cancel/reuse cases, and confidence/logit parity. Do not attach a quality or latency claim to this candidate without those runtime results.

