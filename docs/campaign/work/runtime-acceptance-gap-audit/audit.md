# RUN-04/RUN-05 runtime acceptance-gap audit

Observed at `2026-09-13T04:49:43.940631+00:00`. This is a CPU read-only audit. It did not launch a server, model, editor, benchmark, network operation, or source/state mutation. It covers the selected theta0 Q8 runtime route and the bounded RUN-04/RUN-05 evidence needed before RUN-10.

## Decision

Runtime acceptance is incomplete. The selected theta0 Q8 route remains a viable ordinary serving/fallback candidate, with no promotion or quality claim from the evidence reviewed. Root should spend the next bounded run on one fresh selected-theta0 4K editor cycle plus profile-correct 2K/4K native traces. RUN-10 remains blocked until the final target/draft hashes and paired correctness/latency evidence exist.

The named RUN-04 and RUN-05 output receipts are absent at audit time:

- `docs/campaign/receipts/RUN-04-cache-slot-profile.json`
- `docs/campaign/receipts/RUN-05-context-latency-profile.json`
- `docs/campaign/receipts/RUN-10-final-draft-benchmark.json`

The relevant task specification is `docs/campaign/tasks.json` (SHA-256 `edf63269e4d2e976b37704e1c7970506ed28c2984a96ee0ec335eea90a14004f`).

## Pinned identity and profile

The admitted selected theta0 identity is:

- model: `/home/m0hawk/.local/share/sepalith-campaign-20260915/models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf`, SHA-256 `22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559`;
- server: `/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-vulkan-avx2/bin/llama-server`, SHA-256 `92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f`;
- accepted editor profile: context 4096, output cap 192, batch/ubatch 256, parallel 1, six CPU threads, HTTP threads 2, `-ngl 99`, port 18403;
- argv suffix after model/port: `-t 6 -tb 6 --threads-http 2 --parallel 1 -c 4096 -b 256 -ub 256 -ngl 99 -lv 4`;
- native request contract: `/tokenize` with `add_special:false`, `parse_special:false`, `with_pieces:false`; `/completion` with integer IDs, one manual BOS `0`, `n_predict:192`, `temperature:0`, `stream:false`, `cache_prompt:true`, `return_tokens:true`, greedy decoding, canonical EOS `1`, vocabulary `130560`, normal deadline 5000 ms and diagnostic deadline 60000 ms.

The primary runtime validator hardcodes `contextSize === 4096` and `maxOutputTokens === 192` in `runtime.ts`; the prepared primary manifest does the same. Therefore a server launched with `-c 2048` cannot be represented as a valid primary editor profile by relabeling the 4096 manifest. A profile-aware derivative of the native transition client is required for the 2K trace. The 8K run is stress/latency-only and is excluded from supported editor quality or promotion denominators.

## Existing runnable evidence

| Harness/evidence | Pinned files and SHA-256 | Invocation/profile | Event/evidence coverage and limit |
|---|---|---|---|
| Canonical synthetic transition panel | `README.md` `1bc7349d234555166c9c9ce8b4e8c9771960506b7af64e3c85de97b9365fb17c`; fixture `e6aaf98c888cdcc3c3908d98e0e817ae65d8c94c5a17c966a9019a9d8e2bc8ea`; replay `341a96dd370a67bcbccfb9d59d50d146dd4be8b5b34c819edb8181d4c055c739`; check `0e79ca4731b305694693a1aa97c1f2b04fa79d416c070101d3eebc3ace1b4a08` | CPU check: `node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types docs/campaign/work/serving-transition-panel/check-transition-panel.ts`; native: `node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types docs/campaign/work/serving-transition-panel/replay-transition-panel.ts --fixture docs/campaign/work/serving-transition-panel/transition-fixture.json --server http://127.0.0.1:PORT --mode serial --out TRACE.json` | Nine deterministic events: baseline, two typing edits, cursor move, history change, diagnostics refresh, anchor move, file switch, unchanged-repeat control. Rebuilds exact protocol/selection/history/client sources and records prompt/token/response/application plans. It performs no editor edit. Replay client and fixture pin context 4096; cancel mode proves only client transport abort and records `server_cancellation_unverified`. |
| Primary managed editor preparation | route spec `e8c35ee3d4526789b9780c1328196a52b8245c98ee0b4821c5896db6f5ceb40d`; launcher `dbf94fc191ba84d9e38c01880cdbedfab0108a2a3257a6438fc7cb1204be0a91`; extension harness `b052363ca397bee08ae94af7f441b7ead7c41a47306431fd39347a5dba22c4b2`; old host result `9d720592f10e60f666c8ce19f0bd3a5c73f423e8c2b3252534ffe9d4d6516bb2` | `node run_primary_managed_route.mjs --code /usr/bin/code --vsix ACCEPTED.vsix --target-root TARGET --candidate Q8_0 --port 18403 --run-root FRESH --timeout 600000` (use a fresh selected-theta0 manifest/run root) | Existing host route covered settings, synthetic R files, open/active editor, selection, typing/history prompt identity, file switch, concurrent edit document integrity, one fresh inline commit, and four quiet windows. It has no direct diagnostics subscription/anchor assertion, no visible ghost observation, no raw server request trace, and no defensible first-visible latency. Existing host result used the SFT500 model (`f0be11...`), so it is not selected-theta0 admission evidence. |
| Native readiness transition probe | `runtime_native_probe.py` `f4f556a046f801eb233106c6d773accef317eb83eae5e261c0e0102388ca51a7`; `runtime_transition_probe.py` `f7cf0a0b8d0340c95f7735e602f87f5107516d9c95f859a2033bfd66eb8e4306` | Native probe accepts `--fixture`, `--manifest`, `--url`, `--arm baseline|ngram-mod`, `--cap 192`, `--context`, `--reps`, `--user-deadline-ms 5000`, `--diagnostic-timeout-ms 60000`, `--out`. Transition probe takes `--batch 64|128|256`, `--cancel-deadlines-ms 1000,5000`, `--retry-deadline-ms`, `--cap 192`, `--context`, `--out`. | Four selected TRAIN fixture rows (prompt counts 345, 886, 1216, 1902) and retry/cancel cases. It validates token IDs and native metadata but has no editor ghost, cursor, or actual diagnostic surface. Keep its batch/profile and cancellation fields separate from editor acceptance. |
| Native prewarm A/B pilot | `docs/campaign/work/prewarm-native-live-v3/harness/native_live_ab_harness.ts` SHA-256 `fcd6fcb11179c2fff44e61f79d024761ed2a6ba9811f7d20ee5b9b7e80c678ad` | `PYTHONDONTWRITEBYTECODE=1 node --experimental-strip-types .../native_live_ab_harness.ts --arm off|on --fixture-index 3 --fixture .../native-probe-train-fixture.jsonl --manifest .../native-probe-train-fixture.manifest.json --context-capsule .../context-capsule.json --warm-lead-ms 10|1500 --base-url http://127.0.0.1:18403 --run-id UNIQUE --output-dir OUT` | Records raw native JSON, token usage, cache fields, stop metadata, HTTP/warm queue and full-cycle timings. Only one short completed pair; both long cells timed out; no editor application or quality claim. The 10 ms arm is a contention case, not full prewarm. |
| Accepted theta0 synthetic readout | admission `e0216af6d8a6912f5ecedd618c63a88522ca6d5ce7e811d84a4c8602f307512d`; readout `64269f22e2d728af4bc0347c2693a7f885d793138bba3e090b72c59ce3ae572c`; independent review `0de0f730a02eaff52481b6322e740cf321e5d20ab35e44c4c9726775089b4cb4` | Existing selected theta0 launch used the argv above, with short trace SHA `046dd49615e3b3f4c79d4bce518892c7143495e01dd4952864caeb46d9559549` and long trace SHA `382f7015f2ca6f3c9f4d21237dab27cb55df409fa73f8f4daa6c9cba57bcd256`. | Nine short synthetic native rows completed; eight fresh rows quality-eligible only for protocol/applicability bookkeeping, one unchanged repeat excluded. No real editor application, cancellation acknowledgement, diagnostics subscription, long-context support, or promotion claim. |
| b4 host baseline anchor | receipt `f4d85faf91bbd99efcb12ef1b6283b8db54b3392416b283eb9a955fedcb035ba` | Prior notebook-local b4 host cycle; use only for comparability. | Two fresh/existing phases and 25 checks each establish a host baseline. It explicitly leaves no-op applicability, stale cancellation/ghost nonpublication, raw stop metadata, and final quality open; its model is legacy b4, not selected theta0. |

## Exact missing acceptance evidence

1. **RUN-04 cache/slot profile:** one selected-theta0 run must stratify the nine event IDs by fresh, duplicate/warm, stale/cancelled and timeout; report host index/cache time separately from native prompt-cache time, prompt recomputation, slot/cache reuse, duplicate requests, and cancellation/recovery. The current short readout has nine native responses but no editor application and no event-stratified cache receipt.
2. **RUN-05 supported traces:** produce profile-correct selected-theta0 2K and 4K traces with explicit context size, prompt/output token counts, overflow/rejection, host load, full-cycle p50/p95, parser status and useful-suggestion denominator. The canonical panel can supply event identities but its 4096 hardcode must be adapted for 2K. Use 8K only as a separately labeled stress/latency trace; do not include it in quality, support, or promotion claims.
3. **Real editor transition evidence:** for the 4K managed route, capture exact prompt hash, URI/version/complete-buffer hash, cursor/anchor, request start/finish, parser/applicability and document hash before/after for every event. The existing harness covers typing and file switch but not a direct diagnostics/anchor assertion. `selectPromptContext` currently supplies `diagnostics: []` and `extension.ts` has no diagnostics-change subscription, so the synthetic diagnostics row cannot be upgraded to real editor diagnostics without a reviewed source seam; mark it unsupported if the current extension is used.
4. **Visible ghost and cancellation:** a copied prompt, command return, or unchanged document hash does not show whether an inline ghost was visible. A client abort/cancel barrier does not acknowledge server cancellation. Require a captured visible-ghost-before-accept observation, visible-ghost-after-cancel absence, fresh response after supersession, and correlation to sidecar/server request logs. Keep `server_cancellation_unverified` separate when those observations are unavailable.
5. **RUN-10:** final selected target/draft hashes, paired correctness and end-to-end acceptance are absent. Do not infer them from the b4 host baseline, synthetic native rows, cache reuse, or one inline commit.

## Smallest bounded sequence

Run the following nine IDs in order for the native trace, and for the actual 4K editor route where the current extension supports the event. Record the request identity before dispatch and check that a response is applicable only when URI, document version and complete-buffer SHA-256 still match. Exclude the last row from natural typing denominators.

| # | Event ID | Transition exercised | Required observation |
|---:|---|---|---|
| 1 | `baseline-a1` | Open file A and establish anchor/cursor | fresh prompt and identity; quiet-window gate recorded |
| 2 | `typing-a2-number` | Text insertion in A | history delta, new version/hash, parser plan/no-op, quiet window |
| 3 | `typing-a3-comment` | Second typing edit in A | second history delta and request identity |
| 4 | `cursor-a3-label` | Cursor/selection move with unchanged complete-buffer hash | prompt identity changes only where cursor/region requires; stale prior response cannot apply |
| 5 | `history-a4-result` | Edit that creates the third history entry | history included in prompt and document identity current |
| 6 | `diagnostics-a4-refresh` | Diagnostics refresh | native fixture may inject one diagnostic; current editor path has no diagnostics subscription, so report unsupported/synthetic explicitly |
| 7 | `anchor-a4-result-use` | Anchor/region move to the result line | scope/region prompt changes when `scopeContext` is enabled; applicability uses current identity |
| 8 | `file-switch-b1` | Switch to file B | URI/version/hash reset; old A response rejected; no cross-file reuse |
| 9 | `unchanged-repeat-b1` | Identical request control | duplicate/warm cache and unchanged identity; exclude from typing denominator |

A single extra bounded race is needed for cancellation evidence: dispatch the `typing-a3-comment` request, change to `cursor-a3-label` before completion, then require the old response to be rejected and the current response to remain applicable. This is a stale/cancel control, not a quality row.

## Concrete root actions

1. Rebuild a fresh primary manifest using the selected theta0 model SHA `22401...`, server SHA `92a39...`, accepted VSIX/source/runtime pins and the 4096/192 profile. Verify all hashes and settings before launching a disposable managed host.
2. Run one 4K managed-editor sequence with the nine IDs and a fresh run root. Set `debounceMs:0` for explicit event boundaries. Use the existing quiet-window gate (the harness's 5.2 s observation is a gate, not first-visible latency); separately timestamp request start, first visible ghost, completion, parser and document mutation. Set `scopeContext:true` only if that exact admitted extension/runtime manifest is hashed and accepted. Record diagnostics as unsupported/synthetic under the current `diagnostics:[]` path.
3. Run a profile-aware derivative of the canonical native replay twice, with fresh server/cache identities: `-c 2048` for the 2K trace and `-c 4096` for the 4K trace, both with the fixed b256/ub256/parallel1/threads6/HTTP2/selected theta0 identity and the exact request contract above. Preserve the nine event IDs, raw response/stop/cache metadata and full-cycle timing; reject overflow explicitly. Do not label the 2K server as a primary editor profile.
4. Run the one cancellation race on the 4K route and correlate client cancellation, server log completion/stop and visible ghost before/after. If any correlation is absent, leave cancellation and ghost nonpublication pending and retain ordinary fallback.
5. Write `RUN-04-cache-slot-profile.json` and `RUN-05-context-latency-profile.json` with finite denominators and explicit timeout/cancel exclusions. Keep 8K in a separate stress section. Leave RUN-10 pending until final target/draft evidence is admitted.

No reviewed evidence supports a lossless speedup, quality improvement, or promotion decision.
