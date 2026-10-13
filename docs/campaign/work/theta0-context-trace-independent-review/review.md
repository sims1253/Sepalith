# RUN-05 theta0 context trace independent review

This is a CPU-only, read-only review of the saved `RUN-05-theta0-cuda-context-a` artifacts. It did not load weights or a framework, start a server, use SSH/network/GPU, read sealed content, or edit campaign state. The saved run is the three sequential Q8 CUDA cells at contexts 2048, 4096, and 8192.

## Result

The inference records are mechanically consistent across the three context sizes. Each cell has 9/9 completed trace rows with the same ordered event IDs, prompt token arrays, generated token arrays, protocol fields, and stored response hashes. The 8192 cell is short-panel allocation stress only; this does not establish support for 8192-token prompts. There is no quality claim in these traces.

The run is not a clean runtime pass. The clients returned 0, but the server/timeout records are nonzero after the requests had completed: context 2048 returned 1, 4096 returned -6 (SIGABRT/core), and 8192 returned -11 (SIGSEGV/core). These failures remain part of the result. The logs prove that the failures occurred during post-request teardown after the slots were idle. The duplicate-signal explanation is plausible from the controller calling `g.reap` on a GNU-timeout process group after timeout already forwards TERM, but the saved artifacts do not prove that this is the root cause.

## Evidence checked

| context | support class | props `default_generation_settings.n_ctx` | trace SHA256 | client | server/timeout | rows | eligible |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2048 | `native_diagnostic_2048` | 2048 | `467372c8c4a74977a5e349603cef9fc82053d1fe163d9019757e39cedcdd9be5` | 0 | 1 | 9 | 8 |
| 4096 | `primary_editor_4096` | 4096 | `6533015f324831134d74d79bff4bbb6c0466eef4632cef3322c4fd68aeeaac10` | 0 | -6 | 9 | 8 |
| 8192 | `stress_only_8192` | 8192 | `58955fa5c76ee1405f77890506f2a0c4b511384e0373f2a5a0b0922cfa3a9f35` | 0 | -11 | 9 | 8 |

The terminal is complete (`13.1243713220465` seconds), with no reported cell failure. Its SHA256 is `5513c52f67484b200c9bf18fedb083ccd0f8bec9a9cb69ab5581230bbde9b160`; the launch manifest SHA256 is `2605be98fa05e5f440cfbd7d1ecc406c2661695de4bec19cc82f8cf31c0eb750`.

All three cells have the same denominator: 9 events, 9 fresh responses, 8 quality-eligible fresh responses, 1 duplicate warm control, and zero stale, timeout, cancellation, overflow, or protocol-rejected responses. The ordered operation counts are 2 `replace` and 7 `no_op`; all 9 parser results are accepted with `stopType=eos`. Prompt token lengths are `[178, 211, 244, 244, 277, 300, 300, 148, 148]`, and generated lengths are `[14, 14, 16, 13, 14, 14, 11, 13, 13]`.

The prompt-token SHA sequence and generated-token SHA sequence match pairwise for all 9/9 rows between every context pair. The protocol tuple `(parserStatus, operation, stopType)` matches for 9/9 rows between every pair. The saved `rawResponseSha256` also matches for 9/9 rows between every pair; timing and timestamps are expected to differ. No context overflow was observed in this short panel.

Observed elapsed times (seconds), from the per-cell terminal records, were:

| context | mean | median | p95/max | min |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 83.406 | 83.699 | 111.076 | 41.123 |
| 4096 | 83.364 | 85.823 | 104.112 | 40.020 |
| 8192 | 78.652 | 81.682 | 92.971 | 41.350 |

These are end-to-end request timings from the saved trace, not first-visible editor latency.

## Runtime and source identity

The saved `props.json` files are the native raw `/props` body. Each contains the requested nested `default_generation_settings.n_ctx`, one slot, Q8_0 model metadata, and the same build/model identity. The per-cell props hashes are:

- 2048: `65dd573f79bb5fe81c774fa66983cc60e0e70c29f09bc21b13f89e5150fb87c8`
- 4096: `7313973bcb30c99d144b0380b7e1de50ebc77e5043cf592d0c6f2145f3c6d814`
- 8192: `a3e291d498dc2c1715affd2021fed038a5ab0f611876ae7e1fa52f1a753888bb`

The server logs for all three cells report CUDA graph mode `USE_GRAPHS = 1` and `offloaded 43/43 layers to GPU`. They report the requested native contexts and the expected increasing KV allocation (84 MiB, 168 MiB, 336 MiB). The server-log hashes are:

- 2048: `aa96a65fe6e8ad58d6288b178accc36a81eeadec1b7d4da05840c96ec1c937b3`
- 4096: `23a0c3b4f430671f1c2211cfeabc8e77c702723c537b79d333b19f97bceec76b`
- 8192: `2438f63c8962ddb7de83e101d38b79c81a1899200316f36fcf2a34d23bdb921b`

The launch manifest binds selected model SHA256 `22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559`, CUDA binary SHA256 `e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee`, and the reviewed source closure. The direct source hashes recorded in the trace are:

- `campaign_protocol.ts`: `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824`
- `campaign_selection.ts`: `70a03d86c8d1c2cd0e117edcbd69cfbac193382b20b555e2d0a327e7e006e8e3`
- `history_provider.ts`: `b2763b7d1b146d398835f4a15eb84e3f0bccfa3bb07b2e9faf5ca9a4eda0a251`
- `context_select.ts`: `a72edaf733395d7b4839f2ca9ed0c2092e4051359279f6839c93e2d5d04b9d59`
- `campaign_requests.ts`: `4fdc6a881b93ae8beb9614c2b210f80fd5b0a2d5592ae02c8174345612a17535`
- `campaign_client.ts`: `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333`
- `campaign_protocol.py`: `5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156`

The trace also records renderer `zeta2-prm03-v1`, tokenizer revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`, and the native no-BOS/manual-BOS0/EOS1 tokenization policy.

## Teardown detail and recommendation

- **2048:** all 9 trace responses completed and the server log reaches idle slots, then records `Received second interrupt` and exits through the timeout record with code 1. No core dump is shown. The nonzero exit keeps this cell mechanically useful but runtime-incomplete.
- **4096:** all 9 responses completed and slots reached idle. The log records a second interrupt, a signal-handler/backtrace path, `terminate called without an active exception`, and `timeout: dumped core`; the timeout record is -6. This blocks clean primary-editor runtime acceptance.
- **8192:** all 9 responses completed and slots reached idle. The log records two second-interrupt messages and `timeout: the monitored command dumped core`; the timeout record is -11. This remains stress evidence only.

The saved run supports prompt/token/protocol parity and short-panel context allocation observations. It does not support a clean CUDA serving admission, first-visible latency claim, long-context quality claim, or model-quality promotion. Root should use the corrected wrapper-only teardown experiment for runtime disposition. The corrected follow-up `RUN-05-theta0-cuda-context-b` is outside this independent review; its saved terminal reports client and server/timeout exit 0 for all three contexts, with trace SHAs `1b2321ec9e537c2848099595149b09032dffff8475c5694737b7c145079b2d44`, `cda10bae8d99f184aeedaec085ded3b7fb5b5a4c38e85ef61faf6986850ffb16`, and `6c6ae66a1721eea0eec704ce0908d873ce1d9a2df2e9c8ce8243e9d5a808176f`. Its exact prompt/generated-token hash sequence matches run A; full semantic equivalence was not independently re-audited here.

No additional scans or experiments are required for this receipt. Root retains the runtime and admission decision.
