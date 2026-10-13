Both arms pass all three observer controls. The 350 ms arm retains 4,444 frames, zero drops, 95 frames with all three multiline sentinels, and 13/13 trusted CDP input/document matches. Its actual URL, startup UUID, binding, renderer, VSIX and runtime pins match. Four PNGs corroborate the visible controls and the two product value suggestions.

| Observation | 1500 ms arm | 350 ms arm |
|---|---:|---:|
| Typing-pauses first-visible interval | 285.8–303.5 ms | 288.3–305.9 ms |
| Accept-save first-visible interval | 389.8–405.8 ms | 621.0–638.6 ms |
| Incoming completion requests | 10 | 12 |
| Completion dispatches | 10 | 11 |
| Forwarded completion HTTP 200 responses | 6 | 6 |
| Requester cancellations on completion route | 4 | 6 |
| Native processing/release pairs | 10 | 11 |
| Evaluated prompt tokens, native aggregate | 913 | 914 |

The 350 ms cancellations comprise one before completion dispatch, four while transport remains unsettled, and one after the backend response completes. Native logs do not identify cancellation or map gateway UUIDs to native task IDs. The explicit cancellation edit occurs 52 ms after notebook observation of dispatch. No cancellation savings are established.

Both accepted and saved buffers have SHA-256 19f01186b77d91ce85166c420259d790928809d4e5cd8c4210e45313e3452e05 and pass base-R parsing. The unfinished typing and switch buffers are user buffers, not accepted malformed model outputs.

The pair uses the same key sequences, final buffer hashes, scored collector/startup, analyzers, harness/provider and VSIX. It is one sequential pair with competing automatic request paths, so it establishes neither debounce causality nor statistical p95 or a recommended default. Native timing and token sums do not measure host index/cache time.

RUN-04 coverage: typing is exercised; switching from R to plaintext is exercised with zero completion dispatch delta. Cursor movement appears incidentally in setup/typing selection events. Dedicated cursor-only, controlled history, diagnostics refresh and anchor-move strata are absent. Event-stratified cache correctness and context rebuilding remain open. Renderer diagnostic metadata does not establish code-diagnostics refresh coverage.

The observer ends gracefully after bounded CDP timeout, retaining 69 ms after the final case. Four initial long gaps precede controls. The terminal guard has no recorded survivors. Root retains independent release and metric admission.
