# RUN-04 serving transition panel

This packet prepares a deterministic editor-transition panel for serving hill-climbing. All source buffers are synthetic R documents authored for this packet. The panel does not launch VS Code, `llama-server`, a tunnel, or a model.

`transition-fixture.json` is the golden v1 fixture. `replay-transition-panel.ts` rebuilds it with the exact execution-tree `campaign_protocol.ts`, `campaign_selection.ts`, `history_provider.ts`, `context_select.ts`, `campaign_requests.ts`, and `campaign_client.ts` modules. The fixture records those source hashes and the PRM-03 Python source hash. The pin covers the exact sources plus direct runtime dependencies used by selection, rendering, and the client; it is explicitly not a complete dependency closure. The fixture covers sequential typing, cursor movement, a history change, diagnostics refresh, anchor movement, a file switch, and an unchanged-repeat control.

Each event carries the complete pre/post document identity, exact UTF-16 change range and offsets, retained history, renderer-valid context, complete prompt text, prompt SHA-256, UTF-16/UTF-8 lengths, and shared-prefix measurements. The source and prompts are synthetic only and are separate from serving4TRAIN identical replays.

Run the renderer/history and mock-transport checks from this directory:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types check-transition-panel.ts
```

The checks regenerate byte-identical fixture JSON, re-render every context, verify complete-buffer URI/version/SHA identity, recover exact history text, prove anchor rebuild and file-switch history reset, reject an older identity with `checkStale`, preserve canonical `[NO_EDIT]` semantics, and exercise delayed completions, cursor-only requests, document supersession, client abort, duplicate sharing, timeout, and parser rejection. The mock transport never opens a socket.

After an already-running native endpoint is available and root review admits native replay, the root worker can run the normal measurement command:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-transition-panel.ts \
  --fixture transition-fixture.json --server http://127.0.0.1:PORT --mode serial --out run04-transition-trace.json
```

The driver sends `/tokenize` with `add_special:false`, `parse_special:false`, and `with_pieces:false`, then sends `/completion` with integer prompt IDs, one manual BOS ID `0`, greedy `temperature:0`, `n_predict:192`, `stream:false`, `cache_prompt:true`, and `return_tokens:true`. The managed profile is context 4096, one request slot (`--parallel 1` belongs to the root-owned server launch), and canonical EOS ID `1`.

For each request the trace records wall timing, request/response hashes, full prompt and generated token IDs, stop/parser evidence, response applicability, cancellation and cleanup, and denominators. Applicability is one of `fresh`, `stale`, `cancelled`, `timeout`, `not_run`, or `error`. A fresh row only means that the response identity still matches. The driver computes a plan but performs no editor edit. Protocol evidence is kept separately as `fresh_plan`, `no_op`, `rejected`, or `unavailable`; therefore a parser rejection or no-op cannot be reported as an applied edit.

For completion responses, `TracingTransport` retains only numeric fields known from the native probes: timing fields `prompt_n`, `n_prompt_tokens_processed`, `prompt_ms`, `predicted_n`, and `predicted_ms`; usage fields `prompt_tokens`, `completion_tokens`, and `total_tokens`; and top-level `tokens_evaluated` and `tokens_predicted`. When both total prompt usage and processed prompt count are present, it records the derived cache observation with its source field and computed difference. Missing server fields remain absent, and no cache-hit flag or other counter is invented. These response metrics are separate from prompt text shared-prefix lengths.

Use the separate sixty-second deadline only for diagnosing an operational stall:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-transition-panel.ts \
  --fixture transition-fixture.json --server http://127.0.0.1:PORT --diagnostic --mode serial --out run04-transition-diagnostic.json
```

The cancellation and duplicate controls are explicit diagnostics:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-transition-panel.ts \
  --fixture transition-fixture.json --server http://127.0.0.1:PORT --mode cancel --out run04-transition-cancel.json
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types replay-transition-panel.ts \
  --fixture transition-fixture.json --server http://127.0.0.1:PORT --mode duplicate --out run04-transition-duplicate.json
```

Cancellation mode waits on a bounded observable barrier for each attempted `/completion` dispatch, advances a controlled current-event identity getter, and then aborts superseded clients. A row reports client abort separately from `server_cancellation_unverified`; the transport has no server cancellation acknowledgement. The barrier results and superseding event IDs are recorded in `cancellationDiagnostic`.

The five-second deadline is the normal measurement contract. A timeout is an operational failure and is excluded from quality/rejection denominators. Stale and cancelled responses are excluded from quality denominators. The unchanged repeat is an identical warm control and is excluded from natural-typing denominators. Prefix-cache reuse and latency are serving observations and do not establish quality, parity, or losslessness. Native tokenization and HTTP timing remain unresolved until the root worker supplies a live endpoint.

When `--out` is used, the driver opens the destination with create-only semantics (`wx`) and refuses an existing file, preserving prior traces for review. Root owns endpoint launches, integration, repeated comparisons, and promotion.
