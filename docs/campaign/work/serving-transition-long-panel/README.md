# RUN-04 long serving transition panel

This is a separate stress fixture for serving latency. It uses newly authored synthetic R code in one coherent rainfall-station reporting script. It contains no user editor contents, TRAIN/DEV/final material, model, or server launch.

The fixture is rendered by the exact EXEC production `context_select.ts`, `campaign_selection.ts`, `history_provider.ts`, `campaign_protocol.ts`, and `campaign_client.ts` modules. Their SHA-256 pins, plus `campaign_requests.ts` and the PRM-03 Python source, are recorded in the fixture and receipt. The pin covers direct runtime sources used by selection/render/client and is not a complete dependency closure.

The selector budget is the production 6000 UTF-16 units. The generated document is about 7.4k characters; each event records the selector's exact used/required UTF-16 units, spans, omissions, complete document identity, and the full rendered prompt string with its hash and character lengths. Shared-prefix lengths are computed from those complete prompt strings. Current source character lengths are reported; token counts remain unresolved until native `/tokenize`.

Events are: baseline with an intentionally unfinished `weekly_mean <- mean(` expression; a typing completion that inserts `values)`; cursor movement into report assembly; anchor movement into anomaly formatting; a real threshold history edit with retained history evidence; and an unchanged repeat control. The anchor and history edits preserve exact URI/version/content SHA identity and range geometry. The repeat is excluded from natural-typing denominators.

Run preparation checks:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types check-long-transition-panel.ts
```

The check regenerates byte-identical JSON, re-renders contexts, verifies selector use stays between 5500 and 6000 UTF-16 units, checks exact change offsets and complete-buffer identities, verifies history old/new text and stale rejection, proves cursor/anchor selection changes, and confirms the unchanged prompt repeat. It performs no network or model operation.

After root review admits native replay and an endpoint is already running, use:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types long-transition-panel.ts \
  --fixture long-transition-fixture.json --server http://127.0.0.1:PORT \
  --mode serial --out run04-long-transition-<target>-<new>.json
```

Add `--diagnostic` for the separate 60-second operational-stall diagnostic. The normal deadline is 5 seconds. The driver sends pinned no-special `/tokenize` requests and integer greedy `/completion` requests with manual BOS 0, cap 192, `cache_prompt:true`, and `return_tokens:true`. It records per-event wall timing, request/response hashes, token IDs, response metrics present in the native response, protocol evidence (`fresh_plan`, `no_op`, `rejected`, `unavailable`), applicability (`fresh`, `timeout`, `error`), and cleanup. It computes plans but performs no editor edit.

Trace denominators keep fresh responses, quality-eligible fresh responses, no-op responses, protocol rejections, and timeouts separate. This panel makes no model-quality or losslessness claim. Output paths use create-only `wx` semantics and refuse existing files so runs remain immutable. Root owns endpoint launch, repeated comparisons, and promotion.
