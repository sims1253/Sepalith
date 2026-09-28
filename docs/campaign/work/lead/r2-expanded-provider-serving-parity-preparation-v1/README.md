# Expanded provider / serving parity preparation v1

Status: source-only preparation for a future expanded SFT target. It does not modify or replace the active E750 extension, its 75×3 cap benchmark, b4, a VSIX, or any serving process.

## Exact gap

The active E750 extension calls its local `selectPromptContext` synchronously and renders that result. Its `campaign_protocol.ts` is byte-identical to the expanded TRAIN provider, so the wire serialization agrees. Context selection does not agree: active `context_select.ts` is `a72edaf7…`, while the provider uses `413f0a1c…` together with `full_document_policy.ts`, semantic selection, source-import parsing and NAMESPACE evidence. Matching the renderer therefore does not establish matching inputs, replacement geometry, or full-document/complete-span mode.

The copied provider closure is the review candidate for a future expanded model. `expanded_primary_context.ts` adds the editor integration seam without importing VS Code. It binds an immutable document snapshot, cursor, workspace/document paths, NAMESPACE identity, tokenizer and policy options; checks cancellation and document freshness across async work; caches only supported or explicit parse-unavailable evidence under the full snapshot/cursor/NAMESPACE key; and enforces a total context-selection deadline. It returns no model request on stale, cancelled, deadline, or infrastructure outcomes. It never falls back to the active E750 selector.

`source_import_evidence_unavailable` is an explicit parser outcome and may use the provider's reviewed full-document fallback. A missing `Rscript`, missing package root, malformed helper output, or helper timeout is infrastructure and fails closed. A notebook with no R runtime therefore produces no expanded-model request unless root later admits a separate, evidence-backed policy. No-op family, label, or target is available to this path.

## Integration boundary

A future extension adapter should construct one `ExpandedPrimaryContextCoordinator`, call `providerSelector(client)`, and supply the copied provider options (`path`, history, scope, semantic flag, exact tokenizer hash and generation reserve). Its `stillCurrent` callback must include the existing request lease, runtime generation, restart state, cancellation, URI, document version and content hash. The selected `PromptContext` must feed both `client.complete` and `planNextEdit`; do not reconstruct its replacement range in the VS Code edge.

The current `namespace_runtime.ts` remains unsuitable for a five-second editor request as shipped. It executes the NAMESPACE and source R helpers serially with a two-second timeout each, does not accept `AbortSignal`, and catches infrastructure errors into an unresolved result. The coordinator's deadline prevents a late result from reaching the model, but cannot stop those child helpers. Before VSIX integration, add an owned-child runtime adapter that accepts the coordinator signal, kills only its helper process on cancellation/deadline, and applies one total helper budget. Proposed budget: at most 2,000 ms for selection, leaving at least 3,000 ms of the existing five-second request for tokenization/model response. That value is a candidate requiring latency measurement, not an accepted serving setting.

Cache invalidation must be content based. Namespace evidence is keyed by verified package-root/NAMESPACE identity. Source-import evidence is additionally keyed by document URI/version/content SHA, cursor line/column, and workspace/document path. Keep at most 16 entries, clear on workspace/model generation changes, and recheck current document identity before use. A TTL alone is insufficient.

## Evidence and next checks

The approved exact TRAIN fixtures are copied unchanged:

- `e677ee6a8da38436f4bdb6b6`: malformed source parsing yields explicit `source_import_evidence_unavailable`, then provider `full_document` selection.
- `43b24d15b32aae89fdf72245`: valid source inventory remains byte-identical to the prior provider result.

The bounded provider regression runs the real R helpers, tokenizer bridge and renderer on only those two fixtures. The coordinator test proves exact-key cache reuse/invalidation, stale document rejection, cancellation, deadline and infrastructure separation. A later root-owned integration must also run an editor-shaped test with an abortable owned helper child, then the unchanged 75-case benchmark for the selected expanded target. Prompt SHA parity is required per fixture, but it is evaluated after context-selection mode and replacement geometry parity.

## Commands

```bash
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
P=$PLAN/docs/campaign/work/lead/r2-expanded-provider-serving-parity-preparation-v1
node --no-warnings=ExperimentalWarning --experimental-strip-types "$P/tests/test_coordinator.ts"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python "$PLAN/docs/campaign/work/lead/r2-noop4100-parse-retry-preparation-v1/tests/test_parse_unavailable.py"
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python "$P/tests/test_packet.py"
```
