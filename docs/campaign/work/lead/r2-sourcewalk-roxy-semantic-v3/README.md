# Source-walk roxygen semantic analyzer v3

This packet consumes only roxygen rows whose independent provenance status is
`provenance_pass_semantic_analyzer_queued`. It reopens each normalized TRAIN R
source, requires the complete target block to occur exactly once immediately
before one named function definition, and preserves the complete target.

The R helper parses source and inventories the unique function's formals,
formal defaults, variable references, and call heads with `codetools`. It does
not execute a function body. The Python gate checks exact `@param`/formal
agreement and resolves references only through base bindings, NAMESPACE
imports, the target itself, or unique same-file top-level definitions. Required
same-file helper spans are emitted in source order. Missing or duplicate helper
spans fail closed.

Unresolved globals and NSE names remain named semantic holds. A prior reviewed
recovery ID is evidence provenance, not permission to reuse the decision under
a new source identity, so it also requires an explicit binding. Member names,
loop variables, defaults, and call heads follow the pinned corrected v2 scope
rules. No names-only allowlist is used.

`run_semantic_queue.py` processes a terminal 41-shard provenance replay one
shard at a time. Each shard is written to a staging directory and atomically
renamed. It reuses a shard only when provenance and candidate-packet hashes plus
the semantic ledger remain exact, then requires exact semantic/provenance ID
closure before publishing its manifest.

The actual 20-row run is deferred while checkpoint 66 is writing because it
would scan eight candidate-packet files. The implementation's five synthetic
and source-parser controls pass. No full semantic run or training admission has
occurred.
