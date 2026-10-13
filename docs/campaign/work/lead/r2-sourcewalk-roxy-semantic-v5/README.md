# Source-walk roxygen semantic analyzer v5

This packet consumes only TRAIN roxygen rows with independent provenance status
`provenance_pass_semantic_analyzer_queued`. It requires the complete target to
occur exactly once immediately before one named function definition.

The Python gate writes the exact independently hashed source bytes to an
isolated temporary file. The R helper hashes and parses that copy, then uses
`codetools` to inventory formals, defaults, variables, and call heads without
executing function bodies. Documented parameters must match actual formals;
ellipsis remains a formal and requires `@param ...`.

References resolve only through base bindings, NAMESPACE imports, the target,
or unique same-file function definitions. Helper dependencies are followed
recursively with cycle detection. Every required helper span must be positive,
unique, and source-backed. Spans are emitted in source order. Unresolved globals
and NSE references remain named holds; no name allowlist is used.

Only exact bytes or uniform CRLF-to-LF conversion may establish target support.
NAMESPACE stability is a gate. The target remains complete and untruncated.

`run_semantic_queue.py` processes a terminal provenance replay serially with
atomic shard publication. Resume binds provenance, packet, analyzer, R-helper,
and output hashes. Exact per-shard and global ID closure is mandatory.

The v4 bounded run remains separate evidence. V5 does not reread candidate
packets before root review. Eleven synthetic, mutation, and source-parser
controls cover the inherited integrity gates and the two lexical fixes. No full
semantic run or training admission has occurred.

V5 fixes two lexical-resolution defects found in v4 review. Function formals are scoped to their owning function while recursive dependencies are walked, so a target formal cannot satisfy a helper free variable. Same-file top-level bindings resolve before imports and base bindings, so shadowing helpers are retained in the context closure.
