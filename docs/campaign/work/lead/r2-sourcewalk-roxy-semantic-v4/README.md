# Source-walk roxygen semantic analyzer v4

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

The bounded run analyzed all 13 roxygen rows from the frozen 20-row TRAIN
sample. All 13 have source-backed target/function spans and remain root-review
candidates. Eight synthetic, mutation, and source-parser controls pass. No full
semantic run or training admission has occurred.
