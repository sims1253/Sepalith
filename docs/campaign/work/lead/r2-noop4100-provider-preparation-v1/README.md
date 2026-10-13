# No-op 4100 provider preparation

This packet binds a terminal, independently reviewed no-op4100 reconstruction
manifest before preparing two CPU provider lanes on cores 4 and 6.  All 41
shards are verified, including empty shard artifacts.  The policy renders at
16K with a 2,048-token completion reserve, retries provider holds at 32K, and
preserves all 4,100 row identities through context selection.

The provider sees only raw preedit text, a UTF-16 cursor, path/workspace data,
and reviewed dependency hints.  `build_plan.py` rejects any top-level input key
whose name contains `target` or `gold`.  The fixed no-op training output is the
protocol `NO_EDIT`; it is joined only after prediction-time context selection.

The frozen provider-v2 source-import helper treated a valid cursor after the
last function as a process failure.  This packet adds one general, label-blind
case: a fully parsed source with no function after the cursor emits
`source_import_inventory_no_following_function` with an empty dependency set.
File/path/R runtime failures, parse failures, malformed helper output, and
namespace failures remain fatal.  The branch never inspects family, operation,
target, or gold data.

Real integration controls cover a terminal closing-brace cursor, a raw CRLF
blank cursor, and an existing cursor with a following function.  The latter is
byte-identical to the frozen provider output after omitting the new evidence
field.  Removing `Rscript` from `PATH` remains a fatal provider error.

Provider rendering can run after root replaces the terminal-manifest hash
placeholder in `commands.json`.  Final materialization remains blocked until
root binds dedup manifests for the accepted 20,191-row pool, finalized 10,948
semantic output, and eventual 9,535 semantic output.
