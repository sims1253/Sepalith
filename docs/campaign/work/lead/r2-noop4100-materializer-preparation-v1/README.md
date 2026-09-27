# No-op4100 materializer preparation

This packet joins the fixed `NO_EDIT` target only after target-free provider
context selection. It validates the terminal 4,100-row reconstruction and each
raw preedit, authoritative UTF-16 cursor, selected replacement range, sidecar,
prompt hash, tokenizer boundary, target-only labels, and terminal EOS.

Deduplication never uses the shared target alone or bare source hash. It removes
exact prompt+target duplicates and equivalent source+cursor+matched-geometry
rows. It retains same-file rows at different cursors and no-op rows from
different files. A prompt or exact geometry associated with different target
text becomes a named contradiction hold.

The registry template binds the accepted 20,191 rows and the root-reviewed
10,682 semantic rows. It intentionally leaves the eventual 9,534 semantic
cohort null. The executable command fails closed until root supplies a fresh
registry with that terminal manifest and token-row hash. No materialization or
training admission was run by this preparation.
