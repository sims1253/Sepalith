# Sourcewalk no-op expansion preparation

This packet reconstructs each provenance-supported no-op candidate from the immutable normalized source, not from the short scenario window. It locates the reviewed window exactly, converts its local zero-width cursor to a full-document logical cursor, preserves raw UTF-8 and CRLF/LF bytes, and emits a target-free prediction input plus a separate no-op training sidecar. A one-row proof reconstructs a 7,183-byte source from a 668-byte window at global line 38 and reapplies the no-op byte-exactly.

The current pinned census covers 38 completed shards and 3,813 supported candidates plus 111 provenance holds. It is deliberately partial. The preparer also accepts a later pinned 41-shard census, but reports global closure only when `partial=false`, all 41 receipts are present, and the pending list is empty. Shard 38 completed after the pinned census; shards 39–40 remain pending, so none are silently omitted or admitted.

Use the already reviewed semantic provider closure (`r2-semantic10948-provider-materialization-v2`, source manifest SHA `7546d545...`) on the emitted full-document inputs. The materializer then rebuilds the ordinary PRM03 no-op target `[NO_EDIT]\n>>>>>>> UPDATED`, retokenizes it with tokenizer `3e065a...`, validates the strict token-row contract, and deduplicates against all five cohorts in the exact 20,191-row review union. It checks ID, prompt+target, prompt contradiction, and source-projection+target. A target-only duplicate is recorded but is not an exclusion because every valid no-op shares the same target.

Everything remains review-only. Provider rendering, training admission, and the final 41-shard census require separate root decisions.
