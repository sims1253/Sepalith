# Source-walk semantic materialization v2

This review-only template converts root-accepted semantic-v6 TRAIN rows into the pinned PRM03 editing protocol without changing editor geometry.

For each candidate, `semantic_context_geometry.context_buffer_text` reconstructs the exact contiguous pre-edit buffer from `prefix`, the physical empty insertion line, and `suffix_lines`. The materializer requires that buffer's SHA-256 to equal both `selection_source.content_sha256` and `replacement_range.content_sha256`. It also requires the zero-width line/character range to address the reconstructed empty line. Applying the complete target at that range must produce a contiguous window found exactly once in the reopened normalized source.

The target function remains in the editable buffer. Recursively required helper definitions outside that buffer become separate typed `selected_references`, with exact source line spans and hashes. Omitted source gaps are not represented as blank lines and are never concatenated into the editable document.

If the candidate window contains only part of the target function, the materializer derives a contiguous full-source pre-edit document. It removes only the exact target roxygen block, inserts one physical empty anchor, retains all original intervening blank lines and all other source lines, and recomputes the cursor, zero-width range, document EOL, and content hash. Applying the target must reconstruct the reopened source byte-for-byte. A stale candidate hash or range is a hold and cannot trigger this expansion.

`semantic_context_geometry.py` is a small, pinned, side-effect-free seam for later serving-context integration. This packet uses the same PRM03 renderer and application planner as the current evaluation stack, but renderer agreement alone is not evidence that a future serving context constructor selected the same buffer. Serving must import or independently verify this helper before parity can be claimed.

The pinned tokenizer and strict validator verify complete, untruncated targets, BOS/EOS, terminal tokens, token bounds, and prompt/continuation geometry. The command remains a per-shard template. It does not run corpus materialization or admit training data.
