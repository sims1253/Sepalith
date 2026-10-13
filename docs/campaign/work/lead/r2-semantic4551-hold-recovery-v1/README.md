# Semantic 4,551 hold recovery (review only)

This fresh packet preserves the frozen 4,435-row candidate artifact. It
replays the single output-cap hold from target-free input with a fixed 16K
context and a fixed 2,048-token generation reserve. The prompt remains exactly
433 tokens with the same SHA as the original 1,024-reserve selection. The
complete target is 1,675 tokens, is not truncated, reapplies to the pinned
TRAIN source, passes strict token validation, and is new against the frozen
20,190-row review union.

The exact minimum reserve for this row is 1,675 tokens. A 2,048-token fixed
profile is the reviewable operational choice. Selection never reads the target
or gold. Root must review the serving cost and admit the row separately.

`hold-categories.json` accounts for all 115 provider/policy holds from the
predecessor. No supported row remains held solely by context length after the
32K fallback. The mixed-EOL and semantic/namespace groups remain explicit
repair queues.
