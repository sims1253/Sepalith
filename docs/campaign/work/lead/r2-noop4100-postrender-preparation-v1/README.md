# No-op 4100 post-render preparation

This packet verifies the completed 16K parse-retry render and prepares target-free 32K fallback/context selection. It does not scan the full output during preparation, launch a provider, join `NO_EDIT`, deduplicate, or admit training data.

The verifier requires the immutable 4,100-row plan (`093fd437…`) and 4,227-candidate reconstruction manifest (`38736227…`). It accepts exactly 31 nonempty shards 0010–0040, checks all input/output/log/terminal hashes and row counts, requires ordered input/output ID equality and global uniqueness, rejects infrastructure reasons encoded as holds, and preserves the 127 upstream holds (121 provenance plus six mixed-EOL geometry holds). The malformed pre-edit control `e677ee6a8da38436f4bdb6b6` must remain a supported row with explicit unavailable source-import evidence.

After the verifier succeeds, the copied reviewed policy creates 32K inputs only for genuine 16K policy holds. The prepared runner uses the exact parse-unavailable rich provider closure, two CPU lanes on cores 4 and 6, a 32K context and 2,048-token reserve. It hash-checks that runner/source closure before starting. Empty fallback shards get no synthetic terminal. Final context selection retains all 4,100 IDs and selects 16K results first, then 32K results; a remaining context-budget hold remains a named hold and is not an infrastructure failure or silent exclusion.

Fixed `NO_EDIT` targets and prompt/context dedup remain deferred until the Semantic9534 terminal materialization is pinned. No provider selection input contains a target, family-driven branch, or reward.
