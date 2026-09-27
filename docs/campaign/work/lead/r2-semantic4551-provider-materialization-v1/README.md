# Semantic 4,551 provider materialization (review only)

This packet applies the frozen `r2-semantic-provider-integration-v1` selector
to the 4,551 target-free prediction snapshots prepared by root. Selection is
performed before the complete TRAIN targets are joined. The materializer then
re-tokenizes the exact prompt and target, reapplies the edit to the pinned full
source, validates the strict protocol, and deduplicates against the existing
15,006 + 616 + 133 review pools.

The prediction policy first measures a 32K context. Full-document prompts that
also fit 16K with a 1,024-token generation reserve are reused as 16K. Remaining
rows receive an actual 16K selector rerun; supported 16K output wins, while a
row supported only at 32K retains its 32K result. Targets are never truncated.

`render-01`, `render-02`, and `render-03` are failed attempts and are excluded.
`render-02` records the relative helper-path diagnosis. The accepted run is
bound only to `render-04`, whose first 20 rows per shard passed the explicit
infrastructure-error launch gate after a separate 10-row production-working-
directory control.

This prepares candidates for root review. It does not admit them to training or
change production. Provider v2 is a separate safety revision; these artifacts
remain v1 and need a stable-source v2 rerender/delta proof before representing
v2 serving behavior.

