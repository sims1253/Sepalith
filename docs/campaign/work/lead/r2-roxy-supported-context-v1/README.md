# DAT-10 source-supported roxygen context

This packet prepares review-only context candidates for the 10,017 IDs frozen
by the root context review. It reconstructs the normalized TRAIN before-state
in memory, selects the complete top-level R definition containing the target
function, and follows same-file named function or expression references to a
fixed point. Roxygen tag and formal-name evidence is recorded separately.

The selector policy is `dat10-roxy-source-supported-context-v1`. It is a
shared source-span identity for later serving and training review. AST and
anchor preservation do not establish semantic documentation support, so every
row remains `review_only_unadmitted`.

Run the CPU pass from the plan worktree:

```text
python3 docs/campaign/work/lead/r2-roxy-supported-context-v1/materialize_supported_context.py \
  --output /mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-supported-context-v1
```

The output contains 10,017 metadata profiles and text-free token rows, an
explicit long-context queue, source-context evidence, and a hash manifest.
Targets are passed unchanged. Contexts and targets over the current budget
are retained for a later bounded-context decision; no source or target text is
written and no admission or training launch is performed.

