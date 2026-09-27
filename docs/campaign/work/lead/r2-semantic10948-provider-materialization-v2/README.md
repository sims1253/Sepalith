# Semantic 10,948 provider materialization v2 (review only)

This fresh packet preserves all 24 frozen provider files from `r2-semantic-provider-integration-v1` byte for byte. Its wrapper fixes failure classification before the 10,948-row render: namespace helper infrastructure, malformed helper output, and source-inventory-invalid outcomes terminate the shard before expected-dependency or ordinary policy-hold logic can rewrite them. Tokenizer bridge failures also terminate the process.

The target-free prediction policy uses 16K then 32K context with a fixed 2,048-token output reserve. Context-only holds after 32K enter an exact 64K/128K follow-up queue. Targets are joined only after selection. Targets over 2,048 tokens enter a larger fixed-reserve profile queue and are never truncated. A later larger-reserve profile must be selected uniformly without target access.

Each shard has fresh outputs and a terminal binding input/output/log hashes, exit status, CPU, context size, reserve, and elapsed time. Two serial lanes own cores 0 and 2. Any unexpected integration exception leaves a failed shard receipt and cannot become a data hold. Expected invalid R/source cases require an upstream evidence-backed repair classification; an ambiguous runtime `source_import_inventory_invalid` is never silently counted as an exclusion.

Actual TRAIN row `4fcd7ab2627cee15dc04e6cc` exercised missing and explicitly failing R helpers; both exited nonzero with no hold row. A missing tokenizer bridge also exited nonzero with no hold row. The final wrapper passed the same 40-row production-working-directory smoke: 40 supported, zero holds or infrastructure errors. The policy test covers exact 10,948 closure, 16K/32K fallback, 64K/128K queueing, and terminal mutation rejection.

The earlier 115 provider/policy holds from the frozen 4,435 set remain queued for a separate retrospective audit because the old precedence could have masked infrastructure failures as `runtime_dependency_missing_reviewed_name`. Their supported rows and historical receipts remain unchanged.

`commands.json` contains the exact phased commands. This packet prepares review-only candidates and does not admit rows to training or edit production. Root review is required before the long render.
