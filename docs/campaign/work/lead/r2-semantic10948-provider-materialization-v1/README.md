# Semantic 10,948 provider materialization (review only)

This packet applies the byte-pinned `r2-semantic-provider-integration-v1` prediction-time provider to the 10,948 target-free inputs in `Semantic10952-preparation-v1`. The first policy profile uses a 16K context and a fixed 2,048-token generation reserve. Rows unsupported there are rerun at 32K with the same fixed reserve. Neither selector reads target or gold content.

A context-only hold after 32K is retained in an exact 64K/128K follow-up queue. A target that exceeds the fixed 2,048-token reserve is retained in a larger-fixed-reserve queue after target join; it is never truncated or counted as an exclusion. Larger fixed-reserve rerenders must remain target-free and cover the whole eligible pool for that profile.

Unexpected parser, tokenizer, source-identity, or integration failures terminate the shard. They cannot become ordinary semantic holds. Each shard uses a fresh output and terminal receipt binding the input, output, log, CPU, context, reserve, status, and elapsed time. The two lanes own only CPU cores 0 and 2 and run one shard at a time.

The final materializer joins unchanged complete TRAIN targets only after context selection, re-tokenizes the full prompt/target sequence, checks the strict token protocol and full-source reapplication, and deduplicates against the reviewed 15,006 + 616 + 133 + 4,435 + 1 pools. Every one of the 10,948 prepared IDs receives a candidate, duplicate/contradiction, provider/policy hold, context follow-up, or larger-reserve decision.

The final provider wrapper and exact production working directory passed a 40-row health smoke. A corrupted pre-edit identity returned exit 1 without emitting a data hold. `test_policy_pipeline.py` covers all 10,948 synthetic IDs through 16K/32K selection, 64K/128K queueing, exact closure, and terminal hash mutation rejection.

`commands.json` contains the reviewable serial phases. The long render requires root review of this frozen packet. Outputs remain preparation artifacts and do not admit any row to training.
