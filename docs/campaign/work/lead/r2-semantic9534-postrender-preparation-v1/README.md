# Semantic9534 post-render preparation

This packet prepares the review-only merge after the active Semantic9534 16K
retry. It does not launch a provider, read model or optimizer state, join
targets, or admit materialized rows to training.

The immutable provider input is the Semantic9535 input manifest at
`/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-inputs-v1/manifest.json`,
SHA-256
`db1dd374d1d029695c9b8daef59151272d0db0a912276f8073b24b0df4f8e9a0`. It has
9,534 provider rows in shards 0027 through 0040. One geometry preparation hold
from the 9,535 upstream supported denominator remains explicitly accounted for,
so the review denominator is 9,535 while provider and context-policy rows stay
9,534. The preserved failed 16K root remains
`.../render-16k-v1`; the active retry output is the separate
`.../render-16k-retry-rich-v1` root.

Run the commands in this order after the root confirms that all 14 active retry
terminals are complete:

1. `commands/verify_render16.sh` records every terminal SHA, output SHA, row
   count, and row-ID SHA. It rejects missing, partial, failed, infrastructure,
   changed-input, and changed-output records.
2. `commands/prepare_fallback_9534.sh` writes only the 16K policy holds to the
   fresh `render-32k-semantic9534-postrender-v1/inputs` root. It rejects a
   failed or partial 16K shard; such a shard cannot be turned into a policy
   hold.
3. Root may then run `commands/run_fallback32_lanes.sh` with that fallback
   input root. It uses the pinned parse-unavailable rich provider closure at
   32K with the same 2,048-token reserve. Empty fallback shards are not
   launched and must not acquire synthetic terminals.
4. `commands/finalize_context_policy_9534.sh` selects supported 16K contexts,
   then supported 32K fallbacks, and records 64K/128K context-only follow-up
   inputs. Selection remains target-free and preserves all 9,534 provider IDs.
5. After root creates a bound copy of `dedup-binding.template.json`,
   `commands/materialize_review_only_9534.sh` can produce a review-only
   candidate directory. Its binding requires 20,191 current rows plus the
   finalized 10,682 Semantic10948 rows (30,873 combined), with zero future
   noop rows. Target lines are joined only after context selection.

The 16K retry source is the already reviewed parse-unavailable closure at
`r2-provider-parse-unavailable-preparation-v1/semantic9534-retry`, source
manifest SHA
`9725d968382c98fe5313985f5407fd6aa01a3e34063393b88c3bfb45193572e8`. The
materializer is a pinned copy of the reviewed Semantic10948-compatible
materializer and retains strict source reapplication, tokenizer, target
reserve, contradiction, and duplicate checks.

The root-bound dedup file is intentionally absent from this packet. The
template is fail-closed and has `training_admission: false`. No materialization
admission or corpus mutation is implied by any output from this packet.
