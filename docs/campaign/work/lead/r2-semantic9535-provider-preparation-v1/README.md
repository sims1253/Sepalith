# Semantic 9,535 provider preparation

This frozen preparation converts only the 9,535 reviewed semantic-supported
rows from shards 27–40 into target-free prediction inputs. The 14,795 semantic
evidence holds and 4,423 provenance rows outside the semantic queue remain
explicit upstream accounting.

The provider is the exact 24-file closure reviewed for the 10,948-row run. It
uses a fixed 2,048-token target reserve, attempts 16K first, and reruns only
policy holds at 32K. It never truncates a target. Longer context-only rows stay
queued for 64K/128K review.

`run_shard.sh` bounds each shard to 1,800 seconds. `run_lane.sh` reads the
actual input manifest and assigns alternating non-empty shards to cores 8 and
10. Source, semantic, candidate, and provenance hashes are checked before a
provider input is published atomically.

Final training-row materialization is fail-closed until the ongoing 10,948-row
pipeline has a finalized output manifest. Root must create
`dedup-binding.root.json` from the template with exact manifest/token-row/
provenance paths and hashes. Only then does `materialize9535.py` deduplicate
against the current 20,191 rows plus the finalized 10,948-row result. No output
from this packet is training-admitted.
