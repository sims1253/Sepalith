# No-op 4,106 provider preparation

This packet binds the fresh v3 full-document reconstruction to the exact frozen provider-v2 renderer, tokenizer bridge, namespace helper, and source manifest. The plan enumerates all 41 source shards, including empty shards, verifies each input/sidecar/hold hash and row count, and validates the v3 duplicate-geometry evidence without dropping any row.

`build_plan.py` assigns every nonempty shard to one of two balanced lanes on cores 4 and 6. `bind_commands.py` turns an immutable plan into two exact hash-bound lane commands. Each lane invokes the reviewed `run_shard.sh`, which supplies `taskset`, `nice`, `ionice`, a 1,800-second per-shard timeout, the pinned tokenizer, and the pinned R helper. Lane execution remains root-authorized only.

The policy first renders all 4,106 reconstructed rows at 16K with 2,048 tokens reserved. Only explicit complete `hold` results become target-free 32K inputs. The finalizer verifies both stage terminals, hashes, statuses, and ID closure, retaining complete contexts or explicit post-32K holds. It reports the exact denominator as 121 upstream provenance holds plus 4,106 provider rows. Fixed no-op targets are joined only after context selection; no target or gold is used to choose context and no target is truncated.

The v2 full reconstruction failed before publication. Therefore this packet is prepared against the absent v3 terminal output and no smoke or provider execution was performed.
