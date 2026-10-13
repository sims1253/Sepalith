# DAT-10 semantic queue: committed shards 6+

This packet reuses the byte-pinned, hash-corrected source from `r2-semantic-queue-root-launch-v3`. It owns one bounded CPU review queue over committed replay shards **6–11**, because shards 0–5 were already processed and shards 12–26 remain an explicit next frontier. At intake, replay shards 6–26 had complete provenance receipts; shards 27–40 had index entries but no provenance receipt. The replay controller was live and was not restarted or interrupted.

`run.py` performs a fresh-source and selected-shard closure check, refuses a selected shard that is open by the observed replay PIDs, and launches the reviewed worker with two low-priority CPU workers and a 2,100-second timeout. It writes `launch.json`, `terminal.json`, and `status.json` in this packet; bulk output is on E. It never passes a terminal provenance manifest, so the result remains review-only and cannot claim global closure.

The queue output is a candidate semantic review artifact. It does not admit data or launch training. Root must independently check source/license/global split, exact prompt/target dedup, context/geometry, and any holds before admission.
