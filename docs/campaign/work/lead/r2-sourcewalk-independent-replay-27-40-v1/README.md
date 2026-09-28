# DAT-10 sourcewalk replay shards 27–40

This packet prepares a bounded continuation of the already indexed sourcewalk
replay. It processes only missing shards 27–40 and leaves committed shards 0–26
untouched. The reviewed `full_replay.py` driver remains immutable; the wrapper
binds its hash, the complete 41-shard index, and the parser dependency overlay.

The current intake has 28,753 indexed candidate rows in shards 27–40 and no
committed provenance receipts in that range. The index itself is complete and
hash-bound, but the replay output is not a terminal all-data corpus until these
receipts and a later merge are complete. This packet does not admit data for
training.

Root launch command:

```text
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/parser-dependencies-root-v1 \
  timeout --signal=TERM --kill-after=30s 7200 \
  bash docs/campaign/work/lead/r2-sourcewalk-independent-replay-27-40-v1/run_27_40.sh
```

The wrapper owns the existing output lock, verifies the index identity, and
derives the exact missing list from valid receipts. It invokes the frozen
driver with `replay --shards 27,...,40`; it never calls the all-shards
`run_full.sh` path. A later `full_replay.py merge` is a separate root-reviewed
step after all 41 shard receipts pass their source and output bindings.
