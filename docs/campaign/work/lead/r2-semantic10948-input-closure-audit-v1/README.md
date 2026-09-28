# Semantic 10,948 render-input closure audit

This independent audit checks the actual root-v4 render inputs against the pinned 10,952-row upstream terminal and every original shard 12–26 manifest. It permits only the exact upstream statuses `complete` and `complete_with_explicit_holds`, proves 10,948 prepared plus four disjoint explicit holds, verifies file hashes/bytes/physical row counts, and reconstructs every target-free render row exactly from its prediction and identity sidecar.

The test suite mutates terminal status, shard status/set, row counts, hold counts, IDs, target-shaped keys, and file bytes. All mutations fail. The audit is read-only and does not admit training data or own the running render lanes.
