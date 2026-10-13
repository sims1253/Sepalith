# CPT final union integration v2

This CPU-only preparation preserves v1 union semantics and does not run the full corpus or admit training data.

The final path now scans the main `groups/` directory once with `os.scandir`, requires exactly one directory for every seeded index 775 through 8866, rejects missing, ambiguous, malformed, and out-of-range seeded directories, and validates all 8,092 small receipts before opening the two large base streams.

The two base streams are bound to their reviewed manifests, exact payload SHA-256, byte count, and row count. Payload hashes are computed from the same bytes read during the eventual union run; preparation did not reread the 429 MiB and 274 MiB payloads.

Profile validation reservations are cross-bound to the reviewed profile manifest and exact documents metadata. The builder requires 294 unique complete document identities and the manifest's 370 validation chunks. A repeated `row_id` is excluded only when its complete canonical JSON content is byte-semantically identical; conflicting content under one ID aborts the union.

Root-only candidate command:

```bash
nice -n 10 ionice -c 3 python3 -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-final-union-integration-v2/build_final_union.py --mode final --output /mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v2
```
