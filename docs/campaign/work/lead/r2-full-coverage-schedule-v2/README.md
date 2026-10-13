# Full-coverage seeded batch-mix schedule v2

This preparation replaces v1's longest-to-shortest curriculum with deterministic seeded mixing of whole edit chunks. Edits are grouped by their actual 512, 1,024, 2,048, or 4,096-token sequence bucket in chunks of 12. Complete chunks are seed-shuffled across the horizon. Cross-bucket leftovers form one complete chunk, and the only partial unique edit chunk remains last so that edit alignment replays cannot precede unique coverage.

Each batch has 12 edits and four no-ops. While unique no-ops remain, each no-op slot chooses the closest fitting row from the remaining unique pool. Replays start only after all 1,094 no-ops have appeared. Replay reasons remain explicit: `noop_ratio_length_alignment_replay` and `edit_batch_completion_alignment_replay`.

The concrete 15,008-row artifact uses 1,160 updates and 18,560 draws. It preserves all 13,914 unique edits and 1,094 unique no-ops with no exclusions or target truncation. The first 50 batches contain all four length buckets; v2 has 826 edit-bucket transitions compared with three in v1. Actual-max padding falls by 11,282 tokens and declared-bucket padding falls by 30,914 tokens. Different no-op replay choices increase scheduled input exposure by 63,682 tokens; unique-row denominators are unchanged.

This packet is preparation only. It binds no parent, recipe, optimizer, or launch. Rebuild a later admitted pool with a fresh output:

```text
python3 -B full_coverage_schedule.py --rows ABSOLUTE_TOKEN_ROWS --rows-sha256 EXACT_SHA256 --output FRESH_OUTPUT.json --effective-batch 16 --noop-per-batch 4 --max-sequence 4096 --seed 3407
```
