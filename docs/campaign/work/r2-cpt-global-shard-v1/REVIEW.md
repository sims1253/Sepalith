This separate raw-R CPT candidate shard contains 24,685,393 code tokens plus 11,434 supervised document-end EOS tokens in 18,991 rows. Its 11,434 unique source documents span 750 new groups and package initials A–Z. Materialization stopped after 720.23 seconds. The unchanged validator passed every saved row and source reconstruction in 13.52 seconds. It does not change the active 39M-token stage or authorize a training-data transition. Exact hashes and counts are in `shard/manifest.json`, `source-reconstruction-validation.json`, `global-shard-audit.json`, and the receipt.

The starting order contains all 8,867 remaining eligible CPT-TRAIN package groups, sorted by SHA256 of `DAT10-next-global-v1`, NUL, and group ID. It excludes all 1,296 groups in the previous broader shard and all 556 reserved CPT-validation groups. The builder checks the complete order against the pinned DAT02 registry before traversing any listed package. Within each package it orders R files by a separate fixed path hash. The completed prefix gives a seeded spread across remaining package names; it does not establish whole-corpus coverage.

The private wrapper imports the exact accepted `inventory_raw_train_v2.package` and `raw_cpt_broader` source. It preserves DESCRIPTION/license checks, exact source-stat checks, UTF-8 validation, tokenizer identity and roundtrip, SHA256/SHA1/Git-blob guards, and the existing 2K chunk/mask function. It adds exclusion sets for all 14,098 prior broader-shard document hashes and the 294 profiling-validation document hashes. Each group has a 100,000-code-token cap. A file exceeding the remaining group budget is excluded whole; no retained file loses a tail.

Every retained document is emitted completely. BOS, the one prior-code-token overlap and nonterminal EOS are masked. Every new code token and exactly one actual document-end EOS are supervised. The unchanged validator reconstructs each saved document through the pinned tokenizer and checks its exact source byte hash, contiguous offsets, labels, terminal marker and loss denominator. The supplementary audit checks the seeded package prefix, prior-shard and validation exclusions, and group caps.

Materialization uses two CPU cores, at most two metadata futures and two raw-file futures. It streams token rows and provenance to disk. It does not retain the corpus or all token rows in memory. The 720-second materialization budget is checked between packages and after a complete accepted document; in-flight filesystem calls and a run of excluded files can delay exit. The worker timebox includes source preparation and saved-row validation. Runtime progress alone is not a completed-shard receipt.

CPU commands from PLAN:

```sh
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B docs/campaign/work/r2-cpt-global-shard-v1/test_global_shard.py
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B docs/campaign/work/r2-cpt-global-shard-v1/build_global_shard.py --max-seconds 720 --target-code-tokens 30000000 --group-code-token-cap 100000
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B docs/campaign/work/r2-corpus-preparation-v1/validate_cpt_shard.py --shard docs/campaign/work/r2-cpt-global-shard-v1/shard --documents docs/campaign/work/r2-cpt-global-shard-v1/shard/documents.jsonl --output docs/campaign/work/r2-cpt-global-shard-v1/source-reconstruction-validation.json
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B docs/campaign/work/r2-cpt-global-shard-v1/audit_global_metadata.py
```

The builder requires a fresh `shard` directory. To replay it, copy the wrapper into a new root-owned sibling directory so the existing frozen input paths still resolve. Replay validators into fresh result paths or a private copy to preserve the receipt's bytes.

Limits: exact-document deduplication does not mean token sequences or near-duplicate code are globally unique. The available non-TRAIN parent metadata and 294 reserved document hashes are not a complete heldout source-file inventory. No heldout content was opened to extend that inventory. Raw source syntax was not evaluated or reparsed; no R code, model, GPU, cloud job or training process was run. Root must review the completed shard and explicitly bind any later data transition, draw order, checkpoint identity and budget.
