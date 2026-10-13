# DAT-10 pre-cap CPT recovery

This packet prepares a CPU-only, resumable source walk for the exact
1,999-record `whole_document_exceeds_remaining_group_cap` frontier from the
old global CPT shard.  The frontier is confined to 79 prefix groups (seeded
indices 1 through 765) and has zero exact source-path hits in the terminal
177,190-document provenance.  The old per-group/file/package token caps are
not applied here.

The worker validates the global split and CPT partition before source reads,
rehashes each stable DESCRIPTION and source file, checks the recognized
license evidence, and compares source SHA-256/SHA-1/Git-blob identities with
the terminal provenance, CPT-validation profile reservations, and protected
parent registry.  Empty sources are named exclusions.  Changed, unreadable,
non-UTF-8, NUL-containing, and tokenizer-repair cases go to a named repair
queue.  Files larger than 4 MiB are retained and measured; the worker has no
source-size or token-count cap and never truncates a target.

The materializer writes one atomic directory per group under the E: output.
`progress.json` and each group `manifest.json` make a stopped run resumable.
The resulting packet remains a candidate: terminal global dedup and root
training admission are still required.

## Pinned inputs

- Frontier: `docs/campaign/work/lead/r2-final-union-terminal-accounting-v1/global-cap-exclusion-frontier.jsonl`, 1,999 rows, SHA-256 `acf86ba87bf833b6bd69cf732f2cb4a7d5356bcbd9e38ba2676b385af2b37560`.
- Global split: `/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json`, SHA-256 `c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09`.
- CPT partition: `docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json`, SHA-256 `6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06`.
- Protected parent hashes: `docs/campaign/work/r2-corpus-preparation-v1/known-nontrain-parent-hashes.json`, SHA-256 `7210a342559278c2bd82f3514da899d6cfbbb89104c4d9c8af0df61fbb8133b0`.
- CPT-validation profile metadata: `docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v1/documents.jsonl`, SHA-256 `674d3bf6e2da08b53a0d0fa6d7ae1977c5bc6940938ebe2aaf6c7c1643ff6d68`.
- Terminal document provenance: `/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1/document-provenance.jsonl`, 177,190 documents, SHA-256 `a731974ee8581b9fa571aae9672f6b933743779c24f4253843bff268065c305f`.
- Reviewed raw chunk contract: `docs/campaign/work/r2-corpus-preparation-v1/raw_cpt_broader.py`, SHA-256 `84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab`.
- Tokenizer: `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json`, SHA-256 `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`.

The metadata-only supplemental audit also found four older
`package_code_token_cap` records (2,661,897 current source bytes, two
packages) and fourteen `R_file_empty_or_over_4MiB` records.  The fourteen are
currently zero-byte sources with no payload.  Neither category is silently
merged into this 1,999-row input: the four require their own source rehash and
the fourteen remain named zero-byte exclusions.

## Root-owned commands

Run the metadata and stat preflight first:

```sh
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=''
ionice -c3 nice -n 10 taskset -c 0,2 python3 -B \
  "$PLAN/docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v1/materialize_prefix_cap_recovery.py" \
  --preflight-only
```

After root review, run the full source walk into the fresh E: directory:

```sh
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
OUTPUT=/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-cap-recovery-v1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=''
ionice -c3 nice -n 10 taskset -c 0,2 python3 -B \
  "$PLAN/docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v1/materialize_prefix_cap_recovery.py" \
  --frontier "$PLAN/docs/campaign/work/lead/r2-final-union-terminal-accounting-v1/global-cap-exclusion-frontier.jsonl" \
  --output "$OUTPUT"
```

Observe with `cat "$OUTPUT/progress.json"` and inspect group receipts before
root integrates their `cpt_train.jsonl` and provenance into the next terminal
union.  The recovery rows start at a new dataset/global offset selected by
root; this packet does not alter the current cache or stage schedule.

