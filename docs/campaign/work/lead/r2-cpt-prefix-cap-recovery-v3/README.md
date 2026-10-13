# DAT-10 pre-cap CPT recovery v3

This packet is a fresh, resumable CPU-only source walk for 2,003 omitted CPT
records: the exact 1,999-record global prefix-cap frontier plus the four
broader `package_code_token_cap` records.  The two scopes are retained in the
frozen combined frontier and are checked against the global TRAIN split and
CPT partition independently.  The frozen v1 and v2 packets are unchanged.
V3 fixes resume accounting for valid all-excluded, duplicate-only, and
repair-only groups by writing explicit zero-valued document/row/token
counters.

The worker validates every input pin before reading source payload.  It
rehashes each stable DESCRIPTION and source file, records actual license
evidence, checks source SHA-256/SHA-1/Git-blob identities against terminal
provenance, CPT-validation reservations, and protected parent hashes, and
deduplicates within the recovery stream.  There is no package, group, file,
source-size, or token-count cap.  Files larger than 4 MiB are retained and
measured.  Empty sources are named exclusions; changed, unreadable,
non-UTF-8, NUL-containing, and tokenizer-repair cases go to named repair
queues.  Targets are never truncated.

Each group commits atomically under the E: output.  Resume skips are fail
closed: group manifests, all four payload/sidecar hashes, builder/frontier/
guard pins, and every document/row/exclusion/repair identity must match the
frozen frontier before a group is skipped.  The result remains a candidate;
root must perform the final union dedup and training admission.

## Frozen input

- Combined frontier: `combined-frontier.jsonl`, 2,003 rows, 81 groups, SHA-256 `4708d69f53e04c48754e48095497eb1c281acac998dd099620b8fae293a1019d`.
- Scope counts: `prefix_global_cap=1,999`; `base_broader_package_cap=4`.
- Prefix input: `docs/campaign/work/lead/r2-final-union-terminal-accounting-v1/global-cap-exclusion-frontier.jsonl`, SHA-256 `acf86ba87bf833b6bd69cf732f2cb4a7d5356bcbd9e38ba2676b385af2b37560`.
- Broader-cap source hashes are frozen in `combined-frontier-manifest.json`: AutoPlots `PlotFunctions_NEW.R` SHA-256 `b7bdbb87c9e0269f1aee58d8cf9e6610197f96aadc9b2f3895dae8cf9e094f7b`; AICcmodavg `modavgShrink.R` `ecfedcc7067a769e36a5a0b3bf925d791c2a47307a86c957b9bc88a1b183206b`; `modavg.R` `1acf0ea83c1c71ae58b3a3b175060af75143bad7f000e536edd9f02c9dcadb67`; `aictab.R` `72916d9fa1149f4d3420f8fda94372e30d4fc1642f921e6f9a7726b2ed255c2d`.
- Global split: `/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json`, SHA-256 `c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09`.
- CPT partition: `docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json`, SHA-256 `6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06`.
- Protected parent hashes: `docs/campaign/work/r2-corpus-preparation-v1/known-nontrain-parent-hashes.json`, SHA-256 `7210a342559278c2bd82f3514da899d6cfbbb89104c4d9c8af0df61fbb8133b0`.
- CPT-validation profile metadata: `docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v1/documents.jsonl`, SHA-256 `674d3bf6e2da08b53a0d0fa6d7ae1977c5bc6940938ebe2aaf6c7c1643ff6d68`.
- Terminal document provenance: `/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1/document-provenance.jsonl`, 177,190 documents, SHA-256 `a731974ee8581b9fa571aae9672f6b933743779c24f4253843bff268065c305f`.
- Reviewed raw chunk contract: `docs/campaign/work/r2-corpus-preparation-v1/raw_cpt_broader.py`, SHA-256 `84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab`.
- Tokenizer: `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json`, SHA-256 `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`.

## Root-owned commands

Use the production isolated interpreter.  First run metadata/stat preflight:

```sh
CANON=/home/m0hawk/Documents/Sepalith
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=''
ionice -c3 nice -n 10 taskset -c 0,2 "$CANON/.venv-sft/bin/python" -B \
  "$PLAN/docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v3/materialize_prefix_cap_recovery.py" \
  --preflight-only
```

After root review, run the full source walk into the fresh E: directory:

```sh
CANON=/home/m0hawk/Documents/Sepalith
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
OUTPUT=/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-cap-recovery-v3
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=''
ionice -c3 nice -n 10 taskset -c 0,2 "$CANON/.venv-sft/bin/python" -B \
  "$PLAN/docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v3/materialize_prefix_cap_recovery.py" \
  --frontier "$PLAN/docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v3/combined-frontier.jsonl" \
  --output "$OUTPUT"
```

Observe with `cat "$OUTPUT/progress.json"`.  Inspect each group manifest,
candidate rows, exclusions, repair queue, source hash, DESCRIPTION hash, and
license evidence before root integrates accepted unique documents into a
later terminal union.  This packet does not alter the current cache or stage
schedule.
