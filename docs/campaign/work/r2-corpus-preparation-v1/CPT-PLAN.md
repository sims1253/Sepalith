No separate raw-R continued-pretraining stage was run in the selected campaign branch. The selected SFT1000 run trained on PRM03 rendered prompts and targets with full-text loss. Its first 16,000 sequential draws contained 18,492,737 input tokens: 17,158,062 prompt tokens and 1,334,675 target-plus-EOS tokens. These counts include the task renderer and repeated draws; they are not counts of distinct raw-R tokens.

Those draws exposed 11,475 of the 11,764 admitted rows and all 2,012 package identities in the original pool. Only 289 admitted rows remained unseen, all from finish_block and all from already exposed package identities. The corrected pool retains those 289 rows. Repeating the existing schedule alone would therefore add little source diversity.

The frozen SFT expansion roster contains 6,690 never-attempted TRAIN metadata candidates in 2,311 groups, including 1,239 groups absent from SFT1000. Counts are 4,096 finish_block, 1,024 no_op, 1,024 roxygen, 310 formatting, 172 pipe, and 64 rename candidates. They still require the existing converters, strict finish-boundary reconstruction, tokenizer checks and admission. Root owns that materialization. The 373,984 global TRAIN candidate rows and 46 family labels are a candidate inventory, not an admitted training dataset.

The raw corpus source is `/mnt/h/sepalith/normalized/<package>/<version>/<package>/R/*.R`. Of 14,202 package directories, 10,719 map to DAT02 TRAIN groups. The registry excludes 165 DEV, 187 final-candidate, 600 TU3, 696 historical-evaluation, 185 source-only SFT-v7 quarantine, one other source-only exclusion, and 1,649 unmapped package directories before traversing their contents. No excluded package payload was opened.

The partition in `cpt-train-group-partition.json` was fixed before raw tokenization. SHA256 of `DAT10-CPT-validation-v1`, NUL, and group ID modulo 20 assigns 556 TRAIN groups to CPT validation and 10,163 to CPT training. This is an additional diagnostic holdout within the existing TRAIN split. It does not assert that the original Midtrain pretraining corpus never contained these packages. Campaign DEV and final remain separate.

`profile-shard-v2-2k/manifest.json` pins a frozen profiling dataset. It contains 2,526 TRAIN chunks with 3,867,176 supervised tokens and 499 CPT-validation chunks with 661,360 supervised tokens. Those cover 1,055 and 294 exact source documents from 745 and 50 packages, respectively. The draw schedule is one seeded permutation without replacement. The profile's initial metadata snapshot was partial and alphabetically enumerated; hashing the package selection does not make that snapshot representative of every normalized package.

Every 2K row is BOS, source payload, EOS. A continuation includes one previous source token as masked context. The first BOS, overlap token, and nonterminal EOS have labels -100. Every new source token and exactly one final EOS per document contribute to loss. No source token is silently truncated. The tokenizer roundtrip reconstructs each exact UTF-8 source byte hash. The collator must preserve the supplied labels, pad attention with zero, and pad labels with -100. Actual EOS must remain supervised even though EOS and PAD share ID 1.

The broader fixed shard uses the same chunk rule, a frozen 23,380-file metadata snapshot, package-hash ordering and round-robin file selection. It contains 38,905,169 code tokens plus 14,098 supervised document EOS tokens in 27,430 rows. Its 14,098 unique source files contain 121,031,790 bytes and span 1,296 groups. The largest group contributes 377,673 code tokens, or 0.9708%. It overlaps 1,054 profiling TRAIN documents; 35,493,952 code tokens come from documents outside that profiling TRAIN set. Its separate draw schedule requires an explicit dataset transition if root continues a profiling checkpoint.

The broader selection reads at most 120 MiB of raw source and caps each group at 400,000 code tokens. The 294 profiling-validation document hashes are reserved while building the broader TRAIN shard. The completed pass excluded 16 reserved-validation duplicates, 25 duplicates within TRAIN, 186 files with unrecognized or custom license expressions, and four files that exceeded the per-group token cap. Global exact-byte duplicates are retained once. SHA-1, SHA-256 and Git-blob SHA-1 are checked against available non-TRAIN parent metadata. This is not a complete cross-package near-duplicate audit of held-out source files; no such complete held-out file-hash inventory was available without additional access.

The smallest branch that answers the user's question starts from the verified fresh Midtrain parent, profiles raw-R CPT on the fixed 2K shard, then admits a bounded CPT stage against the larger fixed shard. Keep the selected SFT1000/current release candidate protected. Root initially caps CPT at four hours and uses measured throughput, finite loss, validation NLL, complete checkpoint state and artifact identity to choose the token budget. The 12-hour target and conditional 18-hour envelope remain root scheduling decisions. Dataset contents must stay fixed during each admitted stage.

After CPT, use corrected and expanded PRM03 task SFT with exact complete targets and target-only labels, including the intended terminal EOS. Exclude reserved CPT-validation groups from this new SFT branch when their DAT02 identities can be joined. `sft-cpt-validation-exclusion-options.json` provides the current metadata exclusions and unresolved authored identities. The earlier 25-step target-only result describes that small corrective continuation; it does not establish that broader SFT or raw-R CPT is ineffective. Run RL only if the resulting SFT candidate passes the root's DEV and contract gates and time remains for evaluation and delivery.

Operator preparation commands, from this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false \
  /home/m0hawk/Documents/Sepalith/.venv/bin/python -m unittest test_raw_cpt_v2 -v

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false \
  /home/m0hawk/Documents/Sepalith/.venv/bin/python validate_cpt_shard.py \
  --shard profile-shard-v2-2k --documents profile-shard-v1/documents.jsonl \
  --output profile-revalidation.json
```

These commands validate CPU artifacts. They do not authorize a GPU launch or create a training admission. Root must verify the actual trainer consumes the supplied labels and draw order, prove its fused-loss denominator before the first update, and retain complete model, optimizer, scheduler, RNG and sampler state for any continuation. The worker did not load or hash model weights, launch training, access campaign final content, or alter accepted inputs.
