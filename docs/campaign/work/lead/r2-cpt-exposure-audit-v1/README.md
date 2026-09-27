# Selected CPT lineage: actual R-corpus exposure

The selected lineage did **not** train on the whole prepared R corpus.

Recompute the counts independently with:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 docs/campaign/work/lead/r2-cpt-exposure-audit-v1/recompute_exposure.py --output /tmp/cpt-exposure-recomputed.json
```

The script prints or writes all paths it used, validates the actual schedule-file hashes against the recipes and checkpoint sampler states, streams the row files once, and checks their totals against the shard manifests. It intentionally trusts the recorded large-row-file hashes instead of hashing the 449 MB and 285 MB JSONL files again.

The broad CPT checkpoint at step 750 persisted a sampler cursor of 12,000 draws. Those draws are 12,000 distinct rows from the 27,430-row broad pool: 17,062,230 of 38,973,361 input tokens (43.78%). They touch 7,835 of 14,098 documents, but only 4,633 documents are complete under the materialized-shard definition. Broad-a step 250 is the first 4,000 rows of this same cursor; broad-b resumed it, so those rows are counted once.

Global CPT step 250 then started a fresh LoRA on the merged broad750 parent. Its 4,000 draws are distinct rows from a separate 18,991-row, 750-package shard: 5,255,449 input tokens. Exact row, document, and package overlap with the broad shard is zero. It adds novel material but does not fill the 15,430 unseen broad rows.

Across both stages, the selected weights received gradients from 16,000 distinct rows, 11,155 exact document identities, 1,953 packages, 22,317,679 input tokens, and 22,286,822 loss-bearing tokens. Against the union of the two materialized pools, that is 16,000 of 46,421 rows (34.47%) and 22,317,679 of 63,704,293 input tokens (35.03%). No selected-prefix row was replayed.

These denominators still do not describe the entire R corpus. The split registry has 10,163 TRAIN groups and 556 reserved CPT-validation groups. The two materialized shards contain 2,046 disjoint packages; the selected prefixes touch 1,953, or 19.22% of registered TRAIN groups. Materialization used partial inventories and applied licensing, exact-deduplication, file-size, validation, and package/group token-cap exclusions. Document completeness means all chunks present in the corresponding post-exclusion shard, not all upstream raw files.

The 499-row CPT validation set was scored without gradients and is excluded from training exposure. Token totals are unpadded stored tokens; batch padding is masked and carries no corpus content. Actual exposure is reconstructed from full-checkpoint sampler cursors and immutable schedule prefixes, corroborated by trainer global steps and accepted checkpoint receipts. It is not a per-example device trace, and it cannot say what the upstream MiniCPM parent saw before this campaign.

If more CPT is funded, preserve the incumbent and create a separate branch from the merged global250 parent. Use a fresh LoRA and a new data identity containing all 30,421 unconsumed rows from both frozen shards. One pass plus 11 explicitly named replay rows gives 30,432 draws, or 1,902 updates at effective batch 16. The unique portion contains 41,386,614 input tokens. Observed CPT guard rates imply about 2.40–2.61 hours; reserve roughly three hours before merge and matched evaluation.

Finish and evaluate the active expanded-SFT recovery first. The same CPT window is roughly 1,364–1,479 expanded-SFT updates at the observed 150-step attempt rate, and task editing/no-op quality is the delivery target. If the CPT branch is later run, merge it and reuse the exact expanded-SFT schedule and native DEV route so the additional CPT has a matched comparison. Do not treat the larger metadata inventory as clean training data without a new licensing, split, deduplication, size, parsing, and token-cap audit.
