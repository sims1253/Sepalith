# DAT-10 R2 task mixture: bounded primary subset

This directory contains a CPU-only, unadmitted candidate registry for the new target-only PRM03 SFT stage. The primary packet is `verified-corrected-short-token-rows.jsonl`: 8,115 corrected source-bound rows plus 411 independently audited short rows. Each row keeps the pinned `zeta2-prm03-v1` contract, manual BOS `0`, protocol EOS `1`, complete target tail, and the 4,096-token full-input cap; target length including protocol EOS is capped at 192.

`verified-corrected-short-provenance.jsonl` is a same-order source/group ledger. Corrected rows retain the upstream exact row-ID/DAT-05 join; short rows retain their audited `source_provenance.group_id`. The bounded verifier repeats exact DAT-02 `group_id` membership and explicit CPT partition checks, requiring `cpt_train` and excluding all 556 `cpt_validation` groups. `verified-corrected-short-exclusions.jsonl` preserves the 3,411 rejected corrected rows: 2,438 missing partition entries, 432 reserved groups, and 541 target-cap failures. No prompt duplicate or conflicting-target hash remains in the primary packet.

The packet has 8,526 rows across seven families, 7,432 `replace` and 1,094 `no_op` rows. It contains 313,372 target labels including protocol EOS and 10,505,668 total input tokens. No-op rows are 12.83% of rows and 4.19% of target labels. `verified-corrected-short-draw-manifest.json` is a complete metadata-only 500-step, effective-batch-16 proposal (8,000 draws), with a 10% no-op reservation, 25% family ceiling, and 20% naturally-long ceiling.

`verified-corrected-short-report.json`, `verified-corrected-short-source-manifest.json`, `verified-subset-checks.log`, and `verified-subset-loader-probe.log` bind the counts, hashes, source pins, sampler validation, and existing `campaign_sft_data.inspect_training_data` probe. The probe reads rows and schedule only; it does not load weights, CUDA, a server, or training. The candidate remains preparation-only and requires root rebind/admission.

The earlier 11,515-row extended registry remains available as optional evidence in `candidate-*`, `mixture-report.json`, `source-manifest.json`, and `proposed-draw-manifest.json`. Its structured/completion legacy materializations are deliberately deferred from the primary packet because their authored source joins were not independently rehashed in this bounded pass. `extended-receipt-v1.json` preserves the previous full-scope receipt.

Reproduce the primary freeze and CPU checks with:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 freeze_verified_subset.py
```
