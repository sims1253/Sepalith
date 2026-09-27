# RUN-09 Q4 paired DEV review

`compare_run09_q4.py` reviews the known native F16, Q8 and Q4 quality
receipts against the pinned 75-case DEV panel. It recomputes each case with
the existing `campaign_eval.classify` helper, checks stored denominators, and
compares Q4 with both baselines.

The review compares all prompt and request identity fields row by row:
HF/native prompt IDs, prompt hashes, tokenizer identity, and request settings.
Changed outputs are reported by case ID, token count, terminal status, cap
status, and token-ID SHA256; raw model text is not copied into the receipt.
Caps, no-op false positives, and correct no-ops are reported as separate
counts, with family breakdowns and runtime artifact hashes.

The bounded command used for the receipt was:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src:/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training \
python3 docs/campaign/work/quant-quality-review/q4-review/compare_run09_q4.py \
> docs/campaign/receipts/RUN-09-q4-native-dev-independent-review.json
```

The receipt's mechanical recommendation covers root review of the Q4 run and
does not promote Q4 quality.
