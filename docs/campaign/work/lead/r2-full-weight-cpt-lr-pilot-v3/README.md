# Full-weight CPT lower-LR pilot v3

This fresh packet preserves the v2 scientific configuration: CPT250 parent, 384 distinct-package 8K rows, exact draw order, effective batch 16, 24 updates, two-update warmup, fixed 499-row/661,360-token holdout, Aurora-mix optimizer, and terminal-only full checkpoint. It permits only hidden/side LR pairs `3e-5/3e-6` and `1e-5/1e-6`.

The higher `1e-4/1e-5` arm is excluded after its fixed-panel mean causal NLL regressed from `0.9986833514584186` to `1.0450272190761625`. The unrun `3e-4/3e-5` arm remains rejected.

The v2 evaluator mislabeled `row_metrics.row_id` with document identity. This revision emits the normalized chunk row's `row_id` (falling back only to `id`) and emits `document_id` separately. Multiple chunks from one document therefore remain distinguishable. Package identity, loss numerator, loss-token denominator, mean NLL, row ordering, and aggregate NLL are unchanged.

Root should admit at most one lower-LR arm first. No CUDA work or model loading is authorized by these templates.
