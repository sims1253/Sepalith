# Checkpoint 706 review preparation

This dormant packet verifies the exact root-owned checkpoint450 cadence transition and continuation through checkpoint706 after it is terminal. It requires optimizer updates 451–706, 4,096 draws at positions 6,144–10,239, cursor 10,240, and 381 finite nonzero gradient tensors at every update. It also verifies the admitted identity transition from `46a5…` to `6ac45…`, the complete native and durable full-state inventories, the 85 saved FP32 tensors, and the pinned tokenizer contract.

Checkpoint706 is an internal evaluation step in the frozen recipe. The verifier requires `evaluations/step-706.json`, binds its exact ordered 499-row anchor fixture, and recomputes NLL from per-row loss sums and 661,360 loss tokens. It then prepares a separate root-admitted matched evaluation over the exact 2K, 8K, and 16K fixtures. The external review repeats exact row-ID, denominator, and NLL recomputation checks.

The active third attempt is bound by its controller, guard, attestation, and trainer launch records. Two earlier startup failures occurred before the first optimizer update. Accounting uses the complete ordered optimizer and loss-event ranges 451–706; startup-only events do not count as training.

Nothing in this packet launches CUDA, authorizes evaluation, or promotes checkpoint706. Full payload hashing runs only when root manually invokes `prepare.py` after terminal success.
