# Ideas backlog

Closed by evidence this session (see docs/validation/2026-10-18-specdec-autoresearch.md):
- [CLOSED] Region-copy drafter (brief dir. 2): oracle dominated by ngram-simple
  (1.11x vs 2.3x ceiling); C++ patch not justified.
- [CLOSED] DSpark released drafter (dir. 3): same verify-batch parity blocker;
  CPU draft-model forward cost breaks even only above ~1.65 tok/step sustained;
  acceptance vs moved weights expected low. Proposal-level only (report §10).
- [CLOSED] Trained drafters/MTP/EAGLE (dir. 4): moot until llama.cpp determinism
  fix; training proposal included in report §10 but not queued.

Still open / future:
- Multi-type fallback chain (`--spec-type ngram-simple,ngram-map-k`): first
  non-empty draft wins; could lift draft frequency where simple misses.
  Expected wall gain small (~0.05x) — only worth a run if the gate reopens.
- Quiet-host re-measurement of the full 221 gate table (repetition #2) —
  running as of PR filing; append to report/JSON when complete.
- llama.cpp upgrade evaluation (user decision): newer builds may change both
  the CPU verify-cost curve (B>=4 GEMM path) and batched-verify determinism.
  If a future build makes verification bitwise-lossless, re-run this rig
  unchanged — all tooling is config-driven.
- If spec ships despite near-tie drift: run the C09-style quality gate
  (DEV250 exact-match/no-fp) with speculation enabled (report §10.3).
- [CLOSED run #14] -fa on: inert on b10453 CPU (parity 1.0000, speed
  0.99x) — not a lever for parity or verify cost.
