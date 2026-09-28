# DAT-07 development selector preparation

This CPU-only preparation produced a bounded, non-training development panel from the exact 165-parent `dev_group` allowlist. The locked final and TU3 inputs were not opened. The accepted DAT-03 audit supplied 1,271 allowlisted dev metadata rows across ten sources; the category denominators are completion 651 (51 groups), docs 612 (127 groups), and no-op 8 (6 groups). No allowlisted audit rows supplied the structured rename, pipe, `na.rm`, or format families, so those cases are honest source-derived companions from the normalized versions of allowlisted dev parents.

The frozen candidate artifact contains 118 packets from 118 unique named packages and 118 unique global groups: 22 recoverable existing completions, 32 existing docs candidates, six existing authoritative no-ops, 26 derived state-based no-ops, and eight each of derived rename, pipe, `na.rm`, and format companions. The requested completion quota fell short by ten because 282 deterministic actual-row attempts failed the structured prefix/suffix boundary gate; no unsupported completion text was synthesized. Two oversized normalized source files were skipped during derivation. All 118 packets have a license record inherited from the allowlisted normalized parent, a staged source hash, a full-buffer splice/post-edit hash, exact source range, and an out-of-band `selection_source` object.

No-op semantics are explicit: the semantic replacement region equals `region_old`; the lead renderer must serialize `[NO_EDIT]` with the frozen `>>>>>>> UPDATED` terminal. An empty legacy no-op sentinel is never treated as deletion. Derived no-ops are limited to already-applied native `|>` or `na.rm = TRUE` states and remain chronology-review candidates.

Every packet is candidate-only (`verification=DAT-07-derived-dev-panel`, `split=dev_panel`). Existing rows remain metadata-only pending fresh validation. Derived line transformations pass exact structural and full-buffer splice checks, but the line validator is not a full R parser; rename and format candidates retain a lead source-review gate. All 32 docs packets carry the explicit `roxygen_formal_name_support_pending` gate using `experiments/training/campaign_support_audit.py:roxygen_formal_check`. Prompt collision, tokenizer length, full-source parser/support, and final panel admission remain lead-owned pending gates.

## Reproduction

```text
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 experiments/training/campaign_dev_panel.py --output /mnt/e/sepalith/campaign-20260915/data-work/DAT-07-dev-panel-candidates.json --staging /mnt/e/sepalith/campaign-20260915/data-work/DAT-07-dev-source-snapshots --seed 13
```

The source-only hash manifest is `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-dev-source-manifest.json` (sha256 `f60b1da57d2094c52a0dc3353e6e21d302767405704be6865791fcd62d02b437`). Focused model-free tests: `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m unittest discover -s experiments/training -p 'test_campaign_dev_panel.py' -q` (8 passed). The panel artifact sha256 is `9c6602f7a3d10aad255bce5df9d457ab6b23fe29b3de7f82383312183c32b66d`. The final/TU3 seal and raw input roots remain unchanged.

Observed run timestamps: started `2026-09-12T05:58:24.820897+00:00`, completed `2026-09-12T06:07:42.447109+00:00`; elapsed 557.626 seconds. Staging cleanup removed 60 stale files from earlier superseded local builds and retained exactly 118 source snapshots.
