# Ideas backlog — session complete

All experimental paths are closed by measurement or exact simulation
(27 logged experiments; see docs/validation/2026-10-18-specdec-autoresearch.md
and .auto/log.jsonl):

- Config space: every n-gram mode × depth × lookup grid; multi-type
  stacking; -fa (inert); m-refinement closed by per-position acceptance.
- Patch space: cascade volume-raising (measured wall tie), frequency-voting
  precision-raising (ceiling loss — recency beats frequency), region-copy
  (dominated), region+cascade (no add). Patch kept as negative-result
  evidence in experiments/serving/specdec/patches/.
- Matrix: 6/6 tier×thread cells; flow continuum (warm/mid-abort/cold);
  parity mechanism, quality-neutrality, no-op safety.

Remaining items are USER DECISIONS (report §8/§10), not experiments:

1. Ship `ngram-simple n=2 m=4` accepting near-tie drift of the class the
   warm-up flow already ships (0.87–0.88 vs 0.851 cache drift; Q8-vs-bf16
   0.668 precedent; flips cost zero ground-truth matches and never touch
   no-ops) — or hold for an upstream determinism fix.
2. llama.cpp upgrade / determinism proposal (§10.1): force the greedy
   verify path through the same matmul path as single-token decode.
3. If shipped: re-run the C09-style quality gate with speculation enabled
   (§10.3).

Reopen triggers (what would justify new experiments):
- A newer llama.cpp build lands (re-run the rig unchanged — it is
  config-driven; check whether verify determinism or the B≥4 GEMM cost
  cliff changed).
- The quant mission ships a different tier (matrix rows are cheap to add).
- A trained drafter becomes available AND the determinism fix lands
  (draft-model directions are parity-blocked and CPU-cost-blocked until then).
