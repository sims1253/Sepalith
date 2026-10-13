# C17. Release preparation (user-gated)

- **Where:** PC, notebook, Kaggle
- **Needs:** the user decides that post-training is finished and picks a
  candidate
- **Produces:** a release-qualified GGUF set and, only with the user's
  approval, the sealed final evaluation

This card is intentionally a stub. Start it only when the user asks. Expected
contents:

1. **Quantization.** Qualify the chosen candidate's GGUF quantizations (Q8_0,
   Q6_K, Q5_K_M, Q4_K_M) against bf16 on DEV250, and reuse the existing
   quant-comparison scripts in `scripts/packaging/`.
2. **Serving parity.** Check prompt rendering in the editor extensions
   against the PRM03 contract.
3. **Latency.** Measure on the notebook, on CPU and Vulkan, which is close to
   target-class hardware.
4. **Sealed final evaluation.** Open the sealed final set once, only after the
   user freezes weights and harness. Follow the campaign's existing final
   evaluator and admission guards, which are in the snapshot under
   `docs/campaign/work/final-*`. They are excluded from the public snapshot,
   so read them from the planning worktree.
