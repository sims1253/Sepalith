# C14. GRPO gate: re-census and memory probe

- **Where:** Kaggle for the re-census; PC night for the memory probe
- **Needs:** C13 done, and R4 step 3 reached with `M >= 15%`
- **Produces:** a go or no-go for GRPO, and a fitted GRPO configuration
- **Effort:** one agent-day, plus one night

## Goal

Online GRPO only makes sense if enough prompts still give a learning signal and
full-weight GRPO fits on 32 GB. Nobody has measured the second condition. The
only earlier RL memory profile was a 25M-parameter LoRA run.

## Steps

1. Confirm `M` from the latest census (C13 step 7), per family. Build the GRPO
   prompt pool: prompts with 0.125 ≤ pass ≤ 0.875 at k = 8, plus 10% sampled
   from the all-correct prompts to guard against forgetting. Keep no-op
   prompts at their natural share within the band.
2. **Memory probe.** Use the campaign driver v2 from C01, with the length
   policy and reward v2.1 from C11. Measure peak memory separately for:
   - rollout: in-process generation with the KV cache, G = 4 and G = 8,
     completions up to 1,024 tokens;
   - scoring: log-probabilities chunked over sequence and vocabulary. The
     vocabulary is 130,560, and full fp32 logits for 32 × 1,024 tokens would
     take about 16 GiB, so chunking is mandatory;
   - backward: micro-batch 1.

   If G = 8 does not fit, try these tricks in order, and stop at the first
   configuration that fits with 1.5 GB to spare:
   1. smaller rollout micro-batches;
   2. moving optimizer state to host memory during generation;
   3. G = 6.

   Record time per update.
3. **Decide.** Go if a configuration fits and one update takes at most 4
   minutes, so that at least about 100 updates fit in a night. Otherwise
   no-go. Record it and stop-for-user with the options. No paid compute is
   allowed.
