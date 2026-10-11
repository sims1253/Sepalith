> Review written on 2026-10-11 by the lead Claude Code session, from two
> research agents: one read the user's wiki (`/mnt/nas/wiki`), one compared
> current trl with this repository's runtime. Page paths and versions were
> checked by those agents; effects on the plan are this session's judgement.
> Nothing here changes a card or rule. Amendments marked **proposed** need
> the user's approval before the run they govern starts.

# Wiki and trl review before editing SFT

## Conclusion

Neither review changes editing SFT (C04, C07 to C10). The evidence bears on
post-SFT work: the GRPO optimizer, cheaper GRPO advantages, quantization
practice, and the fallback if the census shows too little signal.

## trl

**Recommendation: keep `trl==0.24.0` (with `unsloth==2026.8.18`) through C13.
Decide on an upgrade only at the C14 memory probe, in a separate venv.**

- Editing SFT does not use trl. `sft/full_weight_edit_sft.py` builds on
  `transformers.Trainer` with its own optimizer mixin, sampler, collator and
  callbacks. Moving it onto `SFTTrainer` would put unsloth's compiled trainer
  rewrite between us and the exact-resume and hash-binding contract, for no
  gain the collator does not already provide.
- GRPO uses `GRPOConfig`/`GRPOTrainer` 0.24.0 (`rl/campaign_rl_train.py`) and
  overrides the private `_generate_single_turn` and `_get_train_sampler`.
- Pins: unsloth 2026.8.18 caps `trl<=0.24.0`, `transformers<=5.5.0` and
  `datasets<4.4.0`. The newest unsloth (2026.10.x) caps `trl<=1.13.0`. trl
  1.x needs `datasets>=4.7.0`. The latest trl is 1.15.0.
- Upgrading would break the GRPO driver: `max_prompt_length` was removed from
  `GRPOConfig`, `_generate_single_turn` changed signature and return value,
  and `_get_per_token_logps_and_entropies` returns a 3-tuple from 1.7.
- 0.24.0 already has everything C13 and C15 specify: `loss_type="dapo"`,
  `epsilon_high`, `mask_truncated_completions`, `scale_rewards`,
  `importance_sampling_level="sequence"`, and DPO with precomputed reference
  log-probs.
- The one feature that would matter, fused LM-head log-prob scoring (trl
  1.14/1.15, large peak-VRAM savings), is outside unsloth's current cap.

**Corrections to existing notes:**
- C15's "or the equivalent in the driver" fallback for DAPO is unnecessary.
- [rl-objective-alternatives.md](rl-objective-alternatives.md) #3 infers
  from "BNPO" that the trl is old. That inference is wrong: BNPO is an
  explicit choice in the driver, and 0.24 already defaults to `dapo`.

## Wiki findings, ranked

Page paths are under `/mnt/nas/wiki/`.

1. **Muon-family optimizers can collapse under GRPO with binary rewards**
   (`entities/pion-spectral-failures.md`, `entities/muon-agentic-rl.md`,
   `entities/muon-not-special.md`; added 2026-09-27).
   - One paper reports Muon collapsing on Qwen3-1.7B/4B GRPO with binary math
     rewards, while AdamW and Pion train. Another finds Muon stable in
     agentic RL once update size is matched.
   - aurora_mix is in this family. SFT, RFT and DPO are cross-entropy-like
     and unaffected.
   - **Proposed (C14/C15):** add an 8-bit AdamW (or Pion) arm to the C14
     memory probe. Pre-register a collapse check over C15's first 50 updates
     (reward and `score250` both falling with rising grad norm) that triggers
     stop-for-user.
2. **F-GRPO focal weighting** (`entities/f-grpo.md`, 2026-09-14).
   - Scales each group's advantage by (1 − success rate)^0.5. With groups of
     8 it matched GRPO with groups of 32 on pass@256, without lowering pass@1
     (9 of 9 setups, including Llama-3.2-3B).
   - **Proposed (C15):** γ = 0.5. A one-line change.
3. **BASIS: census-informed baselines** (`entities/basis.md`, 2026-10-11).
   - Per-prompt values from reference-policy rollouts let GRPO run with 1 to
     2 rollouts per prompt. At 4B it matched group-of-8 with half the samples.
   - The C12 census (k = 8) is exactly that table. Evidence is one paper on
     math only.
   - **Proposed (optional, C15):** a single-rollout arm after the standard
     pilot, not instead of it.
4. **Strict verifier, no length penalty** (`entities/justrl.md`; the
   post-training recipe of our MiniCPM5 base).
   - A forgiving verifier and DAPO's overlong penalty both hurt.
   - C15 already matches this. It argues against making graded partial
     credit primary in C11 (rl-objective-alternatives #1).
   - Applied as guidance to C11: exact or canonical-equivalent correctness
     stays primary, and any partial credit is secondary and switchable.
5. **Quantization fidelity** (`concepts/statistically-lossless-quantization.md`,
   `entities/why-ptq-works.md`, `concepts/token-identity-gate.md`).
   - Benchmarks survive below 4 bits per parameter, but the token
     distribution only survives at 5 to 6 bits.
   - **Proposed (C13, C17):**
     - Generate RFT/DPO samples on the fp16 (C03) or bf16 path, never from Q4
       GGUF. This supersedes the Q4 suggestion in
       [post-sft-routes.md](post-sft-routes.md).
     - Gate Q4_K_M by its DEV250 exact-match delta against bf16 under
       llama.cpp, not by perplexity.
6. **TailSFT** (`entities/tailsft.md`, `concepts/coverage-principle.md`).
   - Dropping already-fit sequences during SFT improved pass@k and
     post-GRPO pass@1 in 15 of 18 OLMo-3 7B cases.
   - It conflicts with R1/R2's greedy exact-match selection, so it is not
     for arm A.
   - **Proposed (R4 step 4):** list it as the evidence-backed alternative to
     "a second offline iteration at T = 1.2" when the census mixed band M is
     below 15%.
7. **Checkpoint averaging** (`entities/kolibri.md`, `concepts/mergemix.md`).
   - An average of two SFT runs beat both, but merges rank candidates poorly.
   - R2/R3 do not define merge candidates. Not proposed now; possible later
     as one extra evaluated candidate.
8. **N-gram speculative decoding** (`concepts/speculative-decoding-mtp.md`).
   - Edit outputs mostly copy the editable region, which favours drafting
     without a draft model.
   - Worth a latency measurement in C17, with a greedy token-identity check.

Not pursued this cycle:
- Self-distillation (SDPO; weak below about 1.5B).
- On-policy distillation (no R-capable teacher shares our tokenizer).
- Critic-based methods (no memory budget for a critic).
