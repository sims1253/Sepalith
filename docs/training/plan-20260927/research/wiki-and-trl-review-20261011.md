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

## Addendum 2026-10-12: quantization for shipping (C09, C17)

Sources: the user's wiki (`concepts/statistically-lossless-quantization.md`,
`entities/why-ptq-works.md`, `concepts/token-identity-gate.md`,
`entities/starling.md`), Unsloth's Dynamic 2.0/3.0 and QAT docs, HF Optimum,
NVIDIA Model Optimizer, AMD Ryzen AI and Quark, llama.cpp's quantizer source,
and arXiv 2610.09346 (OnlineQAT).

- NVIDIA Model Optimizer, AMD Quark/Ryzen AI and HF Optimum export to
  TensorRT, vLLM, ONNX or NPU formats. Only their workflow (calibrate,
  evaluate held out, protect sensitive layers) carries over to GGUF.
- The Unsloth library is open source (Apache-2.0). `save_pretrained_gguf`
  wraps llama.cpp convert/quantize and accepts an imatrix and tensor-type
  overrides. The per-model layer maps and calibration data behind Unsloth's
  own Dynamic uploads are not published, so we build our own imatrix and
  test a single override arm.
- OnlineQAT targets 2 to 3-bit weights on a 1.7B model, with no 4-bit
  results. Not needed at the tiers we ship.

**Proposed (C09, C17):**

1. Convert to a bf16 GGUF, not f16. Use it as both the quantization source
   and the reference.
2. Build an importance matrix from TRAIN prompts plus gold completions in
   the PRM03 render:
   - family-balanced, about 0.3 to 1M tokens;
   - context 2,048 to 4,096;
   - DEV75, DEV250 and the sealed set excluded by id and near-duplicate
     hash.
3. Produce these tiers:
   - Q8_0, Q6_K, Q5_K_M and Q4_K_M with the imatrix;
   - an uncalibrated Q4_K_M as a control;
   - IQ4_XS only if it passes the gate and is not slower on Vulkan.
4. Embedding and head precision:
   - keep `token_embd` at Q8_0 (disk cost only);
   - A/B the output head at Q8_0 against Q6_K.
5. Try one "Q4_K_M+" arm: `attn_k`/`attn_v` at Q8_0 and the first and last
   two `ffn_down` at Q6_K. Keep it only if it beats Q4_K_M on the gate.
6. No QAT unless Q4_K_M and Q5_K_M both fail and a smaller file is still
   needed.
7. Gate every tier against the bf16 GGUF in the same build, greedy on CPU:

| Metric | Q8_0 | Q6_K, Q5_K_M | Q4_K_M |
| --- | --- | --- | --- |
| DEV250 exact-match change | ≥ −0.4 pp | ≥ −1.0 pp | ≥ −2.0 pp |
| Paired McNemar on exact match | p > 0.05 | p > 0.05 | p > 0.05 |
| `nofp` and `valid` | no worse | at most +1 pp | at most +1 pp |
| Whole-output token identity vs bf16 | ≥ 97% | ≥ 90% | recorded |
| Mean KLD on held-out TRAIN text | ≤ 0.002 | ≤ 0.01 | ≤ 0.03 |
| Top token agrees with bf16 | ≥ 99.5% | ≥ 98.5% | ≥ 97% |

   Every shipped tier must also have a one-sided 95% upper bound of at most
   3 pp on net exact-match loss.
8. Measure prefill and decode latency on the target laptop, on CPU and
   Vulkan, with prompt-cache reuse. Ship Q8_0 plus the smallest tier that
   passes.

Thresholds are proposals for the user's sign-off.

**Exporter prerequisites:** rebuild the pinned llama.cpp (lost in the
reinstall), add bf16 conversion, Q5_K_M and `--tensor-type` passthrough to
`experiments/training/export_gguf.py`.

## Addendum 2026-10-12: no-op restraint threshold (proposed new card after C09)

The user asked whether a decision model (for example
`vllm-sr/Decision-2.0-Kai-0.6B`) could decide no-op cases. That model is a
Qwen3-0.6B classifier with a custom head, built for routing and filtering:

- it needs `trust_remote_code` and transformers ≥ 5.17;
- it has no llama.cpp path;
- it knows nothing about R edits.

Wiki: `concepts/decision-models.md`, `concepts/calibrated-relevance-filtering.md`.

The cheaper equivalent uses our own model: suppress an edit when
log P(`[NO_EDIT]`) − log P(edit) > τ. That is one prefill per case
(teacher-forced) and one knob that can become an extension setting.

**Proposed experiment:** daytime, CPU or Kaggle, arm-A checkpoint and its
Q4_K_M.

1. Pre-register the method and the rule for choosing τ.
2. Fit τ on DEV groups not used by DEV75 or DEV250, or by k-fold on DEV250
   if that pool is too small. Never fit on TRAIN: the model saw those no-op
   rows 2 to 3 times.
3. Report on DEV75 and DEV250: `nofp` against exact-edit recall, and AUROC.
4. Compare with Decision-2.0-Kai zero-shot, and optionally a hidden-state
   probe.
5. Adopt τ if it removes at least 3 `nofp` on DEV75 for at most 1 lost edit,
   without lowering `score250`.

The sealed set stays untouched.
