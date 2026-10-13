> Research report written on 2026-09-27 by a Hermes agent (zai/glm-5.3-flash) with web search and the user's wiki.
> The planning session verified the arXiv ids and corrected the wrong ones (GRESO is 2506.02177; Bae et al. is 2504.03380).
> A cited "xLAM next-edit" paper (2509.28928) does not exist, so it was removed. Non-arXiv claims (blogs, product posts) were not verified.
> The report assumed a 2.52B dense model on one 32 GB GPU, as in CONTEXT.md.

# Sepalith post-SFT: alternatives to online GRPO

Scope: what should follow the 15,006-example editing SFT (1x RTX 5090 32GB, full-weight aurora_mix, ~7-8GB headroom, beta=0 TRL GRPO, 92.5% zero-gradient groups; objective = dev quality gain per collected completion). All routes start from the SFT checkpoint.

## Ranked routes

### 1. Census-filtered RFT / expert iteration (rejection-sampling fine-tuning on verified samples)
- What: Roll out the SFT model G=4-8 per training prompt, keep only verified-correct completions (normalized exact match, R-parse, `air` canonicalization; no-op prompts verified by emitting the no-op token), then run your existing full-weight CE stack on the kept set. Iterate once or twice. This is RAFT/expert-iteration: no reference model, no vLLM, no group machinery — generation can run quantized in llama.cpp overnight.
- Primary source: RAFT (reward-ranked fine-tuning), arXiv:2304.06767 (2023); "A Minimalist Approach to LLM Reasoning" arXiv:2504.11343 (Apr 2025): RAFT matches GRPO/PPO at 7B, and GRPO's edge is mostly *discarding all-wrong groups*, not reward normalization.
- Evidence: Qwen2.5-Math-7B: RAFT 49.9% vs GRPO 53.9% avg; RAFT++ 52.5%.
- Fit: every collected completion becomes training data or a diagnostic — exactly the gain-per-collected-completion metric. Directly monetizes the 58/80 all-correct groups GRPO throws away. No new memory; uses the 0.2xF1 shaping for selection only.
- Cost: one generation pass over ~15k prompts + cheap CE training. ~0.2-0.5x GRPO compute. Effort: low (reuses SFT pipeline).
- Risks: reinforces already-correct behavior — limited headroom on the 16 all-zero prompts; mitigate with elevated sampling temperature to mine rare successes.

### 2. DPO on verifier-labeled pairs from the same rollouts
- What: From the census generations, build pairs per prompt: chosen = model's verified-correct sample (or gold when the model never succeeds); rejected = plausible-but-wrong sample (wrong region, over-edit, or a *false suggestion on a no-op prompt*). Train plain DPO with reference logprobs **precomputed once** on the SFT checkpoint (forward-only pass fits the 7-8GB headroom; no resident reference model). Cheapest on-policy negative signal without RL infrastructure.
- Primary source: DPO arXiv:2305.18290 (2023); SimPO arXiv:2405.14734 (2024) reference-free variant; 20-variant study arXiv:2603.19335 (2026): at 1.5B SFT > IPO > KTO > DPO > SimPO, SimPO collapses (-15.7pp vs SFT) — vanilla DPO at 2.5B; avoid SimPO/ORPO.
- Evidence: Zed fixed Zeta1's edge cases with ~150 DPO pairs (Feb 2025); JetBrains Mellum ships DPO post-SFT at 4B (arXiv:2510.05788).
- Fit: pairs are free byproducts of route 1's generation; static pairs mean no sampling during training; no-op prompts get explicit negative mass, which GRPO's all-correct groups never provide. Cost ~0.3x GRPO; ref-logprob precompute is one-off. Effort: low-medium (TRL DPOTrainer; avoid degenerate pairs where chosen is always gold — prefer model-generated chosen).
- Risks: DPO degrades chosen logprob; exact-match label noise (alternative valid edits marked rejected) — filter pairs through the same validators, let 0.2xF1 decide borderline rejections.

### 3. Fix the GRPO run itself: mixed-band prompt filtering + multi-gold canonical reward
- What: Re-run the census on the *new* SFT model and train GRPO only on prompts in the mixed band (0 < pass-rate < 1 at G=4). Widen the reward: exact match after `air` canonicalization on both sides (accepts alternative valid edits), R-parse as a gate, AntGroup-style hierarchy (1.0 exact / partial credit 0.5x-similarity / -1.0 wrong non-empty output on no-op prompts). Keep beta=0.
- Primary sources: JustRL II (OpenBMB blog, Sep 2026) — difficulty recalibration cut discarded groups 55%→25%, ~1.7x more gradient per rollout; AntGroup NES arXiv:2508.02473 (Aug 2025): hierarchical EMR/edit-similarity reward with DAPO on 4B-8B code models; f-grpo arXiv:2602.06717 down-weights obvious-success groups if you prefer reweighting.
- Evidence: AntGroup: SFT alone ~6%→72% location accuracy, DAPO refines; Sweep RL (Jan 2026, 1.5B, 2000 steps, tree-sitter + size rewards) shows small-model RL works with non-exact rewards.
- Fit: the industry-standard stage-2, and the census is the prerequisite either way. Filtering converts the 92.5% zero-gradient waste into per-completion efficiency; canonicalization directly fixes the "penalizes alternative valid edits" complaint.
- Cost: same per-step GRPO cost, far less waste; TRL in-process generation is slow on one GPU, so consider one short resumable cloud run (colocated vLLM + training) for the GRPO phase. Effort: medium.
- Risks: training only on hard prompts shifts the distribution (mix in a fraction of easy prompts); entropy collapse on long completions; still sparse if the mixed band is tiny (see decision rule).

### 4. Distillation from a larger code teacher (off-policy now; on-policy OPD optional)
- What: Have a large open-weight coder rewrite the editable region on your SFT prompts (R-aware prompting; treat recent user edits as intentional — Zed's "reversal problem" lesson), validate against gold/parse, SFT the student where the teacher beats the model. Stronger variant: on-policy distillation — student samples, teacher scores every token via reverse KL used as dense per-token advantage. Teacher scoring runs in the cloud; only per-token logprobs travel home.
- Primary source: Thinking Machines, "On-Policy Distillation" (thinkingmachines.ai/blog/on-policy-distillation, Oct 2025): matches RL at 1/10-1/30 the compute, 7-10x fewer gradient steps; Qwen3: AIME 74.4 at ~1,800 vs 17,920 GPU-hours for RL. Wiki: one-shot OPD (arXiv:2609.04172) shows the data side is nearly free.
- Evidence: Zeta2 = Sonnet-teacher distillation into an 8B student, +30% acceptance over Zeta1 (Mar 2026); MiniCPM5-2B (your base lineage) finishes with multi-teacher OPD.
- Fit: dense signal on *why* an alternative edit differs — what binary rewards can't give — and never needs a local reference model. But good R teachers are scarce (coders are Python/TS-centric; audit z.ai glm-5.3 on R first), and OPD needs tokenizer-aligned teacher logits, so full-OPD is cloud-native.
- Cost: one-time off-policy relabeling ~$10-30 cloud; OPD recurring per step. Effort: highest of these routes.
- Risks: teacher style drift, reversal-problem prompt tuning, license (use open-weight teachers; API distillation into a competing model breaches most provider terms).

### 5. Dedicated abstention/no-op calibration pass
- What: The 25% no-op class is a threshold + data problem before it is an RL problem. Rebalance no-op examples (VS Code: over-eagerness was dataset imbalance; rebalancing fixed jump *and* no-jump accuracy); add Cursor-style asymmetric reward if RL is used (+0.75 accept / -0.25 reject / 0 nothing encodes "suggest only if p(accept) > 25%"); or simply make no-op prompts the primary source of rejected members in route 2's pairs.
- Primary sources: Cursor Tab RL (cursor.com/blog/tab-rl, Oct 29 2025): -21% suggestions, +28% accept rate; VS Code long-distance NES (Feb 2026): rebalancing before RLVR; AntGroup "-keep" samples teach not-suggesting (81.6% preservation accuracy).
- Cost: negligible; effort: low. Risks: threshold tuning without retraining doesn't transfer across users. Best folded into routes 1-2 rather than run alone.

### 6. SimpleSD self-distillation (fallback only)
- What: SFT on raw own-samples at shifted temperature (T_eff ≈ 0.6-0.8), no verifier (wiki: SimpleSD, Apple, arXiv:2604.01193; +30% rel. LiveCodeBench on Qwen3-30B). Trivially cheap but unverified for edit tasks; risks sharpening false suggestions on no-op prompts. Fallback if routes 1-2 stall and generation is free.

## What next-edit model builders do after SFT

- **Cursor Tab**: online policy-gradient RL on production accept/reject signals, reward +0.75/-0.25/0; 1.5-2h deploy-collect cycles; -21% suggestions, +28% accept (cursor.com/blog/tab-rl, Oct 2025). Composer's real-time-RL post (Mar 2026) documents two reward hacks: aborted tool calls and edit-deferral (wiki: cursor-real-time-rl).
- **Zed Zeta1 → Zeta2/2.1**: Zeta1 = SFT (~400 real examples) then DPO (~150 pairs) (zed.dev/blog/edit-prediction, Feb 2025). Zeta2 = teacher distillation (Sonnet → Seed-Coder-8B, ~100k opt-in states), DPO experiments named as the next stage (Mar 2026); Zeta2.1 = prompt-format change only (multi-region, -67% output tokens, May 2026). Acceptance/rejection-signal DPO still listed as future - UNVERIFIED as shipped.
- **GitHub Copilot NES**: SFT on curated real editing sessions → RL against an LLM grader that defines "bad suggestions" over unlabeled data; later releases add no-edit samples, LLM data filtering, synthetic distillation (github.blog, 2025). Long-distance NES: dedicated location model, dataset rebalancing, then RLVR with eventual-cursor-movement as reward, +23% code-written (code.visualstudio.com, Feb 2026). The 3-in-1 model (Sep 2026) re-runs SFT→RL on a unified diff-patch format with pseudolabel bootstrapping.
- **JetBrains Mellum**: pretrain → context-aware SFT → DPO on LLM-judged pairs including deliberately degraded completions (placeholders/TODOs) to kill unhelpful generations (blog.jetbrains.com, Apr 2025; arXiv:2510.05788).
- **AntGroup NES**: SFT → DAPO with hierarchical EMR/edit-similarity reward; "-keep" samples encode abstention; SFT+DAPO datasets open-sourced (arXiv:2508.02473).
- **Sweep Next-Edit (1.5B, local)**: SFT on ~100k examples (4h, 8xH100) → RL 2000 steps with tree-sitter parse reward + diff-size regularization (blog.sweep.dev/posts/oss-next-edit, Jan 2026).
- **Continue Instinct**: 4k real consensual edits; post-SFT recipe not public - UNVERIFIED (likely SFT-only).
- **Windsurf/Codeium**: no public completion-model post-training recipe - UNVERIFIED.
- Independent replication: RLinf reproduced Cursor-style online RL on Qwen2.5-Coder-1.5B with accept signals (+50% on their metric) - small-model online RL works but needed small LR and no KL (rlinf.readthedocs.io, Oct 2025).

Pattern: everyone ships SFT → (distillation and/or offline preference) → online RL only with production-scale reward signal. Nobody runs GRPO on a static 15k-example corpus without first fixing prompt difficulty and reward shaping.

## Recommended sequence for us

1. **Census v2 (local, ~1 GPU-day)**: roll out the new SFT model G=4-8 over all 15k prompts at moderate temperature; record pass-rate, F1 distribution, no-op false-suggestion rate. Everything else keys on this.
2. **RFT round 1 + DPO pairs (local)**: train on verified-correct samples (route 1); build pairs from the same generations (route 2, no-op prompts contributing rejected=false-suggestion pairs); precompute ref logprobs once. Gate (your kill-test discipline): adopt if the 75-case dev panel gains ≥2pp exact with no-op false-suggestion rate not worse.
3. **Decision rule**: if census mixed-band ≥ ~15% of prompts, proceed to **filtered GRPO** (route 3) with canonicalized reward — locally if TRL throughput suffices, else one short resumable cloud run; re-census the RFT+DPO checkpoint first, since steps 1-2 move the mixed band. If mixed-band < ~15%, stay offline: second RFT/DPO iteration at higher temperature, and/or teacher relabeling (route 4, off-policy) to break hard prompts before any RL.
4. **Optional cloud OPD**: only if the dev panel shows quality (not abstention) as the residual gap and an R-capable open teacher passes your audit; short, checkpointed, resumable.

## What the wiki had vs what is new

- Wiki covers the NES cluster well (zeta2, antgroup-nes, vscode-long-distance-nes, instinct-continue, tab-tab-bug) and the method cluster (one-shot-opd, simple-sd, distributional-post-training, tailsft, justrl, group-vs-critic-debate, pedagogical-rl, privileged-critic-inputs, reward-hacking).
- **New via primary sources**: Cursor Tab-RL mechanics (reward values, 1.5-2h cycles, -21%/+28%) — cursor-real-time-rl.md is Composer-only, partially stale; Zeta2.1 (May 2026) — zeta2.md stops at "DPO as future"; Copilot NES training post (SFT→RL-with-LLM-grader, no-edit expansion) — wiki has only the Feb 2026 long-distance post; Mellum (no wiki page; arXiv:2510.05788, SFT→DPO with degraded completions) — a direct template for Sepalith; Sweep's RL reward design (tree-sitter + size) — instinct-continue.md mentions Sweep only in passing; RAFT≈GRPO (arXiv:2504.11343) and the 1.5B ranking inversion (arXiv:2603.19335) — not in wiki; RLinf small-model replication — not in wiki.
- **Stale flags**: projects/sepalith.md (Sep 13) still describes the stage-1 data-pipeline phase — pre-dates the 2.52B model, 472M-token CPT, editing-SFT plan; zeta2.md "Future" understates shipped work; cursor-real-time-rl.md lacks Tab-RL specifics; instinct-continue.md's acceptance-numbers question remains open.
