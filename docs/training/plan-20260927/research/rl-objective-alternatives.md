> Research report written on 2026-09-27 by a Hermes agent (zai/glm-5.3-flash) with web search and the user's wiki.
> The planning session verified the arXiv ids and corrected the wrong ones (GRESO is 2506.02177; Bae et al. is 2504.03380).
> A cited "xLAM next-edit" paper (2509.28928) does not exist, so it was removed. Non-arXiv claims (blogs, product posts) were not verified.
> The report assumed a 2.52B dense model on one 32 GB GPU, as in CONTEXT.md.

# Sepalith post-SFT RL: what to run instead of / on top of GRPO

Scope: objectives, advantage/baseline estimation, rollout/prompt selection. Priorities: dev-panel quality per generated token, one RTX 5090 (32 GB, ~8 GB headroom), no reference model, G=4, 92.5% zero-variance groups, sparse exact-match reward, 1,024-token completions.

## Top recommendations (ranked)

### 1. Reward first: replace exact-match with graded partial credit (token/diff-similarity + validator ladder)
- **What:** Score every completion with a smooth similarity to the reference edit (normalized token/line F1 or SWE-RL-style sequence similarity, computed after `air` canonicalization), plus small binary validator bonuses (R parse, format-clean), plus the no-op/emit-nothing bit. Exact match becomes the top of a ladder, not the only nonzero reward.
- **Sources:** SWE-RL, https://arxiv.org/abs/2502.18449 (Feb 2025). Ant Group's NES system (per your wiki: dual location/edit models, SFT+DAPO datasets) is UNVERIFIED secondhand.
- **Evidence:** SWE-RL showed RL on similarity-of-gold-patch rewards works at 70B on real GitHub edits; the A2R code-RL study (ACL Findings 2026, https://aclanthology.org/2026.findings-acl.886/) documents that code-editing rewards are continuous and "completely successful rollouts are rare," so binary-strict pipelines stall.
- **Fit:** Directly attacks your two stated defects (alternative-valid-edit penalty, sparsity) *and* the 92.5% statistic: under graded reward, an "all-correct by EM" group still has reward variance, so groups stop being zero-signal. No extra memory; you already compute line-F1.
- **Implementation:** TRL custom reward function (low effort, pure Python; `air` + parse validators exist).
- **Cost/risk:** Low. Main risk is reward hacking of the cheap validators (AntNES observed F1-gaming); keep exact-match dominant and monitor the dev panel.

### 2. Pass-rate-targeted prompt selection: census + balanced-band curriculum + DAPO-style refilling
- **What:** Stop feeding the trainer zero-variance prompts. Estimate each prompt's pass rate p̂ under the SFT policy, train only where 0.1–0.2 ≤ p̂ ≤ 0.8–0.9, and refill each step from an oversampled buffer until the batch is filled with mixed groups (DAPO dynamic sampling).
- **Sources:** DAPO, https://arxiv.org/abs/2503.14476 (Mar 2025); Online Difficulty Filtering (Bae et al.), https://arxiv.org/abs/2504.03380 (v3 2025; EACL 2026); GRESO, https://arxiv.org/abs/2506.02177 (NeurIPS 2025); Prompt Replay, https://arxiv.org/abs/2603.21177 (Mar 2026).
- **Evidence:** Bae et al.: balanced filtering beats plain GRPO with <half the updates (+12% AMC at 7B, 3B/7B models). GRESO: zero-variance status is temporally consistent; skipping known-dead prompts frees budget for resampling (1.5B–32B). Prompt Replay: reusing *prompts only* (not trajectories) near p=0.5 raises mean |advantage| per rollout; also reports spurious-reward artifacts on Qwen2.5-Math — a caution for any single-benchmark ablation.
- **Fit:** With 74/80 groups dead, this is the largest single lever. Generation is your scarce resource; every rollout not spent on a certain-certain prompt is a rollout available for a mixed one. Your no-op prompts sit mid-difficulty for a 25%-no-op SFT model, so the band keeps restraint learnable. Fits 32 GB trivially (it is a sampling-policy change, not memory).
- **Implementation:** Not built into TRL — wrap GRPOTrainer with a custom prompt sampler + oversample/refill loop (moderate effort, ~a few hundred lines). GRESO-style pre-rollout skipping is optional later; post-rollout filtering is the simple version.
- **Cost/risk:** Low–medium. Risk: band starvation early (if p̂ is degenerate) — mitigate with the census in the recipe below.

### 3. Advantage/baseline fixes at G=4: Dr.GRPO scaling off, JS-shrunk LOO baseline, DAPO loss + truncation masking
- **What:** (a) Stop dividing by group std (`scale_rewards=False`, Dr.GRPO) or use batch-level std (`scale_rewards="batch"`); (b) replace the per-prompt mean baseline with a James–Stein shrunk estimator b = (1−λ)(r̄_group−own) + λ·batch-mean, λ closed-form; (c) `loss_type="dapo"` (token-normalized), `mask_truncated_completions=True`, `epsilon_high=0.28`.
- **Sources:** Dr.GRPO, https://arxiv.org/abs/2503.20783 (Mar 2025); JS-shrinkage baselines, https://arxiv.org/abs/2511.03710 (Nov 2025, ICML 2026); DAPO (above); Lite-PPO batch-std trick (verified present in current TRL docs).
- **Evidence:** Dr.GRPO removes std-division/length bias; shrinkage paper: at G=2/4/8 the JS baseline beat GRPO, RLOO, ReMax, REINFORCE++, BLOO on Qwen 0.5B–8B (math, logic, VL, RLHF), gradient variance −11% to −67%. Exactly your G regime.
- **Fit:** At G=4 with a 130k vocab and mixed-length completions, the group std is mostly noise; shrinkage pools across the batch with zero new hyperparameters and no extra memory. Truncation masking matters because ~19% of targets are long and 1,024 tokens will truncate some rollouts.
- **Implementation:** (a)/(c) are TRL config flags (verified in current GRPOTrainer docs; note TRL's default `loss_type` is now `"dapo"` and β=0 — your "BNPO" baseline suggests an older TRL). (b) is ~30 lines in the advantage callback (custom, low effort).
- **Cost/risk:** Low. Risk: no-std scaling makes update magnitude reward-scale-dependent — monitor grad norms for the first steps.

### 4. Harvest the all-zero tail: RL-ZVP-style advantage on zero-variance groups
- **What:** For groups where all rewards are equal, don't drop them: give all-correct completions a small positive advantage and all-zero completions a negative one, with per-token entropy as the magnitude so high-entropy (decision) tokens move most.
- **Source:** RL-ZVP, https://arxiv.org/abs/2509.21880 (Sep 2025).
- **Evidence:** Qwen3-1.7B/8B on six math benchmarks: +4.0 Acc@8 avg over GRPO, and it beats GRPO-DAPO-dynamic-sampling and GRESO even when those get 3–5× more rollouts.
- **Fit:** Your 16/80 all-zero groups currently buy nothing; with the shaped reward from #1, the 58 all-correct groups regain variance anyway, and RL-ZVP covers the remainder. No memory cost.
- **Implementation:** ~40 lines in the advantage computation (custom; no framework support found).
- **Cost/risk:** Medium: reinforcement-without-contrast can amplify reward artifacts (an all-correct-by-EM degenerate pattern gets reinforced). Only run after #1; gate on the dev panel's no-op precision.

### 5. Generate more, train on less: PODS max-variance down-sampling + bounded high-|A| replay
- **What:** Burst-generate n=8–16 completions per prompt, then update on the m=4 max-variance subset (binary case: m/2 best + m/2 worst). Separately, replay *individual* high-|advantage| rollouts for ≤10 steps in fresh-anchored batches (half replay ratio) to get >1 gradient step per generated token.
- **Sources:** PODS, https://arxiv.org/abs/2504.13818 (Apr 2025); rollout-level advantage-prioritized replay, https://arxiv.org/abs/2606.04560 (Jun 2026); A2R for code RL, https://aclanthology.org/2026.findings-acl.886/ (2026).
- **Evidence:** PODS: ≥1.7× faster to vanilla-GRPO peak accuracy (OLMo2-3B-Base, Qwen2.5 7B/32B; binary + shaped rewards). Replay study: +1.66 pp at 4B with staleness bounded; A2R: up to +6.8 points over GRPO/DAPO on code-editing models (3B–14B, incl. Qwen2.5-Coder), with an on-policy refresh variant that regenerates instead of replaying stale rollouts.
- **Fit:** Your metric is quality per collected completion — this multiplies gradient signal per generated token and cuts optimizer-side memory per step (you backward over m not n). Replay adds off-policy noise; TRL has no replay buffer (custom; A2R's "replay prompts, regenerate rollouts" variant avoids ratio correction entirely).
- **Implementation:** PODS is a trivial post-reward filter (low effort); replay is a buffer + sampler (medium effort, do after 1–3).
- **Cost/risk:** PODS low; replay medium (staleness → instability; keep τmax small, monitor KL-to-last-step implicitly via clip ratios).

### 6. One cheap cloud day: full census + optional largest-G run; plus cheap insurance flags
- **What:** On a single 80–96 GB rental (~$1–2 for a few hours): sample k=8–16 per prompt at T=1.0 over the whole RL pool once → per-prompt p̂ and a Beta posterior (ThinkPrior-style init so selection works from step 0); optionally run the G=16-variant of recipe 1–4 there. On the 5090, set `importance_sampling_level="sequence"` (GSPO) if/when you replay, and consider `top_entropy_quantile=0.2` (high-entropy-token filtering) as cheap token-level credit assignment.
- **Sources:** GSPO, https://arxiv.org/abs/2507.18071 (Jul 2025, Qwen3); ThinkPrior, https://arxiv.org/abs/2609.09075 (Sep 2026); Beyond the 80/20 Rule, https://arxiv.org/abs/2506.01919 (Jun 2025; TRL flag verified).
- **Evidence:** GSPO stabilizes sequence-level-reward training (adoption across Qwen3 models); ThinkPrior more than halves early silent groups on Qwen2.5-Math-7B (single-paper evidence, UNVERIFIED independently); 80/20 shows most GRPO gradient lives in ~20% high-entropy tokens.
- **Fit/implementation/cost:** Census is one offline generation sweep; flags are config-level. Risk: minimal. Note: colocating a vLLM engine in TRL on your 32 GB card with a 22 GB training state does not fit without `vllm_mode="colocate"` + sleep tricks — UNVERIFIED for a 2.5B dense model on one GPU; treat generation via plain transformers as the known-good path.

## Directly addresses the 92.5% zero-signal groups — concrete recipe

1. **Census:** post-SFT, sample k=8, T=1.0, per prompt over the whole pool (5090 evenings or one A100 hour). Score with the #1 reward.
2. **Curriculum band:** train on prompts with 0.125 ≤ p̂ ≤ 0.875 (balanced-band rationale: Bae et al.; their band 0.2–0.8 with a large pool, widen if starved). Expect heavy curation toward *mixed edit regions* and mid-difficulty no-ops; DEPO (https://arxiv.org/abs/2602.06375, UNVERIFIED) reports difficulty estimators retain ~35–40% no-op-like prompts vs ~5% base rate — keep no-ops near their natural share so restraint stays trained.
3. **Group size & selection:** G=8 on kept prompts (halves baseline noise vs G=4; shrinkage handles the rest). Per step: draw ~1.5–2× the batch, drop groups with zero reward variance, refill (DAPO); feed dropped *all-zero* groups to RL-ZVP; optionally park mid-band prompts in a prompt-only replay buffer (Prompt Replay, cooldown ~10 steps, ≤5 reuses).
4. **Temperature/entropy:** sample at T=1.0; `epsilon_high=0.28` (clip-higher); no entropy bonus (DeepCoder found it unstable).
5. **Update:** `loss_type="dapo"`, `scale_rewards=False` + JS-shrunk LOO baseline, `mask_truncated_completions=True`, `num_iterations=1`.
6. **Track** per-step: fraction of zero-variance groups, mean |A|, completion length, dev-panel exact + no-op precision every N steps.

Expected effect: from ~7.5% to ~100% of collected groups carrying gradient (mixture of #1–#4), i.e., an order-of-magnitude more signal per collected completion even before quality gains.

## What the wiki had vs what is new

- Wiki already covers the critic-vs-group cluster (EVPO, TETHER, shrinkage, BPCO, BPO, RPG's KL-misweighting, POISE, coverage/TailSFT), the frameworks (TRL/verl/prime-rl/slime), DAPO/DeepCoder/SWE-RL context (via the Sepalith and AntNES pages), and the 2025–26 difficulty-filtering family (Bae et al., GRESO, DEPO, CERO, ThinkPrior, Prompt Replay, RL-ZVP, PODS, replay/PER line). It is strong and current.
- New / not in wiki: A2R's code-editing replay evidence (ACL Findings 2026); the fact that current TRL defaults to `loss_type="dapo"`, β=0, and ships `scale_rewards`/`epsilon_high`/`mask_truncated_completions`/`top_entropy_quantile`/`importance_sampling_level` — wiki GRPO/TESA pages predate this and your "BNPO" setup suggests the wiki's TRL picture is stale.
- Stale/weak flags: OAPL wiki mixes blog-grade numbers (Qwen "3.5 27B", 48% vs 44%) with the paper's — treat blog claims as anecdotal; several method pages rest on Qwen2.5-Math ablations, which the Prompt Replay paper shows can produce spurious-reward artifacts; wiki GRPO page still describes the update as KL-penalized "standard PPO" before its own RPG section corrects it — cosmetic but misleading.

## Not recommended and why

- **PPO / BPCO-style value models:** no memory headroom (22 GB used), critic instability is worst exactly at small batch/G; EVPO's own data show the critic hurts until it matures.
- **KL to a reference model (KLPO, β>0):** costs a second 2.5B in memory; RPG shows GRPO's KL term is even mis-weighted off-policy; entropy/clip controls plus the dev panel are cheaper insurance. If you must regularize, RPG's corrected estimator is the principled version — still custom.
- **Wholesale swap to RLOO/REINFORCE++:** same zero-variance failure mode; RLOO ≈ GRPO minus std at G=4; recommendation #3 captures their useful parts.
- **NTF / FlashREINFORCE (G=1):** solves a memory problem you don't have; math results trail GRPO-G16 at your scale; group-cancellation is still helping you.
- **Token-level Monte-Carlo credit (your KLPO as-is):** unbiased but very high variance at G=4 without extra rollouts; `top_entropy_quantile=0.2` gets the spirit for free.
- **Overlong soft-punishment shaping:** with a 1,024 cap and median-short completions, masking truncated samples is safer than shaping length penalties that fight the exact-match objective.
- **Optimizer changes (Muon-for-RL variants):** your aurora_mix already covers this; the wiki's own muon-not-special page argues RL gains there are not robust. Leave the inner loop alone this stage.
