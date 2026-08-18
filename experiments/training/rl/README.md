# RL — verifiable-reward training

Status: scaffold. The environments exist. Trainer selection is staged.

| Layer | Choice | Why |
|---|---|---|
| Environments | Prime Intellect `verifiers` | Our keystroke simulator, exact-reward scenarios, and simulation-verified tasks map onto its env/rubric pattern. Rollout viewers and eval harnessing come free. |
| Judge | glm-5.3, reasoning low | Validated on anchors: 40/40 ground truth, 38/40 degraded, 39/39 unrelated. Use only where exact rewards cannot apply. Log every call. |
| Trainer v1 | unsloth GRPOConfig, `importance_sampling_level="sequence"`, `loss_type="dr_grpo"` | Two flags turn GRPO into GSPO. Same stack as our SFT runs. |
| Trainer alt | prime-rl | PI's algorithm layer, multi-env training, less ops surface than Miles. |
| Scale-up | Miles | SGLang rollouts plus Megatron weight sync, built for many-GPU clusters. It ships a verifiers integration, so environment code carries over. Not worth it at 2B on one GPU. |

## Environments (built, awaiting trainer)

1. Edit scenarios (`../../synthetic-data/scenarios.py`): rename, pipe, format,
   doc-sync, na.rm. Exact-match rewards. No-op baseline scores 0.
2. Keystroke simulator (`../../eval/keystroke_sim.py`): cold and warm
   prefix-cache episodes against any llama-server model.
3. Paper-to-R (`../../synthetic-data/paper_to_r.py`): rewards are statistical
   properties — coverage, type-I error, bias — checked by running the
   generated validator. A validator must fail a corrupted twin before use.
4. No-edit tasks: the correct target is an unchanged region. This penalizes
   eagerness, the top UX complaint about edit suggestion.

## Reward rules

- Exact rewards dominate. The judge fills gaps, never leads.
- Every validator faces a corrupted twin before we trust it.
- Keep no-edit episodes in the mix.
- Log every judgment. Mix reward types so no single signal gets gamed.
- Difficulty gate: a random policy scores near 0. glm-5.3 scores well. Check
  both before a family enters training.
