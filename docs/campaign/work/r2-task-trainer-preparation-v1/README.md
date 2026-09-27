# SFT-11 task-stage preparation

This directory prepares the separate PRM03 task SFT stage that follows raw-R
CPT. The entrypoint is `source/experiments/training/campaign_task_sft.py`.
It delegates target-only collation, fused Trainer loss, checkpoint sealing,
tokenizer restoration, and the pre-update fused-loss witness to the reviewed
SFT closure copied under `source/`. The old 50-step corrective pilot policy is
left unchanged and is not reused as this stage's admission policy.

The new identity is `task_sft_prm03_v1` with objective
`prm03_target_only_next_token_v1`, a fresh rank-32/alpha-64 LoRA, BF16 fused
AdamW, learning rate `2e-4`, effective batch 16, context 4,096, and maximum
step 1,000. Its policy also pins cosine scheduling, warmup `0.03`, zero weight
decay, and seed `3407`. The parent must be either an explicitly bound fresh merged CPT
parent or a fresh Midtrain control. The adapter rejects the old corrective
parent and placeholder parent hashes. The candidate schedule records full
development checkpoints at steps 250, 500, and 1,000. Root may make a
milestone a decision stop in a newly admitted recipe; the identity and full
checkpoint must remain matched on continuation.
The target-only pre-update witness accepts fresh step 0 or a matched full
step 250/500 resume. The copied SFT seam keeps the legacy pilot's step-25
resume default unchanged.

Every admitted training row must be a complete prediction-time PRM03 row. Its
target tail is `target_body_tokens + target_terminal_tokens + [1]`, including
the terminal EOS, and must fit within 192 tokens. The full row must fit within
4,096 tokens. The adapter rejects truncation, incomplete terminal metadata, and
over-cap targets. It consumes only exact rows supplied by root/editor acceptance;
the template and unit fixtures do not constitute a data admission.

Native final DEV at cap 192 is the acceptance route after post-training merge.
An HF evaluator can be bound only as `hf_diagnostic_only`, with its result
explicitly excluded from acceptance authority. CPT loss and HF diagnostics do
not establish task quality or authorize RL continuation.

`recipe-template.json` intentionally contains `ROOT_TO_BIND` placeholders and
`launch_authorized: false`; it is a schema aid, not a launch command. Root must
bind exact train rows, draw schedule, DEV panel/case IDs, renderer contract,
parent/model/tokenizer/source hashes, evaluator factory, deadline, and fresh
output/archive paths before running `--preflight-only`. No candidate data was
fabricated or admitted in this preparation.
