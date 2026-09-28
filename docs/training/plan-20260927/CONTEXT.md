# Shared context for plan cards

Read this before any card. Facts here were verified on September 27, 2026.
If you find one that no longer holds, record it in your status file.

## Project

Sepalith is a local-first R next-edit-suggestion model. It is served as GGUF
through llama.cpp inside VS Code, Positron and Zed. Continued pretraining (CPT)
is finished. This plan covers editing SFT, the post-SFT stage (offline
preference training, optional online RL) and the infrastructure both need.

## Model

- Architecture: dense `LlamaForCausalLM` from the MiniCPM5-2B lineage
  (`openbmb/MiniCPM5-2B-Midtrain`, Apache-2.0). It is **not** the Qwen3.5
  gated-DeltaNet hybrid that older docs and `scripts/cloud/` describe.
- 42 layers, hidden size 2048, 16 attention heads, 2 key-value heads, vocabulary
  130,560, untied embeddings, BOS 0, pad 1, end-of-generation ids 1 and 130073.
- 2,516,756,480 trainable parameters in 381 tensors. Weights are bf16, about
  5.03 GB. Full-weight optimizer state (aurora_mix, fp32) takes about 12.2 GB.
  A full checkpoint with optimizer state is about 17 to 19 GB.
- Static full-weight training state is about 22 GB. The editing-SFT smoke
  test at about 4K tokens peaked at 24.2 GB allocated on the 32 GB RTX 5090.
- Tokenizer: `tokenizer.json` sha256
  `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`. The
  runtime restores added tokens exactly. Feed pre-tokenized prompt ids to any
  inference engine rather than re-tokenizing text.
- Prompt and output contract for editing: `zeta2-prm03-v1`, implemented in
  `sepalith.campaign_protocol` (see the snapshot under `docs/campaign/`). A
  no-op answer is `[NO_EDIT]`.

## Checkpoints

| Role | Local path | Public copy |
| --- | --- | --- |
| Selected SFT parent (step 11,586) | `/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586` | `scholzmx/sepalith-2b-cpt`, folder `checkpoint-11586/` |
| Preserved alternative (step 11,649) | `/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11649` | same repo, folder `checkpoint-11649/` |

The parent's `model.safetensors` sha256 is
`4bf617f9ea66a31e547397b93f0998372686e076d03ba22961780abd285a3c51`. Initialize
editing SFT from the parent's weights only. Its CPT optimizer and sampler
state are not an SFT resume state. Public copies contain no optimizer state.

## Data

| What | Local path | Public copy (`scholzmx/sepalith` dataset) |
| --- | --- | --- |
| Editing-SFT TRAIN rows, 15,006 (pre-tokenized) plus context sidecar and repair ledger | `/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/` | `campaign-20260915/sft/DAT10-finish-source-repair-v3/` |
| Prepared draw schedules (1,160-update and one-pass 938-update) | planning packets, see snapshot | `campaign-20260915/sft/schedules/` |
| Roxygen queue, 10,017 candidates (not admitted) | `/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-supported-context-v1/` | `campaign-20260915/sft/roxygen-supported-context-v1/` |
| DEV75 panel (43 edit, 32 no-op) | `docs/campaign/work/lead/corrected-dev75-v1/` in the planning worktree | `campaign-20260915/dev/corrected-dev75-v1/` |
| CPT cohort rows and draw schedule | `/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-extension-v1/` | `campaign-20260915/cpt/CPT-prefix-extension-v1/` |
| CPT 2K validation panel | planning worktree `docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v2-2k/cpt_validation.jsonl` | `campaign-20260915/cpt/validation/cpt_validation-2k.jsonl` |
| Campaign receipts, manifests and code (no row-level data) | planning worktree `docs/campaign/` | `campaign-20260915/provenance/campaign-docs-snapshot-20260927.tar.gz` |

Family counts in the 15,006 TRAIN rows: finish_block 7,785, pipe_rewrite
1,983, format_propagation 1,615, rename_propagation 1,578, no_op 1,094,
roxygen_drafting 815, na_rm_propagation 136. TRAIN, DEV and the sealed final set
are split-disjoint by identity group (`DAT-02-global-split-v2.json`). The DEV
split holds 169 identity groups and 2,846 candidate rows, which is enough for a
larger panel.

## Sealed final evaluation set: never touch

Do not open, print, copy, upload or train on:

- `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-*`
- `/home/m0hawk/.local/state/sepalith/campaign-20260915/sealed-final/`
- DAT-08 final reference artifacts under `/home/m0hawk/.local/state/sepalith/`

Also avoid upstream pools that predate the split, such as row audits and source
snapshots. They can contain final rows. The final set opens only when the user
approves a release freeze.

## Code

- The full-weight runtime was never in git before this plan. A verbatim snapshot
  of the planning worktree's code and small manifests now lives in
  `docs/campaign/`, with the same relative paths as the original
  `/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/`.
  Supervisor scripts from `~/.local/state/sepalith/resume-20260921/` are in
  `docs/campaign/state-snapshot/resume-20260921/`.
- The packets this plan builds on (all under `docs/campaign/work/lead/`):
  - `r2-full-weight-edit-sft-preparation-v1/`: editing-SFT trainer, collator,
    binder and schedule builder.
  - `r2-full-weight-edit-sft-eval-gate-v1/`: DEV generation and gate control.
  - `r2-cpt450-cadence64-continuation-preparation-v2/`: CPT trainer and
    optimizer.
  - `host-memory-guard-v4/`: host memory guard.
  - `r2-full-weight-rl-production-driver-v2/`: GRPO driver, reward and RL data.
- Card C01 consolidates these into `packages/sepalith/src/sepalith/training/`.
  Later cards use the consolidated package.

## Environments

- PC: RTX 5090 32 GB under WSL2, 47 GiB RAM visible to Linux, 24 threads.
  `/` has about 113 GiB free. `/mnt/e` has about 1.2 TB free and holds
  checkpoints. `/mnt/h` is the NAS.
- Training interpreter: `/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python`
  (torch 2.11.0+cu130, transformers 5.5.0, trl 0.24.0, unsloth 2026.8.18,
  accelerate 1.14.0, peft 0.20.0, datasets 4.3.0).
- R 4.6.1 and `air` are installed. The Kaggle CLI is at `~/.local/bin/kaggle`.
- Research Python: `uv run python ...` from the repository root.
- Notebook: AMD laptop with an integrated GPU and no CUDA. Useful for llama.cpp
  CPU or Vulkan latency checks on target-class hardware.
- Kaggle: T4 GPUs only (16 GB each, two per session, no bf16), 30 GPU-hours per
  week, 12-hour sessions, 20 GB saved output. Internet sessions reach Hugging
  Face.

## Secrets

Secrets come only from the environment. Load them with
`eval "$(grep -E '^export (HF_TOKEN|ZAI_API_KEY|KAGGLE_API_TOKEN)=' ~/.zshrc)"`.
Never write a secret to a file, log, receipt, commit or kernel source.

## Public repositories

- Code: `github.com/sims1253/Sepalith` (public).
- Weights: `huggingface.co/scholzmx/sepalith-2b-cpt` (public). Put new model
  checkpoints in new public model repos or folders, weights only.
- Data: `huggingface.co/datasets/scholzmx/sepalith` (public), under
  `campaign-20260915/`. Its older top-level folders are stale since
  September 1; card C16 re-syncs them.
