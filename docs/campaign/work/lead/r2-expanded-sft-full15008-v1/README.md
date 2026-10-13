# SFT-11 fresh E750 to full15006 preparation

This packet prepares one fresh LoRA stage on the verified merged E750 parent. It resets AdamW, cosine scheduler, RNG, and sampler at step 0 because the data and schedule identities changed. It does not claim checkpoint resume continuity from E.

The immutable TRAIN view holds exactly two root-confirmed contradictory pipe labels for repair and retains the other 15,006 rows, including every one of the 3,503 newly admitted finish rows. The seeded schedule contains 1,160 updates and 18,560 draws at effective batch 16 (12 edits and 4 no-ops per update). Every eligible ID is supervised at least once with its full target body, protocol terminal, and EOS; the observed maxima are 3,064 sequence tokens and 933 target tokens including EOS.

The new source adapter validates the structured v2 draw order rather than treating its eligible-set `row_ids` inventory as an order. It retains the reviewed target-only collator, pinned tokenizer checks, standard GPU gradient checkpointing, per-update telemetry, pre-update gate, and full checkpoint lifecycle. Full checkpoints occur every 40 updates, so terminal step 1,160 persists adapter, optimizer, scheduler, RNG, trainer, and sampler state. HF diagnostics run at 240, 480, 720, and 1,160; native DEV remains the acceptance authority.

Root review and launch sequence:

```sh
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/source/packages/sepalith/src:/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/source/experiments/training /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/source/experiments/training/campaign_expanded_sft.py /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/recipe.json --preflight-only
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src/sepalith/runner.py --state /home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-full15006-v1 snapshot --repo /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/source --include experiments --include packages
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src/sepalith/runner.py --state /home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-full15006-v1 enqueue /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/runner-recipe.json
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/run_owned.py
```

The first command is CPU-only but hashes the 5 GB parent weight input. The last two commands require root admission and the root CUDA lease. This preparation did not run them.
