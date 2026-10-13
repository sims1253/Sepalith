# Expanded SFT E: exact D/full500 continuation

This packet resumes the exact full D checkpoint at step 500 and continues the unchanged 1,000-step, 16,000-draw expanded SFT schedule. It reuses the frozen D source identity `bb2f9fdc1d016679533beab7d9906c9b996c9ce59d36e54ff98101adbaeed384` without a source compatibility exception. The checkpoint identity and recipe identity are equal, including the 11,505-row data SHA, draw schedule SHA, selected merged parent, tokenizer, renderer, target-only policy, effective batch 16, 4,096-token sequence limit, and 1,024-token TRAIN target cap.

The actual resume parent is `/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-d/full/checkpoint-500/campaign-manifest.json`, SHA256 `6a4fe603566af95a09c3fd3bdadea37dac2717db55b08063a0bc9a1d93a1223b`. It is a full step-500 checkpoint with adapter, optimizer, scheduler, RNG, trainer state, and campaign state. Its sampler cursor is 8,000 draws; terminal step 1,000 corresponds to all 16,000 frozen draws.

The frozen D source requires milestones `[250, 500, 1000]` and permits mandatory decision stops only at 250 or 500. E therefore schedules HF diagnostic evaluations at `[250, 500, 750, 1000]`, with 750 and 1000 reached during this continuation, but does not claim a mandatory stop at 750. Full and adapter checkpoint cadence remains every 50 steps, so full checkpoint 750 is still archived. Training terminates at 1,000 unless the existing host-safety guard stops it earlier. Native D500 evaluation remains a root admission prerequisite; HF diagnostics are not acceptance authority.

Fresh paths are reserved for:

- training: `/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-e`
- checkpoints: `/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-e`
- runner: `/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-sft-e-v1`
- host supervision: `/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-e-host-supervision`

All four paths were absent during initial CPU preparation. `commands.json` contains the root-owned preflight, exact-source snapshot, enqueue, and launch commands. `run_owned.py` refuses changed packet/source/checkpoint pins and refuses non-fresh output paths. The worker created no runner state, enqueue, CUDA process, or training launch. A later root-created runner state is valid only when it contains the exact `bb2f9f` snapshot; output, archive, and host-supervision paths must remain fresh before launch.

CPU verification:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -B -m unittest -v tests/test_resume500_e_packet.py tests/test_runner_snapshot_replay.py
```

The seven tests hash every file named by the full checkpoint manifest, inspect optimizer/RNG/sampler continuity, verify all frozen source bytes and modes, replay the Runner snapshot ID, enqueue into a temporary disposable runner, compare the unchanged training identity/schedule/data, and check all E paths are fresh. The real recipe CPU preflight also passed with 11,505 rows, 16,000 draws, 1,000 updates, and `CUDA_started=false`.
