# Full-weight RL runtime proof v1

This packet runs the real installed TRL 0.24 `GRPOTrainer`, the reviewed
`FixedIDGRPOTrainerMixin`, and `FullWeightOptimizerTrainerMixin` on a tiny CPU
causal model. The optimizer uses its actual `aurora_mix` implementation. The
proof performs an uninterrupted two-update lane and an interrupted step-one
plus resumed step-two lane.

The terminal model, optimizer, and scheduler hashes match exactly. Resume
processes only the second source row. All checkpoints are sealed and verified
as `full_weights`, and checkpoint one records `consumed_rows=4`, rollout group
cursor one, optimizer step one, and `rollout_buffer_state=empty`.

Run the contract tests:

```bash
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/rl-fullweight-runtime-tests \
PYTHONDONTWRITEBYTECODE=1 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  -m unittest discover -s docs/campaign/work/lead/r2-full-weight-rl-runtime-proof-v1/tests -v
```

Run a fresh proof using a new E output directory:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=docs/campaign/work/lead/r2-full-weight-rl-runtime-proof-v1/source/experiments/training \
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
 docs/campaign/work/lead/r2-full-weight-rl-runtime-proof-v1/runtime_proof.py \
 --output /mnt/e/sepalith/campaign-20260915/data-work/RL11-full-weight-runtime-proof-review-run
```

This is a mechanical proof, not training admission. The synthetic model is CPU
FP32, generation is deterministic, and the reward callable checks exact token
IDs without running PRM03/R. Production still needs the selected dense editing
parent, 381-tensor dispatch, tokenizer repair, PRM03 reward, sampled-RNG and GPU
memory proof, data identity, recipe, and evaluation gates.
