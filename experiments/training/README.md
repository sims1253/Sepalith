# Training

This stage fine-tunes a base model with LoRA SFT on the assembled mixtures,
merges the adapter, and exports a Q8_0 GGUF for llama.cpp serving and
evaluation. The prompt format is Zeta-2 (background:
[edit prediction at Zed](https://zed.dev/blog/edit-prediction)); the trained
model is scored with `../eval/run_eval.py --model zeta2`.

Products: checkpoints and the final LoRA under
`/mnt/h/sepalith/runs/sft_v1_<model>/`, and `<stem>-Q8_0.gguf` under
`experiments/models/`.

## Before you start

- Run `uv sync` from the repo root. `unsloth` is in `pyproject.toml`, so
  `uv run python` can import it. The script docstrings also name a second
  virtualenv, `.venv-sft`; on the dev machine that is the tested
  environment. If `uv run` fails on a CUDA mismatch, use
  `.venv-sft/bin/python` directly.
- Training data: a directory with `train.jsonl` and `eval.jsonl` in Zeta-2
  text format. The default is `/mnt/h/sepalith/datasets/sft_v1`, built by
  `../post-processing/format_sft_v1.py`. Pass another directory as the third
  argument if your data lives elsewhere.
- GPU: `train_sft.py` runs `nvidia-smi` first. It aborts when more than
  8 GB is in use. That is the shared-machine policy; do not bypass it.
- `export_gguf.py` needs a llama.cpp checkout at `/tmp/llamacpp-src/` (for
  `convert_hf_to_gguf.py`) and `llama-quantize` at
  `experiments/bin/llama/llama-b10453/llama-quantize`. The binary is not in
  git. The script writes into `experiments/models/` through an absolute
  checkout path, so the repo must sit at
  `/home/m0hawk/Documents/Sepalith` or that path must exist.

## Run it

1. Fine-tune. Both extra arguments are optional.

   ```bash
   uv run python experiments/training/train_sft.py openbmb/MiniCPM5-1B 3000
   ```

   Consumes: `train.jsonl` and `eval.jsonl` from the data directory
   (default `/mnt/h/sepalith/datasets/sft_v1`). Produces: checkpoints every
   1000 steps, `final_lora/`, and one smoke generation on an eval prompt,
   all under `/mnt/h/sepalith/runs/sft_v1_<model-name>/`.

2. Point at a different dataset.

   ```bash
   uv run python experiments/training/train_sft.py <model_id> <steps> <data_dir>
   ```

3. Merge and export for llama.cpp.

   ```bash
   uv run python experiments/training/export_gguf.py <model_id> <lora_dir> <out_stem>
   ```

   Consumes: the `final_lora` directory from step 1 and the tools listed
   above. Produces: `experiments/models/<out_stem>-Q8_0.gguf` (the f16
   intermediate is deleted).

4. Serve and evaluate.

   ```bash
   <path-to>/llama-server -m experiments/models/<out_stem>-Q8_0.gguf \
     --port 18080 -c 8192 --host 127.0.0.1
   ```

   Then run `../eval/run_eval.py --port 18080 --model zeta2 --examples ...`
   against the same eval set used in training.

## How it works

| Script | What it does |
|---|---|
| `train_sft.py` | LoRA fine-tune with unsloth. Takes a model id, a step count, and a data directory. Saves checkpoints to the NAS store. |
| `export_gguf.py` | Merges a trained LoRA, converts to GGUF, and quantizes to Q8_0 for llama.cpp serving and eval. |
| `rl/` | Verifiable-reward training scaffold. See its README. |

## Notes

- Fixed settings in `train_sft.py`: LoRA r=32, alpha 64, all projection
  layers, learning rate 2e-4 cosine, sequence length 2048, at most 48,000
  train rows, 500 eval rows, seed 3407.
- Checkpoints are written every 1000 steps (last two kept), but the script
  does not pass `resume_from_checkpoint`. A restart begins from step 0.
- Both scripts print progress. `train_sft.py` ends with a smoke generation
  so you can see at a glance whether the model learned the format.
- `rl/README.md` describes the reward-training scaffold. Nothing in `rl/`
  trains yet; the environments and the judge calibration exist.
