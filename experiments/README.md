# Experiments

The tools behind Sepalith. Each folder is one stage of the pipeline. Run
them in order to build a dataset from scratch.

| Folder | Stage |
|---|---|
| `data-mining/` | Collect R code and real edit history. |
| `synthetic-data/` | Generate training examples. A three-layer gate checks every record before it is written. |
| `post-processing/` | Normalize code, attach licenses, assemble training mixtures. |
| `eval/` | Measure edit quality and latency. |
| `training/` | Fine-tune, export to GGUF, serve in llama.cpp. `rl/` holds the reward-training scaffold. |

## Conventions

- Data never enters git. It lives on the NAS store and on the HF dataset
  repo.
- Credentials come from the environment: `HF_TOKEN`, `ZAI_API_KEY`,
  `OPENCODE_API_KEY`, `OPENROUTER_API_KEY`.
- Setup: run `uv sync` from the repo root. Then run any script with
  `uv run python <path>`.
- GPU scripts check `nvidia-smi` first. They abort if the card is busy.
- Latency numbers from a shared machine are pessimistic. Re-bench on a
  quiet machine before you cite them.
