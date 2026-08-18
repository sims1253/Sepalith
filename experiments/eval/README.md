# Sepalith

An open, R-specialized next-edit-suggestion model. It runs on your machine,
serves through llama.cpp, and never sends your code anywhere.

R users in pharma and biostatistics cannot use cloud autocomplete: their
compliance rules block it. Sepalith targets that gap. A small model makes
local inference fast enough to feel instant.

## Status

Research phase. The data pipeline, evaluation harness, and first fine-tunes
run end to end. The design doc and research notes stay private; the tools
are here.

## Repository

```
experiments/
  data-mining/       collect R code and real edit history
  synthetic-data/    generate training examples, verified at birth
  post-processing/   normalize, license-tag, assemble datasets
  eval/              measure edit quality and latency
  training/          LoRA SFT, GGUF export, RL scaffold
```

See `experiments/README.md` for how the parts fit together.

## Dataset

The training data lives at
[huggingface.co/datasets/scholzmx/sepalith](https://huggingface.co/datasets/scholzmx/sepalith).
Every record carries its source URL and license. One package, one file:
a takedown is a file deletion.

## Run it

```bash
uv sync
uv run python experiments/data-mining/ingest_cran.py 100
```

## License

Apache 2.0. See `LICENSE`.
