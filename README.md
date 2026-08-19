# Sepalith

An open, R-specialized next-edit-suggestion model: local-first (llama.cpp/GGUF),
privacy-preserving, aimed at R's center of gravity in pharma/biostatistics where
cloud autocomplete is compliance-blocked. Served through editor integrations
(VS Code / Positron inline completion; Zed via its OpenAI-compatible
edit-prediction provider).

Status: research + data-pipeline phase. See `DESIGN.md` for the product design
and kill-test plan.

## Components

All work lives under `experiments/`, one directory per pipeline stage. Each
stage has a README that lists its inputs and how to run it.

- `experiments/data-mining/` — corpus building: CRAN ingestion with download
  ranking, repo selection and cloning, git edit-pair mining, and the
  hidden-R harvest from general code datasets.
- `experiments/synthetic-data/` — synthetic generation: the tree-sitter-r
  finish-block extractor, programmatic scenario families (rename, pipe,
  format, doc-sync), LLM comment-to-code and analyst-script generators, and
  the judge-validation harness.
- `experiments/post-processing/` — dataset finishing: `air format` + `jarl`
  normalization, provenance and license enrichment, dedup, SFT mixture
  assembly, and the HF dataset push.
- `experiments/eval/` — the edit-prediction harness: Zeta-1/2/2.1 prompt
  rendering, midtyping construction, keystroke latency simulation, and the
  conditioning-ablation scorer.
- `experiments/training/` — LoRA SFT with unsloth, GGUF export, and the RL
  environment stubs.

## Data

Datasets live on a private HF repo (`sepalith-cran`) and a local NAS store,
including per-package provenance, license texts, and derived finish-block pairs.
No credentials are stored in this repository; tools read `HF_TOKEN` /
`ZAI_API_KEY` from the environment.

## Python environment

Managed with [uv](https://docs.astral.sh/uv/): `uv sync` creates `.venv` from
`pyproject.toml` + `uv.lock`. Run tools via `uv run python <script>`.

## Resource policy

This machine is shared with other workloads: GPU inference only when explicitly
free, CPU-heavy jobs capped (≤8 threads) and `nice`d, latency numbers measured
under contention are labeled pessimistic.
