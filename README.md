# Sepalith

An open, R-specialized next-edit-suggestion model: local-first (llama.cpp/GGUF),
privacy-preserving, aimed at R's center of gravity in pharma/biostatistics where
cloud autocomplete is compliance-blocked. Served through editor integrations
(VS Code / Positron inline completion; Zed via its OpenAI-compatible
edit-prediction provider).

Status: research + data-pipeline phase. See `DESIGN.md` for the product design
and kill-test plan.

## Components

- `experiments/pipeline/` — CRAN ingestion: download-rank via mirror logs, tarball
  fetch, `air format` + `jarl check --fix` normalization, provenance/license
  extraction, one-shard-per-package dataset layout (takedown = delete shard),
  HF dataset push, token-budget estimation.
- `experiments/synthetic/` — synthetic-data generation: coverage grid
  (analyst-style R), z.ai glm-5.3 client with JSON structured output, 3-layer
  validation gate (jsonschema -> R parse -> jarl), thinking-level experiment
  harness, and the tree-sitter-r finish-block extractor (roxygen+signature ->
  function body).
- `experiments/stage0-latency/` — CPU latency benchmarks: llama-bench matrix and
  a keystroke-to-suggestion simulator (cold vs warm prefix-cache).
- `experiments/stage0b-niche/` — held-out edit-replay evaluation: example
  construction from real git commits, official Zeta-1/2/2.1 prompt formats,
  scoring with copy-from-context baseline and bootstrap CIs.
- `experiments/stage1-data/` — mined-example audit tooling.

## Data

Datasets live on a private HF repo (`sepalith-cran`) and a local NAS store,
including per-package provenance, license texts, and derived finish-block pairs.
No credentials are stored in this repository; tools read `HF_TOKEN` /
`ZAI_API_KEY` from the environment.

## Resource policy

This machine is shared with other workloads: GPU inference only when explicitly
free, CPU-heavy jobs capped (≤8 threads) and `nice`d, latency numbers measured
under contention are labeled pessimistic.
