# Sepalith

An open, R-specialized next-edit-suggestion model: local-first (llama.cpp/GGUF),
privacy-preserving, aimed at R's center of gravity in pharma/biostatistics where
cloud autocomplete is compliance-blocked. Served through editor integrations
(VS Code / Positron inline completion; Zed via its OpenAI-compatible
edit-prediction provider).

Status: active research and editor integration development. Start with the
[project map](docs/PROJECT-MAP.md) for workflows, environments and checks.
Local design notes in `DESIGN.md` are not distributed with a clone.
See [code quality checks](docs/CODE-QUALITY.md) for TypeScript linting, the Python
package gate, and the optional research audit.

## Components

Research work lives under `experiments/`, grouped by pipeline stage and research
family. Each stage has a README that lists its inputs and how to run it. The shared
core lives in [packages/sepalith](packages/sepalith/README.md). Product integration
lives in `extensions/` and builds separately from the Python training stack.

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

## Quick checks

```bash
python3 scripts/check_core.py
```

Core checks use Python’s standard library and need no model weights, NAS,
credentials or GPU. For product development, install editor dependencies once
with `npm --prefix extensions/vscode-sepalith ci`, then run
`python3 scripts/check_product.py`. This uses fake assets and tiny test processes.

## Research Python environment

Managed with [uv](https://docs.astral.sh/uv/): `uv sync` creates `.venv` from
`pyproject.toml` + `uv.lock`. Run tools via `uv run python <script>`.

## Resource policy

This machine is shared with other workloads: GPU inference only when explicitly
free, CPU-heavy jobs capped (≤8 threads) and `nice`d, latency numbers measured
under contention are labeled pessimistic.

## Cleanup implementation

The independent [core package](packages/sepalith/README.md) contains the new local
runner and versioned edit-context/latent-memory contracts. It can be checked without
the training environment. See the [research audit](docs/RESEARCH-AUDIT.md),
[runner cutover](docs/EXPERIMENT-RUNNER.md), and
[learned workspace-memory proposal](docs/LATENT-WORKSPACE-MEMORY.md).
The v5 assembler and paired-statistics audit now use shared modules through
their existing entry points. Managed runtime and packaging tools have separate
product checks. Live experiment dispatch remains with its current manager until
the documented cutover.
