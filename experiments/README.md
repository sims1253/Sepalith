# Experiments

Research and tooling behind Sepalith. Everything here runs on CPU-friendly
budgets by default; GPU jobs (SFT, exports) live in `sft/` and check
`nvidia-smi` before launch on shared machines.

## Layout

| Directory | What it is |
|---|---|
| `pipeline/` | Data collection and processing: CRAN ingestion (rank → tarball → air/jarl normalization → provenance shards), GitHub edit-pair mining, PR-review instructed-edit mining, license/`ry`-type provenance enrichment, external-code normalization, dataset assembly, HF push, token-budget estimation. |
| `synthetic/` | Synthetic data generators with a 3-layer validation gate (jsonschema → R parse → jarl): analyst scripts (coverage grid), comment-to-code (real corpus blocks + LLM comments), paper→R (simulation-verified statistical implementations), programmatic edit scenarios (rename propagation, pipe rewrite, format propagation, doc-sync, ...). |
| `stage0b-niche/` | Held-out edit-replay evaluation: example construction from real git commits, official Zeta-1/2/2.1 prompt renderers, midtyping variant, copy-from-context baseline, bootstrap CIs. |
| `stage0-latency/` | CPU latency benchmarks: llama-bench matrix and a keystroke-to-suggestion simulator (cold vs warm prefix-cache). |
| `stage1-data/` | Mined-example audit tooling (auto-flags + human verdicts). |
| `hidden_r_harvest/` | Detectors that recovered ~74k hidden R rows from general code-instruction datasets. |
| `sft/` | LoRA fine-tuning (unsloth), merge + GGUF export for llama.cpp serving. |

## Data & credentials

Datasets live on a NAS store and a private HF dataset repo — never in git.
All scripts read credentials (`HF_TOKEN`, `ZAI_API_KEY`, `OPENCODE_API_KEY`,
`OPENROUTER_API_KEY`) from the environment.

## Environment

From the repo root: `uv sync`, then `uv run python <script>`.
