# Experiments

Research and tooling behind Sepalith. CPU-friendly by default; GPU jobs
(training/) check `nvidia-smi` before launch on shared machines.

| Directory | What it is |
|---|---|
| `data-mining/` | CRAN ingestion (rank → tarball → air/jarl normalize → shards), GitHub edit-pair mining, PR-review mining, hidden-R harvest |
| `synthetic-data/` | Generators with a 3-layer gate (schema → R parse → jarl): analyst scripts, comment-to-code, paper→R (simulation-verified) |
| `post-processing/` | Provenance enrichment, external-code normalization, style tagging, dataset assembly, HF push |
| `eval/` | Held-out edit replay (Zeta prompt formats, midtyping), copy-baseline scoring, keystroke latency simulator |
| `training/` | LoRA SFT + GGUF export; `rl/` for verifiable-reward training |

Data lives on the NAS store and a private HF dataset repo — never in git.
Credentials read from the environment. Setup: `uv sync` from the repo root.
