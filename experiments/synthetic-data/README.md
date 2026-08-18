# Synthetic data

Generates training examples. Every record passes a 3-layer gate:
jsonschema, R parse, jarl lint. Records carry the full prompt, the model,
and timestamps.

| Script | What it does |
|---|---|
| `grid.py` | The coverage grid: domains, package sets, constructs, styles. Diversity lives here, in the inputs. The generator only optimizes output quality. |
| `validate.py` | The 3-layer gate shared by all generators. |
| `generate_analyst.py` | Analyst-style scripts. Three API sources, failover, per-source pacing. |
| `analyst_direct.py` | Minimal single-source client (glm-5.3). Writes to `analyst_direct.jsonl`. |
| `comment_to_code.py` | Two variants: real comment-block pairs mined from the corpus, and LLM comments attached to real code blocks. glm-5.3 primary; free tiers as fallback. |
| `finish_block.py` | tree-sitter extraction of roxygen + signature -> function body pairs from the corpus. |
| `scenarios.py` | Programmatic edit families with exact ground truth: rename propagation, pipe rewrite, format propagation, doc-sync, na.rm. Each passes a splice validator and scores 0 on a no-op baseline. |
| `paper_to_r.py` | Statistical method -> R implementation, verified by simulation. The validator checks a statistical property — coverage, type-I error, bias — not text. A validator must fail a corrupted twin before use. |
| `run_experiment.py` | Thinking-level comparison harness. |
| `judge_validation.py` | Calibrates glm-5.3 as an edit judge on anchor classes. |

Run tests: `uv run python test_scenarios.py && uv run python test_comment_to_code.py`.
