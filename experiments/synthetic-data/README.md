# Synthetic data

This stage generates training examples. Every LLM record passes a three-layer
gate at generation time: jsonschema, R parse, jarl lint (hard fail at five or
more warnings). Records carry the full prompt, the model name, and a
timestamp. After generation, `../post-processing/normalize_external.py` runs
`air format` + `jarl check --fix` on every code block and keeps both versions
(`code_original`, `normalized`).

Products: JSONL family files plus stats under `/mnt/h/sepalith/datasets/`
(`synthetic_analyst_v1/`, `scenarios_v1/`, `paper_to_r_pilot/`), a
finish-block sample in this directory, and two self-contained test suites.

## Before you start

- Run `uv sync` from the repo root.
- Tools on PATH: `Rscript` and `jarl`. The gate calls both on every candidate.
- Environment variables:
  - `ZAI_API_KEY` — glm-5.3 access. Required by `analyst_direct.py`,
    `run_experiment.py`, `judge_validation.py`, and `paper_to_r.py`. Optional
    source in `generate_analyst.py` and `comment_to_code.py`.
  - `OPENCODE_API_KEY`, `OPENROUTER_API_KEY` — optional free-tier fallbacks
    for `generate_analyst.py` and `comment_to_code.py`.
- Corpus: `finish_block.py`, `scenarios.py`, and `comment_to_code.py` read
  `/mnt/h/sepalith/normalized/` and `/mnt/h/sepalith/tarballs/`. Only
  `../data-mining/ingest_cran.py` builds those. Without the NAS store these
  three scripts cannot run. To run them, first ingest CRAN packages with that
  script.
- Output paths are hardcoded to `/mnt/h/sepalith` except: `analyst_direct.py`
  reads `ANALYST_OUT` for its output file (its error log stays on the NAS
  path), `scenarios.py` and `comment_to_code.py` take `--out`, and
  `run_experiment.py` writes into `results/` in this directory.
- The two test files need no corpus, no keys, and no network.

## Run it

1. Run the tests. They check the exactness validators: real examples pass,
   tampered examples fail.

   ```bash
   uv run python experiments/synthetic-data/test_scenarios.py
   uv run python experiments/synthetic-data/test_comment_to_code.py
   ```

   Consumes: nothing outside this directory. Produces: pass or fail on
   stdout.

2. Pilot the LLM generator. This makes `--pilot N` analyst attempts plus
   `N/4` na.rm attempts, then stops.

   ```bash
   uv run python experiments/synthetic-data/generate_analyst.py --pilot 20
   ```

   Consumes: API keys, the grid in `grid.py`. Produces: a small run under
   `/mnt/h/sepalith/datasets/synthetic_analyst_v1/` (`analyst_scripts.jsonl`,
   `na_rm_contexts.jsonl`, `rejects.jsonl`, `stats.json`, `state.json`).

3. Full generation. Restart any time; `state.json` resumes the run.

   ```bash
   uv run python experiments/synthetic-data/generate_analyst.py
   ```

4. Extract finish-block pairs from the corpus (needs the NAS store).

   ```bash
   uv run python experiments/synthetic-data/finish_block.py
   ```

   Consumes: `/mnt/h/sepalith/normalized/`. Produces:
   `finish_block_sample.jsonl` and `finish_block_stats.json` in this
   directory.

5. Build the programmatic edit families (needs the NAS store).

   ```bash
   uv run python experiments/synthetic-data/scenarios.py --packages 200
   ```

   Consumes: normalized trees and raw tarballs. Produces: one JSONL file per
   family (rename, pipe rewrite, na.rm, format, doc-sync) plus `stats.json`
   under `/mnt/h/sepalith/datasets/scenarios_v1/`.

6. Build comment-to-code pairs (needs the NAS store).

   ```bash
   uv run python experiments/synthetic-data/comment_to_code.py --packages 150
   ```

   Consumes: the same corpus. Produces: `comment_to_code_real.jsonl` and
   `comment_to_code_synthetic.jsonl` in the same output directory.

7. Generate paper-to-R examples. One glm-5.3 call per method; the code is
   verified by simulation under `Rscript`, not by text match.

   ```bash
   uv run python experiments/synthetic-data/paper_to_r.py
   ```

   Consumes: `ZAI_API_KEY`, `Rscript`. Produces: `examples.jsonl` and
   `stats.json` under `/mnt/h/sepalith/datasets/paper_to_r_pilot/`.

8. Optional experiments:

   ```bash
   uv run python experiments/synthetic-data/run_experiment.py --dry-run
   uv run python experiments/synthetic-data/run_experiment.py --arms low,high --n 50
   uv run python experiments/synthetic-data/judge_validation.py --n 60
   ```

   `run_experiment.py` consumes `ZAI_API_KEY` and writes
   `results/summary.json` here. `judge_validation.py` consumes eval outputs
   (`../eval/results_zeta2.jsonl` and `../eval/examples.jsonl` by default) and
   writes `results/judge_calibration.jsonl`.

## How it works

| Script | What it does |
|---|---|
| `grid.py` | The coverage grid: domains, package sets, constructs, styles. Diversity lives here, in the inputs. The generator only optimizes output quality. |
| `validate.py` | The three-layer gate shared by all generators. |
| `generate_analyst.py` | Analyst-style scripts. Four API sources, failover, per-source pacing. |
| `analyst_direct.py` | Minimal single-source client (glm-5.3). Writes to `analyst_direct.jsonl`. |
| `comment_to_code.py` | Two variants: real comment-block pairs mined from the corpus, and LLM comments attached to real code blocks. glm-5.3 primary; free tiers as fallback. |
| `comment_drafting.py` | Reverses comment-to-code records into code->comment drafting examples (code visible below the cursor, the comment is the target); writes `comment_drafting.jsonl`. |
| `roxygen_drafting.py` | tree-sitter mining of the corpus: one record per rich roxygen block (>= 1 `@param`/`@return`) and its function — docs are the target at a cursor above the full signature+body suffix; writes `roxygen_drafting.jsonl`. |
| `finish_block.py` | tree-sitter extraction of roxygen + signature to function-body pairs from the corpus. |
| `context_builder.py` | The shared scope-aware context module (spec: docs/prompt-format.md): enclosing-function span, top-level signature outline, and the scope-pin split — one implementation so dataset renderers and the extension cannot drift. Tree-sitter-r only; `context_builder.py <file.R> <cursor_line>` prints the JSON. |
| `scenarios.py` | Programmatic edit families with exact ground truth: rename propagation, pipe rewrite, format propagation, doc-sync, na.rm. Each passes a splice validator and scores 0 on a no-op baseline. |
| `paper_to_r.py` | Statistical method to R implementation, verified by simulation. The validator checks a statistical property — coverage, type-I error, bias — not text. A validator must fail a corrupted twin before use. |
| `run_experiment.py` | Thinking-level comparison harness. |
| `judge_validation.py` | Calibrates glm-5.3 as an edit judge on anchor classes. |

## Notes

- Gate layers: a record fails at the first layer it breaks — `json`,
  `parse`, or `jarl`. Every reject is logged with its layer in `rejects.jsonl`
  so prompts can be iterated.
- Free sources are paced per source (`min_interval` 1 to 6 s) with long
  cooldowns after HTTP 429 (60 to 300 s). Expect the free tiers to stall for
  long stretches; `--wait-on-outage` makes the runner wait instead of exit.
- Dedup guard: whitespace-collapsed code must be unique among accepted
  records, in every generator.
- `paper_to_r.py` has a discrimination check: corrupt a saved example with
  `--corrupt-run <id> --find ... --replace ...` and confirm the validator
  fails the corrupted twin.
- The grid draws core cells without replacement, so a full run covers every
  domain x package x construct cell before it recycles style and length.
