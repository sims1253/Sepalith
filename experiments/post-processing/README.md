# Post-processing

This stage prepares datasets after mining or generation. It normalizes R code
with `air format` + `jarl check --fix`, attaches `source_url` and `license`
to every record, tags style, and assembles the SFT train/eval mixtures.

Products: dataset files rewritten in place (atomically, after verification),
license texts under `provenance/`, the HF dataset repo, and
`/mnt/h/sepalith/datasets/sft_v2/{train,eval}.jsonl`.

## Before you start

- Run `uv sync` from the repo root.
- Tools on PATH: `air` and `jarl`. Both run CPU-only, under `nice`.
- Environment variables: `HF_TOKEN` is required only by `push_hf.py`.
- Every script reads datasets that earlier stages wrote to the NAS store.
  Without `/mnt/h/sepalith`, no script in this stage can run. To run the
  stage, first run data mining and synthetic generation against that store.
  No script here honors `SEPALITH_ROOT`.
- `format_sft_types.py` needs the `ry` binary at
  `/home/m0hawk/Documents/ry-worktrees/dump-types/target/release/ry`. That
  path is outside this repo.
- `estimate_tokens.py` needs `llama-tokenize` at
  `experiments/bin/llama/llama-b10453/llama-tokenize` and the tokenizer model
  `experiments/models/qwen0.5b-q8_0.gguf`. Neither is in git.

## Run it

Run the steps in order. Each one consumes the outputs of the step before it.

1. Normalize R code inside harvested and synthetic records.

   ```bash
   nice -n 19 uv run python experiments/post-processing/normalize_external.py
   ```

   Consumes: `hidden_r_instruction_v1/ling_coder_r.jsonl`,
   `codex_r_strict.jsonl`, `synthetic_analyst_v1/analyst_scripts.jsonl`, and
   `paper_to_r_pilot/examples.jsonl` on the NAS store. Produces: the same
   files with `code_original`, `normalized`, and `dropped_reason` fields.
   Use `--only ling,codex,analyst,paper` to pick targets and `--workers N`
   to set parallelism.

2. Enrich provenance.

   ```bash
   uv run python experiments/post-processing/enrich_provenance.py --dry-run
   uv run python experiments/post-processing/enrich_provenance.py
   ```

   Consumes: the datasets listed in the script plus `provenance/` and the
   git clones on the NAS store. Produces: `source_url`, `license`, and
   (for synthetic records) `full_prompt` on every record. Writes are atomic:
   each file is verified, then replaced.

3. Tag style.

   ```bash
   uv run python experiments/post-processing/style_tag.py
   ```

   Consumes: the NAS edit-pair and scenario files listed in the script.
   Produces: a `style` field (`tidyverse`, `base`, lean variants, `neutral`)
   on every row. You can also pass your own JSONL paths as arguments.

4. Extract license texts.

   ```bash
   uv run python experiments/post-processing/pull_licenses.py
   ```

   Consumes: `provenance/<pkg>.json` and the tarballs on the NAS store.
   Produces: `provenance/<pkg>.license.txt` for each package.

5. Push to the HF hub.

   ```bash
   uv run python experiments/post-processing/push_hf.py
   ```

   Consumes: `HF_TOKEN`, the card text in the script, `provenance/`, and
   `datasets/` on the NAS store. Produces: a private dataset repo named
   `<your-user>/sepalith` (or pass a repo id as the argument). The public
   mirror lives at
   [huggingface.co/datasets/scholzmx/sepalith](https://huggingface.co/datasets/scholzmx/sepalith).

6. Estimate the token budget.

   ```bash
   uv run python experiments/post-processing/estimate_tokens.py
   ```

   Consumes: the package shards and normalized trees on the NAS store, plus
   the tokenizer binary. Produces: per-area token counts and ratios on
   stdout.

7. Assemble the SFT v2 mixture.

   ```bash
   uv run python experiments/post-processing/assemble_sft_v2.py
   ```

   Consumes: all family datasets on the NAS store. Produces:
   `/mnt/h/sepalith/datasets/sft_v2/{train,eval}.jsonl`. `--out <dir>`
   targets another directory. Missing inputs are skipped with a note.

8. Optional: exact and minhash dedup (work in progress).

   ```bash
   uv run python experiments/post-processing/datatrove_dedup.py
   ```

   Consumes: `datasets/packages/*.jsonl`. Produces:
   `datasets/corpus_dedup/` plus a stats report.

## How it works

| Script | What it does |
|---|---|
| `normalize_external.py` | Runs `air format` + `jarl check --fix` on R code inside harvested and synthetic records. Keeps the original and the normalized text. Unparseable blocks stay, tagged with a reason. |
| `enrich_provenance.py` | Adds `source_url` and `license` to every record. Joins CRAN provenance for corpus-derived sets, parses repo licenses for mined sets, reconstructs full prompts for synthetic sets. Templates come from the generator sources by AST, so they cannot drift. |
| `style_tag.py` | Tags each record tidyverse, base, or neutral. A TOSEM study found mixed-style training hurts; the tag lets mixtures stratify. |
| `pull_licenses.py` | Extracts each package's LICENSE file from its tarball into provenance. |
| `push_hf.py` | Pushes shards, manifest, and provenance to a private HF dataset repo. |
| `estimate_tokens.py` | Token-budget estimate for pretraining: stratified sample, exact tokenizer counts, per-area ratios. |
| `format_sft_v1.py` | Renders finish-block records into the Zeta-2 prompt format. |
| `format_sft_types.py` | Builds the type-conditioning ablation pair: same records, with and without a `<filename>types` section from `ry dump-types`. |
| `assemble_sft_v2.py` | Mixes all families into train/eval sets. `--out` to target a directory; skips missing inputs with a note. |
| `datatrove_dedup.py` | Work in progress: exact-seq plus minhash dedup over the corpus. |

## Notes

- `normalize_external.py` is incremental. Progress is checkpointed to
  `<stem>.normalize_state.json` every batch, so an interrupted run resumes.
  If a live appender touches the file mid-run, the replace is aborted and the
  result is kept as `<name>.normalized.jsonl`.
- `enrich_provenance.py` never edits an original directly. It writes
  `<name>.enriched.jsonl`, verifies it, then replaces the original. On
  verification failure the original stays untouched.
- `style_tag.py` rewrites each input file in place after a successful pass.
  Keep a copy if you need the untagged version. `assemble_sft_v2.py` only
  reads its inputs and writes the mixture.
- Resource politeness: run the CPU-heavy steps under `nice` (step 1 shows the
  pattern). `assemble_sft_v2.py` is single process, no GPU.
- `push_hf.py --finish-block` is subject to the stale path above.
