# Evaluation

This stage measures edit quality and latency of next-edit-suggestion models.
You serve a GGUF model with llama.cpp `llama-server`. The harness renders each
example in the Zeta-1, Zeta-2, or Zeta-2.1 prompt format (as published by Zed
Industries; background: [edit prediction at Zed](https://zed.dev/blog/edit-prediction)),
sends temperature-0 completions, and scores the predicted region against
ground truth.

Products: one JSON row per example plus an aggregate JSON block on stdout, a
sanity check against the official published sample pair, and cold/warm
latency numbers from the keystroke simulator.

## Before you start

- Run `uv sync` from the repo root.
- A `llama-server` binary and a GGUF model. The dev machine's copies
  (`experiments/bin/llama/llama-b10453/`, `experiments/models/`) are not in
  git. Build llama.cpp yourself and download a model, for example
  [zeta-2](https://huggingface.co/zed-industries/zeta-2) from HF.
- Examples: `run_eval.py --examples` takes any file in the edit-pair JSONL
  format. Two sources:
  - `/mnt/h/sepalith/datasets/edit_pairs_v1/eval.jsonl` from the NAS store,
    built by `../data-mining/`.
  - `examples.jsonl` in this directory, built by `build_examples.py`. The
    file and the clones it needs (`experiments/.cache/repos/`) are not in
    git; clone R and Python repos into that layout yourself.
- No API keys. The optional `--official` check downloads one sample pair from
  the HF hub, so it needs network access.
- Tools on PATH: `git` (only for `build_examples.py`), `curl` (only for
  `run_sims.sh`).

## Run it

1. Start the server. Wait until it answers on `/health`.

   ```bash
   <path-to>/llama-server -m <model>.gguf --port 18080 -c 8192 --host 127.0.0.1
   ```

   Add `-ngl 99` to put layers on the GPU, or `-t <n>` CPU threads. The
   context must cover prompt plus completion; 8192 is enough for the
   edit-pair examples.

2. Check the harness against the official sample.

   ```bash
   uv run python experiments/eval/run_eval.py --port 18080 --model zeta2 --official
   ```

   Consumes: the `sample.prompt` and `sample.output` files of the
   `zed-industries/zeta-2` HF repo. Produces: one JSON block with exact,
   first-line, and line-F1 scores and latency on stdout.

3. Build examples (skip this if you use the NAS eval file).

   ```bash
   uv run python experiments/eval/build_examples.py experiments/eval/examples.jsonl 8
   ```

   Consumes: git clones at `experiments/.cache/repos/{r,python}/<repo>`.
   Produces: `experiments/eval/examples.jsonl` with up to 8 examples per
   repo.

4. Run the eval. Redirect stdout to keep the rows.

   ```bash
   uv run python experiments/eval/run_eval.py --port 18080 --model zeta2 \
     --examples experiments/eval/examples.jsonl > results_zeta2.jsonl
   ```

   Consumes: the server and the examples file. Produces: one JSON row per
   example (`exact`, `first_line`, `line_f1`, `latency_s`, the prediction)
   and a final aggregate block per language.

5. Resume an interrupted run.

   ```bash
   uv run python experiments/eval/run_eval.py --port 18080 --model zeta2 \
     --examples experiments/eval/examples.jsonl --resume results_zeta2.jsonl
   ```

   Already-scored examples are skipped.

6. Measure latency (needs the `data.table` clone in `.cache`).

   ```bash
   uv run python experiments/eval/keystroke_sim.py --port 18081 --ctx 4096
   ```

   Consumes: the server and R sources from the clone. Produces: one JSON
   line per phase, cold and warm, on stdout.

7. Sweep context sizes with a fresh CPU server each time. Run from the
   `experiments/` directory; the script uses relative paths.

   ```bash
   bash eval/run_sims.sh <model.gguf> 18082 8 /tmp/sims.txt
   ```

   Consumes: the model and `llama-server` at
   `bin/llama/llama-b10453/llama-server` (relative to `experiments/`).
   Produces: one keystroke-sim block per context size (2048, 4096, 8192) in
   the output file.

8. Aggregate (pilot sets only; see Notes).

   ```bash
   uv run python experiments/eval/analyze.py experiments/eval/results_zeta2.jsonl
   ```

   Consumes: results files named `results_<model>.jsonl` next to
   `examples.jsonl`. Produces: per-language metrics, the copy-from-context
   baseline, and a bootstrap confidence interval on the Python-R gap.

## How it works

| Script | What it does |
|---|---|
| `build_examples.py` | Builds next-edit examples from commit diffs in local clones: parent state becomes prefix/region/suffix, a sibling hunk becomes the event. |
| `run_eval.py` | Renders prompts in Zeta-1/2/2.1 format, sends temperature-0 completions, scores regions (exact, first line, line F1). `--official` reproduces the published sample pair. |
| `eval_ablation.py` | Scores whole-region predictions on the conditioning-ablation prompt/target rows; reports the aggregate split by `has_types` (the dropout arm's both-ways eval) and by record kind. |
| `eval_scenarios.py` | Serves a GGUF itself (CPU `-t 8 --parallel 1 -c 8192`; readiness = a real completion POST returning 200; teardown signals only the tracked child PID), renders the held-out rows of the five programmatic scenario families with the training-time zeta2 render (`assemble_sft_v2.edit_row`), and scores them with the exact `synthetic-data/scenarios.py` validators. Writes `results_scenarios_<model>.jsonl`; the per-family aggregate is printed last. |
| `keystroke_sim.py` | Cold versus warm request latency at a given context size, with keystroke-sized deltas on a shared prefix. |
| `run_sims.sh` | Starts a fresh CPU `llama-server` per context size and runs the keystroke sim against it. |
| `analyze.py` | Aggregates result rows per language, computes the copy-from-context baseline, and bootstraps the language gap. |
| `audit.py` | Writes a human-review worksheet with auto-flags for constructed examples. Verdicts go into `verdicts.tsv`. |

## Notes

- `--examples` is a flag, not a positional argument. The usage line inside
  the script shows a positional form that argparse does not accept.
- `--variant midtyping` moves the cursor into a partial line and scores the
  completion suffix. `--align suffix` realigns whole-region answers to that
  suffix. State the variant when you cite numbers.
- Each format gets its own stop sequence (`>>>>>>> UPDATED`, markers), so
  trailing text does not burn tokens.
- `--official --model zeta1` reads `reference/zeta1_row5.json`, which is not
  in the repo. Only the zeta2 official check runs today.
- `analyze.py` assumes the earlier mixed-language pilot set (`lang` is
  `python` or `r`). On R-only examples it fails on the missing Python rows.
  The midtyping and R-scale results are better read from the aggregate block
  of step 4.
- Latency numbers from a shared machine are pessimistic. Re-run on a quiet
  machine before you cite them.
