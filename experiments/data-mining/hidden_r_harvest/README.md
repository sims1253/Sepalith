# Hidden-R harvest

This stage recovers R instruction rows that hide inside general
code-instruction datasets.

Ling-Coder-SFT ships a `languages` field, so one plain filter keeps its R
rows: 64,249 kept (per `ling_stats_tmp.json`). CodeX-7M-Non-Thinking has no
language field, so a two-stage detector was tuned by hand and run over all
227 parquet shards: 13,380 detected (per `codex_stats.json` and the committed
`codex_filter.log`).

Products, on the NAS store under
`/mnt/h/sepalith/datasets/hidden_r_instruction_v1/`: `ling_coder_r.jsonl`,
`codex_r.jsonl`, and stats files. One record per row, each with a provenance
block (`source_url`, `license`, `derived_from`, `harvester`/`detection`).
Everything is CPU-only and writes incrementally.
`../../post-processing/normalize_external.py` normalizes both files later.

## Before you start

- Run `uv sync` from the repo root. `pyarrow` comes with the lock.
- Environment variables: none. Both datasets are public on HF.
- Disk and time: the full CodeX scan is CPU-only and took about 88 minutes
  (5,270 s) on the dev machine. The downloads took 446 s (CodeX) and 100 s
  (Ling).
- The two filters write to
  `/mnt/h/sepalith/datasets/hidden_r_instruction_v1/`. That path is
  hardcoded. Without a writable `/mnt/h/sepalith`, the filters cannot run.
  Make it a symlink to your storage if you need to move it.
- `probe_codex.py` and `tune_codex.py` go further: they hardcode the dev
  machine's cache path
  `/home/m0hawk/.cache/huggingface/hub/datasets--Modotte--CodeX-7M-Non-Thinking/...`.
  On any other machine, adjust the glob or link that path to your HF cache.

## Run it

1. Download the CodeX shards.

   ```bash
   uv run python experiments/data-mining/hidden_r_harvest/download.py Modotte/CodeX-7M-Non-Thinking
   ```

   Consumes: the HF hub. Produces: the parquet shards under
   `~/.cache/huggingface/hub/datasets--Modotte--CodeX-7M-Non-Thinking/`.

2. Download the Ling shards.

   ```bash
   uv run python experiments/data-mining/hidden_r_harvest/download.py inclusionAI/Ling-Coder-SFT
   ```

   Produces: the shards under
   `~/.cache/huggingface/hub/datasets--inclusionAI--Ling-Coder-SFT/`.

3. Probe the CodeX shards (optional). Recon before you commit to a full
   scan: fence info-string histogram and prefilter hit rate.

   ```bash
   uv run python experiments/data-mining/hidden_r_harvest/probe_codex.py 3
   ```

   Consumes: three shards spread across the dataset. Produces: stats on
   stdout.

4. Tune the detector on a sample (optional).

   ```bash
   uv run python experiments/data-mining/hidden_r_harvest/tune_codex.py 3
   ```

   Consumes: sample shards. Produces: `_tune_accepted.jsonl` and
   `_tune_rejected.jsonl` in the NAS output directory, plus printed samples
   for manual inspection.

5. Filter Ling. The argument is the shard index to start from; the default
   is 0.

   ```bash
   uv run python experiments/data-mining/hidden_r_harvest/filter_ling.py
   ```

   Consumes: the Ling shards in the HF cache. Produces: appends to
   `ling_coder_r.jsonl` (flush and `fsync` every 1000 rows) and rewrites
   `ling_stats_tmp.json`.

6. Filter CodeX.

   ```bash
   uv run python experiments/data-mining/hidden_r_harvest/filter_codex.py
   ```

   Consumes: all CodeX shards. Produces: `codex_r.jsonl` (rewritten from
   scratch each run) and `codex_stats.json` (checkpointed every 10 shards).

7. Normalize the harvest.

   ```bash
   nice -n 19 uv run python experiments/post-processing/normalize_external.py --only ling,codex
   ```

   Consumes: the two JSONL files above. Produces: normalized copies with
   `code_original` kept. See that stage's README for details.

## How it works

| Script | What it does | Usage | Inputs to outputs |
|---|---|---|---|
| `download.py` | `snapshot_download` of a dataset's `data/*.parquet` shards into the local HF cache. | `uv run python download.py <repo_id>` | HF hub to `~/.cache/huggingface/hub/datasets--<owner>--<name>/` |
| `r_detect.py` | The two-stage detector (module). Stage 1 is a cheap regex prefilter: an R fence tag, a strong R token, or any `<-`. Stage 2 confirms inside fenced blocks: R mass against other-language mass, plus a weighted R token score. | imported by `filter_codex.py` / `tune_codex.py` | - |
| `probe_codex.py` | Recon over shards spread across the dataset: fence info-string histogram plus R-prefilter hit rate. | `uv run python probe_codex.py [n_shards=3]` | CodeX parquet to stdout stats |
| `tune_codex.py` | Runs the detector over sample shards and dumps accepted and stage-2-rejected rows for manual inspection. | `uv run python tune_codex.py [n_shards=3]` | CodeX parquet to NAS `_tune_accepted.jsonl`, `_tune_rejected.jsonl` |
| `filter_codex.py` | Full CodeX scan with the detector; incremental writes, stats checkpointed every 10 shards. | `uv run python filter_codex.py` | CodeX parquet to NAS `codex_r.jsonl` + `codex_stats.json` |
| `strict_filter.py` | Reconstructed strict cut of `codex_r.jsonl` (R fence or strong tokens); writes `codex_r_strict.rebuilt.jsonl`, never the authoritative file. | `uv run python strict_filter.py` | `codex_r.jsonl` to NAS `codex_r_strict.rebuilt.jsonl` |
| `filter_ling.py` | Scans Ling-Coder-SFT for rows whose `languages` list contains `"R"`; appends, flush and `fsync` every 1000 rows. | `uv run python filter_ling.py [start_shard=0]` | Ling parquet to NAS `ling_coder_r.jsonl` + `ling_stats_tmp.json` |

## Notes

- Stage 2 exists because `<-` alone is not R-specific; Bluespec and Verilog
  use it too. The fence-aware confirmation is what keeps precision up.
- `filter_ling.py` resumes by shard index. The prior-run counts (52,860
  detected through shard 20) are carried as constants in the script and were
  verified from the log. Pass the shard index you stopped at.
- `filter_codex.py` rewrites its output from scratch each run. Only the
  stats file is checkpointed mid-run. Expect the counts to replay exactly:
  the detector is deterministic and the shard order is sorted.
- `codex_r_strict.jsonl` (the cut `normalize_external.py` consumes) is built
  by `strict_filter.py` — a reconstruction of the rule documented in the
  research notes ("R fence OR >=2 strong tokens"). The original 2026-08-18
  cut (9,661 of 13,380 rows) was made by a script that was never committed;
  the reconstruction keeps 11,246 rows and writes
  `codex_r_strict.rebuilt.jsonl` by default so it cannot silently replace
  the authoritative file. Review the delta before using a rebuilt cut.
- `codex_download.log`, `ling_download.log`, and `codex_filter.log` are the
  committed run logs: download timings and the final shard and detection
  counts.
