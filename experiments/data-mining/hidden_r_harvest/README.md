# Hidden-R harvest

Recovers R instruction rows hiding inside general code-instruction datasets.
Ling-Coder-SFT ships a `languages` field, so its R rows are a plain filter
(64,249 kept, per `ling_stats_tmp.json`). CodeX-7M-Non-Thinking has no
language field, so a two-stage detector was tuned by hand and run over all
227 parquet shards (13,380 detected, per `codex_stats.json` / the committed
`codex_filter.log`). Everything is CPU-only and writes incrementally to the
NAS under `/mnt/h/sepalith/datasets/hidden_r_instruction_v1/`, one record
per row with a provenance block (`source_url`, `license`, `derived_from`,
`harvester`/`detection`). `post-processing/normalize_external.py` later
normalizes the harvested `ling_coder_r.jsonl` / `codex_r_strict.jsonl`.

| Script | What it does | Usage | Inputs -> Outputs |
|---|---|---|---|
| `download.py` | `snapshot_download` of a dataset's `data/*.parquet` shards into the local HF cache. | `uv run python download.py <repo_id>` | HF hub -> `~/.cache/huggingface/hub/datasets--<owner>--<name>/` |
| `r_detect.py` | The two-stage detector (module): stage 1 = cheap regex prefilter (R fence tag, strong R token, or any `<-`); stage 2 = fenced-block confirmation comparing R mass vs other-language mass plus a weighted R token score. | imported by `filter_codex.py` / `tune_codex.py` | - |
| `probe_codex.py` | Recon over shards spread across the dataset: fence info-string histogram + R-prefilter hit rate. | `uv run python probe_codex.py [n_shards=3]` | CodeX parquet -> stdout stats |
| `tune_codex.py` | Runs the detector over sample shards and dumps accepted and stage-2-rejected rows for manual inspection. | `uv run python tune_codex.py [n_shards=3]` | CodeX parquet -> NAS `_tune_accepted.jsonl`, `_tune_rejected.jsonl` |
| `filter_codex.py` | Full CodeX scan with the detector; incremental writes, stats checkpointed every 10 shards. | `uv run python filter_codex.py` | CodeX parquet -> NAS `codex_r.jsonl` + `codex_stats.json` |
| `filter_ling.py` | Scans Ling-Coder-SFT for rows whose `languages` list contains `"R"`; appends, flush + `fsync` every 1000 rows. | `uv run python filter_ling.py [start_shard=0]` | Ling parquet -> NAS `ling_coder_r.jsonl` + `ling_stats_tmp.json` |

Notes:

- `filter_ling.py` resumes by shard index; the prior-run counts (52,860
  detected through shard 20) are carried as constants, verified from the log.
- `filter_codex.py` rewrites its output from scratch each run; only the
  stats file is checkpointed mid-run.
- Stage 2 exists because `<-` alone is not R-specific (other languages use
  it too); the fence-aware confirmation is what keeps precision up.
- `codex_download.log` / `ling_download.log` are the committed run logs
  (download timings plus the final shard/detected counts).
