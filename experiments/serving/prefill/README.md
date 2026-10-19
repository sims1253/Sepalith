# Prefill autoresearch (time-to-suggestion on CPU llama.cpp)

`build_sessions.py` freezes realistic PRM03 edit sessions from the DAT10 TRAIN
context sidecar (never DEV/final). `bench.py` replays them against a
config-driven llama-server (spawn args + warm-up policy + request flow in
`configs/*.json`) and reports TTFT / cached-fraction / warm-up overhead per
event type. The report, with the experiment log, is
`docs/validation/2026-10-17-prefill-autoresearch.md`. `sessions.jsonl` is not
committed (`*.jsonl` is gitignored): regenerate it with `build_sessions.py`, and
`bench.py` checks it against `manifest.json`.

Laptop runs: `laptop_run.sh` wraps `bench.py` (fixed thermal pre-burn before
each timed run, 30 s host-state sampling, per-run server-log copies, driven by
`SEPALITH_*` env vars), and `pool_runs.py` re-summarizes several interleaved
single-pass runs with exactly the aggregation `bench.py` uses, so an
A/B/A/B sequence reports like one multi-pass run. Winner config:
`configs/warmup-full-eager.json` (desktop 8T); the laptop validation used
`configs/warmup-full-eager-6t.json` and `configs/baseline-extension-6t.json`
(all physical cores on a 6C/12T notebook).

`probe_abort_cancel.py` checks whether a client abort cancels an in-flight
prefill, from server-log evidence (fresh server per scenario, `--batch`
variants); see `docs/validation/2026-10-19-llama-upgrade-eval.md` §2.
