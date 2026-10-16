# Prefill autoresearch (time-to-suggestion on CPU llama.cpp)

`build_sessions.py` freezes realistic PRM03 edit sessions from the DAT10 TRAIN
context sidecar (never DEV/final). `bench.py` replays them against a
config-driven llama-server (spawn args + warm-up policy + request flow in
`configs/*.json`) and reports TTFT / cached-fraction / warm-up overhead per
event type. See `.auto/prompt.md` at the repo root for the running experiment
log, and `docs/validation/2026-10-17-prefill-autoresearch.md` for the report.
