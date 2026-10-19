# Speculative-decoding autoresearch (lossless CPU spec-dec)

`freeze_traces.py` freezes the 221-row TRAIN trace set (never DEV/final) into
`traces.jsonl`. The file is gitignored (`*.jsonl`): regenerate it and compare
the per-row sha256 values with `manifest.json`. `build_dev75_traces.py` does
the same for the one-shot DEV75 speed panel (`dev75_manifest.json`).
`bench_spec.py --config configs/<name>.json` runs the paired baseline/candidate
llama-server bench, interleaved per trace. Use `--model`, `--server-bin` and
`--out-root` to override the local defaults. `sim_draft.py` is the offline
draft/accept simulator. `parity_repro.py` is the minimal repro of the
verify-batch numerics flip that blocks the lossless gate. `patches/` holds the
experimental llama.cpp cascade patch. `cross_build.py --old-bin B --new-bin B`
runs two llama.cpp builds side by side (no-spec, plus spec arms unless
`--no-spec`) for upgrade evaluations (report
`docs/validation/2026-10-19-llama-upgrade-eval.md`).

Report: `docs/validation/2026-10-18-specdec-autoresearch.md`. Champion:
`configs/ngsimple-n2m4.json`.
