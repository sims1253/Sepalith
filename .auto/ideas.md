# Ideas backlog (prefill autoresearch)

- `--cache-reuse 64` (smaller min chunk): accept-events' identical suffix chunks are often <256 tokens; may enable tail reuse — but tail contains changed region; likely still no match. Low priority.
- Warm-up policy variants to A/B: "full" on settle events only (open/accept/diag/cursor/switch) vs eager-everything (incl. type). Cost/benefit via warmup_total_s.
- `--ub 2048`: +5-6% pp2048 in llama-bench; numerics-changing (ubatch split) — only if a big-win config already breaks parity.
- `-ctk/-ctv q8_0`: SLOWER prefill (61.8 vs 85.4 pp2048) + numerics — reject for CPU.
- `--ctx-checkpoints`/`--checkpoint-min-step`: switch_back already ~free via default KV pool; test `--cache-ram 0` (disable pool) to quantify what the default buys (report note for extension: keep defaults).
- `--poll 0` vs 50: untested; tiny expected effect on TTFT overhead. Maybe one session run.
- CPU pinning `--cpu-range` + strict: helps under contention; laptop has fewer cores — report-only.
- Multi-slot `--parallel 2` + per-file slot pinning for file switches: complexity + memory; likely unnecessary given switch_back is free and switch warm-up works.
- Layout proposal simulation (token-LCP cached-fraction under PRM03 vs volatile-last orderings) — pure-python, for the report.
- 4-thread numbers for final config (laptop-like) — required by brief.
- llama-bench quant-type prefill table (bf16/Q8_0/Q4_K_M/Q5/Q6/imat tiers) at 8T and 4T — for the quant mission.
