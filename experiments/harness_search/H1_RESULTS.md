# H1 — harness-search method bake-off (RESULTS)

Queue: `docs/EXPERIMENT-QUEUE.md` §3 H1. Runbook:
`docs/research/2026-09-04-whale-harness-weight-plan.md` (§1 rig, §2 H1).
Run: 2026-09-05 (started 2026-09-04T23:26+02). Agent: zcode-h1-harness.
Status: RUNNING — this file is finalized at verdict time; sections marked
**[PENDING]** fill from `results/`.

## 1. Rig (experiments/harness_search/)

- **D_harness** (`data/d_harness.jsonl`, 256 rows): TRAIN-side carve from
  `scenarios_v1` under the sft_v3 materialized eval-split authority
  (`/mnt/h/sepalith/datasets/sft_v3/eval.jsonl`). 52 rename_propagation +
  51 each pipe_rewrite / format_propagation / doc_sync / na_rm_propagation,
  file order after a seed-42 shuffle per family, every row validated by
  `scenarios.validate_example` and rendered through the exact
  `assemble_sft_v2.edit_row` (zeta2). Disjoint from the eval split at BOTH
  the package level (plan's rule) and the rendered-prompt level
  (belt-and-braces; unit-tested). `holdout_rule.py` N/A per the plan
  (synthetic constructed families, no CRAN package dimension); the 2%-rule
  verdict per package is recorded as an audit column in
  `data/d_harness_manifest.json` (0 held-out-rule packages in the carve).
- **Search space** (`harness_config.py`): the plan §1.3 table verbatim
  (caps 2000/4000/6000/8000; pin off|2000|4000; outline off|{20,60}×{800,
  1500}; suffix truncation protect-pin|hard-cut; max_tokens 128/320/640;
  stops 7-marker|minimal; parse 3-identical-cut and marker-line-drop
  toggles; comment-line heuristic on/off; full-region-replace on/off).
  Render markers FROZEN (zeta2 canonical). DEFAULT_CONFIG = the banked eval
  convention: scope OFF (`eval_scenarios`' zeta2 render and
  `eval_noop_fp`'s scope=null port both ship without pin/outline), the
  extension's gen/parse defaults (320 tokens, 7 stops, both parse toggles
  on), both post-heuristics on, cap 6000/protect-pin. Unit-pinned:
  default-config render is BYTE-IDENTICAL to `edit_row` prompts and to
  `eval_noop_fp.build_prompt`.
- **Offline knob mapping** (documented semantics; see also §6 confounds):
  the prefix cap truncates from the prefix START (tail kept, the v0.0.6
  rule), budget = cap − suffix chars; an active pin protects its span from
  the cut and drops its own outline entry (doc rule 1), falling back
  outline-only when it cannot fit (buildScope rule); the suffix-direction
  knob distinguishes older-lines-cut-deepest (protect-pin) from
  cut-strictly-at-budget (hard-cut) in the noopFP builder. The two
  post-heuristics map offline as: comment-heuristic prepends "" when the
  cursor line is a non-empty comment and the first prediction is
  code-looking; full-region-replace OFF scores the glue interpretation
  (typed partial + first predicted line) instead of whole-line replace.
- **Scorer** (`scorer.py`): pre-registered lexicographic rule — MAXIMIZE
  validator-exact (`scenarios.validate_example` via the eval_scenarios
  code path) on D_harness with the plan's noise control (2 rollouts per
  candidate-example pair at temp 0; a row passes only if BOTH rollouts
  pass; disagreement = unstable, counts as not-pass); guardrails noopFP
  (scored classes a/c/d) ≤ baseline + 2pp and p95 request wall-time ≤
  1.3× default. Selection key (exact, −unstable, −p95); ties resolve to
  the first-evaluated candidate. Search-time noopFP guardrail set = seeded
  120-row subset of the `eval_noop_fp` construction (its documented
  floor); the verdict battery uses the FULL case set.
- **Serving**: llama-b10453 CPU-only, `-t 8 -ngl 0 --parallel 1 -c 8192`,
  port 18310, tracked PID, POST-completions readiness — the plan's
  convention verbatim EXCEPT `--cache-reuse 1024` (added so identical
  prompts and shared prefixes reuse KV): determinism verified (0/15 text
  mismatches on repeat pairs), all candidates + baselines measured under
  the same mechanism, absolute latencies slightly optimistic vs the
  no-reuse convention (deviation noted; ratios — the guardrail's actual
  form — unaffected).
- **Proposer** (`proposer.py`): backends.py ZaiBackend (glm-5.3) default,
  OpencodeSpark fallback; sees current best + archive (≤8) + per-example
  outcome digest (fail mix + up to 10 failing rows with pred/target heads)
  + the knob-space schema with semantics; M=3 candidates/iteration, STRICT
  JSON; invalid candidates dropped and refilled by seeded fallback
  mutations (logged as origin=fallback). EVERY call ledgered with the API
  usage block (prompt/completion/reasoning tokens) in
  `results/<arm>/proposer_ledger.jsonl`.
- **Arms** (`methods.py`): (a) hill — single lineage, best-so-far, revert
  on regression (guardrail-failing candidates never become best);
  (b) population — archive of all evaluated configs, 2 demes, proposer may
  re-mix archive parents, cross-deme parent choice forced at iterations
  7/10/13; (c) GEPA — render config FROZEN at defaults, evolves only
  instruction_line / checklist_line (outline_header slot inert under
  frozen scope-off, documented to the proposer). 13 iterations × M=3 = 39
  evaluated candidates per arm + the SHARED default-config baseline
  evaluation. Budget accounting: per-candidate completions attributed to
  the first candidate that needed each (prompt, stops, max_tokens) key —
  later hits count as pair reuses (the archive semantics; identical
  requests are not re-spun).
- **Tests** (`test_harness_search.py`, 19 passing): carve
  disjointness+determinism, default render byte-parity (scenarios + noop),
  knob effects, rscan port semantics, parse toggles, both-rollout rule,
  guardrail edge cases (+2pp / 1.3× exactly at the line), selection
  ordering, config validation, pair-cache ledger math, usage
  normalization, fallback determinism, per-candidate budget attribution.

## 2. Baseline (default config on D_harness)

Two-regime note: the run started single-server (18310), then moved to the
two-server sharded regime (18310+18311) for throughput; the baseline was
RE-MEASURED fresh under the co-running regime for guardrail consistency.
exact is text-derived and IDENTICAL across regimes (0.6797 both — a
determinism check); only latencies differ.

| regime | exact | unstable | p95 | mean | noopFP | wall |
|---|---|---|---|---|---|---|
| 1-server | 0.6797 | 0.0117 | 8.76s | 2.04s | 0.7444 | 1117s |
| 2-server (guardrail baseline) | 0.6797 | 0.0117 | 17.75s | — | 0.7444 | 1118s |

Per-family (2-server regime): rename .9231 / pipe .9608 / na_rm .8235 /
format .6863 / doc_sync .0000. Banked v7 HELD-OUT profile (same model,
eval_scenarios battery): rename .947 / pipe 1.0 / format .731 / na_rm .80 /
doc_sync .000 — the rig reproduces the model's behavior profile on the
train-side carve; doc_sync is a model-capability gap (0.0 held-out too), not
a harness artifact, so the search's headroom lives in rename/pipe/format/
na_rm (and in the noopFP/latency guardrail interactions).

Throughput reality (measured, board 01:44): two-server sharding gives ~zero
net gain on this box (decode bandwidth-bound; both servers ~790% CPU);
kept for regime consistency. ETA at full pre-registered budget ≈ 15-20h
wall, dominated by the GEPA arm (every candidate all-novel).

## 3. Search-phase results (D_harness) **[PENDING]**

## 4. Held-out verdict battery **[PENDING]**

## 5. Budget accounting (rollouts + proposer compute) **[PENDING]**

## 6. Verdict (pre-registered) **[PENDING]**

## 7. Confounds and limitations (known before reading §6)

1. **Scenario prefixes are short** (median ~270 chars, max 813): the
   prefix-cap knob (2000–8000) is INERT on D_harness scenario rows by
   construction, and pin/suffix-truncation are near-inert there (they bind
   only via the noopFP guardrail set, whose cases carry real file
   contexts). The code-knob subspace that D_harness can exercise is
   outline/max_tokens/stops/parse/heuristics. This is a property of the
   offline scenario eval, not a rig defect — but it means a GEPA win is
   partially explained by code-knob inertness rather than by text
   superiority alone. H2/H4 (if run) inherit this caveat; a LIVE on-device
   A/B is where the cap/pin knobs actually bite.
2. **noopFP guardrail noise**: at n=120 subset rows (≈96 scored classes)
   the binomial SE on the proposal rate is ~4pp, larger than the +2pp
   line; near-boundary rejections are noisy. The verdict battery's full
   258-case leg (~213 scored, SE ~3pp) is the authoritative read.
3. **The latency guardrail is a ratio on a shared cache-reuse server**:
   absolute p95s are optimistic vs the no-reuse convention; candidate-vs-
   default ratios are internally consistent.
4. **Proposer non-determinism**: glm-5.3 (temp 0.95 per the repo's
   backend) proposes different candidates per run; the arm comparison is
   one draw per method, not an average over restarts (WHALE-style
   replication would need multi-seed arms — out of H1 budget).
