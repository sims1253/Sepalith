# H1 — harness-search method bake-off (RESULTS)

Queue: `docs/EXPERIMENT-QUEUE.md` §3 H1. Runbook:
`docs/research/2026-09-04-whale-harness-weight-plan.md` (§1 rig, §2 H1).
Run: 2026-09-04T23:26+02 → 2026-09-05T16:45+02 (~17.3h wall incl. ~5h of
CPU contention from co-running batteries). Agent: zcode-h1-harness.
Weights FROZEN: `sft_v7_minicpm5-Q8_0.gguf` (the banked v7-class serving
GGUF; no B-α winner exists yet — swap per plan §1.2 when it does).

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

## 3. Search-phase results (D_harness, never the verdict set)

Baseline (seed, shared by all arms): exact 0.6797, noop 0.7444, p95 17.75s.

| arm | cands (M=3×13) | guard-passing | best exact | best config / texts | Δ vs baseline |
|---|---|---|---|---|---|
| (a) hill | 39 | 17 | **0.6875** | max_tokens 640 + pin 2000 + cap 8000 | +0.78pp |
| (b) population | 39 | 31 | **0.6875** | max_tokens 640 (fp 67d21c11f63c) | +0.78pp |
| (c) GEPA | 39 | 27 | **0.7109** | instr+checklist text (fp a9e95f441d5b) | +3.12pp |

GEPA's winning texts (guard-passing, noop BELOW baseline at 0.7222):

    instruction_line: Copy the current code verbatim except for the single
      intended edit: one line changed stays one line, a multi-line call keeps
      its exact original breaks, and every added argument appears with a value.
    checklist_line: Check: number of output lines equals the replacement
      length, no call is collapsed onto one line, indentation matches,
      nothing follows the final line.

Search-phase observations (honest method behavior):

- The optimum both code arms converged to is the 640-token decode budget
  (+0.4pp alone); pin/cap add nothing further on D_harness (inert knobs —
  see §7.1) but are harmless on the guardrails.
- Five hill candidates and several GEPA candidates scored 0.6914+ on exact
  and were BLOCKED by the noopFP guardrail (all outline-section configs:
  the outline inflates no-op proposals, +4 to +9pp). The pre-registered
  guardrail is the binding constraint in this space, not the objective.
- Population: 21/39 candidates were seeded-fallback origin — after the
  neighborhood of the best exhausted, glm-5.3 kept re-proposing archive
  duplicates which the dedupe rejected. The 2-deme structure degenerated
  to deme 0 (the best never left deme 0, so cross-deme migration had no
  deme-1 material; the islands were decorative under this budget). The
  archive's value was cache reuse, not diversity.
- Hill kept its lineage discipline: best only moved baseline → 0.6836
  (iter 2, max_tokens 640) → 0.6875 (iter 6); regressions reverted.

## 4. Held-out verdict battery (NEVER D_harness)

eval_scenarios held-out selection (255 rows: 150 rename / 18 pipe / 67
format / 15 doc_sync / 5 na_rm) + FULL noopFP set (258 cases: 204 scored
a/c/d + 54 judgment b) + intent suite (44 rows, glm-5.3 judged, anchors
all OK), one rollout per row at temp 0, all four configs measured in one
uncontended window (16:00-16:45).

| config | scenarios valid | scenarios exact-str | noopFP (a/c/d) | p95 vs default | intent mean (frac 2) |
|---|---|---|---|---|---|
| default (shipped) | **0.8353** | 0.7529 | 0.6961 | 1.00× | 1.5455 (0.7045) |
| hill (mt640+pin2000+cap8000) | **0.8353** | 0.7490 | 0.7010 (+0.5pp) | 1.326× | **1.5682 (0.7273)** |
| population (mt640) | **0.8353** | 0.7490 | 0.7010 (+0.5pp) | 1.240× | 1.5455 (0.7045) |
| gepa (best texts) | 0.8275 | 0.7333 | 0.6912 (−0.5pp) | 1.340× | 1.2955 (0.5909) |

Per-family held-out valid: default = hill = population on every family
(rename .9467 / pipe 1.0 / format .7313 / doc_sync .000 / na_rm .80).
GEPA: rename UP .9467→.9733 but pipe DOWN 1.0→.8333 and format DOWN
.7313→.6866 — the prompt text overfits the carve's family mix.

Latency note: search-time guardrails all passed under their measurement
regimes; the unified-window re-measurement puts the 640-token configs at
1.24–1.34× (the 1.3× line is regime-sensitive; none of them buy held-out
quality anyway).

## 5. Budget accounting (matched at 39 candidates/arm = 19,968 row-rollout-pairs each; fresh server completions + proposer tokens reported per the plan)

| arm | fresh completions | archive reuses | proposer calls (ok) | proposer tokens (p+c, reasoning) | wall |
|---|---|---|---|---|---|
| baseline (shared) | 752 | 0 | — | — | 19 min ×2 regimes |
| (a) hill | 3,896 | 12,560 | 14 | 28,562 (25,148+3,414; 178 reasoning) | ~2.1h |
| (b) population | 2,804 | 13,106 | 15 | 31,234 (27,273+3,961; 350) | ~35 min |
| (c) gepa | 29,016 | 0 | 17 (incl. 2 repairs) | 33,833 (27,909+5,924; 613) | ~12.7h |
| verdict battery | ~4,700 singles | — | — | intent judge ~180 glm calls | ~1.7h |

All proposer traffic went to zai glm-5.3 (fallback spark never fired).
Cache-reuse semantics: a (prompt, stops, max_tokens) key's completions are
charged once to the first arm/candidate needing it; later hits are archive
reuses. GEPA has zero reuses by construction (every candidate re-renders
every prompt).

## 6. Verdict (pre-registered rule: winner = best held-out exact at matched rollouts + proposer tokens)

1. **Method winner for the H2/H4 slot: (a) hill-climb, by tie-break.**
   hill and population tie exactly on the primary metric (held-out
   validator-exact 0.8353 each, identical per-family); GEPA is −0.78pp.
   Tie-breaks: hill spent 8.5% fewer proposer tokens (28,562 vs 31,234),
   and won the intent leg (frac_2 0.7273 vs 0.7045, +4.5pp over
   population/default). Margins vs (c): +0.78pp scenarios valid, +0.25
   intent mean, +13.6pp intent frac_2, and 16% fewer proposer tokens than
   GEPA.
2. **The flat branch fires: no method beat the DEFAULT config held-out**
   (hill 0.0pp, population 0.0pp, GEPA −0.78pp). Per the plan's
   pre-registration: the extension-only space is already near-optimal for
   this frozen θ (v7-class weights) ⇒ H2's regime question is moot until
   weights move; record and skip to H3-S0 / H5-S0 / H4-decision points.
3. **(c) did NOT win ⇒ the space is NOT prompt-bound.** No scope shrink,
   no code-search-line closure on this evidence. The FST-analogue control
   lost held-out DESPITE winning the harness set by +3.12pp — the
   D_harness→held-out transfer gap is the headline methodological finding:
   a 256-row harness set at temp-0 with the 2-rollout agreement rule STILL
   lets prompt-text search overfit (train +3.12pp inverts to −0.78pp).
   Consequence for H4: any harness-phase winner must be confirmed on a
   held-out slice (or a larger D_harness) before adoption; the rig already
   supports this (`verdict_battery.py`).
4. **Recommended H2/H4 config**: harness baseline UNCHANGED (shipped
   defaults; max_tokens stays 320 — 640 buys nothing held-out and sits at
   the 1.24–1.34× latency line). Search machinery for any future harness
   phase: single-lineage hill-climb, M=3, ~13 iters, guardrails as built.
   Treat max_tokens/stops as θ-coupled: re-search only after weight moves
   (B-α winner, H4 SFT steps).

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
5. **Contention windows**: co-running batteries (B13 chain) contended the
   CPU for ~5h mid-run; exact scores are unaffected (text-deterministic)
   but latency measurements span different contention regimes. The §4
   battery re-measured all four configs in ONE uncontended window, which
   is the authoritative latency column.
6. **Held-out family mix ≠ D_harness mix** (rename is 59% of held-out vs
   20% of D_harness; doc_sync 15 vs 51 rows): the D_harness→held-out gap
   partially reflects mix differences, not only overfitting. Both effects
   argue the same direction (adopt nothing without held-out confirmation).
