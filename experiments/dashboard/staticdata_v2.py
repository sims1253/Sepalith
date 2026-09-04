#!/usr/bin/env python3
"""Static ground truth for dashboard v2 (glossary, tree, timeline seed, inventory).

Everything in this module is REPO GROUND TRUTH, mined by hand from:

  docs/EXPERIMENT-QUEUE.md            statuses, series definitions, W-numbers
  experiments/training/base_bakeoff/RESULTS.md   B-series numbers, gates B-a/B-b
  experiments/training/poc_diff/RESULTS.md       poc_diff / M1 / D-grid numbers
  experiments/training/rl/O3_S0_RESULTS.md       O3-S0 verdict
  experiments/synthetic-data/TU1_RESULTS.md      TU1 verdict
  docs/research/2026-09-04-p12-roofline-results.md  P12 numbers
  comms/board.md                      live owners/ETAs, 2026-09-04/05 events
  experiments/synthetic-data/cases/specs/*.json  family descriptions
  experiments/synthetic-data/scenarios.py        family semantics
  experiments/training/rl_smoke.py + rl/README   RL pools, quotas, priors
  /mnt/h/sepalith/...                counted live (see inventory())

No numbers are invented here; where a number could not be grounded the
entry says so. The DYNAMIC state (running, timeline appends, next,
decisions) lives in state_v2.json and is patched by muse-spark via
refresh_cycle_v2.py. This module is only read at build time.

The inventory counts are cached (mtime-keyed) under results/ so a 30-min
cycle does not rescan /mnt/h.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE = RESULTS / "inventory_cache_v2.json"
DS = Path("/mnt/h/sepalith/datasets")

# ---------------------------------------------------------------------------
# 1. timeline seed - the verdict history in plain language
# ---------------------------------------------------------------------------
# verdict_class: good | bad | neutral | pending  (render color only)
TIMELINE_SEED = [
    dict(date="2026-08-22", id="sft_v7", title="SFT v7 becomes the serving model",
         ask="Does a quadrupled synthetic program (33 families, 264k rows) buy a better serving model?",
         measured="264,516 train / 9,252 eval rows. Scenario families 0.772 valid / 0.677 exact; rename 0.967/0.883, pipe 1.0/1.0, format 0.700/0.550. Intent layer 1.636, best ever (v6: 1.386). doc_sync stayed at 0.",
         verdict="WIN", verdict_class="good",
         implies="sft_v7_minicpm5-Q8_0.gguf is the model we serve; every later SFT question is measured against it."),
    dict(date="2026-08-22", id="LANDSCAPE", title="v7 measured honestly against the frontier",
         ask="How big is the gap to glm-5.3 really, once format unfamiliarity is controlled?",
         measured="158 identical held-out rows. v7 0.772/0.677 vs glm-5.3 0.424/0.335 zero-shot and 0.601/0.494 with three worked examples. The few-shot control closes about half of glm's deficit.",
         verdict="WIN (17 points, not 35)", verdict_class="good",
         implies="Citation framing is settled: quote the 17/18-point controlled spread, never the zero-shot 35. The residual edge lives in rule-knowledge families (format, pipe, rename)."),
    dict(date="2026-08-22", id="FIM-DOSE", title="FIM dose ladder settled the A2/B2 dispute",
         ask="How much fill-in-the-middle data should pretraining mix in, and does the causal floor ever pay?",
         measured="Four paired arms (0/10/20/35 percent PSM-FIM share, 350.2M tokens each). Doses 0.2 to 0.8 percent over control. Span BPB saturates by 10 to 20 percent; free-running serving collapses at 35 percent; dose 20 best.",
         verdict="ADOPT 20 to 35 percent, center 25 to 30", verdict_class="good",
         implies="A2-prime's mixture arithmetic is evidence-backed end to end. One falsifier stays open: the masked-vs-unmasked replica arm (parked, queue section 2)."),
    dict(date="2026-08-22", id="COMPOUND-FB", title="Compound finish_block attacked the glm corner",
         ask="Can deterministic derivation plus authored functions close the finish_block gap (v7 0.5 vs glm 2.0)?",
         measured="7 registry rules, 39/39 selftests. 9,574 corpus-exact rows from 1,562 base samples; 297/300 pass the real renderer unchanged. Authored wave passed the 400-sample quality gate. Commit e7aec65.",
         verdict="WIN (data landed)", verdict_class="good",
         implies="The one corner where the frontier beats us now has a data program aimed at it: domain-diverse authored functions times cut-point derivations."),
    dict(date="2026-08-23", id="NOOP-FP", title="The no-op false-positive suspicion was confirmed",
         ask="Does v7 propose edits where the right answer is to change nothing?",
         measured="v7 false-suggests on 70.6 percent of decoy prompts (v6: 68.6, so it is structural). 100 percent proposal on every UNTRAINED stop geometry (interior line-end, mid-identifier, blank line, refactorable); about 30 percent on trained geometries.",
         verdict="CONFIRMED (bad news, banked)", verdict_class="bad",
         implies="Restraint is geometry-specific. v8 levers: no_op training on the three untrained in-function geometries and a client-side completeness gate as the cheap immediate fix."),
    dict(date="2026-08-27", id="POC-DIFF", title="Masked diffusion beat autoregression on edit spans",
         ask="At matched harness, does a masked-diffusion twin beat an AR-FIM twin at infilling R edit spans?",
         measured="216-row paired eval. Span exact: MD@32 0.0694 vs AR 0.0000. p95 latency 855ms vs 7,689ms. Both arms 0.000 on the 51-to-256-token bucket (80 rows).",
         verdict="VALIDATED (all three kill-test disjuncts)", verdict_class="good",
         implies="The diffusion line earned its follow-ups. Long spans stayed unsolved and became the D-grid and X-series questions."),
    dict(date="2026-08-29", id="POC-DDOT", title="OT position coupling killed",
         ask="Does an optimal-transport position field help the diffusion head place edits?",
         measured="The position field converges before 200M tokens; value routing poisons quality.",
         verdict="KILLED", verdict_class="bad",
         implies="Family E is closed. X3's length-aux head is the only remaining honest-length path from this line."),
    dict(date="2026-09-01", id="Q1-Q6", title="The contraction question set closed",
         ask="Six pre-from-scratch questions (Q1 to Q6) settled before any big training spend.",
         measured="All DONE 2026-09-01. Verdicts live on the board and in the 2026-08-31 contraction close-out doc; the queue section 1 rows were pruned.",
         verdict="DONE (verdict map in the close-out doc)", verdict_class="neutral",
         implies="The program's scope was cut to what the evidence supports; the queue file became the single lookup for everything after."),
    dict(date="2026-09-02", id="X1", title="Cross-paradigm span eval: SFT arms measure zero",
         ask="Do the serving SFT models (v3/v7/v8_2) carry the span skill on the poc_diff harness?",
         measured="All SFT arms exact 0.0000 vs MD@32 0.0694; the 51-256 bucket is 0.000 paradigm-wide. edit_sim: v3 0.066, v8_2 0.058, v7 0.052, base 0.024.",
         verdict="DONE (measurement closed)", verdict_class="neutral",
         implies="Span-exact is a pretraining-scale skill. The serving models keep their scenario-battery wins; no serving change."),
    dict(date="2026-09-02", id="X4", title="CAL full recipe killed",
         ask="Can the paper's full length-search recipe fix span length prediction?",
         measured="MAE 158.1 (v1 was 190.7), exact 0.0000. Mechanism: the oracle peak needs a visible suffix; our cursor insertion is suffix-free.",
         verdict="KILLED", verdict_class="bad",
         implies="The CAL line is closed at v1 and v2. X3 (length-aux head) remains the only parked salvage."),
    dict(date="2026-09-02", id="M1", title="Micro-specialist probe killed",
         ask="Is a small, curated corpus enough at 76M params (the mdlARC thesis)?",
         measured="Both arms exact 0.0000 vs the 206M/2.0B anchor's 0.0694; the kill test (M1b below 0.0347) fired. Curation delta: +0.024 edit-sim, zero exact.",
         verdict="KILLED", verdict_class="bad",
         implies="Scale and general-corpus exposure are load-bearing at every size tested. No curation term in the W5 arithmetic; the M-series is closed."),
    dict(date="2026-09-02", id="B-SERIES-RUNGS", title="Bake-off rungs B-anchor, B1, B2, B3, B12 measured",
         ask="Which external base classes survive a uniform recipe (3000 steps, sft_v7, LoRA r32a64)?",
         measured="b12_cal anchor passed pre-gate 2 (unsloth vs PEFT delta -3.1pp valid, p=0.039). B1 dense ladder: 76.9/72.2/56.9/12.5 valid at 1.08B/967M/854M/741M. B2 GDN 0.8B: 82.7/73.3. B3 conv+GQA 350M: 15.7 valid, 95.0 noopFP, 94.1 t/s. B12 SWA spark 1.7B: 85.1/77.3 on the TRL path.",
         verdict="MEASURED (gate input)", verdict_class="neutral",
         implies="Enough data for the first architecture verdict; B3's speed cannot buy back its accuracy."),
    dict(date="2026-09-02", id="GATE-B-ALPHA", title="Gate B-alpha: the architecture verdict",
         ask="Which architecture class should the next base come from?",
         measured="SWA 3:1 leads raw (85.1 valid) but at 1.71B with no matched-size rival; GDN at 752M beats dense at 1.08B (82.7 vs 76.9, minus 30 percent params). Dense does not survive as the next-build substrate; conv+GQA eliminated. doc_sync 0/15 on every arm.",
         verdict="SWA leader (size-provisional), GDN conditional on B4, conv out", verdict_class="good",
         implies="B4 (GDN at 2B) and B5 (dense at 3B) became the decisive size controls. doc_sync tilts construction, not capacity."),
    dict(date="2026-09-03", id="D-GRID", title="D-grid: budget is the lever, breadth is not",
         ask="Is the span-exact anchor's edge parameters, training budget, or unique-data breadth?",
         measured="D2 (76M, 2.0B tokens): 0.0602. D1 (206M, 0.5B): 0.0000. D3 (76M, 2.0B tokens over 7x unique data): 0.0000. Size at matched budget buys +0.0092.",
         verdict="DONE: budget/anneal depth is the lever", verdict_class="good",
         implies="At this task size, deep anneal on a small universe beats coverage. The A2 pretraining program (more total tokens, bigger universe) is exactly the untested regime."),
    dict(date="2026-09-04", id="B4-B5", title="Size controls landed: a three-way tie",
         ask="Do GDN at 2B and dense at 3B confirm or kill spark's lead?",
         measured="b4 GDN 1.88B: 85.1/76.5 at 19.2 t/s. b5 granite 3B: 87.8/78.0 at 10.65 t/s (native-FIM format 79.1, best of all arms). McNemar spark-vs-b4 p=1.0000 (exact tie); b5's +2.7pp is p=0.119, not significant.",
         verdict="TIE at the top", verdict_class="neutral",
         implies="Spark's gate-alpha lead was size, not SWA. The classes now separate on decode and product fit, not quality."),
    dict(date="2026-09-04", id="GATE-B-BETA", title="Gate B-beta: the final base verdict",
         ask="Which base do we production fine-tune on?",
         measured="Three-way statistical tie (b5/spark/b4). GDN holds the best validator-at-decode at both sizes (b2 82.7 at 36.2 t/s, b4 85.1 at 19.2). doc_sync 0.0 at every size from 350M to 3B.",
         verdict="GDN wins the product axis; recommendation is b4-config", verdict_class="good",
         implies="The winner track (B8 midtrain, B9 SeleKT, B10 WiSE-FT, W16 pipeline, production fine-tune) is unblocked pending the base-pick GO. B13 (LFM2.5-2.6B) runs first per the user's call to try untried candidates. doc_sync is a data problem: B8 plus the data program own it."),
    dict(date="2026-09-04", id="P12", title="Roofline profiling: kernel-day GO",
         ask="Is CPU decode bandwidth-starved, and do Q4 quants realize their byte advantage?",
         measured="STREAM triad 25 GB/s on this WSL2 box (about 43 bare-metal). Q8/Q6/Q5 decode rides the read ceiling (92-97 percent of Copy, about 34 GB/s). Q4_K_M vs Q8_0 speedup 1.37-1.45x, under the 1.5x theoretical-realization bar at every thread count.",
         verdict="KERNEL-DAY GO via the realization arm", verdict_class="good",
         implies="A GGML Q4 dequant-path kernel day is worth scheduling (prize about +16 percent if closed), behind the 13B gate. CPU-tier quant prior: Q8_0, Q4_K_M, Q4_0."),
    dict(date="2026-09-05", id="O3-S0", title="Suffix-entropy telemetry did not land",
         ask="Does output-position entropy predict RL stall before the exact metric does?",
         measured="The apparent 97-157-step stall lead is a run-family confound (partial r -0.48 to -0.16, ns, when controlled). Banked runs are 220-300 steps vs the papers' 3K-10K+. Retry is hard-blocked by save_total_limit=2.",
         verdict="NOT LAND", verdict_class="bad",
         implies="O3-S1 (curriculum arm) stays closed. What survives: a validated stop/continue instrument (frac_reward_zero_std at 0.70 or above, reward_std at 0.15 or below) and a MetricsCb fix spec for the next RL build window."),
    dict(date="2026-09-05", id="TU1", title="Sufficiency judge-gate dead",
         ask="Can glm-5.3 label which rendered rows have derivable targets, and does that label predict exact outcomes?",
         measured="Pooled gap +56.6pp (Fisher p=1e-6) but direction-consistency 1/5 families vs the required 3/5: the judge rates 92.2 percent of rows SUFFICIENT and 3 of 5 families had zero insufficient rows. Reported as written, no post-hoc rescue. Free readout: doc_sync sufficiency 0.267 and its 4 derivable rows are 0/4 exact.",
         verdict="DEAD per pre-registered rule", verdict_class="bad",
         implies="TU2 runs solve-gated only (arm d cancelled). The doc_sync construction verdict from B-beta is independently re-confirmed and sharpened: broken beyond context sufficiency."),
    dict(date="2026-09-05", id="W29", title="Cloud SFT capability proven",
         ask="Can the repo train on rented GPUs without PAT or baked images?",
         measured="60-step LoRA smoke on Anyscale A10G: loss 1.495 to 1.326, time-to-ready 3.5 minutes, burn 0.55 dollars of a 10-dollar cap. A10G gives 2,746 tok/s (0.63x the 5090). A B13-class rung costs about 6.5-8 dollars.",
         verdict="CLOUD-READY", verdict_class="good",
         implies="About 85 dollars of credit is reserved for the production fine-tune (about 12 B13-class rungs). W1 FP8 does not belong here (57 dollars vs about 7 on the A2 rental)."),
]

# ---------------------------------------------------------------------------
# 2. the experiment tree (genealogy of program lines)
# ---------------------------------------------------------------------------
# status: done | win | killed | gate | running | parked | proposed | blocked
SKILL_TREE = [
    dict(id="contraction", label="Q1-Q6 contraction", status="done",
         note="Six pre-from-scratch questions settled 2026-09-01; scope cut to what evidence supports.",
         children=[
            dict(id="pocdiff", label="POC-DIFF (MD twin vs AR twin)", status="win",
                 note="VALIDATED 2026-08-27: MD@32 span exact 0.0694 vs AR 0.0000; long spans 0.000 both.",
                 children=[
                    dict(id="x1", label="X1 cross-paradigm eval", status="done",
                         note="SFT arms 0.0000 on the span harness; measurement closed.", children=[]),
                    dict(id="x2x3", label="X2 AR-init / X3 length-aux", status="parked",
                         note="Plan-frozen follow-ups; pre-registered kill tests.", children=[]),
                    dict(id="x4", label="X4 CAL full recipe", status="killed",
                         note="MAE 158.1, exact 0.0000; oracle needs a visible suffix.", children=[]),
                    dict(id="dgrid", label="D-grid D1/D2/D3", status="done",
                         note="Budget is the lever (D2 0.0602); breadth loses (D3 0.0000). D-series closed.", children=[]),
                    dict(id="m1", label="M1 micro-specialist", status="killed",
                         note="76M arms 0.0000 vs 0.0694 anchor; scale and general-corpus exposure load-bearing.", children=[]),
                 ]),
            dict(id="pocddot", label="POC-DDOT (OT position field)", status="killed",
                 note="KILLED 2026-08-29: position field converges under 200M tokens; value routing poisons quality.",
                 children=[]),
         ]),
    dict(id="bseries", label="B-series base bake-off", status="done",
         note="External bases under one uniform recipe; two pre-registered gates.",
         children=[
            dict(id="brungs", label="B-anchor, B1 dense ladder, B2 GDN-0.8B, B12 SWA spark", status="done",
                 note="b12_cal pre-gate 2 passed; dense has no floor above 24L; GDN 82.7/73.3 at 752M; spark 85.1/77.3 at 1.71B (TRL path).", children=[]),
            dict(id="b3", label="B3 conv+GQA 350M", status="killed",
                 note="ELIMINATED: 15.7 valid / 95.0 noopFP; its 94.1 t/s cannot buy the gap back. B7 rescue conditional.", children=[]),
            dict(id="balpha", label="Gate B-alpha (architecture)", status="gate",
                 note="SWA leader size-provisional; GDN conditional on B4; dense out as substrate; conv eliminated; doc_sync 0 tilts construction.", children=[]),
            dict(id="b4b5", label="B4 GDN-2B / B5 granite-3B", status="done",
                 note="Exact tie spark-vs-b4 (p=1.0); b5 +2.7pp not significant; granite native-FIM format 79.1 best.", children=[]),
            dict(id="bbeta", label="Gate B-beta (final verdict)", status="gate",
                 note="Three-way tie; GDN wins the product axis; production rec = b4-config.", children=[
                    dict(id="b13", label="B13 LFM2.5-2.6B rung", status="running",
                         note="User call: try untried candidates before the final pick. Blocks the base pick GO.", children=[]),
                    dict(id="winner", label="Winner track: B8 midtrain, B9 SeleKT, B10 WiSE-FT, W16, production fine-tune", status="blocked",
                         note="B8 train_sft patch built and tested (31 tests), not fired. Waits on the base pick.", children=[]),
                 ]),
         ]),
    dict(id="sftline", label="SFT serving line (v-series mixtures)", status="done",
         note="v6 to v7 to v8/v8_2: v7 is the serving champion; the v8 mixture route was closed 2026-08-23 (third data point that mixtures cannot win the serving question; RL owns it).",
         children=[]),
    dict(id="rlline", label="RL line (verifiable-reward GRPO)", status="done",
         note="Runs v1 to v5, t1_dapo banked; no_op reward 0.39 to 0.97 on trained geometries; stop/continue instrument banked by O3-S0.",
         children=[
            dict(id="eseries", label="E-series: environment evolution", status="proposed",
                 note="E1 EL-scheduler patch built (26 tests), GPU run armed; E2 evolver, E3 axes, E4 head-to-head gated behind it.", children=[]),
            dict(id="oseries", label="O-series: OPD transfer to GRPO", status="proposed",
                 note="O1 diverse-16 selection built; O3-S0 telemetry NOT LAND (S1 closed); O2 filter, O4 alignment-price queued.", children=[]),
            dict(id="tuseries", label="TU-series: trajectory to environment", status="proposed",
                 note="TU1 judge-gate DEAD; TU2 runs solve-gated (teacher solve in flight); TU3 cross-file, TU4 multi-round, TU5 budget-split queued.", children=[]),
         ]),
    dict(id="vseries", label="Eval v2 (feel-of-use battery)", status="running",
         note="V1a episode metrics and V1b AST-equivalence landed (+14.9pp format recovery); V1c TTFT bench pending; V1d pairwise preference judging now.",
         children=[]),
    dict(id="hseries", label="H-series: harness x weights (WHALE)", status="running",
         note="H1 search-method bake-off rig built, smoke passed, full pipeline running; H2 to H5 gated behind it.",
         children=[]),
    dict(id="sseries", label="S-series: serving wall-clock", status="proposed",
         note="S0 trace freeze running (doubles as Q7's trace set); S1 spec-decode and S2 quant-cycle proposed, fed by P12's priors.",
         children=[
            dict(id="p12", label="P12 roofline", status="done",
                 note="KERNEL-DAY GO; CPU-tier quant prior Q8_0/Q4_K_M/Q4_0; WSL2 bandwidth tax measured.", children=[]),
         ]),
    dict(id="dataprog", label="Data program (feeds every line)", status="running",
         note="Normalized CRAN corpus, astfim_v1 (276k rows / 220.5M tokens), 40+ family bank, A2 strata program complete (13/13 strata, 7.55B tok).",
         children=[]),
]

# ---------------------------------------------------------------------------
# 3. family glossary (plain-language decode of every synthetic family)
# ---------------------------------------------------------------------------
# source: spec file | scenarios.py | curated (v1 bank notes) | rl_smoke.py
# rows_key joins the live inventory count; example is plain R text.
GLOSSARY_GROUPS = [
    ("Propagation (an edit at one site must reach the next site)", [
        dict(name="rename_propagation", plain="An identifier (argument, local variable, column name) appears three or more times in one function. The user renames one occurrence; the model must rename the next one the same way.",
             example="event: threshold <- cutoff at one site\ntarget: the next  threshold  becomes  cutoff",
             source="scenarios.py family 1", rows_key="rename_propagation"),
        dict(name="na_rm_propagation", plain="Inside a dplyr summarise or mutate call, two or more mean/sd/var calls lack na.rm. The user adds it to one; the model must add it to the next.",
             example="mean(x, na.rm = TRUE)  # the event\nsd(y)                 # the target gets na.rm = TRUE",
             source="scenarios.py family 3", rows_key="na_rm_propagation"),
        dict(name="format_propagation", plain="The raw-to-normalized diff of a package file (what air format changes) has two or more clean hunks. One hunk is shown reformatted; the model must reformat the next hunk.",
             example="event: one hunk switched to air style\ntarget: the next hunk, same style change",
             source="scenarios.py family 4", rows_key="format_propagation"),
        dict(name="doc_sync", plain="A function's signature gains one benign argument (verbose = FALSE style); the model must insert the matching roxygen @param line in the doc block above. The maintenance skill, not fresh drafting.",
             example="#' @param verbose Show progress messages.\nverbose = FALSE  # added in signature",
             source="scenarios.py family 5", rows_key="doc_sync"),
        dict(name="namespace_qualify_propagation", plain="In a file where pkg:: prefixes were stripped from the prompt, one earlier call is already re-qualified; the cursor sits before the next bare call, and the model must re-add the package prefix.",
             example="dplyr::mutate(...)   # already restored\ngroup_by(...)        # target: dplyr::group_by",
             source="spec namespace_qualify_propagation.json", rows_key="namespace_qualify_propagation"),
        dict(name="pkg_metadata_sync", plain="A real export()/importFrom()/S3method() line (or a DESCRIPTION Imports entry) is removed from its sorted slot; the model re-inserts the exact line.",
             example="export(parse_data)   # removed; model re-inserts it",
             source="spec pkg_metadata_sync.json", rows_key="pkg_metadata_sync"),
    ]),
    ("Completion (finish what the user started typing)", [
        dict(name="finish_block", plain="Complete a function body from the opening brace. The backbone family of the mixtures (90,728 train rows in v7).",
             example="summarise_stats <- function(x) {\n  # model completes the body",
             source="finish_block.py", rows_key="finish_block"),
        dict(name="mid_body_edit", plain="The counter-skill to finish_block: exactly one line inside a function body changed (a constant edited, na.rm inserted, a comparison flipped); the model must predict only that line, never re-emit the function.",
             example="mean(x)  becomes  mean(x, na.rm = TRUE)",
             source="spec mid_body_edit.json", rows_key="mid_body_edit"),
        dict(name="astfim_partial", plain="The user deleted a semantic span and started retyping it. The first k retyped lines sit above the cursor; the model continues the retyping. Derived purely from the astfim corpus, no LLM.",
             example="# deleted span, first 2 lines retyped\n# model types the rest",
             source="spec astfim_partial.json", rows_key="astfim_partial"),
        dict(name="pipe_chain_link", plain="A real multi-line pipe chain is cut right after the |> or %>% that ends a line; the model completes the next chain link, verbatim from the corpus.",
             example="x |>\n  # model completes: filter(!is.na(value))",
             source="spec pipe_chain_link.json", rows_key="pipe_chain_link"),
        dict(name="expectation_completion", plain="Complete the author's expect_ assertion inside a real test_that block; the cursor sits right after the typed expect_ partial.",
             example="test_that(\"na dropped\", {\n  expect_   # model: equal(nrow(out), 2L)\n})",
             source="spec expectation_completion.json", rows_key="expectation_completion"),
        dict(name="tidyselect_completion", plain="A real dplyr select/relocate/across line has its tidyselect helper call (starts_with, contains, where, ...) removed at a mid-line cursor; the model re-completes it.",
             example="select(where(is.numeric),   # model: starts_with(\"pt_\"))",
             source="spec tidyselect_completion.json", rows_key="tidyselect_completion"),
        dict(name="trycatch_handler_completion", plain="Complete the condition handlers of an existing tryCatch or withCallingHandlers call, from the comma after the guarded block to the closing paren.",
             example="tryCatch(fit(), error =   # model: function(e) NULL)",
             source="spec trycatch_handler_completion.json", rows_key="trycatch_handler_completion"),
        dict(name="removed_block_comment", plain="A run of 3 to 8 statements was removed from inside a function and a one-line comment marks the site; the target re-inserts the removed block verbatim.",
             example="# compute fold changes\n# model re-inserts the removed lines",
             source="spec removed_block_comment.json", rows_key="removed_block_comment"),
        dict(name="mid_roxygen", plain="Complete a roxygen doc block at a mid-line cursor inside the block (suffix convention: everything after the cursor is given).",
             example="#' @param x the values to summar   # model: ize.",
             source="scenarios_v1 (curated)", rows_key="mid_roxygen"),
        dict(name="hidden_r_instruction", plain="R rows harvested from general code-instruction datasets (39,000 train rows in v7); general R knowledge, not a specific edit skill.",
             example="", source="hidden_r_instruction_v1/", rows_key="hidden_r_instruction"),
    ]),
    ("Rewrite and repair", [
        dict(name="rewrite_lint_fix", plain="Author models fix real lint findings on corpus code (T/F booleans, seq(), paste0, sapply, class checks), gated on behavior preservation.",
             example="if (x == T)  becomes  if (isTRUE(x))",
             source="cases/ authors (curated)", rows_key="rewrite_lint_fix"),
        dict(name="fix_issue_inject", plain="Reverse-strip injection: a dirty twin (char swap, wrong variable, boundary operator) is injected as the prompt; the corpus original is the exact ground truth.",
             example="length(x) = n  becomes  n <- length(x)",
             source="cases/ authors (curated)", rows_key="fix_issue_inject"),
        dict(name="loop_rewrite", plain="Rewrite a for-loop into vectorized R, behavior verified. Corpus supply measured thin (about 0.06 percent of functions), so the family stays small.",
             example="for (i in seq_along(x)) s[i] <- x[i]^2\n# becomes: s <- x^2",
             source="rewrite_verify_proto.py (curated)", rows_key="loop_rewrite"),
        dict(name="edit_pairs", plain="Real git commit diffs mined into edit rows (15,427 train rows in v7); also the source of the midtyping eval rows.",
             example="", source="edit_pairs_v1/", rows_key="edit_pairs"),
    ]),
    ("Restraint (when NOT to edit)", [
        dict(name="no_op", plain="The prompt contains a plausible edit hook, but the correct completion is to change nothing (emit only the terminator). Trains the emit-nothing discipline; the RL no_op reward went 0.39 to 0.97 on trained geometries.",
             example="# bait: a line that LOOKS unfinished\n# correct output: (nothing)",
             source="scenarios_v1 (curated)", rows_key="no_op"),
    ]),
    ("Drafting (write what has no code yet)", [
        dict(name="roxygen_drafting", plain="Given a function, write its full roxygen doc block, mined corpus ground truth (38,593 train rows in v7; 246,401 mined, mixture-capped).",
             example="#' Summarise with a guard\n#' @param x numeric vector",
             source="roxygen_drafting.py", rows_key="roxygen_drafting"),
        dict(name="comment_drafting", plain="One-line comments drafted for real comment-free corpus code; a five-style pool with a comment gate.",
             example="# keep the last observation per group",
             source="comment_drafting.py", rows_key="comment_drafting"),
        dict(name="comment_insert", plain="Insert a comment at the right place mid-function; the cursor-positioning skill.",
             example="", source="comment_insert.py", rows_key="comment_insert"),
        dict(name="comment_to_code (real / synthetic / gemini / styles)", plain="Reverse-synthetic: the corpus code is the ground truth and the model writes the one-line comment a developer would have put above it. Variants cover real blocks, LLM-synthesized blocks, a Gemini-authored wave, and the 5-style spec.",
             example="df <- df[order(df$id), ]  # model: order by id",
             source="comment_to_code.py + spec comment_to_code_styles.json", rows_key="comment_to_code"),
        dict(name="synthetic_analyst", plain="Analyst-style scripts generated from a domain/construct grid, then validated.",
             example="", source="generate_analyst.py", rows_key="synthetic_analyst"),
        dict(name="paper_to_r", plain="Reimplement a paper's reported analysis as R; rewards are statistical properties (coverage, type-I error, bias) checked by running the generated validator, which must first fail a corrupted twin.",
             example="", source="paper_to_r.py", rows_key="paper_to_r"),
        dict(name="pr_instructed", plain="Instructed pull-request-style edits (506-row pilot).",
             example="", source="pr_instructed_v1/", rows_key="pr_instructed"),
    ]),
    ("Compounding (one sample becomes many)", [
        dict(name="compound (finish_block_compound, compound_spark)", plain="One base corpus sample becomes many cases: multi-family derivations per function via a rule grid or author models. 9,574 deterministic rows from 1,562 samples in the finish_block wave; about 10 rows per sample measured.",
             example="", source="finish_block_compound.py + compound_author_spark.py", rows_key="compound"),
        dict(name="registry (transform rules)", plain="Deterministic detector/rewrite/verify plugins over real corpus code. verify() is a pure re-derivation, so a rule can serve as a deterministic RL reward. 16 rules registered.",
             example="", source="cases/rules/ (curated)", rows_key="registry"),
    ]),
    ("RL refinement", [
        dict(name="refine_* (rename / format / pipe / no_op)", plain="Recursive validator feedback: rows built from failed RL rollouts plus a natural-language complaint, the corrected completion as target (v1: 494 rows, v2: 2,046 rows).",
             example="", source="rl_refinement_v1/v2.jsonl", rows_key="refine"),
    ]),
]
# Row-count fallbacks for families whose rows do NOT live in cases_v1 /
# scenarios_v1 (the bank the builder counts live). Values are grounded in
# the sft_v7 stats, the pool scan, and the rules registry size.
ROWS_FALLBACK = {
    "finish_block": "90,728 train rows in sft_v7",
    "hidden_r_instruction": "92,958 harvested rows on disk (39,000 in v7)",
    "edit_pairs": "16,951 rows on disk (15,983 examples + 968 eval)",
    "loop_rewrite": "0 train rows in v7 (spec only; supply measured thin)",
    "synthetic_analyst": "7,638 rows on disk",
    "paper_to_r": "12 pilot rows",
    "pr_instructed": "506 pilot rows",
    "registry": "16 rules registered",
    "refine": "494 rows (v1) + 2,046 rows (v2)",
}
# families joined by prefix-sum over the bank instead of an exact key
ROWS_PREFIX = {"comment_to_code"}

# ---------------------------------------------------------------------------
# 4. jargon (one consistent term per thing, defined once)
# ---------------------------------------------------------------------------
JARGON = [
    ("AR", "autoregressive", "Generates text left to right, one token at a time. The classic decoder."),
    ("AST-FIM", "syntax-tree fill-in-the-middle", "Our FIM variant: spans are semantic units (function bodies, argument lists) picked by parsing the syntax tree, not fixed-size windows. Built on the astfim_v1 corpus."),
    ("BPB", "bits per byte", "Loss measured per input byte rather than per token, so models with different vocabularies compare fairly."),
    ("edit_sim", "edit similarity", "Token-level similarity between prediction and ground truth; the partial-credit metric when exact is too harsh."),
    ("eval-v2 (V-series)", "feel-of-use battery", "The 2026-09-01 addition: episode metrics, AST-equivalence scoring, pairwise preference. Additive columns; pre-registered verdict rules unchanged."),
    ("exact / valid", "metrics", "exact: prediction equals ground truth. valid: parseable R. Quoted as a pair, e.g. 82.7/73.3 means 82.7 percent valid, 73.3 percent exact."),
    ("GDN", "Gated DeltaNet", "A linear-attention layer class: fast, constant memory. Qwen3.5 mixes 3 GDN layers per 1 full-attention layer (the 3:1 you see in B2/B4)."),
    ("GGUF / Q8_0 / Q4_K_M", "model file formats", "GGUF is the single-file format llama.cpp serves; Q8_0 and Q4_K_M are quantization levels within it (8.5 and about 5 bits per weight)."),
    ("GRPO", "the RL algorithm", "Reward here is exact match plus 0.2 times line-F1; advantage comes from comparing 4 completions of the same prompt. Unanimous groups give zero signal (the variance problem O2 targets)."),
    ("kernel-day", "scheduled GGML work", "One day spent on llama.cpp compute kernels. P12's gate ruled it worth scheduling for the Q4 dequant path."),
    ("LoRA", "low-rank adapter", "Train a small patch on a frozen base model. r32a64 in the bake-off recipe."),
    ("MD / MD@32", "masked diffusion", "Fills a span by iteratively unmasking tokens; MD@32 means 32 denoising steps. The poc_diff twin validated it against AR."),
    ("McNemar rule", "significance discipline", "Any before/after claim at n under about 300 carries a McNemar test line and a verdict from the fixed vocabulary (WINNER, TIE-UNDERPOWERED, ...). Per-example rows are persisted."),
    ("midtyping", "eval battery", "Rows from real git diffs where the cursor sits mid-line; the model completes a partially typed line."),
    ("noopFP", "no-op false-positive rate", "Share of decoy prompts (nothing should be edited) where the model proposes an edit anyway. Lower is better; v7 sits at 70.6 percent, structural."),
    ("OPD", "on-policy distillation", "RL with a teacher signal on every rollout, even wrong or unanimous ones. The O-series maps its transferable parts onto our teacher-free GRPO."),
    ("PSM", "prefix-suffix-middle", "A FIM prompt layout: prefix and suffix are given, the middle is filled. zeta2 renders PSM with the suffix placed first."),
    ("RLVR", "verifiable-reward RL", "RL where the reward is a program (exact match, validator, statistical property), not a model opinion."),
    ("SWA", "sliding-window attention", "Attention restricted to a window (512 tokens in Spark-X2.5), mixed 21 windowed + 7 full layers. Constant memory per layer, cheaper long context."),
    ("t/s (tg128)", "decode speed", "Tokens per second generating 128 tokens, llama-bench, 8 CPU threads, contended-box convention. The product-axis metric of the bake-off."),
    ("zeta2", "the canonical prompt render", "Suffix first, then the raw edit-history diff, then the file prefix with the edit region in merge markers (<<<<<<< CURRENT / ======= / <[fim-middle]>). All SFT data and evals are keyed to it; H3 prices breaking it."),
]

# ---------------------------------------------------------------------------
# 5. live inventory (three sections, mtime-cached)
# ---------------------------------------------------------------------------

_VARIANT_RE = re.compile(
    r"_(zai|spark|gemini|xpreview(?:-free)?|orfree_[A-Za-z0-9.\-]+|free)$")
_SKIP_FILES = {"base_samples_spark", "suffix_scenarios"}


def _family_of(stem: str) -> str:
    base = stem
    m = _VARIANT_RE.search(base)
    while m:
        base = base[:m.start()]
        m = _VARIANT_RE.search(base)
    if base == "rewrite_fixissue":
        base = "fix_issue_inject"
    return base


def _count_lines(p: Path) -> int:
    n = 0
    with open(p, "rb") as f:
        b = f.read(1 << 22)
        while b:
            n += b.count(b"\n")
            b = f.read(1 << 22)
    return n


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except ValueError:
            pass
    return {"files": {}, "computed_at": 0}


def _cached_lines(p: Path, cache: dict) -> int:
    key = str(p)
    st = p.stat()
    ent = cache["files"].get(key)
    if ent and ent["mtime"] == st.st_mtime and ent["size"] == st.st_size:
        return ent["lines"]
    n = _count_lines(p)
    cache["files"][key] = {"mtime": st.st_mtime, "size": st.st_size, "lines": n}
    return n


def _bank_families(cache: dict) -> dict:
    """Live per-family row counts over cases_v1 + scenarios_v1 (train side)."""
    fams: dict[str, int] = {}
    for d in (DS / "cases_v1", DS / "scenarios_v1"):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jsonl")):
            if p.name.endswith((".done.jsonl", ".bak", ".stats.json")):
                continue
            stem = p.stem
            if stem in _SKIP_FILES or stem.endswith("_eval"):
                continue
            fams[_family_of(stem)] = fams.get(_family_of(stem), 0) + _cached_lines(p, cache)
    return fams


def _dir_count(p: Path) -> int:
    try:
        return sum(1 for _ in p.iterdir())
    except OSError:
        return -1


def _manifest_sums() -> dict:
    """normalized/ ingest log: package count + summed n_bytes (fast)."""
    mf = DS / "manifest.jsonl"
    pkgs, nbytes = 0, 0
    if mf.exists():
        with open(mf, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                pkgs += 1
                nbytes += d.get("n_bytes") or 0
    return {"packages": pkgs, "bytes": nbytes}


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:                                  # noqa: BLE001
        return None


def _sft_mixtures() -> list[dict]:
    out = []
    for v in ("sft_v1", "sft_v2", "sft_v3", "sft_v4", "sft_v5", "sft_v6",
              "sft_v7", "sft_v8", "sft_v8_1", "sft_v8_2"):
        d = DS / v
        if not d.is_dir():
            continue
        st = _json(d / "stats.json")
        if st and isinstance(st.get("report"), dict) \
                and st["report"].get("total_train") is not None:
            r = st["report"]
            out.append(dict(name=v, train=r["total_train"], eval=r["total_eval"],
                            families=len(r.get("families", {}))))
        else:
            tr = d / "train.jsonl"
            ev = d / "eval.jsonl"
            if tr.exists():
                cache = {"files": {}}
                out.append(dict(name=v, train=_count_lines(tr),
                                eval=_count_lines(ev) if ev.exists() else 0,
                                families=0))
    return out


def _a2_strata() -> list[dict]:
    st = _json(Path("/mnt/h/sepalith/a2/r/stats.json"))
    if not st or not isinstance(st.get("streams"), dict):
        return []
    rows = []
    for k, v in st["streams"].items():
        if isinstance(v, dict) and "tokens" in v:
            rows.append(dict(stream=k, docs=v.get("docs", 0),
                             tokens=v["tokens"]))
    rows.sort(key=lambda r: -r["tokens"])
    return rows


def inventory(persist: bool = True) -> dict:
    """Three-section data inventory. Counts real files; caches per-file line
    counts by mtime so the 30-min cycle does not rescan /mnt/h."""
    cache = _load_cache()
    inv: dict = {}
    soft = lambda fn, default: _soft(fn, default)    # noqa: E731

    # (a) real-world corpus
    real = {}
    real["normalized_dirs"] = _dir_count(Path("/mnt/h/sepalith/normalized"))
    mf = soft(_manifest_sums, None)
    if mf:
        real["manifest_packages"] = mf["packages"]
        real["manifest_bytes"] = mf["bytes"]
    real["packages_shards"] = _dir_count(DS / "packages")
    real["a2_strata"] = soft(_a2_strata, [])
    real["stack_staging_items"] = _dir_count(Path("/mnt/h/sepalith/stack_staging"))
    inv["real"] = real

    # (b) prepared synthetic data
    syn = {}
    syn["mixtures"] = soft(_sft_mixtures, [])
    bank = soft(lambda: _bank_families(cache), {})
    syn["bank_families"] = bank
    syn["bank_total"] = sum(bank.values())
    ast = _json(DS / "astfim_v1" / "stats.json")
    if ast:
        syn["astfim"] = dict(rows=ast.get("rows", 0),
                             est_tokens=ast.get("est_tokens", 0),
                             packages=ast.get("packages_done", 0),
                             eval_pkgs=ast.get("eval", {}).get("packages", 0))
    astr = DS / "astfim_random_v1" / "train-000.jsonl"
    if astr.exists():
        syn["astfim_random_train"] = _cached_lines(astr, cache)
    for label, fn in (("rl_refine_v1", "rl_refinement_v1.jsonl"),
                      ("rl_refine_v2", "rl_refinement_v2.jsonl")):
        p = DS / fn
        if p.exists():
            syn[label] = _cached_lines(p, cache)
    ds_pools = ("edit_pairs_v1", "commit_goals_v1", "hidden_r_instruction_v1",
                "synthetic_analyst_v1", "paper_to_r_pilot",
                "sim_trajectories_v1", "pr_instructed_v1")
    pools = {}
    for sub in ds_pools:
        d = DS / sub
        if d.is_dir():
            n = 0
            for p in d.glob("*.jsonl"):
                if p.name.startswith(("judged", "progress")):
                    continue
                n += _cached_lines(p, cache)
            pools[sub] = n
    syn["pools"] = pools
    inv["synthetic"] = syn

    # (c) RL environments / scenario pools
    rl = {}
    rl["quotas"] = {
        "rename_propagation": 1400, "format_propagation": 1400,
        "no_op": 350, "pipe_rewrite": 150,
    }
    rl["quotas_run2"] = {
        "rename_propagation": 1400, "format_propagation": 1400,
        "no_op": 1600, "pipe_rewrite": 150, "finish_block": 800,
    }
    rl["priors"] = {
        "rename_propagation": "v6 exact 0.820 (headroom, not cold)",
        "format_propagation": "v6 exact 0.522 (most headroom)",
        "pipe_rewrite": "0.944 near-ceiling; about 90 percent zero-std groups (dead for GRPO variance)",
        "no_op": "emit-nothing discipline; RL reward 0.39 to 0.97 on trained geometries",
    }
    rl["env_classes"] = [
        "Edit scenarios (rename, pipe, format, doc-sync, na.rm): exact-match reward, no-op baseline scores 0.",
        "Keystroke simulator: cold and warm prefix-cache episodes against a live llama-server.",
        "Paper-to-R: rewards are statistical properties (coverage, type-I error, bias); validators must fail a corrupted twin first.",
        "No-edit tasks: the correct target is an unchanged region; penalizes eagerness.",
    ]
    rl["el_tiers"] = "pipe_rewrite, then rename_propagation, then format_propagation; next tier admitted at 0.75 fully-solved-group rate over 5 steps (E1)."
    rl["difficulty_gate"] = "A random policy scores near 0; glm-5.3 scores well. Both are checked before a family enters training (rl/README)."
    div = DS / "rl_diverse_select"
    if div.is_dir():
        rl["diverse_select"] = {
            p.name: _cached_lines(p, cache)
            for p in sorted(div.glob("*.jsonl"))}
    inv["rl"] = rl

    if persist:
        cache["computed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            RESULTS.mkdir(exist_ok=True)
            tmp = CACHE.with_name(CACHE.name + ".tmp")
            tmp.write_text(json.dumps(cache))
            os.replace(tmp, CACHE)
        except OSError:
            pass
    return inv


def _soft(fn, default):
    try:
        return fn()
    except Exception:                                  # noqa: BLE001
        return default


if __name__ == "__main__":
    t0 = time.time()
    inv = inventory()
    print(json.dumps(inv, indent=1)[:6000])
    print(f"... computed in {time.time() - t0:.1f}s")
