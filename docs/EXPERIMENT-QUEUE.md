# EXPERIMENT-QUEUE — the single, central experiment queue

One file, all queued/parked/proposed experiments, live status. Verdicts are
NOT recorded here — they live in each experiment's RESULTS.md / research doc;
this queue links to them. Created 2026-08-31 (supersedes the queue sections
of `2026-08-31-contraction-closeout.md` §1 and its supplement, which remain
the verdict map for everything closed before that date).

Protocol: `comms.md`. Governance unchanged — **nothing below fires without a
user GO** except entries already covered by one (§1 chain). GPU work also
claims/releases in `comms/gpu.md`. Status edits happen HERE; announcements
and verdicts go to `comms/board.md`.

Status vocabulary: `RUNNING` (user-GO'd, owner named) · `PARKED` (user-known,
awaiting GO) · `PROPOSED` (new, awaiting user triage) · `INDEXED` (decided
elsewhere; listed so this file is the single lookup) · `DONE` / `CLOSED`
(verdict landed — linked, then pruned on the next sync).

Last synced: 2026-09-02T00:0x+0200 (zcode-queue-mgr takeover — user
activation GO: queue-mgr triages PARKED/PROPOSED at discretion, useful +
interesting; GPU chain claimed 23:4x: X1 → b12_cal anchor → B1 a0-a3 →
B2 → B3, gate B-α after; B4/B5/spark/M1/P10 behind. Repo-wide backlog
sweep integrated this sync: new §2 FIM-dose replica + §3 V1/KV-cache
entries, W28-W34 ops/close-out items, INDEXED additions (μP, batch-warmup,
PVF/TETHER condition, judge governance, McNemar, Zed PR path), Q1-Q6 rows
pruned per the prune-on-next-sync rule — verdicts remain on the board +
§5 links. Builds policy change: nothing canonical in /tmp anymore;
permanent builds under `experiments/bin/` (CUDA b10453 rebuild at
`experiments/bin/llama/llama-cuda-b10453`, source at
`experiments/bin/src/llamacpp-b10453`).)

---

## 1. RUNNING — user activation GO 2026-09-01T23:4x, owner zcode-queue-mgr (chain in `comms/gpu.md`)

Q1–Q6 pruned this sync (all DONE 2026-09-01, verdicts on the board +
close-out docs; the contraction queue is closed).

| # | Experiment | Status | Entry point |
|---|---|---|---|
| X1 | Cross-paradigm span eval (SFT GGUFs on the poc_diff 216-row harness) | **DONE 2026-09-02 01:0x — measurement closed** (board + `poc_diff/CROSS_EVAL.md`: ALL SFT arms exact 0.0000 vs MD@32 0.0694; 51-256 bucket 0.000 paradigm-wide; edit_sim v3 0.066 > v8_2 0.058 > v7 0.052 > base 0.024) | `poc_diff/cross_eval.py` → `CROSS_EVAL.md` |
| B-anchor | b12_cal_minicpm5 | **DONE 2026-09-02 — pre-gate 2 PASSED** (delta -3.1pp valid p=0.039 / -2.4pp exact p=0.18) | RESULTS.md §pre-gate-2 |
| B1 | a0-a3 ladder | **DONE 2026-09-02** (76.9/72.2/56.9/12.5 valid @ 1.08B/967M/854M/741M; no floor above 24L; collapse order format→na_rm→pipe→rename; l16 v1 crashed on ctx-race, clean retry) | RESULTS.md §B1 |
| B2 | b2_qwen35_08b | **DONE 2026-09-02** (82.7/73.3 @ 752M, noopFP 71.3, 36.2 t/s; vision-stripped per 2b-text precedent; Base ships mtp.* — Q7 datapoint) | RESULTS.md |
| B3 | b3_lfm25_350m | **DONE 2026-09-02 — ELIMINATED** (v1 INVALID 983K-param LoRA; full-attachment rerun 15.7 valid / 95.0 noopFP, train loss BETTER than v1 = overfit-fragility; 94.1 t/s = class's one surviving claim; B7 rescue condition triggered, stays conditional) | RESULTS.md §B3 |
| B12 | b12_spark17b | **DONE 2026-09-02** (85.1/77.3 raw on TRL path ≈88 adjusted, noopFP 67.8, 17.4 t/s CPU; smoke-print bug + converter-path bug both fixed, zero training lost) | RESULTS.md |
| **Gate B-α** | the architecture verdict | **DONE 2026-09-02 18:5x — SWA LEADER / GDN conditional #2 (B4 decisive) / dense OUT as next-build substrate / conv eliminated / doc_sync 0-everywhere tilts construction** | RESULTS.md §gate-B-alpha + board |
| D1 | 206M @ 0.5B undertraining disambiguation (user-reopened; readout pre-registered in script header) | **RUNNING** (ETA ~21:00) | `scripts/run_d1_chain.sh` |
| D2/D3 | budget-vs-unique-data pair | **DONE 2026-09-03** (D2 0.0602 / D3 0.0000 — breadth<depth at fixed budget; full D-grid in poc_diff/RESULTS.md §D-grid) | RESULTS.md |
| B4 | GDN@2B decisive size control | **DONE 2026-09-04** (85.1/76.5 @1.88B, 19.2 t/s, unsloth+knobs after 5-attempt saga; EXACT TIE with spark p=1.0) | RESULTS.md §gate-B-beta |
| B5 | granite-3B ceiling + doc_sync | **DONE 2026-09-04** (87.8/78.0 nominal leader, p=ns, 10.65 t/s; format 79.1 = native-FIM signal; doc_sync 0.0 at 3B = CONSTRUCTION) | RESULTS.md §gate-B-beta |
| **Gate B-β** | final verdict | **DONE 2026-09-04 22:0x** (3-way tie; GDN wins product axis; production rec = b4-config; plan doc opened) | RESULTS.md + `2026-09-04-production-finetune-plan.md` |

Chain script: `scripts/run_b_series_chain.sh` (anchor → B1 a0-a3 → B2 →
B3, one trainer at a time, batteries CPU-side under flock, midtyping
--limit 18 with the runbook's row-id join check). Gate B-α fires after B3,
owner-run. B4/B5/B12-spark/M1/P10 slot behind the gate. Battery scripts
now persist FULL raw outputs (`raw` field — eval-v2 enabler, see §3 V1).

## 2. PARKED — known to the user, awaiting GO (GPU items also wait for the card)

| # | Experiment | Cost | Why it's queued | Entry point |
|---|---|---|---|---|
| Q4 | Flash E2/E3 (holdout-interference probe ~2.5h; blocked-vs-interleaved SFT A/B) | ~2.5h GPU | RL/SFT-methodology; relevant only when that phase resumes | `docs/research/2026-08-28-flash-derived-poc-plan.md` |
| Q5 | stabtok P2/P3 + T1-tokenizer (~35h GPU) | ~35h GPU | 4x-LR stress margin of pinned recipe (P2 = the only pre-launch value; partially answered by Q3's stress pair); tokenizer compression + forced R-pattern tokens | `poc_stab/` (GatedNorm folds in via Q3) |
| Q7 | RT-2 spec-acceptance probe (n-gram lookup + MTP acceptance on mined r-universe traces) | 1–2 days, NO GPU | Sets training spend for any MTP work + n-gram-first question before A2 freezes the head; 1.4–1.9x rollout speedup if real | design-A §3 (near-term item 3) |
| FIM-Replica | FIM-dose replica arm — unmasked-FIM @35% probe2-replica control (the one untested leg of the adopted 20-35% dose verdict; pre-registered masked-vs-unmasked ≥2x gate) | ~1.5h GPU (ladder-arm class) | The adopted FIM-dose verdict has a never-run falsifier; `dashboard_state.json:21` tracks it as "untested (replica arm pending)" | `2026-08-22-night-session.md` §verdict + `fromscratch-design-A2.md:444` |

### 2b. B-series — external-base bake-off × param-floor ladder (PARKED 2026-08-31 23:0x per user trust verdict)

Runbook = `docs/research/2026-08-31-base-bakeoff-plan.md` — exact commands,
common harness, pre-registered verdict rules, artifact naming; written to be
executable cold by a queue manager. Run order B1 → B2 → B3 → gate B-α →
B4 → B5; B8–B10 fold onto the gate winner; B10/B11 are CPU/API-class,
anytime. B12 joins the gate B-α input set if its two pre-gates (draft-backend
parity, trainer-stack calibration) have passed by the time the gate fires;
otherwise it runs post-gate as an added column. GPU rungs additionally wait
for the card.

| # | Experiment | Cost | Why (the datapoint) | Entry |
|---|---|---|---|---|
| B1 | In-family param ladder: MiniCPM5-1B layer-drop 24→20/16/12L + retrained 24L reference, LoRA each | ~6h GPU + CPU battery | Params as the only variable → the size floor + the restraint (no-op FP) floor; instrument landed + dry-run tested | runbook §2 B1 |
| B2 | Qwen3.5-0.8B-Base rung (GDN 3:1 at 0.8B; supersedes the Qwen3-0.6B frankenmodel idea) | ~2h GPU + pull | GDN column's sub-1B point; pre-checks: GGUF throughput #20072, MTP weights, text-strip precedent | runbook §2 B2 |
| B3 | LFM2.5-350M-Base rung (conv+GQA at the stage-0 latency floor) | ~1h GPU + pull | conv-hybrid column at the floor; no-chat-template serve; lfm1.0 license check | runbook §2 B3 |
| — | **Gate B-α** (analysis after B1–B3) | CPU | Top-2 architecture classes survive; two curves; the pick | runbook §2 gate |
| B4 | Qwen3.5-2B-Base straight-SFT rung — base ALREADY LOCAL (`models/qwen3.5-2b-base-text-hf`) | ~3h GPU | GDN column at 2B; the honest redo of the instrument-broken bake-off leg (no midtrain) | runbook §2 B4 |
| B5 | granite-4.1-3b-base rung (3B dense, native FIM in pretraining) | ~5h GPU + pull | Ceiling check + the doc_sync capacity-vs-construction disambiguation + native-FIM sample efficiency | runbook §2 B5 |
| B8 | AST-FIM midtrain re-probe with the FIXED instrument (masking+packing), on the gate winner | ~4-6h GPU + train_sft patch | Does midtrained edit-span ability help at all; cross-checks B5's granite datapoint | runbook §2 B8 |
| B9 | SeleKT gradient-importance masking A/B on the winner | GPU-hours | The one documented post-train-only failure mode (naive SFT loses edit ability) | runbook §2 B9 |
| B10 | WiSE-FT LoRA interpolation (α∈{0.3,0.5,0.7}) | CPU-class, anytime | Multi-task shipping path + the pre-registered RL-phase guardrail | runbook §2 B10 |
| B11 | NextCoder-style R edit-seed pack | API+CPU, anytime | The edit-shaped data construction for post-train v1 | runbook §2 B11 |
| B12 | Spark-X2.5-1.7B-Base rung — **4th arch class** (SWA 3:1 hybrid: 21 SW-512 + 7 full attn, 8Q/2KV, tied emb, Apache-2.0) on the TRL/PEFT trainer path (`train_sft_trl.py`) + PR-build llama.cpp | ~2-3h GPU + ~1.5h calibration anchor | SWA-hybrid vs GDN 3:1 / conv+GQA / dense GQA, sized between B2 (0.8B) and B4 (2B). **PREP DONE 2026-09-01** (weights local, PR #27868 CUDA build, `.venv-spark`, scripts, base Q8 GGUF) and **pre-gate 1 (backend parity) PASSED 3/3**; remaining pre-gate: trainer-stack calibration anchor `b12_cal_minicpm5` (needs GPU). Runbook §2 B12 + `docs/research/2026-09-01-local-builds.md` | runbook §2 B12 |

### 2c. X-series — paradigm follow-ups (PARKED/prepared 2026-08-31 23:4x)

Runbook = `docs/research/2026-08-31-paradigm-followup-plan.md` —
pre-registered metrics, verbatim kill tests, tasks, costs. Promoted from
§3 (P1/P2/P3/P6) per the user's queue-preparation directive. X1 needs no
GPU claim; X2/X3 slot BEHIND the §1 chain (no preemption).

| # | Experiment | Cost | Status / kill test | Entry |
|---|---|---|---|---|
| X1 | Cross-paradigm span eval — SFT GGUFs (sft_v3/v7/v8_2 + base) on the poc_diff 216-row harness, raw-PSM render (format-transfer caveat pre-registered) | ~4h CPU (no claim) or ~30min GPU | **PREPARED — code landed + smoke-verified** (`cross_eval.py` + tests; poc_diff suite 34-pass; eval triples regenerated deterministically = exactly 216, matching the banked run); measurement, readout rule in plan | `experiments/training/poc_diff/cross_eval.py` → `CROSS_EVAL.md` |
| X2 | AR-init diffusion span head (the un-triggered Task 8) | ~7h GPU | plan-frozen; kill: EXCEED 0.0694 exact within 1.0B continuation tokens or write the negative | plan §X2 (Task 1 = AR→MD weight conversion + test) |
| X3 | Length-aux hybrid (aux length head, NO value routing; the DDOT-signal salvage) | ~4h GPU | plan-frozen; kill: exact ≥ 0.0347 with predicted lengths AND length-MAE ≤ 58.2 | plan §X3 (Task 1 = aux head + warm-start `md_final.pt`) |
| X4 | CAL-full recipe — **DONE 2026-09-02: KILLED** (board 16:4x; full recipe + refit bias: MAE 158.1 v1 190.7, exact 0.0000; mechanism: oracle peak needs visible suffix, our cursor insertion is suffix-free; CAL line closed v1+v2; X3 is the remaining honest-length path) | ran 2026-09-02 | `poc_ddot/RESULTS.md` §X4 |

## 3. PROPOSED — remaining, awaiting user triage

Grounding: POC-DIFF VALIDATED (MD exact 0.0694 vs AR 0.0000; 51–256-tok
spans 0.000 on BOTH arms), POC-DDOT KILLED (position field converges <200M
tok; value routing poisons quality), survey = `nse-ot-flows-survey-2026-08.md`.
P1/P2/P3/P6 promoted to §2c (X-series) 2026-08-31 23:4x. Remaining
suggested order: P7 → P4, P5 → P8, P9; M1 (added 2026-09-01, the
cheapest P7-adjacent item at ~4h GPU) slots ahead of P7 if GO'd — its
verdict tells P7 whether the corpus-scale spend is the right lever.
User arbitrates.

| # | Experiment | Class / cost | Why (the datapoint) | Entry point |
|---|---|---|---|---|
| P4 | **Block-diffusion span head** (family D targeted at spans) | GPU ~13–26h + build | The 51–256-tok bucket is 0.000 everywhere; AR anchors structure, diffusion fills within blocks; KV-cacheable | new `poc_bd/` reusing poc_diff data + sampler |
| P5 | **Edit Flows arm** (family C: insert/delete/substitute CTMC) | GPU ~13–26h + new trainer | Survey's named candidate for long spans (+138% over mask-only at 1.3B on code); native variable length + deletions — the primitive AR-PSM and MD both lack | new `poc_editflows/`; survey §3.2 (2506.09018) |
| P6 | **Full CAL recipe** (not our failed v1 half-peak rule) | CPU + GPU-hours on existing ckpts | v1 failed hard (length-MAE 191) but the paper's full recipe reports +47.7% Pass@1 on code infilling; cheap to test on `md_final.pt` | `poc_ddot/cal_length.py` upgraded to the paper's full search |
| P7 | **Data-scale disambiguation rerun** — corpus dependency RESOLVED 2026-09-02 (strata already on disk at `/mnt/h/sepalith/a2/r/`, 7.55B tokens available per mixture smoke; D-series micro-arms running first as the cheap probe: see §1 D1/D2/D3) | retrain class | Tests paradigm-vs-data for the long-span zero; survey notes the regime is data-starved | `poc_diff/data_prep.py` + a2 strata |
| P8 | **OT as training loss** (FMPE-style continuous relaxation over span embeddings, family F) | GPU, speculative build | Pure white space — no text/code result exists anywhere; the "NSE as OT" idea's last untested form | survey §3.3 (FMPE 2305.17161 + minibatch OT) |
| P9 | **Differentiable edit-distance loss** trained into a code model | GPU, speculative build | Survey white-space item #4 (edit-distance-as-OT exists for graphs/trees, never trained into a code model) | survey §2 white-space list |
| P10 | **GatedNorm-v2 (near-identity init)** ladder arm — σ-init ≈1 (bias the gate) instead of the standard 0.5, same 668-step paired discipline + 2x-LR stress | ~1h GPU | Directly tests the user's scale question (2026-09-01): Q3's +2% BPB cost may be an init transient (σ≈0.5 halves sublayer outputs until learned open — a large fraction of a 350M-token run, <1% of a 13-25B run; Qwen's "standard init suffices" claim was made at 560B tokens). If v2 closes most of the +2%, the cost amortizes at scale and GN re-enters the 25B conversation; if not, the rejection is structural and scale-proof. Stability leg already CONFIRMED (stress pair: p99.9 1.19x vs 2.28x clip) | `ladder/run_gatednorm.sh` + one-line init change in `model.py` GatedNorm |
| M1 | **Micro-specialist probe (mdlARC-derived)** — **DONE 2026-09-02: KILLED** (board 16:4x; both arms exact 0.0000 vs 0.0694 anchor, kill test "M1b<0.0347" fired; curation delta soft +0.024 edit-sim, 0 exact; scale/general-corpus exposure load-bearing; M-series closed, W5 no curation term, P7 corpus-scale direction stands) | ran 2026-09-02 | `poc_diff/RESULTS.md` §M1 + plan doc |
| V1 | **Eval-strategy v2 (feel-of-use battery)** — the user's 2026-09-01 directive: current non-RL evals may not represent how good the model feels to use; keep existing for anchoring, add the missing axes. Landed pieces: full raw outputs in the battery scripts ✓ · V1a episode metrics ✓ 2026-09-03 (judge_loop + episode_metrics.py wrapper; time-to-edit + interruption rate added; BOTH calibrations reproduced — banked FP-gap 0.99→0.44 p<1e-7, live v8_2 8-accepts-vs-0; battery one-liner in docstring; caveats: n=40 subsample, 2048-slot rejects biased to smaller ctx) · V1b AST-equivalence ✓ (+14.9pp format_propagation recovery). Remaining legs: (1) judge_loop episode metrics as battery member (accept_rate/fp_rate/saved_ratio + time-to-edit), (2) AST-equivalence/parse-validity re-scorer, (3) TTFT + concurrent-load bench, (4) blind pairwise preference on the known-good pair, calibration on v8_2-vs-base + v7-vs-rl_v2c | ~1 day build + CPU/API per model | Every gate from here (B-α onward) gets additive v2 columns without changing pre-registered verdict rules; retroactively applicable to all arms whose raw outputs are stored | design doc `docs/research/2026-09-02-eval-strategy-v2.md` (inventory-backed: simulator + judge_loop + panel judges already built, never in the battery cadence) |
| KV-AR | KV-cached AR re-baseline (honest serving-latency table for the poc_diff AR arm; current 7689ms p95 is cacheless) | CPU/GPU-minutes | Sharpens X1's cross-paradigm latency readout; gated by X1 landing | `poc_diff/RESULTS.md` "Next" |
| P11 | **Vocab probe — R-weighted 32K vs generic (tokenmonster)** — staged, pre-registered. **S1 tokenizer-only** (CPU ~half-day, chain-safe): build three vocabs on a FIXED corpus sample — G32 (generic BPE slice) / R32 (tokenmonster filtered fusion, R-weighted from CRAN `normalized/` + Stack R token stats) / G65 (size control); measure bytes/token density, single-token rate on R-signature tokens (`%>%`, `<-`, `[[`, top tidyverse identifiers), byte-fallback rate, on held-out R text NOT in the tokenizer sample. **Gate A: R32 < +3% density on R text → KILL** (generic vocab, spend nothing further). **S2 LM legs** (only if S1 passes; ~2-3h GPU behind the chain): three M1-class 76M arms, fixed 0.2B tokens, identical corpus, scored in **BPB** on held-out R + a general-text control. **Verdict: WINNER-R32 iff >2% BPB better on R AND <1% regression on the general control; else generic** — bpb makes cross-vocab comparison legal by design (the exact trap the tokenmonster thread hit with token-loss) | CPU half-day + GPU ~3h conditional | A2's vocab is a one-way door and the only datapoint is LFM vocab-scaling (survey Q4). tokenmonster record: tokenizer-only change ≈ 40% token-efficiency, realized as a filtered 28,416 vocab. Carries the M1 lesson the other way: external priors do not auto-adopt — the probe IS the instrument version of the vocab question | `experiments/training/vocab_probe/` (to build) + `docs/research/2026-09-02-external-intel-batch.md` §4 |
| P12 | **Roofline profiling half-day** — llama-bench matrix on the b1_ref24 GGUF: {Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0} × {t4, t8, t12} × {pp512, tg128}, 3 reps, **idle-box window only** (no TRAIN leg active; the contended-box convention corrupts roofline math — coordinate on the board before running). Report implied GB/s (= tg t/s × weight-GB) and %of-STREAM per format, plus the Q4/Q8 realization ratio (speedup achieved vs bytes halved). **Verdict outputs: (a) CPU-tier prior table for the 3-arm quant A/B (§5, design-A §312); (b) kernel-day GO iff Q4 implied bandwidth <70% of STREAM or Q4/Q8 speedup <1.5x; NO-GO otherwise until the 13B gate** | CPU ~2-3h idle window, zero GPU | Q8_0 decode is already at/near roofline on this box's 5900X (implied ~36-44GB/s demand vs ~40-48 achievable; board 11:0x). Open questions: per-format utilization, and whether Q4-class actually realizes its ~2x bytes advantage — that number prices the CPU tier of the ship matrix AND gates any GGML kernel work | `scripts/roofline_bench.sh` (to build) + `docs/research/2026-09-02-external-intel-batch.md` §5 |
| P13 | **Looped-depth probe (compute-for-bandwidth)** — staged, pre-registered. **S0 CPU paper gate** (~half day, chain-safe, may run anytime): redo the design-A2 §4.3 bytes/token table for K=12×R≤2 and K=8×R≤3 vs the 24L dense reference (Q4_K_M + Q8_0, 8K, q8 KV) counting REPEAT weight loads (R·W_K) + full N-layer KV *and* the KV-shared variant + head; model the adaptive-R target (exit-rate grid E[R], the (exit-rate, quality) point the loop must hold to beat dense at equal bytes); serving spike (confirm GGUF cannot express tying — expected no — scope the K-block/R-pass/per-pass-KV/exit-hook runner fork in days). **Kill: no quality-adjusted average-bytes win even optimistic, OR fork we won't own → KILL before any GPU.** Footprint-only variant may PASS explicitly labeled "capacity, not tokens/sec". **S1 ladder quality probe** (ONLY if S0 passes; ~1-2h GPU/arm at 76M, strictly serial behind D3→B4→B5 per W37): poc-twin/micro rig, fixed order + seed 1273 + fixed vocab/corpus; arms dense-K (A) vs looped K×R layer-loop-first w/ DeepLoop scaling (B) vs model-loop iff cheap (C) vs dense-N compute-matched ref (D); held-out R BPB + general control + task battery + stabtok stability telemetry; quality-only (training tok/s claims nothing about looped serve). **Verdict: WINNER-LOOP iff beats (A) clearly AND stands against (D) at equal streamed bytes (hits S0's target); beating only an untuned loop baseline = FAIL** (McNemar go-forward rule, §5). **Latent-feedback branch** (2608.08888, <1%/tok gated fusion) conditional on S1-positive-plus-heavy-fork — answers recurrence-vs-passes; never jumps the order. Sits BEHIND the queued bytes/token levers (quant A/B+QAD, single-mode spec/Q7, Matryoshka draft, SWA KV, P12) until S0 shows otherwise; R1 + W37 hold throughout | S0 CPU ~half day, zero GPU; S1 GPU ~4-8h conditional, behind the §1 chain | User 2026-09-03 question: bandwidth-bound decode → trade compute for memory? DeepLoop (2607.13491) is stability-only (no memory/FLOP numbers); honest accounting: fixed-R looping streams the SAME bytes as dense at matched unrolled depth (repeat loads R·W_K = W) — only adaptive-R exit + KV-sharing can win average bytes, and stock GGUF unrolls to duplicated layers (no loop op, verified vs vendored tree) so any win needs a runner fork. Wiki cluster backs the staging (loopie layer-loop + compute-matched recipe; HRM-Text-1B 09-02 note; latent-feedback as the cheap sibling). **+ SMELT 2609.01343 (2026-09-04): middle-50%-twice MoE under triple budget-match (FLOPs/params/KV), 6.8–18.0% frontier CE gain, code leads at 20.4%, long-sample + ICL tilt, sink-collapse mechanism; does NOT redeem dense looping (MoE-only recourse, no wall-clock match) — S0 gains a shrink-to-equal-quality conversion, S1 gains a middle-loop topology arm, kill rule unchanged; see plan §6** + **L-only asymmetric variant (2026-09-04, §7): loop blocks 16–23 only, S/M taps byte-identical (loop is post-exit in `trunk_taps`), M doubles as the free control, S draft path untouched — preferred loop form if S0 passes, still gated by S0, no separate experiment** | plan `docs/research/2026-09-03-looped-depth-probe-plan.md` (survey: DeepLoop + ALBERT/Huginn/MoEUT/Relaxed-Recursive/Griffin + wiki looped cluster + SMELT §6) |

| E1 | **EL-scheduler retrofit on rl_smoke.py** — ordered difficulty scheduling vs the current random fixed-quota draw. Order families pipe_rewrite → rename → format (plus compound when present); admit the next tier only when the current one exceeds a pass-rate threshold (paper: 6/8 over 8 rollouts; adapt to our num_generations=4 groups). Pre-registered readout: partial-solved-group rate + mean reward over the first 50 steps (paper Fig 5). **Kill: no partial-solved-rate gain vs random at matched rollout budget → KILL before any evolver build** | GPU ~smoke + 50-step short run, serial per W37; sampler patch only, no new env infra | Cheapest paper win; directly tests whether our pipe-near-ceiling vs format-headroom split is wasting GRPO group variance today | `experiments/training/rl_smoke.py` (sampler + MetricsCb) + `experiments/training/rl/README.md` |
| E2 | **Length-axis evolver lineage (format → compound)** — build the Loop-1/Loop-2 harness at our scale: Proposer inserts one dependent second edit site (single-pair edit = paper's low effort), Modifier via `cases/` authors, gates = Oracle pass + empty/no-op fail + corrupted-twin fail (our existing reward rules as the three verifiers). 5-generation lineage from format_propagation; GRPO vs fixed-quota baseline at matched env count. **Verdict: WINNER-EVOLVER iff evolved generations hold partial-solved groups after the seed saturates AND eval_scenarios exact beats baseline (McNemar per §5 rule); else KILL** | ~half-day build + short GPU run, serial | Paper's strongest single-step effect (−7.1pp pass rate at the smallest mutation); maps exactly onto our compound tasks + saturation problem (pipe ≈ dead at 90% zero-std per §5 RLVR notes) | `experiments/synthetic-data/cases/` (authors + `validators.py`) + `compound_author_spark.py` + `rl_smoke.py` |
| E3 | **Scenario- + skill-axis evolver + cross-mode fallback** — same harness as E2, other two axes adapted to R-edits: scenario = same transform in rarer surrounding context (uncommon package, style, file layout); skill = rarer transform in the same context. Per-generation random axis order with fallback when reviewer/verifiers reject (paper §4.1). Tests whether direction-specific difficulty profiles persist over multi-step lineages (paper Table 1) and whether mixed lineages generalize beyond length-depth | Build on E2 + CPU/API generation; GPU only for the closing run. Do NOT run before E2 passes its gates | Length-only lineages risk depth overfitting; the paper's generality claim rests on all three axes + random ordering | E2 harness + `experiments/synthetic-data/cases/specs/` + `samplers.py` |
| E4 | **Evolution vs co-evolution head-to-head** — matched-env-count comparison: (a) off-policy evolver lineages (E2/E3 output) vs (b) on-policy weakness-bank baseline (failed GRPO trajectories → new envs around failures, paper §5.5 recipe) vs (c) ensemble baseline (merge failed seed envs pairwise). Offline eval_scenarios at fixed intervals to the run budget; pre-registered peak-accuracy + step-to-peak readout (paper Fig 7). **Only fires if E2 or E3 lands** | GPU full-run class, serial; the most expensive E-item by design | The paper's headline (+14.4/+18.0pp over co-evolution/ensemble) is the claim that would justify evolver infra as permanent; without this A/B we cannot separate evolver gains from more-data gains | E2/E3 lineages + `experiments/eval/eval_scenarios.py` + RLVR annotations (§5) as the co-evolution baseline spec |

### H-series — joint harness–weight optimization (WHALE line; user directive 2026-09-04)

Premise: WHALE (arXiv:2609.00196, 31 Aug 2026) treats agent performance as
J(θ,h) — model weights × executable harness — and reports +7.7–24.4pp over
single-component optimization and +4.2–13.0pp over prompt-only joint
optimization (FST control), with small alternating steps beating one-shot
stagewise by 5–9pp. Our harness = `extensions/vscode-sepalith/src/`
(renderer, pin/outline, caps, truncation, parsing, stops, debounce) +
the RENDER registry in `experiments/eval/run_eval.py`; our weight step =
`train_sft.py` filtered-SFT. Compatibility boundary is the extension APIs
only (vscode/positron `inlineCompletionProvider` = full prompt control;
Zed = W34 upstream PR or a second checkpoint) — everything else is
rebuildable. Serving re-target per the same directive: iGPU is the floor
(2–3x-over-CPU claim), CPU is the fallback. Method note: ShinkaEvolve is
population evolution over marked code blocks (islands/archive/migration);
pi-autoresearch is greedy single-lineage hill-climbing
(propose→bench→keep/revert) with autonomy/resumability machinery. Both are
generate-and-test on a fitness score; they differ in exploration structure,
not in kind. H1 tests which one fills the harness-search slot; GEPA/DSPy
cover the prompt-only sub-slice (the FST-style control), not the code
decisions (truncation, parsing, caps) that WHALE found load-bearing.

Runbook/plan (shared rig, D_harness construction, pre-registered scoring,
proposer + budget accounting): `docs/research/2026-09-04-whale-harness-weight-plan.md`.

| # | Experiment | Class / cost | Why (the datapoint) | Entry point |
|---|---|---|---|---|
| H1 | **Harness-search method bake-off** — three methods on the FIXED extension-only space (caps, truncation direction, scope on/off, max_tokens/stops, parse rules; render markers FROZEN to keep the model in-distribution): (a) single-lineage hill-climb (pi-autoresearch pattern, greedy best-so-far), (b) population evolution (ShinkaEvolve pattern: islands/archive, M=3 candidates/iter, revert allowed), (c) prompt-only GEPA (text subcomponents only — the FST-style control). Fixed 256-row D_harness (carve from scenarios/intent families, disjoint from eval; `holdout_rule.py` applies), scoring = edit exact + noopFP + latency; selection on harness-train, verdict on HELD-OUT eval with proposer compute counted. **Verdict: the H2/H4 method is the held-out winner; (c) winning means the space is prompt-bound → shrink the space and close the code-search line** | CPU/API-class, chain-safe (~40 candidates/arm via CPU llama-server + proposer API) | Decides the search machinery before spending regime/re-render budget; pre-registers the expectation that code-search beats prompt-only (the WHALE FST gap, 4–13pp) or the space is mis-scoped | `experiments/harness_search/` (to build) + `eval_noop_fp.py` + `cache_bench.py` |
| H2 | **Harness-only diagnostic (regime test)** — H1-winner method, weights FROZEN at the B-α winner (smoke now on the v7 class): I≈12–18 iters × M=3 on the extension-only space, same scoring as H1. **Verdict: harness-dominant iff harness-only Δ matches/exceeds the reference SFT Δ at <10% of a full-SFT cost → H4 GOes harness-first; flat → weight-dominant → one small SFT (0.2 epochs) then re-search (the catalytic test); re-search still flat → space exhausted, close the line** | CPU/API-class, chain-safe | The cheap WHALE diagnostic: tells us which regime R-edit prediction lives in before any joint spend (their SearchQA matched peak weight-only at ~6% of rollouts; their Math moved only after a weight update) | same rig as H1 |
| H3 | **Render-format liberation (break zeta2)** — render_zeta2 is anchoring, not optimality: all SFT data, all evals, and the extension-parity rule are keyed to it, and its suffix-first ordering is cache-hostile per `docs/prompt-format.md`. **S0 paper/cache gate** (CPU, chain-safe): candidates = orderings (prefix-first PSM vs suffix-first zeta2 vs hybrid) × marker vocabs (merge markers vs numbered vs PSM tags) × history placement × cursor encoding (signature empty-region vs midtyping partial); score zero-shot format-fail + `cache_bench.py` arms + byte math, kill obvious losers. **S1 adaptation** (GPU ladder-class, behind the chain): top 3–5 survivors each get a SHORT adaptation SFT (0.2-epoch LoRA on re-rendered sft_vX, fixed seed), then held-out edit replay + noopFP + latency. **Verdict: WINNER-RENDER iff ≥ parity on exact AND a serving win (or a quality win large enough to pay the retrain + Zed-path cost); else zeta2 stays and the line closes. Ships to vscode/positron immediately (full prompt control); Zed via W34 PR or second checkpoint** | S0 CPU half-day; S1 ~1h GPU/survivor behind the §1 chain | Prices exactly what breaking zeta2 costs (re-render + adaptation + re-eval); the PSM-vs-zeta2 cache question is otherwise argued from principle, never measured | RENDER registry in `run_eval.py` + `context_build.ts` + `train_sft.py` re-render |
| H4 | **Full WHALE alternating loop** — alternate filtered-SFT (E=0.2–0.6 epochs) with the H1-winner harness search (I=6) for K cycles, vs stagewise control (full SFT then full harness search once) at matched cumulative budget (rollouts + GPU-hours + proposer compute counted — the paper excluded proposer cost). Adaptive arm: patience rule (min 0.2 epochs / 6 iters, patience 0.2 / 2, training-signal only, no validation peeking). **Verdict: WINNER-WHALE iff beats the stronger single-component baseline AND beats stagewise at matched budget; else joint optimization closes and components ship separately** | GPU-heavy, behind §1/B-series; fires post-B-α on the winner (pilot on the 76M poc rig if cheap) | The actual joint-optimization claim, with the paper's two load-bearing comparisons (alternating-vs-stagewise, adaptive-vs-fixed) re-run on our task and our budgets | `train_sft.py` + `experiments/harness_search/` |
| H5 | **iGPU serving tier (re-target the ship matrix)** — **UPDATED 2026-09-04: hardware exists.** User-measured on a notebook (mobile AMD, ASR models 600M–3B): iGPU = 2–3× CPU — but that is compute-bound encoder work; our decode is bandwidth-bound and an APU iGPU shares system DDR, so the transfer is plausible for prefill, unproven for decode. **S0 = verify on OUR pattern** (one llama.cpp Vulkan build on the notebook): llama-bench {Q8_0, Q6_K, Q5_K_M, Q4_K_M} × {CPU t8, iGPU, hybrid} × {pp512, tg128}, 3 reps; PRIMARY metric = the cache_bench keystroke-cycle (prompt_ms + 16-tok gen), not raw tg alone. **Kill: median iGPU speedup on the primary <1.5x → CPU stays primary, close.** **S1 cache re-read** (only if S0 passes): `cache_bench.py` arms a/b/c/d on iGPU — does prefix-first still win when prefill is 2–3x cheaper? **S2 re-price**: P12 roofline + quant-A/B priors + the prompt-format ordering argument recomputed for the iGPU tier. Outputs feed H3's serving term and H4's cost accounting | S0 = one Vulkan build + bench on the notebook (CPU-class, chain-safe); S1/S2 CPU-class after S0 | Every serving conclusion in the repo (prompt-format.md ordering, P12 roofline, quant A/B priors) was derived for slow-CPU prefill; the user's 2–3x measurement, if it holds on the decode+prefill mix, re-prices all of them | plan §6 `docs/research/2026-09-04-whale-harness-weight-plan.md` + `scripts/igpu_bench.sh` (to build) + `cache_bench.py --backend` |

Order: H1 → H2 (regime) → H3-S0 anytime / H3-S1 behind the chain →
H4 post-B-α → H5-S0 once hardware exists, H5-S1/S2 after S0. Build
prerequisites fold into the arms (D_harness 256-row carve, render
registry, adaptation re-render script) — no separate W items.

## 4. WORK — the engineering/build backlog (2026-09-01 expansion; user directive: queue must carry ALL remaining work)

The program's remaining work beyond experiments: build, port, harden,
decide. Grouped by phase; everything here is PARKED under the same
governance (nothing fires without a user GO unless covered by one).
Sourced from the runbook's documented next steps, design-A banked
decisions, the simulator roadmap, and the 2026-08-26→09-01 sessions.

### 3a. Pre-launch (before the A2 cluster GO — do when GO is imminent)

| # | Work item | Size | Source / notes |
|---|---|---|---|
| W1 | **FP8 validation cell** — write the rental-hour-1 script (H100 E4M3 path; B200 MXFP8 alt), gate ≤0.004 val-loss delta at ~0.5BT | ~half day | runbook §7 launch-checklist item; nothing exists yet |
| W2 | **13B gate hardening**: port canary/regurgitation evals to the A2 32K run (currently MiniCPM/ladder-anchored) + automate the causal-floor-vs-twin-anchor curve comparison (currently a manual read) | ~1 day | runbook §4 gates; the high-epoch R streams make canaries load-bearing |
| W3 | **GGUF export dry-run** at full 1.5B config on a shakeout ckpt (24L + 8/16 tier prefixes + MTP head) — converter verified at twin scale only | ~half day | runbook §5; de-risks the post-train path before the 5-day run |
| W4 | **run.py polish**: doctor `--fix` (installs deps), HF-download path integration test in a clean env, `gates` on 24L exit-set sanity | ~half day | 2026-08-26 session |
| W5 | **Decision package assembly**: §3.2 re-cut options memo (so_r_qa curve incl. 4x, epochs table, B-series floor when it lands, 1.5B-vs-2B arithmetic under rental throughput) — the one-table brief for the GO call | ~2h writing | close-out §3 + 2026-09-01 size discussion |

### 3b. Data program continuation (the epochs levers)

| # | Work item | Size | Source / notes |
|---|---|---|---|
| W6 | **`git/` GitHub-R pack**: license audit (2,586 repos) → holdout carve-out → pack → manifest entry | ~2-3 days | biggest real-world R lever (~91GB tree); R-eval holdout rule applies |
| W7 | **R-eval holdout rule implementation** — **DONE 2026-09-02** (`experiments/data-mining/holdout_rule.py` + packer hooks (`pack_r_strata --holdout` default ON, `r_repack_full` default OFF to protect the astfim lineage); audit artifact `datasets/a2/holdout_packages.json`; validated in `runs/holdout_scan.json`) | done | enables W6/W8 safely |
| W8 | **full-CRAN causal render** beyond astfim's 3.5% span sample, with the 284-package eval exclusion wired | ~1 day | closes the r_causal epochs gap from the content side |
| W9 | **CRAN Archive acquisition** (edit-diff stratum, 0.5B/2% design slice) | acquisition + pack | manifest deferral; optional at re-cut |
| W10 | **vignettes/prose slice pack** (10% design slice; man/ stays excluded) | ~half day | design-A mixture table row |
| W11 | **English×R Q&A-prose bridge** — requires SO raw-dumps acquisition decision (so_r_qa is answer-code only) | acquisition decision + pack | design "bridge" slice; unmaterialized |

### 3c. Post-13B / 25B path

| # | Work item | Size | Source / notes |
|---|---|---|---|
| W12 | **Eval battery port to the A2 base**: FP gate battery + intent suite + scenarios on the 32K tokenizer/PSM formats (currently MiniCPM-keyed) | ~1-2 days | runbook §5; needed before any serving claim on the new base |
| W13 | **MTP serving path**: llama.cpp speculative-decode integration for the A2 MTP head (informed by Q7/RT-2) | ~1-2 days | runbook §5; the 1.04x-cost structure pays off only if this lands |
| W14 | **DDP/multi-GPU wrap of train_a2** — only if renting >1 card | ~1 day | train_a2 docstring "documented next step"; single-80GB plan = conditional |
| W15 | **25B GO decision package**: assembled from the 13B gates + canaries + the W5 memo updates | ~2h writing | runbook §4 staged decision |

### 3d. Post-train program (fresh-retrain-per-version house pattern)

| # | Work item | Size | Source / notes |
|---|---|---|---|
| W16 | **SFT pipeline adaptation to the A2 base** (tokenizer, PSM render formats at 32K, eval harnesses) | ~1-2 days | runbook §5; zeta2/scenario data exists |
| W17 | **RL-run-5 design on the new base** — constraints banked: no warm-start (clean negative), no critic blend (v4/v5), keep zero-std groups (T1 verdict), run-2 profile, RL no-op arm (--run2: code + data dry-checked, launch was a user decision; v8's FP 100%→<20% gate makes it necessary) | design + run | board verdicts 2026-08-27→09-01 + `v8-noop-random-cursor.md` layer 3 |
| W18 | **Simulator closed loop (stage 2→3)**: model outputs influence the simulated typist — the RL-integration stage-2 design | build | simulator design.md "RL integration — two stages" |
| W19 | **2c serving productization**: extension acceptance-threshold knob (the v0.0.8 client-side stop-confidence gate, product-critical) + post-accept cooldown as user-facing config; intent-suite regression watch | ~1 day | the standing serving call's engineering side; `2026-08-23-night-session.md` |

### 3e. Design-A banked build decisions (experiments parked inside the build)

| # | Work item | Size | Source / notes |
|---|---|---|---|
| W20 | **Outline A/B per exit** (answers workspace-Q1+Q2 with one experiment) | ladder-scale arm | architecture-questions "banked for A2-prime build decisions" |
| W21 | **Exit-wise self-distillation option** | ladder-scale arm | same bank |
| W22 | **n-best decode product lever** | serving-side | same bank |
| W23 | **Workspace summary adapter** (100-200M encoder, SFT-phase) | build | design-A §Q2 path 2 |

### 3f. Infrastructure / hygiene

| # | Work item | Size | Source / notes |
|---|---|---|---|
| W24 | **SYSTEMS.md update**: run.py, r_repack_full, pack_r_strata, push_cases pretraining/, the venv split (3.10 vs 3.14 dill breakage), CUBLAS/vocab scar | ~1h | 2026-08-26→09-01 sessions' scripts undocumented there |
| W25 | **HF dataset card YAML frontmatter** (silences the repo-card warning; adds license/language tags) + card sync with the adopted manifest | ~30 min | push_cases warns every run |
| W26 | **NAS runs/ retention policy**: q6/gatednorm/e3v2 checkpoints organized or pruned (results JSONs retained) | ~1h | disk hygiene; /tmp scars say keep NAS canonical |
| W27 | **Dashboard state refresh** (v55 → current: verdicts, queue link) | ~1h | house pattern |
| W28 | **Anyscale GPU ladder smoke** (one A10G-class node, ~100M-token ladder smoke → cloud tok/s vs 5090's 48k; decides the fp16 T4/Tier-free port) | ~$1/hr, hours | `2026-08-30-anyscale-setup-and-smoke.md` §open-follow-ups #1 |
| W29 | **Repo packaging at scale** (bake container image vs GitHub-PAT clone — pick before any real cloud run) | ~half day | same doc #2 |
| W30 | **Compute-credit program applications** (Tier A signups; Trelis $500/q, Vast $2.5k, HOSTKEY; Lambda/AMD gated; week-1 sequencing incl. fp16 ladder port) | writing-class | `2026-08-30-compute-credit-stack.md` |
| W31 | **PI Fast Compute Grant send** (email drafted; user calls: size ~$2-3k core vs $5k, HF handle, naming) | send + 2 user decisions | `2026-08-30-pi-fast-compute-grant-application.md` |
| W32 | **Aurora verdict archaeology + poc_twin/RESULTS-arms.md close-out** (dashboard says "ADOPTED into A2-prime Muon recipe", papers-recon plan says "parked as OPT-3 gated on its falsifier", runbook §3.2 recipe has no Aurora; Arms C/D never closed in the canonical file) | ~2h reading + 1 verdict post | `dashboard_state.json:85` vs `2026-08-29-papers-recon-poc-plan.md:36`; sweep contradiction (a) |
| W33 | **v7 ablation clean rerun** (rows ~200-812 of `results_ablation_v7_on_plain.jsonl` are ALL-EMPTY predictions; file marked TRUSTED-UP-TO-ROW-~200) | ~hrs CPU | `landscape-v7-vs-glm53.md` tail; sweep integrity item |
| W34 | **Zed upstream `prompt_format` PR** (path 1 of the prompt-format endgame: contribute custom-prompt API upstream — "the right move once the model works"; path 2 = zeta2-format second checkpoint, already the serving route) | engineering, external | `docs/prompt-format.md` |
| W37 | **WSL2 dual-CUDA-context instability — BROADENED 2026-09-03 13:2x to ONE-CUDA-WORKLOAD-AT-A-TIME, no exceptions**: victims = D3-as-second-context 3x (large-pool poc trainer, async unknown-error at loss.item()) AND B4-v1 as second context beside a 6GB eval (CUDA graph-capture failure at currentStreamCapture, dead in 5min, 09:23). Historical co-runs that survived (X1-server+nothing, M1a+M1b poc pair, spark-TRL+M1-evals) were luck-of-the-draw, not a safe class. All chains strictly serial from here | infra rule, effective immediately | board 05:0x + 13:2x; logs d3_train_v*.log, b4_qwen35_2b_train.log |
| W36 | **Spark serving-path optimization probe** (user directive 2026-09-02: 1-2 days IF spark survives B4/B5): upstream PR #27868 merge check + rebuild, llama.cpp kernel/flag tuning for head_dim-256 + SWA KV, quant variants (Q5/Q4 + quality check per the INDEXED quant A/B), and evaluate the GPU-offload product path (254 t/s measured vs 17.4 CPU — the decode objection may be a deployment choice, not an arch cost). Scoping: observed CPU is ~80% of compute-scaled expectation (21.7 t/s), so CPU headroom is modest | 1-2 days, conditional on gate outcome | RESULTS.md §gate-amendments; runbook §2 B12 |
| W35 | **poc-rig microbatch throughput**: `MICRO_PAD_TOK=4096` is launch-bound at small scales (M1a 76M measured 35% GPU util, ~17k tok/s < the 206M anchor's 43k) — raise to 16-32K for future from-scratch runs AFTER verifying `span_batch_to_loss` aggregates per-real-token (padding masked) so chunking stays math-identical; keep 4096 for any run compared against banked telemetry (M-series, X2/X3) | ~2h verify + one A/B smoke | 2026-09-02 M1 observation (user prompt); ETA math in m1a_train.log |

### B-series — PARKED in §2b above (user trust verdict, 2026-08-31 23:0x)

B1–B5, B8–B11 moved to §2b with a full queue-manager runbook:
`docs/research/2026-08-31-base-bakeoff-plan.md` (common harness, exact
commands, pre-registered verdict rules, gate B-α decision rule, env risks).
B6/B7 → INDEXED (§4, conditional). B1's instrument is landed and dry-run
smoke-tested: `experiments/training/truncate_layers.py`.

## 5. INDEXED — decided elsewhere, listed for single-lookup (do not re-derive)

- **RL-resume miners** (supplement §6, CPU-class, no GPU, when RL resumes):
  pass@8-vs-pass@1 boundary canary, near-miss mining (add edit-sim to A7
  snapshot spec), judge-share-of-reward-mass telemetry, family-band mixture
  policy.
- **RLVR annotations** (supplement §2, fold at RL resume): coverage gate
  (per-family pass@8 prior before entering RL; pipe ≈ dead at 90% zero-std,
  format carries signal), dense-shaping A/B for sparse exact-match families
  (fold into Q4/flash-E2), entropy/no-edit canary in every mix.
- **MuonH reopen recipe** (supplement §3): update-norm pinning + ELR-matched
  LR search — only if an optimizer phase reopens. Puro-2B is the receipt.
- **SWA + attention sinks** (supplement §4): long-context baseline any
  retrofit must beat — only if long-context is ever re-cut.
- **Design-A banked build decisions** (`architecture-questions-2026-08-23.md`):
  outline A/B per exit (answers Q1+Q2 with one experiment), exit-wise
  self-distillation, MTP head (design-A §1.4; probe = Q7/RT-2), workspace
  summary adapter (Q2 path 2).
- **Data levers** (close-out §1, unpursued until the §3.2 re-cut): `git/`
  GitHub-R repos (license audit + holdout rule first), full-CRAN causal,
  CRAN Archive edit-diff stratum, vignettes slice, English×R bridge.
- **External-base shortlists + NOT-USEFUL lists (do not re-review)**:
  `model-survey-2026-08-20.md` §3-4, `model-survey-sub1b-supplement-2026-08-31.md`
  §2/§6-7 (quirks Q1-Q15; frankenmodel verdict §4 — buy Qwen3.5-0.8B-Base,
  don't build; Qwen3-0.6B/min-spark/Boris×2/cagliostro-v2 do not re-review).
- **B6/B7 conditional rungs** (conditions, not queued experiments):
  SmolLM3-3B-Base only if the CPT-into-base route is chosen; LFM2.5-1.2B-Base
  only if B3 collapses on edit accuracy (supplement §6).
- **User calls standing** (close-out §3, unchanged): A2 GO + hardware ·
  §3.2 re-cut at the 13B gate · serving 2c adoption · queue activation.
- **Rental-conditional POCs** (papers-recon plan §deferred): μP
  (Wortsman LR-flattening — makes P3's η* transfer 450M→2B principled;
  trigger = A2-prime rental scheduling, never retrofit mid-line) ·
  batch-size warmup A/B (Qwen §3.2 regime mismatch at 206M; rental
  hardware only — distinct from Q6's keep-512k verdict).
- **PVF/TETHER revisit condition** (close-out §1 + pvf-tether-poc tail):
  parked; reopen only with online-/refreshed-critic training + adaptive ρ
  + normalization match. (W17 banks the run-5 constraints; the RL no-op
  arm --run2 launch decision folds there too — v8's FP failure is the
  evidence it's necessary.)
- **Judge-ensemble governance** (banked for RL resume):
  `judge-calibration-gemini-opus.md` (4 judges, 3 families, majority
  reward, gemini tie-break, opus weekly-audit ~150-200 calls/wk) +
  `judge-calibration-gptsol.md` (NOT an unbiased replicate) +
  `oxalpha-verifier-eval.md` (second-judge/fuzzy-gate roles).
- **McNemar go-forward rule** (`2026-08-29-paired-significance-audit.md`):
  any before/after claim at n<~300 gets a McNemar line; verdict vocabulary
  WINNER-A/B/SIGNIFICANT-AMBIGUOUS/TIE-UNDERPOWERED; persist per-example
  rows for every paired verdict.
- **v8/v8.2 SFT-mixture route closed** (2026-08-23): 3rd data point that
  the mixture route cannot win the serving question; RL (2c) owns it; v7
  stays serving; v8 artifacts preserved for forensics.
- **V0 three-axis verdict rule** (papers-recon plan): recipe adoption
  requires loss-side + task-side + cost + stability; BPB-only wins not
  adoptable — stabtok-line rule, generalized.
- **Frontier-mechanism-not-loss pattern** (Q3 board note): frontier-derived
  deltas at our scale transfer MECHANISM but not LOSS (P1 hygiene, CMA
  tail+SMA6, GatedNorm) — recipe-note class for any future adoption case.
- **Data levers, conditional/stale** (beyond W6-W11): pwc-archive one-shot
  harvest (harvester implemented `experiments/data_pipeline/paper_code_harvester.py`,
  NOT deployed; ~3.8k R-ish repos, +0.03-0.1B tok; awaits user call) ·
  Stack v2 R subset (~141k license-filtered repos / 22.4GB; accept-gate +
  dedup) · RPubs crawl / tidyverse PR-review corpus / CI-failure mining /
  SWE-bench-R (old probes, data lanes OFF per contraction — stale) ·
  Bioconductor ingest SUPERSEDED (E3 dropped bioc share to 0.0; scout
  doc's "do this week" is stale) · gemini-family-ideas.txt = 12
  pharmaverse family specs, partially triaged by case-specs-v1.
- **Quant A/B conditional** (design-A §312; +QAD arm added 2026-09-02,
  user-approved — see board 10:3x + 10:4x notes): stock Q4_K_M vs
  Dynamic-Q4 + imatrix vs QAD-style QAT-distill (BF16 teacher → quant
  student; Liquid LFM2.5 Q4_0 evidence: ~97% BF16 retention, matches
  Q5_K_M at ≤350M, native Q4_0 speed) on intent suite — runs on the
  first from-scratch exports; fold into W3's dry-run when A2 exports
  exist.

## Sync notes

- 2026-09-04 (2nd): H-series plan doc landed (`docs/research/2026-09-04-whale-harness-weight-plan.md`) — shared rig (256-row D_harness from TRAIN-side scenario packages under the sft_v3 eval split authority; lexicographic scoring exact→noopFP/latency guardrails; proposer via backends.py zai/spark with tokens counted; full search-space table incl. the live-only knobs parked for on-device A/B). H5 row UPDATED: user's own notebook measurement (mobile AMD, ASR 600M–3B, iGPU 2–3× CPU) resolves the hardware gap — S0 is now a Vulkan-build verification on that notebook, with the ASR-compute-vs-decode-bandwidth transfer caveat pre-registered and the keystroke-cycle as primary metric. All H-items remain PROPOSED; nothing fired.

- 2026-09-04: added H1–H5 (joint harness–weight optimization line from WHALE arXiv:2609.00196) to §3 PROPOSED per user directive. H1 = search-method bake-off (hill-climb vs population-evolution vs GEPA-prompt-only), H2 = frozen-weight regime diagnostic, H3 = render-format liberation (break zeta2, staged S0/S1), H4 = full alternating loop post-B-α, H5 = iGPU serving re-target. Order H1 → H2 → H3/H5-S0 → H4 → H5-S1/S2. No GO asked, nothing fired.

- 2026-09-04: added E1–E4 (environment-evolution line from arXiv:2609.04128, mapped to our RL pipeline in the 2026-09-04 thread) to §3 PROPOSED per user request. E1 = scheduler-only falsifier, gates E2; E2 = length-axis evolver, gates E3/E4; order E1 → E2 → E3 → E4. No GO asked, nothing fired.

- 2026-09-02T00:1x+0200: zcode-queue-mgr takeover sync (user activation
  GO + backlog-sweep integration). Repo-wide sweep (agent) found two live
  second lists (papers-recon §deferred, anyscale/compute-credit
  follow-ups) + never-queued threads: FIM-Replica (§2), V1/KV-AR (§3),
  W28-W34 (§4), 9 INDEXED additions (§5 incl. judge governance, McNemar
  rule, rental-conditionals, PVF/TETHER condition, data-lever
  conditionals). Contradictions flagged: Aurora adoption status (→W32),
  Bioc ingest (superseded, noted), v7-ablation empty rows (→W33).
  Q1-Q6 pruned (DONE, verdicts on board). §1 = live chain (X1 → B-anchor
  → B1 → B2 → B3, gate B-α next). Battery scripts now persist full raw
  outputs (V1 enabler). Board takeover post 23:4x.
- 2026-08-31T22:44: created; imported close-out §1 (Q1 done/adopted, Q2–Q3
  running, Q4–Q7 parked), added P1–P9 from the paradigm review, indexed the
  supplement + design-A items. Announced on the board.
- 2026-08-31T22:5x: added B-series (external-base bake-off × param-floor
  ladder, B1–B11 + gate B-α) to §3 PROPOSED from the user's post-train-v1
  directive (most-capable architecture + size floor); indexed the two
  model-survey docs. Announced on the board.
- 2026-08-31T23:0x: user trust verdict — B1–B5, B8–B11 PREPARED and PARKED
  as §2b (runbook `docs/research/2026-08-31-base-bakeoff-plan.md`; B1
  instrument `experiments/training/truncate_layers.py` landed, dry-run
  tested); B6/B7 → INDEXED-conditional. Announced on the board.
- 2026-08-31T23:4x: P1/P2/P3/P6 prepared + promoted to §2c as X1–X4
  (runbook `docs/research/2026-08-31-paradigm-followup-plan.md`; X1 code
  `cross_eval.py` landed + smoke-verified, suite 34-pass, triples regen
  = exactly 216 matching the banked eval). P4/P5/P7/P8/P9 remain §3.
  Note for the §1 owner: the "P1 prep" smoke you spotted is X1's — the
  full harness is `cross_eval.py`, runnable now (see §2c).
- 2026-09-01T16:2x: B12 (Spark-X2.5-1.7B-Base, SWA 3:1 hybrid — 4th arch
  class for gate B-α) added to §2b per user GO; prep landed CPU-only (no GPU
  claim, §1 chain untouched): weights `models/spark-x2.5-1.7b-base-hf`,
  llama.cpp build `bin/llama/llama-spark2_5-pr27868` (draft PR — parity
  pre-gate required before verdict-grade numbers), `.venv-spark`,
  `train_sft_trl.py`, `export_gguf.py` env overrides, inventory
  `docs/research/2026-09-01-local-builds.md`. Announced on the board.
- 2026-09-01T23:2x: M1 micro-specialist probe (mdlARC-derived: ~75M
  from-scratch MD twin, 0.5B-token budget, scale-control + curated arms vs
  the banked 206M/2.0B anchor on the unchanged 216-row harness) designed +
  added to §3 PROPOSED per the user's request; plan pre-registered
  (`docs/research/2026-09-01-micro-specialist-probe-plan.md` — kill test,
  curation delta rule, conditional M1c). Reuses poc_diff rigs only; no GO
  asked, nothing fired. Announced on the board.
