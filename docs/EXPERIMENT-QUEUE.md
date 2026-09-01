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

Last synced: 2026-09-01T16:2x+0200 (user GO — "setup the experiment and add
it to the queue": B12 Spark-X2.5-1.7B rung added to §2b, PREP landed CPU-side
while the §1 chain holds the card: weights pulled, llama.cpp PR #27868 CUDA
build, `.venv-spark` (transformers 4.57 — 5.5 breaks the remote code),
`train_sft_trl.py` + export_gguf env overrides, untracked builds inventory).
Prior sync 2026-08-31T23:3x+0200: user queue-continuation GO — Q6 promoted,
P1 prep started; prior 23:0x: close-out §1 + supplement, paradigm review +
base-bakeoff prep sessions.

---

## 1. RUNNING — user GO 2026-08-31 20:4x, owner zcode-main (chain in `comms/gpu.md`)

| # | Experiment | Status | Entry point |
|---|---|---|---|
| Q1 | so_r_qa dose-response v2 (2x/4x share, E3 protocol) | **DONE — ADOPTED at 2x** (board 22:1x; manifest rebuilt, so_r_qa 0.012→0.024) | `poc_cma` run_e3.sh pattern |
| Q2 | T1 DAPO zero-std filter A/B | **DONE — DROP** (board 2026-09-01 08:2x: structural degradation, batch-shrinkage mechanism at K=4; production RL keeps zero-std groups) | `docs/research/2026-08-28-slime-miles-adoption-plan.md` §T1 |
| Q3 | GatedNorm ladder arm | **DONE — REJECTED for adoption, stability mechanism CONFIRMED** (board 2026-09-01 16:1x: +9.3/+7.2 BPB standalone, +2.0/+0.2 stacked; stress p99.9 1.19x vs 2.28x; P10 = the identity-init follow-up) | ladder rig + `stress_metrics.py` |
| Q6 | Batch 512k→1M tokens/step + LR probe | **DONE — KEEP 512k** (board 2026-09-01 21:1x: 1M = +9.0% eval loss for ~11% wall-clock; 2x LR doesn't recover; flat-above-optimum does not transfer to 206M scale) | `poc_twin/run_q6_batchprobe.sh` |

Follow chain as posted: GN stress arms → T1 → gn_only retry → Q6.
CPU-parallel: P1 prep live (llama.cpp CUDA build in background; a 3-row
cross-paradigm smoke already exists at `/tmp/poc_diff/cross_smoke.md` —
sft_v3 exact 0.0000; full P1 harness is the next session item).

## 2. PARKED — known to the user, awaiting GO (GPU items also wait for the card)

| # | Experiment | Cost | Why it's queued | Entry point |
|---|---|---|---|---|
| Q4 | Flash E2/E3 (holdout-interference probe ~2.5h; blocked-vs-interleaved SFT A/B) | ~2.5h GPU | RL/SFT-methodology; relevant only when that phase resumes | `docs/research/2026-08-28-flash-derived-poc-plan.md` |
| Q5 | stabtok P2/P3 + T1-tokenizer (~35h GPU) | ~35h GPU | 4x-LR stress margin of pinned recipe (P2 = the only pre-launch value; partially answered by Q3's stress pair); tokenizer compression + forced R-pattern tokens | `poc_stab/` (GatedNorm folds in via Q3) |
| Q7 | RT-2 spec-acceptance probe (n-gram lookup + MTP acceptance on mined r-universe traces) | 1–2 days, NO GPU | Sets training spend for any MTP work + n-gram-first question before A2 freezes the head; 1.4–1.9x rollout speedup if real | design-A §3 (near-term item 3) |

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
| X4 | CAL-full recipe (bias calibration + peak search — the two pieces v1 lacked) | CPU + GPU-minutes | plan-frozen; kill: length-MAE ≤ 100 AND exact ≥ 0.0347; Task 0 (paper+repo recon) mandatory before code | plan §X4 (`poc_ddot/cal_length.py`) |

## 3. PROPOSED — remaining, awaiting user triage

Grounding: POC-DIFF VALIDATED (MD exact 0.0694 vs AR 0.0000; 51–256-tok
spans 0.000 on BOTH arms), POC-DDOT KILLED (position field converges <200M
tok; value routing poisons quality), survey = `nse-ot-flows-survey-2026-08.md`.
P1/P2/P3/P6 promoted to §2c (X-series) 2026-08-31 23:4x. Remaining
suggested order: P7 → P4, P5 → P8, P9. User arbitrates.

| # | Experiment | Class / cost | Why (the datapoint) | Entry point |
|---|---|---|---|---|
| P4 | **Block-diffusion span head** (family D targeted at spans) | GPU ~13–26h + build | The 51–256-tok bucket is 0.000 everywhere; AR anchors structure, diffusion fills within blocks; KV-cacheable | new `poc_bd/` reusing poc_diff data + sampler |
| P5 | **Edit Flows arm** (family C: insert/delete/substitute CTMC) | GPU ~13–26h + new trainer | Survey's named candidate for long spans (+138% over mask-only at 1.3B on code); native variable length + deletions — the primitive AR-PSM and MD both lack | new `poc_editflows/`; survey §3.2 (2506.09018) |
| P6 | **Full CAL recipe** (not our failed v1 half-peak rule) | CPU + GPU-hours on existing ckpts | v1 failed hard (length-MAE 191) but the paper's full recipe reports +47.7% Pass@1 on code infilling; cheap to test on `md_final.pt` | `poc_ddot/cal_length.py` upgraded to the paper's full search |
| P7 | **Data-scale disambiguation rerun**: both twins at 10–20x corpus, ~4 epochs (vs current 44 epochs/44.5M unique) | Largest: corpus build + 2 retrains (~1.8–3.6B tok total) | Tests paradigm-vs-data for the long-span zero before any big arch bet; survey notes the regime is data-starved; DEPENDS on A2 data program (so_r_qa stratum) | `poc_diff/data_prep.py` + A2 `pretraining/` package |
| P8 | **OT as training loss** (FMPE-style continuous relaxation over span embeddings, family F) | GPU, speculative build | Pure white space — no text/code result exists anywhere; the "NSE as OT" idea's last untested form | survey §3.3 (FMPE 2305.17161 + minibatch OT) |
| P9 | **Differentiable edit-distance loss** trained into a code model | GPU, speculative build | Survey white-space item #4 (edit-distance-as-OT exists for graphs/trees, never trained into a code model) | survey §2 white-space list |
| P10 | **GatedNorm-v2 (near-identity init)** ladder arm — σ-init ≈1 (bias the gate) instead of the standard 0.5, same 668-step paired discipline + 2x-LR stress | ~1h GPU | Directly tests the user's scale question (2026-09-01): Q3's +2% BPB cost may be an init transient (σ≈0.5 halves sublayer outputs until learned open — a large fraction of a 350M-token run, <1% of a 13-25B run; Qwen's "standard init suffices" claim was made at 560B tokens). If v2 closes most of the +2%, the cost amortizes at scale and GN re-enters the 25B conversation; if not, the rejection is structural and scale-proof. Stability leg already CONFIRMED (stress pair: p99.9 1.19x vs 2.28x clip) | `ladder/run_gatednorm.sh` + one-line init change in `model.py` GatedNorm |

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
| W7 | **R-eval holdout rule implementation**: packer hook enforcing 2%-by-package holdout on new R corpora (rule proposed 2026-08-31; code not written) | ~2h | enables W6/W8 safely |
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
| W17 | **RL-run-5 design on the new base** — constraints banked: no warm-start (clean negative), no critic blend (v4/v5), keep zero-std groups (T1 verdict), run-2 profile | design + run | board verdicts 2026-08-27→09-01 |
| W18 | **Simulator closed loop (stage 2→3)**: model outputs influence the simulated typist — the RL-integration stage-2 design | build | simulator design.md "RL integration — two stages" |
| W19 | **2c serving productization**: extension acceptance-threshold knob + post-accept cooldown as user-facing config; intent-suite regression watch | ~1 day | the standing serving call's engineering side |

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

## Sync notes

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
