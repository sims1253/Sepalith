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

## Queue-owner sync — 2026-09-08 (continuous ownership)

Owner: `codex-queue-owner`. Legacy dispatch is retired. The separate-agent heartbeat is disabled: the user requested wakeups of the original conversation, not another agent. Local execution uses
`~/.local/state/sepalith/migration-20260906/prepared-state`, one explicit
`run-next` followed by a review pause; ownership continues into the next eligible job. Earlier live PIDs/ETAs below are historical.

- S1: full GPU measurement and corrected M16/M48 ngram sweep DONE; no qualified GPU winner.
  Corrected attempt `1573aca118fd4f06853f4d368fcc015b` replaces the original invalid ngram-depth claim.
  Reduced CPU ngram scout MEASURED (160 recovered rows); execution interrupted, no confirmation nominee. See recovery readout.
- S2: CPU/GPU timings archived. Fresh b4 Q8/stock-Q4/imatrix-Q4 scenario/no-op comparison DONE (1,539 requests); paired 44-case intent comparison DONE; adoption remains unassessed. See queue-owner readout.
- S2 b4 timing follow-up: GPU and CPU DONE (120 requests each); final records linked in queue-owner readout. Adoption remains unassessed.
- W16/V1a: b4 context-eligible baseline DONE, attempt `8c2ed0c3bbaa46c2b1bdc1e5dceb7d70`: 57 whole trajectories, 1,208 points; three original candidates excluded by frozen context audit.
- V1b: saved b4 scenario column completed; 195/255 exact and 204/255 structural matches, with stated normalization limits.
- W24: operating documentation updated; legacy launch guidance superseded.
- V1c: saved v7 column reconciled and archived; calibration/concurrency acceptance unresolved.
- W33: historical input provenance unresolved; original row remains parked.
- W33-N: user-authorized NEW frozen 921-row plain-split evaluation DONE on
  RTX5090, attempt `9c9a385b45b5460982cd565f09111633`. Separate from historical
  W33 and the stopped partial CPU attempt. All 921 rows scored and archived;
  its GPU claim released. Readout: `docs/validation/2026-09-08-plain921.json`.
- Paid cloud dispatch remains blocked. Preserve roughly USD85 Anyscale and
  remaining Azure credit for production; verify balances/expiry before admission.
- Optimization pre-rolls where valuable (user update 2026-09-08): Pi `opencode/muse-spark-1.3-contributor-free` and
  `zai/glm-5.3`, both `max`, with a runtime-appropriate review limit; retain
  failures/timeouts and review/test changes. LOC1-S1 and Benchmark publication
  decisions remain parked.

Evidence and scope: `docs/migrations/2026-09-08-queue-owner.md` on
`t3code/queue-owner-handoff`. Cloud boundary: `docs/CLOUD-BUDGET-ADMISSION.md`.

Last synced: 2026-09-05T20:47+02 (zcode-frm-intel/main — verdict-sync pass, user-requested; status
edits only, nothing fired: FIM-Replica → DONE (falsifier PASSED 3.8x, board
19:35); P10 → RUNNING on the card (zcode-gpushorts); H2 → MOOT-until-weights
per H1's flat branch; §0 decision-surface block added; earlier this session:
X5 intake + P13 cross-note + O-series re-share note. DONE-row pruning left
to zcode-queue-mgr-2's next sync per convention.)

Last synced: 2026-09-04T23:5x+0200 (zcode-queue-mgr-2 takeover — user GO: run the
queue, NEW series prioritized (E/H/O/TU/S). Round-1 working set fired, all
subagent-owned: P12 roofline (quiet window now), TU1 sufficiency judge
(API), O3-S0 RL telemetry, S0 trace freeze, E1/O1 build-only patches, B8
train_sft midtrain patch (base-agnostic, unblocks the winner track on GO).
§1 B-series chain COMPLETE through gate B-β; winner track (B8/B9/W16)
awaits the §1 base-pick GO — GDN/Qwen3.5 b4-config recommended (gate B-β),
asked. Anyscale ~$100 credit-expiry recon running. Prior-sync notes
preserved below.)

---

## 0. AWAITING USER GO — decision surface (added 2026-09-05T20:47 by zcode-frm-intel; status-only view for triage, authoritative rows below)

- **Production-finetune track on the resolved base pick** (GDN/Qwen3.5 b4-config, resolved by elimination per B13; plan `docs/research/2026-09-04-production-finetune-plan.md`) — the headline GO. B8b (stacked midtrain→sft_v7 arm) lands first and closes the midtrain slot either way.
- **PFT1** LoRA-vs-full-FT A/B (opened 2026-09-06 on the user's training-mode question): the production track's first recipe delta — one matched-budget arm vs the banked b4 anchor, forgetting probe included; also decides whether the RL phase rides a full-FT SFT base (RL itself stays LoRA either way, memory-forced).
- **X5** FRM staged probe (intake 2026-09-05 from the user-shared paper): S0 = CPU-only convergence-residual replay on banked `md_final.pt`, chain-safe anytime; S1 behind the chain.
- **S3** Uno diffusion-draft line (opened 2026-09-05 on the user's fork-warrant read): staged so the runtime fork is gated behind cheap legs — S0 prize math (needs S1-CPU legs first) → S1 adapter training (acceptance measurable with NO fork) → S2 fork only if both gates pass.
- **LOC1** Muninn-on-R localization probe (opened 2026-09-06): S0 = zero-cost CPU eval of the released weights on a self-built R localization set; product-feature decision as much as experiment (adds a context-selection dimension the H-series never searched).
- **O2** (pass-rate admission filter) / **O4** (alignment price + novelty audit) — the remaining O-legs; E1's kill explicitly left O2's mechanism untouched.
- **H3-S0** (render-format paper gate) + **H5-S0** (iGPU bench) — CPU-class, chain-safe, per H1's skip-to.
- **B9 / B10 / B11** — fold onto the gate winner after the base-pick GO.
- Landed today, no action: TU2 (raw-route-stands), O1 (killed), E1 (killed), TU1 (dead), V1d (alive), S0 traces (frozen), P12 (kernel-day GO, behind 13B gate), B8 replacement arm (decisive negative). In flight tonight: P10 evals ~23:05; B8b slots behind the shorts chain.

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
| D1 | 206M @ 0.5B undertraining disambiguation (user-reopened; readout pre-registered in script header) | **DONE 2026-09-02 — readout in poc_diff/RESULTS.md §D-grid** (row had stale RUNNING status; flagged by the dashboard-v2 audit 2026-09-05) | `scripts/run_d1_chain.sh` |
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
| Q7 | RT-2 spec-acceptance probe (n-gram lookup + MTP acceptance on mined r-universe traces) | 1–2 days, NO GPU | Sets training spend for any MTP work + n-gram-first question before A2 freezes the head; 1.4–1.9x rollout speedup if real — **TRACE SET LANDED 2026-09-05 (spec_traces/ 1100 paired; S1's rig answers this, fold flagged)** | design-A §3 (near-term item 3) |
| FIM-Replica | FIM-dose replica arm — unmasked-FIM @35% probe2-replica control — **DONE 2026-09-05 19:3x: FALSIFIER PASSED, the adopted 20–35% dose verdict SURVIVES** (pre-registered masked-vs-unmasked ≥2x gate = **3.8x** — masked/unmasked served line-F1 0.0019 vs 0.0005, POC-floor caveat carried, masked@35 still below the banked dose-20 peak 0.0039; COLLAPSE SIGNATURE RESOLVED: the 35% free-running collapse (0.0% stops, median run to the 384 cap) becomes 4.9% stops under masked loss = the dose-10/20 class (~8.1%) reappears at the top dose; TF stop-acc 15.70% ≈ family best on 12.1% of loss tokens; **NEW MEASURED COST: causal-floor BPB 0.7793 = +3.1% vs the unmasked twin (0.7561), +2.25% vs the 0% control — above the ≤1% design band**, partly an effective-loss-budget artifact (~31% fewer gradient tokens at matched data); persistence at 0.5BT is the A2-prime ladder's question; board 2026-09-05T19:35; artifacts `ladder/results_fimreplica.md` + `logs/bpb_eval.json` tag `ladder_fim35m` + `logs/fim_eval_ladder_fim35m.jsonl` + ckpt `/mnt/h/sepalith/runs/ladder_fim35m/final.pt`) | ~1.5h GPU (ladder-arm class) | The falsifier has now RUN and PASSED; NOTE: `dashboard_state.json:21` cell is stale ("untested (replica arm pending)") — flip on the next dashboard refresh | `2026-08-22-night-session.md` §verdict + `fromscratch-design-A2.md:444` |

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
| B8 | AST-FIM midtrain re-probe with the FIXED instrument (masking+packing), on the gate winner — **RUN #1 DONE 2026-09-05 15:4x: instrument VALIDATED, replacement-arm DECISIVELY NEGATIVE (valid 0.8 / exact 0.0 vs b4's 85.1/76.5, McNemar p≈1e-48; noopFP 93.4; format_propagation 3.0 vs granite's 79.1)** — health signature MATCHED (seam exact 48000/48000 vs 0% in the 08-19 bug, prefix-route 100%, completion 12.2%): the instrument fix is REAL and the gates pattern is reusable. MECHANISM: astfim_v1 REPLACED sft_v7 for the full 3000 steps — the model learned span-completion, lost the zeta2 edit-block contract (the pre-registered format-transfer caveat). **B8b STACKED arm (midtrain-THEN-sft_v7, granite's actual structure) FIRED from the banked artifact** (merge midtrain LoRA → fresh sft_v7 LoRA, ~3h, behind the shorts chain) — **B8b DONE 2026-09-06 00:34: STACKING = EXACT b4 PARITY, ADDS NOTHING — MIDTRAIN SLOT DROPS (both structures measured)** (valid/exact 83.1/74.9 vs b4 85.1/76.5, McNemar 12/7 p=0.359 + 13/9 p=0.523 TIE; noopFP decisions row-identical to b4 0/0 discord; format_propagation TIE at 71.6 — granite's 79.1 class does NOT transfer via stacking on GDN; MECHANISM TELL: b8b eval_loss tracked b4's banked curve within 0.001 at EVERY checkpoint = product SFT annihilates the merged midtrain deltas; route to granite's format class = the midtrain-native base (B5) itself, not a stage; attachment 21,823,488 exact 3rd consecutive run; §B8b in RESULTS.md, b8b_verdict.py validated vs B8's published numbers; persistent merged base kept at runs/b8b_stacked_base_merged) | ~4-6h GPU + train_sft patch | RESULTS.md §B8 |
| B9 | SeleKT gradient-importance masking A/B on the winner — **DONE 2026-09-06 06:4x: NO-ADOPT, masking family closed (PLAIN SFT stands)** (token-space adaptation pre-registered in selekt_data.py header: I_t = ‖softmax−onehot‖ at the b4 init state, global keep-50% quantile τ=0.2963, full 21,823,488 attachment — labels the ONLY delta vs banked b4; valid 72.2 vs 85.1 p=2.1e-9, exact 47.1 vs 76.5 p=1e-15 = 29.4pp BELOW the 1.0pp no-harm band; mechanism: 50/77 lost rows stay valid — format learned (high-I tokens), exact content under-fits (the masked 50% = must-be-exactly-right tokens the base already predicts); format_propagation hardest 34.3 vs 71.6 discord 25/0; noopFP row-identical 0/1 discord p=1 — restraint saturated at b4, the guarded failure mode never manifested in b4's battery; per-MODULE restriction variant untested but bracketed by 3 masking negatives (B8/B8b/B9); **winner track now FULLY RESOLVED: base by elimination B13, midtrain DROP B8/B8b, masking PLAIN B9**; §B9 RESULTS.md) | GPU-hours (ran 01:23→06:34, incl. pre-OOM bs2×8 fallback at ckpt-1000) | The one documented post-train-only failure mode (naive SFT loses edit ability) | runbook §2 B9 + RESULTS.md §B9 |
| B10 | WiSE-FT LoRA interpolation (α∈{0.3,0.5,0.7}) | CPU-class, anytime | Multi-task shipping path + the pre-registered RL-phase guardrail | runbook §2 B10 |
| B11 | NextCoder-style R edit-seed pack | API+CPU, anytime | The edit-shaped data construction for post-train v1 | runbook §2 B11 |
| B12 | Spark-X2.5-1.7B-Base rung — **4th arch class** (SWA 3:1 hybrid: 21 SW-512 + 7 full attn, 8Q/2KV, tied emb, Apache-2.0) on the TRL/PEFT trainer path (`train_sft_trl.py`) + PR-build llama.cpp | ~2-3h GPU + ~1.5h calibration anchor | SWA-hybrid vs GDN 3:1 / conv+GQA / dense GQA, sized between B2 (0.8B) and B4 (2B). **PREP DONE 2026-09-01** (weights local, PR #27868 CUDA build, `.venv-spark`, scripts, base Q8 GGUF) and **pre-gate 1 (backend parity) PASSED 3/3**; remaining pre-gate: trainer-stack calibration anchor `b12_cal_minicpm5` (needs GPU). Runbook §2 B12 + `docs/research/2026-09-01-local-builds.md` | runbook §2 B12 |
| B13 | LFM2.5-2.6B-Base rung — **DONE 2026-09-05 07:40: QUALITY TIE, PRODUCT-ELIMINATED** (87.1 valid / 76.9 exact — joins the top-table statistical tie: vs b4 p=0.33/1.0, spark p=0.30, granite p=0.79; B3's 15.7% collapse REFUTED as class property — was capacity + conv-freeze confound. **noopFP 99.0% vs field 58.8–59.8** (propose-always persists at ceiling, load-bearing FAIL) + decode ~11–14 t/s clean-scaled vs 19.2 bar = product-eliminated. B3 FORENSIC: its "full attachment" was 72 modules with ZERO conv.* — unsloth_zoo's get_peft_regex doesn't know parent name `conv`; B13 used raw-regex targets (train_sft.py `regex:` passthrough, default-off); 48,922,624/166 verified all loads. B7 ORDER RULE RESOLVED: not retired, leave-unspent (class-shaped failures; revisit only for a CPU-latency tier). Ops ledger: 2 harness reaps → detached pattern; bs2×ga8 fixes lfm2's 128k-vocab unfused-CE memory (16.8GB flat). **⇒ BASE PICK RESOLVED by elimination: GDN/Qwen3.5 b4-config** (user try-first directive honored: LFM tried, Ouro/K2 disqualified, MiniCPM prior loss) | ~4-5h wall | RESULTS.md §B13 + `2026-09-04-base-candidate-recon-ouro-k2-lfm.md` |

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
| X5 | **FRM recurrent self-conditioning + Fixed-Point Forcing on the MD span head** — **PROPOSED 2026-09-05 (user-shared paper, awaiting GO)**; arXiv 2606.29150 **v2** (version warning: v1 was a different recipe — renoise-CE + FlowDPO; alphaxiv shows v2; digest + non-transfer list: `docs/research/2026-09-05-frm-intel.md`). Staged. **S0 DONE 2026-09-06 04:40: BOTH BARS PASS — the confidence signal + free-latency lever exist TODAY on the banked head** (AUROC 0.860 [0.741–0.952] @64 / 0.897 @32, bar 0.65; early-stoppable 32.9% strict / 28.2% latency-honest, bar 20%; NOT a length proxy — within-11-50 AUROC 0.851 vs 0.568 for length alone; replay byte-valid, banked anchors reproduced exactly 16/216@64 + 15/216@32; fixed-by-k≤8 only 3.2% vs the paper's 97.5% = residual separates correctness but the schedule converges LATE — early fixation is precisely what FPF would add; **51–256 = unconverged regime confirmed** (median r 0.108 vs 0.089, k* 52 vs 20, post-freeze drift 2.05% vs 1.20% = exposure-bias signature; depth 32→64 halves residual + triples stoppability = long spans USE depth, S1's ≤2×-NFE bar binding exactly where lift-off is needed); **POST-HOC ABSTAIN SIZING bankable now: 28.5% of incorrect rows suppressed at 0/16 correct lost, 48% @1/16, 94% @3/16**; S1-GO recommended by the probe — value = push 0.86→1.00-class + create early fixation + the 51-256 lift-off, bars unchanged; `poc_diff/X5_S0_RESULTS.md`) — original S0 pre-registration (CPU-only, banked-artifact replay, chain-safe anytime): rerun banked `md_final.pt` on the 216-row harness with the existing remasking schedule instrumented — per-row adjacent-step residual r_k = SKL(p_k, p_{k−1}) (their Fig 6 instrument); readouts = (a) AUROC of final-step residual vs span-exact correctness, (b) commit-when-stable early-stop fraction (rows whose argmax stops changing ≥2 steps before the schedule ends). Their numbers: AUROC 1.00 under FPF vs 0.50 vanilla; fixed point by k≤8 in 97.5% of examples. **S0 is diagnostic either way**: AUROC ≥0.65 OR ≥20% early-stoppable rows = the confidence signal + free-latency lever exist TODAY on the banked head (feeds the noopFP-abstain question + S-series latency); ≈chance on both = stability is FPF-created, S1 carries the whole claim (flag, not kill). **S1 self-cond + FPF arm (GPU ~4–6h + `model_md.py` surgery, continuation from `md_final.pt`, behind the §1 chain per W37)**: zero-init carry channel ingesting prev-pass raw span logits (their appendix-A recipe: raw logits, zero-init W, stochastic two-pass training); Stage A = two-pass self-cond continuation → Stage B = FPF (carry = stopgrad rollout prediction from t_start ~ Beta(2,2) capped at supervision t — train on own inference states, objective otherwise unchanged); eval = the 216-row harness verbatim + recurrent depth k∈{2,4,8}. **Verdict: WINNER-SC iff exact > 0.0741 (banked MD@64 anchor) at ≤2× anchor NFE AND the 51–256 bucket lifts off 0.000 (first nonzero long-span anywhere in the program); win only at ≥4× NFE → latency-negative, write-the-negative; 11–50 regression vs 0.110 → KILL.** Secondary readouts: post-FPF residual AUROC (paper predicts →1.0) + residual-gated abstain on no-op rows (noopFP analog, H-series adjacent). Conditional follow-ons, NOT pre-registered until X5 lands: renoise-CE best-of-N verifier (v1; serving-cost conflict), FlowDPO wrong-cell DPO with AST-aligned token masks (v1; code has no vertex correspondence — `ast_equiv.py` tree-diffs would supply the mask). Recon pre-task when arming S1: Yoo et al. 2026 (self-conditioned flow-map LMs, v2-cited concurrent work — the language-domain existence proof; id unlisted, fetch first) | S0 CPU-hours (banked ckpt); S1 ~4–6h GPU + model surgery | The 51–256 bucket is 0.000 paradigm-wide; the paper's exact claim = recurrent carry + FPF converts depth into accuracy precisely where aggressively-parallel token updates are inconsistent (our long-span failure mode); plus the only mechanism class offering a verifier-free confidence signal for abstain/noopFP. Survey tension carried, not edited: §3.5.5 "exposure bias red herring" stands for AR-vs-MD, amended-by-note for the self-conditioning loop (intel doc §2 M2). + **EqR cross-ref 2026-09-06** (2605.21488, ICML'26, locuslab): the attractor baseline FRM benchmarks against — independent replication of fixed-point-residual-as-correctness (X5-S0's instrument), **ACT halting 1024→58.7 NFE at ~0.8pt cost = the adaptive-stopping precedent** for the commit-when-stable readout, and noise-injection (λ=0.05, β=0.01) + randomized-init = candidate S1 stabilizers (`2026-09-06-intel-batch-3.md`) | `model_md.py` + `eval_spans.py` + banked `md_final.pt` + `docs/research/2026-09-05-frm-intel.md` |

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
| SY1 | **ry_diagnostic_repair — ry as synthetic-data oracle, first semantic case family** (added 2026-09-06 from the user's three-model idea screen; build + smoke DONE same day, verdict in the results doc; awaiting GO for full-scale scan + mixture inclusion) | CPU only; full-corpus scan ~40 min | Every existing family is syntactic (tree-sitter + text). This one's eligibility oracle is the product's own checker: a row exists only where a deterministic corruption (is.na→`== NA` RY034, paren-move RY093/RY100, column typo RY060) provably adds EXACTLY one new ry diagnostic on the mutated line, and the target restores the verbatim corpus original (tier-1 exact, diagnosed-by-construction/repaired-by-construction). The rendered diagnostic rides the prompt (the VS Code product surfaces ry squiggles at the cursor — no other edit model can train on this signal); a blind twin spec shares the scan cache for the guided-vs-blind ablation. Smoke: 1667 real CRAN files / 77 pkgs / 228 s → 135 verified rows, 0 gate rejects, resume + cache-share verified, 48/48 tests. Screening casualties on record: RY090-class mutations NOT buildable at ry 0.8.0 (typeshed formals too thin — probed), cross-file counterfactuals + semantic no-ops need a package-level sidecar index (wave-2) | `cases/specs/ry_diagnostic_repair{,_blind}.json` + `results/cases/RY_DIAGNOSTIC_REPAIR.md` |
| P4 | **Block-diffusion span head** (family D targeted at spans) | GPU ~13–26h + build | The 51–256-tok bucket is 0.000 everywhere; AR anchors structure, diffusion fills within blocks; KV-cacheable | new `poc_bd/` reusing poc_diff data + sampler |
| P5 | **Edit Flows arm** (family C: insert/delete/substitute CTMC) | GPU ~13–26h + new trainer | Survey's named candidate for long spans (+138% over mask-only at 1.3B on code); native variable length + deletions — the primitive AR-PSM and MD both lack | new `poc_editflows/`; survey §3.2 (2506.09018) |
| P6 | **Full CAL recipe** (not our failed v1 half-peak rule) | CPU + GPU-hours on existing ckpts | v1 failed hard (length-MAE 191) but the paper's full recipe reports +47.7% Pass@1 on code infilling; cheap to test on `md_final.pt` | `poc_ddot/cal_length.py` upgraded to the paper's full search |
| P7 | **Data-scale disambiguation rerun** — corpus dependency RESOLVED 2026-09-02 (strata already on disk at `/mnt/h/sepalith/a2/r/`, 7.55B tokens available per mixture smoke; D-series micro-arms running first as the cheap probe: see §1 D1/D2/D3) | retrain class | Tests paradigm-vs-data for the long-span zero; survey notes the regime is data-starved | `poc_diff/data_prep.py` + a2 strata |
| P8 | **OT as training loss** (FMPE-style continuous relaxation over span embeddings, family F) | GPU, speculative build | Pure white space — no text/code result exists anywhere; the "NSE as OT" idea's last untested form | survey §3.3 (FMPE 2305.17161 + minibatch OT) |
| P9 | **Differentiable edit-distance loss** trained into a code model | GPU, speculative build | Survey white-space item #4 (edit-distance-as-OT exists for graphs/trees, never trained into a code model) | survey §2 white-space list |
| P10 | **GatedNorm-v2 (near-identity init)** ladder arm — σ-init ≈1 (bias the gate) instead of the standard 0.5, same 668-step paired discipline + 2x-LR stress — **DONE 2026-09-05 22:45: REJECTION STANDS (structural, scale-proof)** (causal 0.7644 vs plain 0.7533 = **+1.5%**, recovers only ~27% of v1's +2.0% → the user's scale question answered with data: NOT an init transient. **BUT GN-v2 strictly dominates GN-v1** — FIM slice 0.7496 vs plain 0.7527 = −0.4%, beats plain; if GN re-enters at 1.5B/25B it enters AS V2 with the causal slice as the bar. Stress 2x-LR: p99.9 1.56x vs plain 2.28x = mechanism survives with less margin than v1's 1.19x; pinned plain+QK-Clip recipe unchanged; σ=0.9817 at init = RMSNorm parity, cfg-gated +36,864 params so banked GN ckpts load unchanged; `poc_twin/ladder/results_gatednorm_v2.md` + ckpts `/mnt/h/sepalith/runs/gatednorm/{gn2_qk,stress_gn2}`) | ~1h GPU | Directly tests the user's scale question (2026-09-01): Q3's +2% BPB cost may be an init transient (σ≈0.5 halves sublayer outputs until learned open — a large fraction of a 350M-token run, <1% of a 13-25B run; Qwen's "standard init suffices" claim was made at 560B tokens). If v2 closes most of the +2%, the cost amortizes at scale and GN re-enters the 25B conversation; if not, the rejection is structural and scale-proof. Stability leg already CONFIRMED (stress pair: p99.9 1.19x vs 2.28x clip) | `ladder/run_gatednorm.sh` + one-line init change in `model.py` GatedNorm |
| M1 | **Micro-specialist probe (mdlARC-derived)** — **DONE 2026-09-02: KILLED** (board 16:4x; both arms exact 0.0000 vs 0.0694 anchor, kill test "M1b<0.0347" fired; curation delta soft +0.024 edit-sim, 0 exact; scale/general-corpus exposure load-bearing; M-series closed, W5 no curation term, P7 corpus-scale direction stands) | ran 2026-09-02 | `poc_diff/RESULTS.md` §M1 + plan doc |
| V1 | **Eval-strategy v2 (feel-of-use battery)** — the user's 2026-09-01 directive: current non-RL evals may not represent how good the model feels to use; keep existing for anchoring, add the missing axes. Landed pieces: full raw outputs in the battery scripts ✓ · V1a episode metrics ✓ 2026-09-03 (judge_loop + episode_metrics.py wrapper; time-to-edit + interruption rate added; BOTH calibrations reproduced — banked FP-gap 0.99→0.44 p<1e-7, live v8_2 8-accepts-vs-0; battery one-liner in docstring; caveats: n=40 subsample, 2048-slot rejects biased to smaller ctx) · V1b AST-equivalence ✓ (+14.9pp format_propagation recovery) · **V1d blind pairwise preference ✓ 2026-09-05: instrument ALIVE** (sanity anchor gt-vs-corrupted 43–0/49; calibration v8_2-vs-base 0.696 [0.618–0.764] = intent-suite-consistent, typing 67–3 / noop 36–2; v7-vs-rl_v2c DIVERGES from intent suite (0.809-vs-0.511) but agrees with FP-discipline — rl_v2c preferred on both-proposed points = additive axis not redundant; position bias mild, 0/5.4/10.1% flips, absorbed by both-orders consistent-win rule; gemini-3.7-flash via agy — glm-3-gate calibrated 120/120, zai quota avoided; 690 calls / 27 tests; NOTE: future arms need V1a-style per-point rows for a V1d column; `eval/V1D_RESULTS.md`). Remaining legs: (1) judge_loop episode metrics as battery member (accept_rate/fp_rate/saved_ratio + time-to-edit), (2) AST-equivalence/parse-validity re-scorer, (3) TTFT + concurrent-load bench, (4) blind pairwise preference on the known-good pair, calibration on v8_2-vs-base + v7-vs-rl_v2c | ~1 day build + CPU/API per model | Every gate from here (B-α onward) gets additive v2 columns without changing pre-registered verdict rules; retroactively applicable to all arms whose raw outputs are stored | design doc `docs/research/2026-09-02-eval-strategy-v2.md` (inventory-backed: simulator + judge_loop + panel judges already built, never in the battery cadence) |
| KV-AR | KV-cached AR re-baseline (honest serving-latency table for the poc_diff AR arm; current 7689ms p95 is cacheless) | CPU/GPU-minutes | Sharpens X1's cross-paradigm latency readout; gated by X1 landing | `poc_diff/RESULTS.md` "Next" |
| P11 | **Vocab probe — R-weighted 32K vs generic (tokenmonster)** — staged, pre-registered. **S1 tokenizer-only** (CPU ~half-day, chain-safe): build three vocabs on a FIXED corpus sample — G32 (generic BPE slice) / R32 (tokenmonster filtered fusion, R-weighted from CRAN `normalized/` + Stack R token stats) / G65 (size control); measure bytes/token density, single-token rate on R-signature tokens (`%>%`, `<-`, `[[`, top tidyverse identifiers), byte-fallback rate, on held-out R text NOT in the tokenizer sample. **Gate A: R32 < +3% density on R text → KILL** (generic vocab, spend nothing further). **S2 LM legs** (only if S1 passes; ~2-3h GPU behind the chain): three M1-class 76M arms, fixed 0.2B tokens, identical corpus, scored in **BPB** on held-out R + a general-text control. **Verdict: WINNER-R32 iff >2% BPB better on R AND <1% regression on the general control; else generic** — bpb makes cross-vocab comparison legal by design (the exact trap the tokenmonster thread hit with token-loss) | CPU half-day + GPU ~3h conditional | A2's vocab is a one-way door and the only datapoint is LFM vocab-scaling (survey Q4). tokenmonster record: tokenizer-only change ≈ 40% token-efficiency, realized as a filtered 28,416 vocab. Carries the M1 lesson the other way: external priors do not auto-adopt — the probe IS the instrument version of the vocab question | `experiments/training/vocab_probe/` (to build) + `docs/research/2026-09-02-external-intel-batch.md` §4 |
| P12 | **Roofline profiling half-day** — **DONE 2026-09-04 23:15: KERNEL-DAY GO via the Q4/Q8-realization arm** (Q4_K_M vs Q8_0 = 1.37–1.45x < 1.5x at every thread count; bandwidth arm clean — Q4 implied 110–126% of triad, NOT <70%; WSL2 tax: triad 25 GB/s vs ~43 bare-metal; Q4 shortfall = dequant-kernel efficiency, prize +16% if closed; CPU-tier A/B prior = {Q8_0, Q4_K_M, Q4_0} — Q6_K dominated by Q5_K_M, Q5_K_M pp-worst; report `2026-09-04-p12-roofline-results.md`; GO sits behind the 13B gate per pre-registration) — original pre-registration: llama-bench matrix on the b1_ref24 GGUF: {Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0} × {t4, t8, t12} × {pp512, tg128}, 3 reps, **idle-box window only** (no TRAIN leg active; the contended-box convention corrupts roofline math — coordinate on the board before running). Report implied GB/s (= tg t/s × weight-GB) and %of-STREAM per format, plus the Q4/Q8 realization ratio (speedup achieved vs bytes halved). **Verdict outputs: (a) CPU-tier prior table for the 3-arm quant A/B (§5, design-A §312); (b) kernel-day GO iff Q4 implied bandwidth <70% of STREAM or Q4/Q8 speedup <1.5x; NO-GO otherwise until the 13B gate** | CPU ~2-3h idle window, zero GPU | Q8_0 decode is already at/near roofline on this box's 5900X (implied ~36-44GB/s demand vs ~40-48 achievable; board 11:0x). Open questions: per-format utilization, and whether Q4-class actually realizes its ~2x bytes advantage — that number prices the CPU tier of the ship matrix AND gates any GGML kernel work | `scripts/roofline_bench.sh` (to build) + `docs/research/2026-09-02-external-intel-batch.md` §5 |
| P13 | **Looped-depth probe (compute-for-bandwidth)** — staged, pre-registered. **S0 CPU paper gate** (~half day, chain-safe, may run anytime): redo the design-A2 §4.3 bytes/token table for K=12×R≤2 and K=8×R≤3 vs the 24L dense reference (Q4_K_M + Q8_0, 8K, q8 KV) counting REPEAT weight loads (R·W_K) + full N-layer KV *and* the KV-shared variant + head; model the adaptive-R target (exit-rate grid E[R], the (exit-rate, quality) point the loop must hold to beat dense at equal bytes); serving spike (confirm GGUF cannot express tying — expected no — scope the K-block/R-pass/per-pass-KV/exit-hook runner fork in days). **Kill: no quality-adjusted average-bytes win even optimistic, OR fork we won't own → KILL before any GPU.** Footprint-only variant may PASS explicitly labeled "capacity, not tokens/sec". **S1 ladder quality probe** (ONLY if S0 passes; ~1-2h GPU/arm at 76M, strictly serial behind D3→B4→B5 per W37): poc-twin/micro rig, fixed order + seed 1273 + fixed vocab/corpus; arms dense-K (A) vs looped K×R layer-loop-first w/ DeepLoop scaling (B) vs model-loop iff cheap (C) vs dense-N compute-matched ref (D); held-out R BPB + general control + task battery + stabtok stability telemetry; quality-only (training tok/s claims nothing about looped serve). **Verdict: WINNER-LOOP iff beats (A) clearly AND stands against (D) at equal streamed bytes (hits S0's target); beating only an untuned loop baseline = FAIL** (McNemar go-forward rule, §5). **Latent-feedback branch** (2608.08888, <1%/tok gated fusion) conditional on S1-positive-plus-heavy-fork — answers recurrence-vs-passes; never jumps the order. Sits BEHIND the queued bytes/token levers (quant A/B+QAD, single-mode spec/Q7, Matryoshka draft, SWA KV, P12) until S0 shows otherwise; R1 + W37 hold throughout | S0 CPU ~half day, zero GPU; S1 GPU ~4-8h conditional, behind the §1 chain | User 2026-09-03 question: bandwidth-bound decode → trade compute for memory? DeepLoop (2607.13491) is stability-only (no memory/FLOP numbers); honest accounting: fixed-R looping streams the SAME bytes as dense at matched unrolled depth (repeat loads R·W_K = W) — only adaptive-R exit + KV-sharing can win average bytes, and stock GGUF unrolls to duplicated layers (no loop op, verified vs vendored tree) so any win needs a runner fork. Wiki cluster backs the staging (loopie layer-loop + compute-matched recipe; HRM-Text-1B 09-02 note; latent-feedback as the cheap sibling). **+ SMELT 2609.01343 (2026-09-04): middle-50%-twice MoE under triple budget-match (FLOPs/params/KV), 6.8–18.0% frontier CE gain, code leads at 20.4%, long-sample + ICL tilt, sink-collapse mechanism; does NOT redeem dense looping (MoE-only recourse, no wall-clock match) — S0 gains a shrink-to-equal-quality conversion, S1 gains a middle-loop topology arm, kill rule unchanged; see plan §6; re-fetch 2026-09-05 adds the arm's design constants: r=2 ONLY (r=3–4 hurts — budget-matched narrowness), loop residuals scaled 1/r, second-pass expert reuse 25–40% (= refinement not repetition), second-pass residuals 1.2–3.5× larger + directionally aligned — `2026-09-05-intel-batch-2.md` §2** + **L-only asymmetric variant (2026-09-04, §7): loop blocks 16–23 only, S/M taps byte-identical (loop is post-exit in `trunk_taps`), M doubles as the free control, S draft path untouched — preferred loop form if S0 passes, still gated by S0, no separate experiment** | plan `docs/research/2026-09-03-looped-depth-probe-plan.md` (survey: DeepLoop + ALBERT/Huginn/MoEUT/Relaxed-Recursive/Griffin + wiki looped cluster + SMELT §6; FRM 2606.29150 cross-note 2026-09-05, `2026-09-05-frm-intel.md` §4: recurrence-in-SAMPLER = adaptive depth with static weights and NO GGUF fork, but flow-line-only (X5) and pays accuracy not streamed bytes — S0 bytes math unchanged; Graph Machine 2609.02881 (2026-09-06): differentiable-edge sequence-model class, KV-access bytes ~0.1% of dense — aimed at our streamed-bytes pain but WATCH-ONLY (no code, prototype kernels SLOWER than dense wall-clock, loss-only evidence at 0.6B/15.7B tokens, +0.003-nat margin) — `2026-09-06-intel-batch-3.md`; SWA-beats-linear-attention 2608.28444 (2026-09-06 addendum): SWA+sinks ≥ linear-attention retrofits everywhere, 2–10× on NIAH/BABILong — strengthens the SWA-KV lever prior, quiet vote for the spark class if the W36 GPU-offload tier opens, anti-retrofit caution for A2 arch design) |

| LOC1 | **R-localization embedder probe (Muninn-on-R)** — **PROPOSED 2026-09-06, awaiting GO**; staged. Premise (`2026-09-06-intel-batch-3.md` §Muninn): Brokk's Muninn = 346M (voyage-4-nano base) + Muninn-small **47M CPU-tier** (granite-embedding base) repo-localization embedders — English query → function-level chunks; recipe = synthetic queries FROM COMMIT HISTORY, gold functions via Bifrost co-edit/import analysis, **margin-MSE distillation from a teacher (their largest single gain)**, multi-positive contrastive, mined+filtered negatives; trained <$100 on 8×4090; Apache-2.0, HF weights. NO R among their 11 languages → weights won't localize R directly; the RECIPE maps 1:1 onto r-universe/CRAN (commits are our native unit). Why: the extension's 2-8K window is cursor-local today (pin/outline/caps = H-series harness space) — a 47M CPU retriever ranking R functions for an English intent is the missing context-SELECTION half; margin-MSE-from-teacher is also a loss-level upgrade candidate for any future scoring distillation leg. **S0 zero-cost probe (CPU half-day, chain-safe)**: (a) run Muninn/Muninn-small AS-IS on a self-built R localization set (50–200 query→function pairs from CRAN commit messages; Bifrost co-edit gold where cheap) — prices cross-language transfer of the released weights; (b) sizing prior banked (Embedder's Dilemma 2608.12875: 212–596M embedders ~3.5pts off frontier at 10–100× lower cost than LLM-scorers; no code-similarity task in their 37 → our own eval is the only evidence). **Gate: recall@10 ≥ 0.3 on R → the released weights might serve as-is (cheap product trial); below → S1 rides the recipe only.** **S1 recipe-on-R pilot (API ~$50–100 + GPU half-day)**: synthetic queries from mined r-universe commit intents (glm-5.3 generator, O-series hygiene rules), gold via co-edit/import graphs, margin-MSE distillation from a frontier embedding teacher, 47M + ~300M arms, Quarry-style held-out R benchmark (disjoint repos). **Verdict: WINNER-LOC iff R recall@10 beats BOTH Muninn-as-is AND a BM25/ripgrep lexical baseline (query-level paired test) AND CPU-tier latency fits the keystroke budget (H3 doctrine); else localization stays lexical (ripgrep) and the line closes** | S0 CPU half-day; S1 API ~$50–100 + GPU half-day | **New-capability flag**: retrieval-into-context changes the H-series harness search space (H1 found the extension-only space near-optimal for cursor-local context — this ADDS a dimension H1 never searched) — a product-feature decision as much as an experiment | `2026-09-06-intel-batch-3.md` §Muninn + HF brokkai/Muninn{-small}/Quarry + Bifrost + `experiments/data-mining/` commit-intent miners |
| PFT1 | **LoRA vs full fine-tuning at matched budget — the training-mode A/B** (user question 2026-09-06: "do we just want to LoRA or do full fine-tuning/RL/posttrain?"; analysis `2026-09-06-lora-vs-fullft.md`; the production plan's first recipe-delta — §2 currently hardcodes LoRA "for comparability" and full FT has NEVER been run on any external base). **Arms (b4-config base, sft_v7 data, 3000 steps, seq 2048, seed 3407)**: (a) the banked b4 LoRA arm AS the anchor (no rerun); (b) full-FT: bf16, paged 8-bit AdamW, gradient checkpointing, lr 1.5e-5 cosine (single pre-registered value, ONE rescue at 5e-6/3e-5 if divergent-or-flat); memory pre-registration ~14–18GB expected, peak-watch MANDATORY (B8b's LoRA transient hit 32.1GB). Build item: `train_sft.py` full-FT mode (unsloth path is get_peft_model-bound; TRL path trains without PEFT natively — b12 precedent, knobs re-check), half-day class. **Readouts**: eval_scenarios exact + per-family (McNemar vs b4), noopFP, V1a episode metrics, **forgetting probe = causal-floor BPB on held-out general R text + general-text control** (existing bpb_eval machinery — the pre-named failure mode of full FT). **Verdict: WINNER-FT iff exact > b4 LoRA (McNemar) AND noopFP not worse AND general-R BPB regression ≤1%; LoRA stands otherwise — and wins the non-exact axes by default (B10 WiSE-FT, B8b-style stacking, runtime adapter selection are all adapter-native).** Interaction decisions pre-registered: WINNER-FT → RL stays LoRA-on-FT-base (memory-forced: full-FT GRPO + frozen ref does not fit 32GB), midtrain slot re-litigates under FT, adapter paths keep a LoRA sibling; LoRA stands → uniform recipe is load-bearing, question closed at this scale, reopen only if noopFP/doc_sync later show capacity signatures. Evidence context: B3 forensic (adapter targeting silently missed ALL conv.* = the LoRA operational-risk class, hit once, fixed in B13) vs B8b telemetry (eval_loss tracks b4 within 0.001 regardless of merged base = weak hint the adapter is the binding object) | one ~2–3h GPU arm + half-day build + CPU battery | The one recipe delta that changes WHAT the trained object is; cheap because the anchor is banked; also the adjacent disambiguator for the B8b attractor reading | `2026-09-06-lora-vs-fullft.md` + `train_sft.py`/`train_sft_trl.py` + b4 banked artifacts + bpb_eval |
| E1 | **EL-scheduler retrofit on rl_smoke.py** — **DONE 2026-09-05 07:30: KILLED** (pre-registered rule fired: psg_rate ordered 0.3650 vs random 0.4225 = **−13.6%** at matched 1,600-rollout budget; MECHANISM: the ≥75%-fully-solved admission gate demands ~0.93/completion exact on tiers that start lower — format (highest psg density 0.463, most SFT headroom 0.522 exact) got **0/1,600 completions vs 20.5% under random** = the scheduler starves the variance GRPO needs; reward/exact deltas were mix effects (budget on near-ceiling pipe/rename); rename admitted step 6, format never; random fixed-quota stands as the default; **E2 stays CLOSED — no evolver GO**; O2's pass-rate FILTER unaffected (different mechanism, composes under random); `rl/results/E1_RESULTS.md` + reusable `e1_readout.py`) — original pre-registration: ordered difficulty scheduling vs the current random fixed-quota draw. Order families pipe_rewrite → rename → format (plus compound when present); admit the next tier only when the current one exceeds a pass-rate threshold (paper: 6/8 over 8 rollouts; adapt to our num_generations=4 groups). Pre-registered readout: partial-solved-group rate + mean reward over the first 50 steps (paper Fig 5). **Kill: no partial-solved-rate gain vs random at matched rollout budget → KILL before any evolver build** | GPU ~smoke + 50-step short run, serial per W37; sampler patch only, no new env infra | Cheapest paper win; directly tests whether our pipe-near-ceiling vs format-headroom split is wasting GRPO group variance today | `experiments/training/rl_smoke.py` (sampler + MetricsCb) + `experiments/training/rl/README.md` |
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
| H1 | **Harness-search method bake-off** — **DONE 2026-09-05 16:xx: WINNER = (a) single-lineage hill-climb; PROMPT-ONLY DID NOT WIN** (held-out exact: hill 0.8353 = population 0.8353 (tie broken on 8.5% fewer proposer tokens + intent frac-2 0.727 vs 0.705); GEPA won D_harness +3.12pp then INVERTED held-out −0.78pp (pipe 1.0→0.83) = **prompt search overfits 256-row temp-0 harness sets — H4 winners must be held-out-confirmed**; FLAT BRANCH: hill/population 0.0pp vs default held-out ⇒ extension-only space already near-optimal for frozen v7-class weights ⇒ **H2 regime question MOOT until weights move**; per plan skip to H3-S0/H5-S0/H4-decisions; noopFP+p95 guardrails did real work (blocked every outline config); doc_sync 0.0 at EVERY config = model gap not harness gap; recommended H2 config: defaults unchanged (max_tokens 320), hill-climb M=3 I=13 for re-search after weight moves; `harness_search/H1_RESULTS.md`) — original pre-registration: three methods on the FIXED extension-only space (caps, truncation direction, scope on/off, max_tokens/stops, parse rules; render markers FROZEN to keep the model in-distribution): (a) single-lineage hill-climb (pi-autoresearch pattern, greedy best-so-far), (b) population evolution (ShinkaEvolve pattern: islands/archive, M=3 candidates/iter, revert allowed), (c) prompt-only GEPA (text subcomponents only — the FST-style control). Fixed 256-row D_harness (carve from scenarios/intent families, disjoint from eval; `holdout_rule.py` applies), scoring = edit exact + noopFP + latency; selection on harness-train, verdict on HELD-OUT eval with proposer compute counted. **Verdict: the H2/H4 method is the held-out winner; (c) winning means the space is prompt-bound → shrink the space and close the code-search line** | CPU/API-class, chain-safe (~40 candidates/arm via CPU llama-server + proposer API) | Decides the search machinery before spending regime/re-render budget; pre-registers the expectation that code-search beats prompt-only (the WHALE FST gap, 4–13pp) or the space is mis-scoped | `experiments/harness_search/` (to build) + `eval_noop_fp.py` + `cache_bench.py` |
| H2 | **Harness-only diagnostic (regime test)** — **MOOT UNTIL WEIGHTS MOVE (H1 flat branch, 2026-09-05)**: hill/population landed 0.0pp vs default held-out ⇒ the extension-only space is already near-optimal for frozen v7-class weights; per plan skip to H3-S0 / H5-S0 / H4-decisions; re-open only after a weight update, as H1's recommended re-search (hill-climb M=3 I=13, defaults unchanged) — original pre-registration: H1-winner method, weights FROZEN at the B-α winner (smoke now on the v7 class): I≈12–18 iters × M=3 on the extension-only space, same scoring as H1. **Verdict: harness-dominant iff harness-only Δ matches/exceeds the reference SFT Δ at <10% of a full-SFT cost → H4 GOes harness-first; flat → weight-dominant → one small SFT (0.2 epochs) then re-search (the catalytic test); re-search still flat → space exhausted, close the line** | CPU/API-class, chain-safe | The cheap WHALE diagnostic: tells us which regime R-edit prediction lives in before any joint spend (their SearchQA matched peak weight-only at ~6% of rollouts; their Math moved only after a weight update) | same rig as H1 |
| H3 | **Render-format liberation (break zeta2)** — render_zeta2 is anchoring, not optimality: all SFT data, all evals, and the extension-parity rule are keyed to it, and its suffix-first ordering is cache-hostile per `docs/prompt-format.md`. **S0 paper/cache gate** (CPU, chain-safe): candidates = orderings (prefix-first PSM vs suffix-first zeta2 vs hybrid) × marker vocabs (merge markers vs numbered vs PSM tags) × history placement × cursor encoding (signature empty-region vs midtyping partial); score zero-shot format-fail + `cache_bench.py` arms + byte math, kill obvious losers. **S1 adaptation** (GPU ladder-class, behind the chain): top 3–5 survivors each get a SHORT adaptation SFT (0.2-epoch LoRA on re-rendered sft_vX, fixed seed), then held-out edit replay + noopFP + latency. **Verdict: WINNER-RENDER iff ≥ parity on exact AND a serving win (or a quality win large enough to pay the retrain + Zed-path cost); else zeta2 stays and the line closes. Ships to vscode/positron immediately (full prompt control); Zed via W34 PR or second checkpoint** | S0 CPU half-day; S1 ~1h GPU/survivor behind the §1 chain | Prices exactly what breaking zeta2 costs (re-render + adaptation + re-eval); the PSM-vs-zeta2 cache question is otherwise argued from principle, never measured | RENDER registry in `run_eval.py` + `context_build.ts` + `train_sft.py` re-render |
| H4 | **Full WHALE alternating loop** — alternate filtered-SFT (E=0.2–0.6 epochs) with the H1-winner harness search (I=6) for K cycles, vs stagewise control (full SFT then full harness search once) at matched cumulative budget (rollouts + GPU-hours + proposer compute counted — the paper excluded proposer cost). Adaptive arm: patience rule (min 0.2 epochs / 6 iters, patience 0.2 / 2, training-signal only, no validation peeking). **Verdict: WINNER-WHALE iff beats the stronger single-component baseline AND beats stagewise at matched budget; else joint optimization closes and components ship separately** | GPU-heavy, behind §1/B-series; fires post-B-α on the winner (pilot on the 76M poc rig if cheap) | The actual joint-optimization claim, with the paper's two load-bearing comparisons (alternating-vs-stagewise, adaptive-vs-fixed) re-run on our task and our budgets | `train_sft.py` + `experiments/harness_search/` |
| H5 | **iGPU serving tier (re-target the ship matrix)** — **UPDATED 2026-09-04: hardware exists.** User-measured on a notebook (mobile AMD, ASR models 600M–3B): iGPU = 2–3× CPU — but that is compute-bound encoder work; our decode is bandwidth-bound and an APU iGPU shares system DDR, so the transfer is plausible for prefill, unproven for decode. **S0 = verify on OUR pattern** (one llama.cpp Vulkan build on the notebook): llama-bench {Q8_0, Q6_K, Q5_K_M, Q4_K_M} × {CPU t8, iGPU, hybrid} × {pp512, tg128}, 3 reps; PRIMARY metric = the cache_bench keystroke-cycle (prompt_ms + 16-tok gen), not raw tg alone. **Kill: median iGPU speedup on the primary <1.5x → CPU stays primary, close.** **S1 cache re-read** (only if S0 passes): `cache_bench.py` arms a/b/c/d on iGPU — does prefix-first still win when prefill is 2–3x cheaper? **S2 re-price**: P12 roofline + quant-A/B priors + the prompt-format ordering argument recomputed for the iGPU tier. Outputs feed H3's serving term and H4's cost accounting | S0 = one Vulkan build + bench on the notebook (CPU-class, chain-safe); S1/S2 CPU-class after S0 | Every serving conclusion in the repo (prompt-format.md ordering, P12 roofline, quant A/B priors) was derived for slow-CPU prefill; the user's 2–3x measurement, if it holds on the decode+prefill mix, re-prices all of them | plan §6 `docs/research/2026-09-04-whale-harness-weight-plan.md` + `scripts/igpu_bench.sh` (to build) + `cache_bench.py --backend` |

Order: H1 → H2 (regime) → H3-S0 anytime / H3-S1 behind the chain →
H4 post-B-α → H5-S0 once hardware exists, H5-S1/S2 after S0. Build
prerequisites fold into the arms (D_harness 256-row carve, render
registry, adaptation re-render script) — no separate W items.

### O-series — OPD / One-Shot transfer to GRPO (user request 2026-09-04)

Premise: THU OPD line — `2604.13016` (Rethinking On-Policy Distillation:
phenomenology, mechanism, recipe; code `github.com/thunlp/OPD`) +
`2609.04172` (One Training Example, submitted 2026-09-03; code
`github.com/Thinking-Space/One-Shot-OPD`). Both distill WITH a teacher
that gives a dense per-token signal on EVERY rollout (even wrong or
unanimous ones) — "data-overfed but algorithm-starved". Our GRPO
(`experiments/training/rl_smoke.py`: reward = exact + 0.2*line_f1,
no teacher) gets zero advantage on unanimous groups, so the algorithm
does NOT transfer — only (a) embedding-diverse prompt selection, (b)
the unanimous-group guard, (c) suffix-entropy/length telemetry plus the
cold-start / template-alignment hygiene we already bank (merged v6 SFT
base → fresh LoRA; BOS assert; byte-identical zeta2 renders).
Explicitly NOT transferred (pre-registered): empty-`<think>`/WildChat
trick (dense-teacher-only, zero reward under sparse exact), hard
never-solved queries (OPD-positive, GRPO-fatal), full-vocab KL
(O(batch*len*vocab) memory; paper's sampled-token ≈ top-16). Holdout
authority unchanged (`sft_v3/eval.jsonl` exclusion in `build_dataset`);
canonical render decisions belong to H3, never here. Re-share note
2026-09-05 (paper re-read, still v1): one unmined pointer — the one-example
paradigm's RLVR origin, Wang et al. arXiv 2504.20571 (NeurIPS 2026), the
teacher-FREE version of the claim; O1 already prices the practical question
at our scale (16-pool ≈ quota within noise on exact, all ties underpowered,
but quota held the variance readout first50_psg 0.42 vs 0.32–0.34 → quota
stands). The absorption-rate instrument v_t=(d_t−d_{t+1})/d_t is
teacher-dependent and would hit O3's run-length wall (banked 220–300-step
runs) — not queued.

| # | Experiment | Class / cost | Why (the datapoint) | Entry point |
|---|---|---|---|---|
| O1 | **16-diverse prompt selection** — **DONE 2026-09-05 14:10: KILLED (count > curation; quota does NOT shrink)** (diverse-16 0.710 vs random-16 0.718 vs quota 0.725 exact on 255 matched rows; McNemar diverse-vs-random discord 2/4 p=0.6875, all 15 pairwise TIE-UNDERPOWERED with zero directional support for diversity; MECHANISM (secondary): 16-prompt pools master their own prompts (train exact 0.85-0.92) without held-out transfer; quota supplies group variance longest (first50_psg 0.42 vs 0.32-0.34) — the quota table stands; O2's pass-rate FILTER unaffected, different mechanism; `rl/O1_RESULTS.md` + paired rows persisted) — original pre-registration: selection script (embed TRAIN-split candidates per family, K-means, one-per-cluster, 16/family across rename/format/no_op/pipe; holdout + BOS + length guards reused from `build_dataset`) + 3-arm GRPO at matched steps: diverse-16 vs random-16 vs quota baseline (staged diverse-vs-random first if GPU-tight). **Verdict: WINNER-DIVERSE iff diverse beats random-16 on `eval_scenarios` exact (McNemar §5) at matched env count; FULL-DATA-PARITY iff it also matches the quota baseline — then the quota shrinks permanently; else KILL (count > curation)** | CPU selection + up to 3x short GPU runs (~300-step class each), strictly serial per W37, behind the §1 chain | Paper 2's cheapest win (1 prompt = 71.5% coverage / 87% of full-data gain; 16 diverse = 98.9% = full-data; single-cluster 16 ≈ diverse 4; order irrelevant) mapped onto our random 1400+1400+350 quota draw | new `experiments/training/rl/select_diverse.py` (to build) + `rl_smoke.py` (--data/quotas) + `eval_scenarios.py` |
| O2 | **Unanimous-group guard (pass-rate filter)** — pre-screen candidates under the frozen base, admit ~0.2–0.8 pass-rate only (adapted to num_generations=4), drop unanimous batches, log partial-solved-group rate + mean reward over the first 50 steps (same readout as E1 so they compare). Implements the INDEXED coverage gate (pipe ≈ dead at 90% zero-std) with the paper's mechanism (same-query OPD >2x RLVR because RLVR dies without variance). Hard never-solved EXCLUDED by design. **Verdict: WINNER-FILTER iff partial-solved rate up vs unfiltered at matched rollouts AND eval exact beats unfiltered (McNemar); else KILL** | CPU/API pre-screen + 2x short GPU runs, sampler patch only, serial | Our variance problem stated plainly; composes with E1 (O2 = admission filter, E1 = ordering) and with O1 (filter underneath the diverse set) | `rl_smoke.py` (sampler + MetricsCb) + `rl/README.md` |
| O3 | **Suffix-entropy + length telemetry (S0) → curriculum gate (S1)** — **S0 DONE 2026-09-05 00:0x: NOT LAND** (apparent +97–157-step stall lead is a run-family confound: partial r −0.48→−0.16 ns when controlled; suffix-entropy collapse CONCURRENT with stall not preceding; v1 dropped suffix H −23% while still improving — kills level-based prediction. Verdict = "not demonstrated at our scale" — banked runs 220–300 steps vs papers' 3K–10K+; 7/10 still improving at stop; retry hard-blocked by save_total_limit=2. Residue: validated stop/continue instrument (frac_reward_zero_std ≥0.70 / reward_std ≤0.15); MetricsCb live-logging spec banked `rl/O3_S0_RESULTS.md` §5 incl. the save_total_limit fix — adopt at the next RL build window; v4_tether entropy ×2.7-while-exact-rises = PVF-line flag; entropy replay done 8 runs × 3 weight points; `rl/O3_S0_RESULTS.md`) — S1 CURRICULUM STAYS CLOSED. Original pre-registration: S0 CPU-only, chain-safe: extend `rl_metrics.jsonl` with output-position entropy curves, grad norm, advantage concentration, top-k stability across ckpts; replay a banked run to test whether suffix-first entropy rise precedes stall/collapse (paper 1: 3K–7K sweet spot, 10K+ suffix→forward entropy spread then collapse; continuation advantage +0.37 → +0.02). S1 GPU only if S0 lands or long-horizon traces enter: short-target curriculum vs extended-horizon at matched env count. **Verdict S0: LAND iff telemetry predicts stall before exact does; S1: WINNER-SHORT iff short curriculum beats extended at matched budget (McNemar); else keep the current 192-completion cap and close** | S0 CPU hours, anytime; S1 short-GPU, behind the chain | The only early-warning instrument the papers offer a teacher-free pipeline | MetricsCb + `rl_metrics.jsonl` + `eval_scenarios.py` |
| O4 | **Template-alignment price + novelty check (zeta2 stays canonical)** — (a) aligned (current byte-identical zeta2 + BOS assert) vs deliberately misaligned render (markers/order perturbed) at matched size, short run; (b) CPU novelty audit: RL pool pass-rate spread + embedding distance from the SFT distribution (same-pipeline distillation stagnates without novelty — paper 1 §3.2–3.3); mix aligned+OOD if aligned-only collapses entropy (paper 1 recipe). Ship decisions stay with H3; this only prices alignment. **Verdict: ALIGNMENT-PRICED iff misaligned underperforms aligned (McNemar) — alignment stays a hard constraint; NOVELTY-GATED iff low-novelty pools stall regardless of size; else KILL** | CPU audit + 2x short GPU, serial, behind the chain | Paper 1 §§3/5: template alone moved gap recovery 80→85%; content-matched wins but risks entropy collapse | `rl_smoke.py` + RENDER registry in `run_eval.py` + `eval_scenarios.py` |

Order: O3-S0 anytime (CPU) → O1 → O2 (compose: O2 filter underneath
the O1 diverse set and the E1 ordering if all GO) → O3-S1 / O4 behind
the chain. O4 perturbations never ship (H3 owns renders).

### TU-series — trajectory→environment reconstruction (Terminal-Universe line; user request 2026-09-04)

Premise: Terminal-Universe (arXiv:2609.04148, Qwen team, 3 Sep 2026)
inverts the rollout direction: it reconstructs executable workspaces FROM
recorded agent trajectories (deterministic replay of file ops → agentic
completion → read-only sufficiency judge), re-queries each workspace
(intent recovery, single-WS, cross-WS breadth, multi-round depth), and
trains on verifier-passed teacher re-solves — 37.3k envs → 32.0k demos →
TB2.1 +11.9 / EvoCode-Bench v2 MT@4 +13.8 on Qwen3.5-27B. Load-bearing
ablations for us: re-solve beats imitate 52.1 vs 36.7 at matched volume
(§6.1); agentic completion +4.2 (§6.2); verifier filtering pays on HARD
tasks only (Cross-WS 53.2→55.4, Single-WS flat — §6.3); cross-WS +2.0 on
top, teacher pass@1 72.3→49.2 (§6.4); multi-round +2.6 MT@4, round-verifier
removal −2.2/−3.7 (§6.5); at fixed budget env-expansion 56.0 >
query-expansion 53.8 ≈ solution-expansion 53.9 (§6.6). Our mapping:
env = parent-commit workspace (CRAN normalized-tree version transitions —
the simulator's trajectory scope); trajectory = the commit diff /
simulated workstream; teacher/judge = glm-5.3 reasoning low
(anchor-validated 40/40 · 38/40 · 39/39, `rl/README.md`); verifier =
exact + `ast_equiv.py` + validators-with-corrupted-twin. Explicitly NOT
transferred (pre-registered): per-task ubuntu containers, 500-turn/4h/256k
rollouts, frontier-teacher scale, TF-IDF relation mining at 38k-env scale,
agent-authored pytest suites (our verifier trio already covers single-span
edits; paper_to_r keeps its statistical validators). Frontier-mechanism-
not-loss rule applies (§5).

| # | Experiment | Class / cost | Why (the datapoint) | Entry point |
|---|---|---|---|---|
| TU1 | **Sufficiency-gate validation (zero training)** — **DONE 2026-09-05 00:4x: DEAD** (judge-gate fails the rule: pooled gap +56.6pp MET (Fisher p=1e-6) but direction-consistency 1/5 vs required 3/5 — glm-5.3 rates 92.2% of rows SUFFICIENT, 3/5 families had ZERO insufficient rows so the rule was structurally near-unreachable; reported as written, no post-hoc rescue. TU2 arm (d) CANCELLED, TU2 runs solve-gated only. Free readout: doc_sync sufficiency 0.267 (4/15) and the 4 derivable rows are 0/4 exact — B-β's construction verdict independently re-confirmed AND sharpened: broken beyond context sufficiency. McNemar pooled b=55/c=4 p=1.7e-12, per-example rows persisted; 256 calls / 185k tokens; `synthetic-data/TU1_RESULTS.md`) — original pre-registration: glm-5.3 read-only judge labels rendered TRAIN-side rows (prompt + target region) sufficient/insufficient ("is the target derivable from the context visible in the prompt alone?"); validate the label against persisted per-row results (raw field, V1 enabler) of a banked arm (v8_2 or the b4 production base): exact(sufficient) vs exact(insufficient), per family, McNemar on both subsets. Five edit families (rename/pipe/na_rm/format/doc_sync); no_op excluded (empty target = different question). **Verdict: SIGNAL iff pooled gap ≥15pp AND direction-consistent in ≥3/5 families → TU2 arm (d) runs; else judge-gate dead, TU2 runs solve-gated only.** Free second readout: doc_sync sufficiency rate ≈0 would independently confirm B-β's construction verdict | CPU/API-class, chain-safe, anytime | The paper's stage-3 filter (their Table 2: 40→93% sufficiency post-completion) at zero GPU; prices whether our rendered rows carry underivable-target noise BEFORE any retraining | new `experiments/synthetic-data/sufficiency_judge.py` (to build) + per-row `results_*_v8_2*.jsonl` / b4 battery files |
| TU2 | **Re-solve-vs-imitate SFT A/B (the §6.1 headline)** — **PREP DONE 2026-09-05 05:0x** (teacher solve pass 2,362/2,362 rows one-attempt @ eval settings: pooled solve 26.4% (exact 21.6% + ast-equiv 4.8%); per-family na_rm 47.5 / pipe 38.5 / rename 31.8 / format 29.1 / **doc_sync 0/550 = THIRD independent construction-verdict confirmation, mechanism = verbatim-pinned author targets vs teacher paraphrase**; arms built contamination-clean (canary 0/307); AMENDMENT (queue-mgr, documented): primary (a)-raw vs (b)-solve-gated re-cut matched at |b|=624, (c)-teacher-target stays the 113-row PAIRED secondary isolating consistent-policy — McNemar rules unchanged; caveat recorded: solve-gate filters through glm-5.3's task-inference profile; 830k tokens / 6.4h; `synthetic-data/TU2_RESULTS.md` + results/tu2_arms/). Training arms: **DONE 2026-09-05 14:2x: RAW-ROUTE-STANDS (not WINNER-RESOLVE) — teacher-in-the-loop data CLOSED on this rig** (a'624 raw 74.12 exact vs b'624 solve-gated 67.06 = **−7.06pp, McNemar b/c 10/28 p=0.0051**; c113 teacher-target 36.47; b113 46.27; the (c)−(b@113) paired identical-prompt isolation = **−9.80pp p=0.0199** — consistent-teacher mechanism NEGATIVE where the teacher had rendering freedom; rename −12.7 p=6.6e-5 carries the primary, format +4.5 ns = the one headroom flicker, drowned; noopFP guardrail: c FAILS +2.94pp, all arms propose-always class 93-99% (300-step short-adaptation structural); MECHANISM PRICED: the 26.4% solve-gate is a teacher-ABILITY filter — the 412 dropped unsolved rows (rename/pipe-heavy) were load-bearing supervision; verbatim-pinned GT > teacher renderings; paper's +15.4 does NOT transfer at our teacher/scale — divergence documented; c100/b100 re-run flagged low-priority only; `synthetic-data/TU2_RESULTS.md` + paired rows ×4 arms persisted; cloud arms $1.4 total). Original pre-registration: 4 arms at matched row count (subsample to smallest arm; the paper matched 35.8k both), one base (gate-B-β production rec, b4-config), fixed recipe + seed: (a) raw rows = status-quo rendering; (b) solve-gated: keep rows glm-5.3 solves in ONE attempt at eval settings (exact OR ast_equiv), train on ground-truth targets = the pure filter effect; (c) teacher-target: on solve-passing rows train on the teacher's rendering where AST-equiv-different = the consistent-policy effect; (d) judge-sufficiency-gated (only if TU1 SIGNAL) = the cheap-filter control. Clean split only (contamination protocol; verbatim canary clean-vs-seen logged). **Verdict: WINNER-RESOLVE iff (b) or (c) beats (a) on eval_scenarios exact (McNemar + per-family) with noopFP not worse; (c)−(b) isolates the consistent-teacher mechanism; nothing beats (a) → raw-diff route stands, teacher-in-the-loop data closed.** Pre-registered expectation (their §6.3 + our O2/E1 variance story): gains concentrate in headroom families (format/na_rm), not near-ceiling pipe | API solve pass + ~4 × 2-3h GPU, strictly serial per W37, behind the production-finetune track | Their largest single ablation (+15.4 imitate→re-solve) mapped onto our data route: is the v-series raw-diff rendering leaving supervision on the table | new `experiments/synthetic-data/teacher_resolve.py` (to build) + `train_sft.py` + `eval_scenarios.py`/`eval_noop_fp.py` |
| TU3 | **Cross-file reference family (the §6.4 breadth analog)** — new family `cross_file_reference`: writable target file + read-only reference file (same package) in the render; subtypes: signature/style match, port-with-adaptation, doc-sync-to-reference (a direct doc_sync rescue lane per B-β). Build via `cases/` authors + validators + corrupted twins (three-gate rule). Difficulty gates BEFORE any training: random ≈ 0 (`rl/README` rule) AND glm-5.3 pass@1 ∈ [30%, 85%] (paper's cross-WS sat at 49.2 vs 72.3 single-WS; outside band = re-scope, not train). **Verdict: WINNER-CROSSFILE iff a +TU3-share mixture beats its TU2-winning control on the new family AND no existing-family regression (McNemar) AND noopFP not worse — reference-in-context is a NEW false-propose risk, pre-registered** | ~half-day build (CPU/API, anytime) + one GPU arm behind TU2 | Their cross-WS data is the only +2.0-on-top datapoint (56.4→58.4) with 1.6-1.9× trajectory complexity; our rename/doc_sync families are the seed; single-package scope keeps it cheap | `cases/specs/` + `samplers.py` + `scenarios.py` registration (`doc_sync_spark.py` precedent) |
| TU4 | **Multi-round version-chain sessions (the §6.5 depth analog; build folds into W18)** — extend the simulator's trajectory generator from single-commit episodes to package version-chains (v_parent→v1→…): each round = one transition + its goal card (`mine_commit_goals*.py`); round styles = extension/revision/conflict mapped to real transition types; ROUND-VERIFIER = the actual next-version diff checked by validator or ast_equiv BEFORE the chain advances (their tests-before-coding rule); solver-visible feedback = natural-language user complaints only, never tracebacks (matches our panel-prompt doctrine). Outputs: V1a episode-metrics rows + a stage-1 RL arm (sequential no-op context native). **Verdict: WINNER-DEPTH iff the multi-round arm beats its TU2 control on episode metrics (accept/fp at matched sim-hours) AND a fail-stop consecutive-rounds score (their MT@4 analog); else depth folds back to single-round data** | build-class + one GPU arm; prerequisites: goal-card corpus at volume + TU2 verdict | Their multi-round data is what moved persistent-session metrics (MT@4 18.4→21.0; round-verifier removal −2.2/−3.7); our simulator already plans full-package workstreams — this supplies the missing round structure | `coding_simulator/simulate.py` + `judge_loop.py` + design.md §multi-round + W18 |
| TU5 | **Budget-split rule (the §6.6 analog) — optional piggyback** — at matched row budget on the TU2 rig: (a) N commits × 1 render-point (env expansion) vs (b) N/2 commits × 2 render-points (query expansion = the simulator's multiple suggestion points per commit). **Verdict: fold the winner into the assembler/A2 quota policy (their env 56.0 > query 53.8 ≈ solution 53.9); KILL the dedicated run if TU2 lands a large effect (interaction risk) — re-cut post-hoc on the TU2 rig instead** | 2 × short GPU arms on the TU2 rig, or post-hoc re-cut | The paper's only budget-allocation datapoint, and it prices exactly our assembler knob: per-commit render multiplicity vs package coverage | `assemble_a2_mixture.py`-style quotas + `train_sft.py` |

Order: TU1 (CPU/API, anytime) → TU2 (GPU, serial, behind the
production-finetune track) → TU3 (build parallel; GPU arm after TU2) →
TU4 (post-TU2 + goal-card corpus; W18 absorbs the build) → TU5 piggyback
or post-hoc. Composability: TU2's solve-gate is O2's admission filter
moved to the SFT side (frozen teacher instead of frozen base) and doubles
as E2's seed-env difficulty gate; TU4's rounds are a natural host for
E2/E3 lineages (generations = rounds). Contamination, judge-logging, and
holdout rules unchanged throughout.

### S-series — serving wall-clock (user request 2026-09-04)

Premise: two vendor receipts landed the same week and both match OUR
serving pattern (single-stream agentic decode, BS=1, ~8K ctx, short edit
outputs) at larger scale — NVIDIA SpeedBench-Coding 8K AIPerf (llama.cpp
on 5090: 1.5x on Qwen3.6-27B, 1.9x on 35B; vLLM 1.2–1.4x) and Unsloth's
GLM-5.3-Flash day-zero llama.cpp PR (B200, 1-bit: kernel-only gain ~0 at
short ctx — pp512 1121→1122 — rising to 2.37x at 64K where the baseline
falls over; MTP n=2 best, 58.6→86.5 at 4K, fading by n=5). Transfer
reading (mechanism-not-loss, §5): kernels pay where the baseline is
KV/cache-bound at long ctx, NOT at our 2–8K CPU tier where P12's note has
Q8 decode already riding the roofline; spec-decode pays where acceptance
is real — which is exactly Q7's unanswered question. Neither receipt
reopens training-step optimization (our graph/OT kills were launch-bound
micro-rigs, a different bottleneck). Both are serving-side, both run on
stock-or-PR llama.cpp — so both are rebases + benches, not kernel
projects. Load-bearing repo fact: `export_gguf.py` strips MTP via
`--no-nextn` on qwen archs, so every GGUF on disk is MTP-less — S1's MTP
arm needs an MTP-preserving export first (converter without the flag,
`mtp-*.gguf` serve path per W13). + **Uno fold (2609.04010, 2026-09-05,
`2026-09-05-intel-batch-2.md` §1)**: diffusion-LoRA drafting INSIDE the AR
model (per-layer dual weights, position-gated Gated LoRA, single KV, block
drafts with first-token-always-AR, spec-style verification = lossless;
DCD-KL + TV-confidence post-hoc distillation). Receipts: 2.2–2.5× BS=1,
1.5× at BS=64, 8B > 26B DiffusionGemma/Mercury 2, **+40% end-to-end RL
rollout speedup persisting under AR-weight drift**. NOT an S1 arm: unlike
the two receipts above it needs adapter training + a dual-weight/Ψ-sampler
serving stack — not expressible in stock GGUF/llama.cpp (runtime fork +
training spend; lossless trained-draft as a MECHANISM is already covered
by S1's arm classes). Where it DOES bite: (a) the **A2 trained-draft
decision** — Uno adapters are the named alternative to MTP weights when
A2 opens its draft-head slot (design-time flag, not a finetune-track
item); (b) **RL-phase cost accounting** — the 40% number prices the
rollout-acceleration ceiling if such a stack ever reaches our rig;
(c) **X2 cross-ref** — "keep AR + diffusion side-weights + AR-teacher
distillation" is the pre-named fallback family if X2's AR→MD conversion
leg fails (X2 row unchanged, intel doc holds it). **→ S3 opened 2026-09-05**
(user fork-warrant interest): the staged line that owns the fork decision —
this note's "stays out of the bench" applies to S1's STOCK arms only; repo
recon moved S3's facts in (Apache-2.0; Nano-vLLM+FA2/3 GPU-only engine, no
export path; Qwen3-8B LoRA training recipe released).

| # | Experiment | Class / cost | Why (the datapoint) | Entry point |
|---|---|---|---|---|
| S1 | **PARTIAL 2026-09-08: saved trimmed CPU measurement reconciled/archived; broader scope untested (queue-owner record).** **Spec-decode wall-clock on our pattern** — staged. **S0 DONE 2026-09-05 02:5x: 1100 traces (550 real edits × paired 2K/8K) frozen + validated** (1100/1100 prompts byte-identical to render_zeta2; 0 band misses; tokenizer cross-check 6/6 vs the serving GGUF; contamination guards vs edit_pairs_v1 train+eval + pr_instructed; MANIFEST at `/mnt/h/sepalith/datasets/spec_traces/`; rig notes: `-c` ≥10240 for the 8k class (p95 9163 — the 8192 default silently truncates), stop `>>>>>>> UPDATED`, max_tokens 64, recount targets if benching a non-Qwen-family ckpt; acceptance = trained-family-unseen-edits, residual documented; Q7 FOLD flagged on board). Original S0 pre-registration: mine ~500–2000 frozen edit continuations · **RIG BUILT 2026-09-05 03:5x, smoke-clean, full legs parked for the quiet window** (`spec_bench.py` 4 single-mode arms + 25 tests; MTP-preserving export path LANDED — upstream Qwen3.5-2B-Base ships the MTP head (15 tensors) our local strip lacks; re-merge + graft + convert-without-`--no-nextn` works, tensor-verified, BOTH draft-mtp serve modes proven on b10453 (`--spec-type draft-mtp` etc.); smoke: ngram-simple 0.556 accept / 1.6–1.8x wall CONTENDED, model-draft@2 0.967 accept (same-family stand-in), draft-mtp@2 0/90 = untrained-with-body head → conservative lower bound for the A2 MTP-freeze decision; WINNER-SPEC realistic weight → ngram + model-draft; 5090-offload legs need a CUDA b10453 rebuild first (W37); `S1_RIG.md` has the full-leg command) (PSM render, 20–60 tok outputs, 2K/8K ctx) from r-universe/CRAN transitions; this is Q7's trace set, so S0 supplies Q7 either way. **S1 bench** (CPU legs anytime; 5090-offload legs idle-card windows only per W37): single-mode arms only (A2 §4.5 rule — no stacked claims): baseline vs `ngram-simple` vs `draft-mtp` (MTP-preserving export required, see premise) vs Matryoshka S-tier `--model-draft` (zero-PR fallback); sweep draft depth (expect n≈2 optimum per Unsloth, report the curve). Metrics: acceptance (tok/step) + wall tok/s + TTFT, per tier (CPU t8, 5090 offload), at 2K and 8K ctx. **Verdict: WINNER-SPEC iff a single-mode arm ≥1.4x wall tok/s on either tier with no crash under suffix churn (spec is distribution-lossless, so no quality re-proof needed); nothing ≥1.3x → close the spec line, ship baseline + cache work (H3).** Acceptance number feeds W13 (MTP serve path) + the A2 MTP-head freeze decision | S0 CPU hours; S1 bench hours, zero GPU-train | The NVIDIA MTP=3 + Unsloth n=2 receipts re-run on OUR traces, OUR renders, OUR tiers — replaces the folk 1.4–2.5x bands with one measured number before any latency table multiplies it | new `experiments/eval/spec_bench.py` (to build; `cache_bench.py` + `keystroke_sim.py` patterns) + `export_gguf.py` MTP-preserving path |
| S2 | **PARTIAL 2026-09-08: CPU timings reconciled/archived; GPU timing column completed in c467679abdc146f2a92abf79572fe6a4; quality gate remains unverified (queue-owner record).** **Quant wall-clock + quality on the keystroke-cycle** — the P12 companion: P12 prices bytes (roofline, kernel-day gate); S2 prices the felt cycle. Matrix {Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0} × {CPU t8, 5090 offload} × {pp2K, tg48 PSM}, 3 reps, on b1_ref24 now / the B-α winner when it lands; PRIMARY metric = keystroke-cycle (prompt_ms + short gen), not raw tg (H5-S0 doctrine). Quality gate per arm: intent suite + eval_scenarios exact within 1pp of Q8_0 (design-A §4.6 rule). **Verdict: WINNER-QUANT = cheapest format holding the 1pp line per tier → writes the ship-matrix row (L/M/S tiers, §312); Q4-class >1pp down → ship Q8/Q6 at that tier, QAD stays the conditional rescue (INDEXED).** Shares P12's idle-window discipline for CPU legs (one contended-free evening covers P12 + S1-CPU + S2-CPU) | CPU hours + eval CPU/API, chain-safe; offload legs idle-window | Stock-quant serving IS the CPU-tier memory-throughput optimization (intel-batch §5: Q8 already at/near roofline) — this measures what the user feels instead of arguing it from bandwidth + **quant priors 2026-09-06** (`2026-09-06-intel-batch-3.md`): (a) **Minima (2609.04098, GDN-hybrid 27B FP4)** — the quant-fragile part of a GDN base is NOT the decay/write gates (log-space/softplus compresses ~11% GEMM error → ~2% output error = qualitative de-risk of Q4-class on OUR GDN base); recurrence noise PLATEAUS (~12.6% flat over 32K, impulses decay 80–1,382 steps — long-file ctx doesn't compound); CAUTION: their per-module-vs-fused-GEMM calibration mismatch silently cost −5.9 AIME with clean-looking PPL, and llama.cpp also fuses projections — verify scale provenance before trusting per-module calibration; (b) **HBQ (2609.00450)** — MXFP4-class W4A4 + 4-bit KV collapses (PPL 6.80→11.39): KV-quant stays OFF the CPU-tier prior table | new `scripts/quant_serve_bench.sh` (to build) + `eval_scenarios.py` + intent suite |
| S3 | **Uno diffusion-draft line — the runtime-fork warrant, staged** (user 2026-09-05: "that much of a benefit would warrant shipping a fork as runtime"; **PROPOSED, awaiting GO**). Repo facts banked (github.com/ifm-ai/uno): Apache-2.0; training pipeline incl. a Qwen3-8B LoRA recipe (conditional-LoRA diffusion adapters, DCD-KL to the AR teacher + TV-confidence loss, progressive block-size curricula); inference = **custom PyTorch on Nano-vLLM + FlashAttention-2/3 — GPU-only, NO GGUF/ONNX export** ⇒ our CPU tier means reimplementing the mechanism in llama.cpp, not adapting their engine; FA3 is Hopper-only and FA2 wheels may lack sm_120 (5090), so their engine on OUR card likely needs an attention-backend swap (PyTorch-level patch, not a fork). **S0 prize gate (CPU + math, chain-safe, after S1-CPU legs land)**: (a) prize math from S1's measured acceptance×cost curve — projected wall-clock of a 1-pass block draft at acceptance a∈{0.6, 0.8, 0.9} on the CPU tier vs the best STOCK S1 arm (the fork buys only the gap; P12's roofline makes the bandwidth arithmetic exact); (b) fork scope vs llama.cpp b10453 in honest days: dual-weight forward with position-gated LoRA, block-draft sampler, first-token-AR, existing spec verify path; Linear sampler only (Tree is spare-GPU-shaped). **Gate: fork proceeds only if projected CPU wall ≥1.8× baseline AND ≥ best stock arm +0.3×; else Uno stays the A2 draft-path flag.** **S1 adapter leg (GPU behind the chain, NO fork needed)**: train Uno adapters on the b4-config base (their recipe + our edit-continuation corpus; AR teacher = our merged weights — B8b merge machinery); acceptance = plain forward passes on held-out spec_traces (draft block vs frozen continuation) + optional wall-clock on their engine over the 5090 post backend-swap. **Gate: held-out adapter acceptance ≥0.80** (reference points: ngram-simple smoke 0.556; stock model-draft stand-in 0.967 — the adapter must beat ngram decisively, and S1's full legs decide whether stock already banks the prize). **S2 the fork (both gates only)**: CPU-first minimal llama.cpp fork; pinned-runtime maintenance rule (the win must survive one llama.cpp rebase or we pin the runtime); Linear sampler only. **Verdict: WINNER-UNO iff served CPU-tier wall ≥1.5× the best stock arm at matched output — spec is distribution-lossless ⇒ no quality re-proof (S1 doctrine)** | S0 CPU-days; S1 GPU hours-class behind the chain; S2 fork = 1–2wk build class + standing maintenance tax | The user's warrant read is mechanically sound on OUR tier: P12 says CPU decode is bandwidth-roofline-bound and spec's win = weights streamed once per accepted block; Uno's block draft costs ~1 LoRA-gated pass vs k AR-draft passes (or free-but-ceilinged ngram lookups); lossless preserves the SFT distribution exactly. And the fork decision is CHEAP to gate: acceptance — the load-bearing number — is measurable with zero runtime work | `2026-09-05-intel-batch-2.md` §1 + S1's rig (`spec_bench.py`) + `/mnt/h/sepalith/datasets/spec_traces/` |

Order: S0 trace freeze anytime (CPU, unblocks Q7 either way) → S1-CPU +
S2-CPU in P12's idle window (one contended-free evening covers all three)
→ S1/S2 offload legs whenever the card idles (W37: never beside a
trainer). Q7's question is answered on S1's rig — flag the fold on the
board when S1 GOs. Both rows feed H3's serving term + H4's cost
accounting.

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
| W13 | **MTP serving path**: llama.cpp speculative-decode integration for the A2 MTP head (informed by Q7/RT-2) — **SERVE PATH PROVEN 2026-09-05 via the S1 rig** (MTP-preserving export `export_gguf_mtp.py`: graft upstream head + convert without `--no-nextn`, tensor-verified; both draft-mtp serve modes work on b10453 with `--spec-type`/`--spec-draft-n-max`; remaining for real acceptance = a head trained WITH the body, i.e. the A2 MTP head itself) | ~1-2 days → serve path done, head awaits A2 | runbook §5; the 1.04x-cost structure pays off only if this lands |
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
| W24 | **DONE 2026-09-08: script contracts, environment/vocab scars and runner cutover documented in SYSTEMS.md §9; no payload executed.** **SYSTEMS.md update**: run.py, r_repack_full, pack_r_strata, push_cases pretraining/, the venv split (3.10 vs 3.14 dill breakage), CUBLAS/vocab scar | ~1h | 2026-08-26→09-01 sessions' scripts undocumented there |
| W25 | **HF dataset card YAML frontmatter** (silences the repo-card warning; adds license/language tags) + card sync with the adopted manifest | ~30 min | push_cases warns every run |
| W26 | **NAS runs/ retention policy**: q6/gatednorm/e3v2 checkpoints organized or pruned (results JSONs retained) | ~1h | disk hygiene; /tmp scars say keep NAS canonical |
| W27 | **Dashboard state refresh** (v55 → current: verdicts, queue link) — **DONE 2026-09-04 by automation** (postplan updater daemon: muse-spark editorial + build_dashboard.py rebuild + upload every ~30 min; STOP flag; see board 2026-09-04 23:xx) | automated | house pattern |
| W28 | **Anyscale GPU ladder smoke** (one A10G-class node, ~100M-token ladder smoke → cloud tok/s vs 5090's 48k; decides the fp16 T4/Tier-free port) | ~$1/hr, hours | `2026-08-30-anyscale-setup-and-smoke.md` §open-follow-ups #1 |
| W29 | **Repo packaging at scale** — **DONE 2026-09-05 00:1x: CLOUD-READY YES** (smoke passed end-to-end: 60-step LoRA on A10G, loss 1.495→1.326 healthy; time-to-ready 3.5 min; total burn $0.55; packaging = 12MB git-archive, no PAT/no image needed; runbook `2026-09-04-anyscale-sft-cloud-runbook.md` + scripts/cloud/). BURN TABLE: A10G $1.006/h = 2,746 tok/s (0.63x the 5090's 4,050-4,360); B13-class rung ≈ $6.5-8, B7 ≈ $2.7, GRPO arm ≈ $2.5-3; W1-FP8 ≈ $57 on Anyscale vs ≈$7 on the A2 rental ⇒ credits NOT spent there. POLICY: ~$85 reserved for the production fine-tune (≈12 B13-class rungs); opportunistic de-serialization spends only. B7 cloud rung PREPPED, fires only on B13-pass per the order rule | ~half day | `2026-09-04-anyscale-sft-cloud-runbook.md` |
| W30 | **Compute-credit program applications** (Tier A signups; Trelis $500/q, Vast $2.5k, HOSTKEY; Lambda/AMD gated; week-1 sequencing incl. fp16 ladder port) | writing-class | `2026-08-30-compute-credit-stack.md` |
| W31 | **PI Fast Compute Grant send** (email drafted; user calls: size ~$2-3k core vs $5k, HF handle, naming) | send + 2 user decisions | `2026-08-30-pi-fast-compute-grant-application.md` |
| W32 | **DONE 2026-09-08: saved evidence reconciled and archived; mechanism supported, blanket adoption not established by the registered gate. See `poc_twin/RESULTS-arms.md` reconciliation and `docs/validation/2026-09-08-w32.json`.** **Aurora verdict archaeology + poc_twin/RESULTS-arms.md close-out** (dashboard says "ADOPTED into A2-prime Muon recipe", papers-recon plan says "parked as OPT-3 gated on its falsifier", runbook §3.2 recipe has no Aurora; Arms C/D never closed in the canonical file) | ~2h reading + 1 verdict post | `dashboard_state.json:85` vs `2026-08-29-papers-recon-poc-plan.md:36`; sweep contradiction (a) |
| W33 | **PARKED 2026-09-08: historical input provenance unresolved; saved rows do not align with current plain split.** **v7 ablation clean rerun** (rows ~200-812 of `results_ablation_v7_on_plain.jsonl` are ALL-EMPTY predictions; file marked TRUSTED-UP-TO-ROW-~200) | ~hrs CPU | `landscape-v7-vs-glm53.md` tail; sweep integrity item |
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
  B13-candidate disqualifications 2026-09-04 (recon
  `2026-09-04-base-candidate-recon-ouro-k2-lfm.md`): **Ouro-1.4B** — no
  llama.cpp arch (declined mradermacher #1490; looped-support PR #25994 is
  arch-specific), transformers pin <4.56, recurrent-depth decode ≈5.5B-equiv
  compute; watch item only. **IFM K2-Horizon-0.9B** — no base checkpoint
  (unsuffixed = MOPD post-trained reasoning-chat), llama.cpp fork-only
  (MBZUAI-IFM, unmerged); revisit iff a true base ships + arch merges
  mainline.
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

- 2026-09-05 (08:0x): **BASE PICK RESOLVED — GDN/Qwen3.5 b4-config** (B13
  quality-tied all rivals at McNemar but product-eliminated LFM via noopFP
  99% + decode; user's try-first directive honored; B7 leave-unspent;
  spark stays W36-conditional challenger; granite = B8 control). E1
  DONE-KILLED (admission gate starves format, the headroom family; E2
  closed; O2 unaffected). Dependent wave FIRED: B8 runner armed behind
  O1's card chain (~13:00), TU2 cloud arms on Anyscale @300 steps
  (~$1.1), B7 stood down. Session verdict tally: P12 GO · TU1 DEAD ·
  O3-S0 NOT LAND · V1d ALIVE · S0 frozen + S1 rig + W13 serve path ·
  TU2 prep · E1 KILLED · B13 tie/elimination · base pick. Cloud ledger
  ≈$0.68 + $1.1 TU2. Dashboard v2 accepted (spark-designed).

- 2026-09-04 (23:2x): P12 DONE — KERNEL-DAY GO (realization arm), CPU-tier
  prior {Q8_0/Q4_K_M/Q4_0}; quiet window lifted 23:15, GPU chain open. B13
  armed (LFM2.5-2.6B-Base rung — user base-pick answer: try untried
  candidates first; MiniCPM5-1B noted as already-run = the B1 ladder,
  lost). Ouro-1.4B + K2-Horizon-0.9B disqualified pre-spend (§5). B7
  ordering rule folded into B13's row. W27 retired by the dashboard daemon.
  W29 cloud packaging fired (Anyscale capability build, $10 smoke cap —
  credits CONFIRMED NON-EXPIRING 2026-09-05: user console shows grant
  window 08/30/2026→08/05/2126; posture = val/$ spend, bulk reserved for
  the production run; repo + HF public ⇒ no-PAT packaging).
  Round-1 still in flight: TU1, O3-S0, S0-traces, E1/O1/B8 patches,
  dashboard daemon bring-up. GPU slots: B13 → E1 → O1 arms (readiness
  order; W37 serial).

- 2026-09-04 (22:0x): added S1–S2 (serving wall-clock line from the
  NVIDIA 1.5–1.9x SpeedBench + Unsloth GLM-5.3-Flash MTP receipts,
  user request 2026-09-04) to §3 PROPOSED. S1 = spec-decode
  wall-clock, single-mode arms (ngram / draft-mtp / Matryoshka-draft)
  on frozen 20–60-tok edit traces at 2K/8K, CPU + 5090-offload; S0
  trace freeze doubles as Q7's trace set. S2 = quant wall-clock +
  1pp quality gate on the keystroke-cycle (P12's felt-cycle
  companion; writes the §312 ship-matrix row). Transfer reading:
  kernels pay at long-ctx KV-bound, spec pays where acceptance is
  real (Q7's question); training-step kills untouched (different
  bottleneck). Load-bearing: export_gguf.py strips MTP today — S1's
  MTP arm needs an MTP-preserving export first. Order S0 → CPU legs
  in P12's idle window → offload legs on idle card; no GO, nothing
  fired.

- 2026-09-04 (21:5x): added TU1–TU5 (trajectory→environment reconstruction
  line from Terminal-Universe arXiv:2609.04148, mapped to our pipeline in
  the 2026-09-04 thread) to §3 PROPOSED per user request. TU1 =
  judge-sufficiency validation (CPU/API, zero training; doc_sync
  sufficiency rate doubles as B-β construction evidence), TU2 =
  re-solve-vs-imitate SFT A/B (the §6.1 headline, 4 matched arms
  isolating filter-effect vs consistent-teacher-effect), TU3 = cross-file
  read-only-reference family (doc_sync rescue lane, noopFP risk
  pre-registered), TU4 = multi-round version-chain sessions with
  round-verifiers (folds into W18), TU5 = env-vs-query budget split for
  the assembler quotas. Explicitly not transferred: container fleet,
  rollout scale, frontier teacher, TF-IDF mining, agent-authored pytest.
  Order TU1 → TU2 → TU3 → TU4, TU5 piggyback. No GO asked, nothing fired.

- 2026-09-04: added O1–O4 (OPD / One-Shot transfer line from THU 2604.13016
  + 2609.04172, mapped to our teacher-free GRPO in the 2026-09-04 thread)
  to §3 PROPOSED per user request. O1 = 16-diverse prompt selection,
  O2 = unanimous-group guard (pass-rate filter, composes under O1/E1),
  O3 = suffix-entropy telemetry S0 (CPU, anytime) → curriculum gate S1,
  O4 = template-alignment price + novelty check (renders stay with H3).
  Algorithm explicitly non-transferred (dense-teacher vs sparse exact);
  order O3-S0 → O1 → O2 → O3-S1/O4. No GO asked, nothing fired.

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
