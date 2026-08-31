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

Last synced: 2026-08-31T23:3x+0200 (user GO ~23:2x — "continue running
experiments from EXPERIMENT-QUEUE.md whenever the current ones are done":
queue proceeds in its own cost-first order. Q6 promoted to RUNNING-QUEUED,
P1 prep started (llama.cpp CUDA building; eval triples present). Prior
sync 23:0x: close-out §1 + supplement, board through 22:5x, paradigm
review + base-bakeoff prep sessions same night).

---

## 1. RUNNING — user GO 2026-08-31 20:4x, owner zcode-main (chain in `comms/gpu.md`)

| # | Experiment | Status | Entry point |
|---|---|---|---|
| Q1 | so_r_qa dose-response v2 (2x/4x share, E3 protocol) | **DONE — ADOPTED at 2x** (board 22:1x; manifest rebuilt, so_r_qa 0.012→0.024) | `poc_cma` run_e3.sh pattern |
| Q2 | T1 DAPO zero-std filter A/B (~80min; mandatory gate before production RL) | RUNNING (in follow chain; 3.14-venv crash fixed via venv-sft 3.10) | `docs/research/2026-08-28-slime-miles-adoption-plan.md` §T1 |
| Q3 | GatedNorm ladder arm (QK-Clip vs +GN vs GN-only + 2x-LR stress) | RUNNING (gn_qk clean; gn_only CUDA-died step 200 — transient, retry queued with resume; stress arms pending) | ladder rig + `stress_metrics.py` |
| Q6 | Batch 512k→1M tokens/step + LR probe (promoted from §2 by the queue-continuation GO) | RUNNING-QUEUED — auto-fires after the Q2/Q3 follow chain (`scripts/queue_continuation.sh` → `poc_twin/run_q6_batchprobe.sh`): 3 arms at matched 400M-token budget (512k control / 1M / 1M+2xLR), loss-at-tokens readout | `poc_twin/run_q6_batchprobe.sh` |

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
anytime. GPU rungs additionally wait for the card.

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

## 3. PROPOSED — from the 2026-08-31 paradigm/architecture review, awaiting user triage

Grounding: POC-DIFF VALIDATED (MD exact 0.0694 vs AR 0.0000; 51–256-tok
spans 0.000 on BOTH arms), POC-DDOT KILLED (position field converges <200M
tok; value routing poisons quality), survey = `nse-ot-flows-survey-2026-08.md`.
Suggested triage order = P1 → P6, P3 → P2 → P7 → P4, P5 → P8, P9 (cheap+high-info
first, speculative last). User arbitrates.

| # | Experiment | Class / cost | Why (the datapoint) | Entry point |
|---|---|---|---|---|
| P1 | **Cross-paradigm eval on one harness**: SFT GGUFs through `eval_spans.py` + KV-cached AR re-baseline | CPU-class (llama-server) + triples regen, ~1 day | The only honest common table; poc_diff RESULTS "Next" already names it; first read on whether the long-span zero is paradigm-specific | `experiments/training/poc_diff/eval_spans.py` + `data_prep.py` (regen triples; /tmp wiped) |
| P2 | **AR-init diffusion span head** (families B×G; the never-triggered Task 8 rescue) | GPU ~13h (2B tok, 206M; warm-start) | AR learns context fastest (held-out 1.57 nats/tok @131M); diffusion only where multimodality pays; trunk keeps a llama.cpp path | `poc_diff/train_md.py` + survey family G (DiffuLLaMA/Dream shift-op) |
| P3 | **Length-aux hybrid**: aux length/position head on the validated MD arm, NO value routing | GPU ~13h (warm-start from `md_final.pt`) | DDOT's one real signal: position field 8.14→0.008 in <200M tok; honest replacement for GT-length conditioning now that CAL v1 failed | `poc_ddot` position-head components folded into `train_md.py` as aux loss |
| P4 | **Block-diffusion span head** (family D targeted at spans) | GPU ~13–26h + build | The 51–256-tok bucket is 0.000 everywhere; AR anchors structure, diffusion fills within blocks; KV-cacheable | new `poc_bd/` reusing poc_diff data + sampler |
| P5 | **Edit Flows arm** (family C: insert/delete/substitute CTMC) | GPU ~13–26h + new trainer | Survey's named candidate for long spans (+138% over mask-only at 1.3B on code); native variable length + deletions — the primitive AR-PSM and MD both lack | new `poc_editflows/`; survey §3.2 (2506.09018) |
| P6 | **Full CAL recipe** (not our failed v1 half-peak rule) | CPU + GPU-hours on existing ckpts | v1 failed hard (length-MAE 191) but the paper's full recipe reports +47.7% Pass@1 on code infilling; cheap to test on `md_final.pt` | `poc_ddot/cal_length.py` upgraded to the paper's full search |
| P7 | **Data-scale disambiguation rerun**: both twins at 10–20x corpus, ~4 epochs (vs current 44 epochs/44.5M unique) | Largest: corpus build + 2 retrains (~1.8–3.6B tok total) | Tests paradigm-vs-data for the long-span zero before any big arch bet; survey notes the regime is data-starved; DEPENDS on A2 data program (so_r_qa stratum) | `poc_diff/data_prep.py` + A2 `pretraining/` package |
| P8 | **OT as training loss** (FMPE-style continuous relaxation over span embeddings, family F) | GPU, speculative build | Pure white space — no text/code result exists anywhere; the "NSE as OT" idea's last untested form | survey §3.3 (FMPE 2305.17161 + minibatch OT) |
| P9 | **Differentiable edit-distance loss** trained into a code model | GPU, speculative build | Survey white-space item #4 (edit-distance-as-OT exists for graphs/trees, never trained into a code model) | survey §2 white-space list |

### B-series — PARKED in §2b above (user trust verdict, 2026-08-31 23:0x)

B1–B5, B8–B11 moved to §2b with a full queue-manager runbook:
`docs/research/2026-08-31-base-bakeoff-plan.md` (common harness, exact
commands, pre-registered verdict rules, gate B-α decision rule, env risks).
B6/B7 → INDEXED (§4, conditional). B1's instrument is landed and dry-run
smoke-tested: `experiments/training/truncate_layers.py`.

## 4. INDEXED — decided elsewhere, listed for single-lookup (do not re-derive)

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
