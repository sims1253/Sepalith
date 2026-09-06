# Production fine-tune #1 — the living plan (v1: recipe resolved)

Status: v1 (2026-09-06 evening). The iteration vehicle the user asked
for: pre-registered blocks that verdicts REWRITE, not re-plan. Every v0
slot has now landed — every component of the recipe carries a measured
verdict (base B-β/B13, midtrain B8/B8b, masking B9, mode PFT1, data
TU2, LR the 4-arm sweep). Owner: zcode-queue-mgr; user arbitrates GO.

## 1. Base pick — RESOLVED by elimination (2026-09-05, B13)

**GDN/Qwen3.5 at the b4 config (`qwen3.5-2b-base-text-hf`, 1.88B
text-only).** The v0 OPEN USER CALL is closed: the production-finetune
track GO landed with the pick resolved by elimination, the user's
try-first directive honored first (queue §2b B13 row) —
- **Gate B-β (09-04)**: three-way statistical tie at the top
  (b5/spark/b4; spark-vs-b4 McNemar p=1.0); GDN wins the
  pre-registered product axis (b4 85.1% valid @ 19.2 t/s; granite's
  +2.7pp NOT significant at 1.8x decode cost); the class holds
  quality-at-decode at BOTH sizes (b2 82.7 @ 36.2).
- **B13 (09-05) eliminated the last challenger**: LFM2.5-2.6B
  quality-TIES the top table (vs b4 p=0.33/1.0, spark p=0.30, granite
  p=0.79) but fails restraint (noopFP 99.0% scored vs the field's
  58.8-59.8) and decode at quant (6.6 t/s contended / ~11-14 clean vs
  the 19.2 bar) — class-shaped failures, so B7 stays unspent; lfm1.0
  license cap ($10M revenue) noted for any production rec.
- **Ouro-1.4B / K2-Horizon-0.9B disqualified pre-spend** (recon
  2026-09-04: no llama.cpp arch + trainer pin / no base variant +
  unmerged fork); **MiniCPM dense prior loss stands** (gate B-α: 8.2pp
  behind at decode parity with GDN).
Challenger: **spark SWA stays W36-conditional** (GPU-offload decode
lever — 254 t/s measured on the PR build; dissolves its decode
objection IF deployment allows). **Granite-3B = the native-FIM
control**: its format class (79.1) does NOT transfer to GDN via a
midtrain stage (B8/B8b) — the remaining route is a midtrain-native
base, i.e. granite itself, not a stage.

## 2. Recipe block — RESOLVED (every slot landed, each with its verdict)

- **Base**: b4-config (§1). Trains ONLY via unsloth-with-knobs on this
  box (`UNSLOTH_COMPILE_DISABLE=1 UNSLOTH_DISABLE_AUTO_PADDING_FREE=1`
  mandatory for this arch; HF path = 25 s/it, the B4 saga).
- **Training mode: LoRA r32/a64** — PFT1 (09-06) is a clean sweep for
  LoRA: full-FT at matched budget loses exact by 17.3pp (p=1.3e-10),
  valid by 16.9pp (p=7e-11), is WORSE on noopFP (63.7 vs 58.8 scored,
  p=0.0020) and exceeds the forgetting gate (general-R BPB +1.287% >
  ≤1%); the rescue LR never fired. All three WINNER-FT conditions
  FAIL → **question closed at this scale**; reopen only if noopFP/
  doc_sync show capacity signatures (they have not — doc_sync is 0.0
  with EVERY parameter trained). LoRA also wins the adapter-native
  axes by default (B10 WiSE-FT, B8b-style stacking, runtime adapter
  selection).
- **LR: 2e-4 cosine** — validated by the 4-arm sweep {5e-5, 1e-4, 2e-4
  anchor, 4e-4} × 300 steps on Anyscale (eval@300: 1.249 / 1.237 /
  1.226 / 1.222): monotone improvement to the anchor; 4e-4 eval-ties
  it within noise (best train loss) with the late-horizon caveat — a
  300-step instrument cannot clear 4e-4 for the 3000-step production
  run's late damage; 5e-5 clearly under-learns. NO case to move off
  the banked 2e-4 anchor.
- **Data: sft_v7 raw route** (48k shuffle(42) cap, ground-truth
  targets) — TU2: solve-gating −7.06pp (p=0.0051, primary 624-row
  pair); teacher renderings catastrophic (−37.65pp, p=7e-26, noopFP
  guardrail FAIL) and the consistent-teacher mechanism itself is
  NEGATIVE (−9.80pp paired at identical prompts+steps, p=0.0199; pipe
  −72.2, format −29.9 — the teacher's multi-rendering freedom buys
  noise). **Teacher-in-the-loop data CLOSED on this rig** (the
  solve-gate filters through glm-5.3's 26.4% one-attempt
  task-inference profile, not derivability). The v2-data slot (B11
  pack) is unchanged — data-program owned.
- **Midtrain stage: NONE** — B8 (full-replacement astfim): 0.8% valid /
  0.0% exact (p≈1e-48), restraint collapse (91.7% scored) —
  catastrophic. B8b (stacked midtrain→SFT, granite's structure): exact
  parity (valid p=0.36, exact p=0.52; noopFP decisions row-identical,
  0/0 discord; format valid tied at 71.6) with eval_loss tracking b4's
  banked curve within 0.001 at every checkpoint — **product SFT
  annihilates the merged midtrain deltas to near-identity**. A ~3h GPU
  stage buying 0.0pp on every measured axis = DROP. Granite's format
  class needs a midtrain-native base, not a stage.
- **Masking: PLAIN SFT (completion masking OFF)** — B9 SeleKT
  keep-50%: exact −29.4pp (p=1e-15), valid −12.9pp (p=2.1e-9),
  format_propagation collapses hardest (34.3 vs 71.6, discord 25/0);
  mechanism = the masked-out half is dominated by the
  must-be-exactly-right tokens the base already predicts, which then
  never enter the loss. Third masking negative on this base
  (B8/B8b/B9) — the family is bracketed; reopening needs a mechanism
  argument, not a variant.
- **Steps: 3000** (series-comparable; nothing measured beyond it on
  the winner — the dose-response stays a one-at-a-time delta, §4).
  seq 2048, seed 3407 — the B-series uniform recipe, comparability
  intact.
- **RL phase: LoRA on the LoRA-trained base** (PFT1 §6 — the
  memory-forced hybrid applied only to a WINNER-FT, which did not
  happen). W17 constraints banked (no warm-start, no critic blend,
  zero-std groups, no-op arm).

## 3. Acceptance gates (pre-registered; carried from v0 unchanged)

- Scenario validator ≥ b4's 85.1% − 2.0pp (no regression vs its own
  base arm) AND no-op FP ≤ b4's 67.4% + 2pp (scored convention 58.8).
- Episode metrics (V1a): accept_rate ≥ b4-baseline arm; time-to-edit
  and interruption reported (no gate yet — first calibration on the
  winner arm).
- AST-equiv (V1b): reported as a column; gate TBD after the first
  winner-arm baseline.
- doc_sync: NOT a gate — construction problem at maximum statistical
  power (0.0% at every size 350M→3B, for the teacher 0/550 solved,
  AND under FULL-FT with every parameter trained); owned by the data
  program (TU3's construction lane), not by training.
- Decode: tg128 ≥ 18 t/s CPU t8 (the b4-class bar; if the product path
  allows GPU offload, W36 reopens spark).

## 4. Sequence to first run (v1: steps 1-2 DONE)

1. ~~User confirms/overrides the base pick~~ — RESOLVED by elimination
   (§1; B13, 2026-09-05).
2. ~~B8 + B9 fill the §2 slots~~ — DONE, plus the deltas v0 did not
   know were coming: midtrain DROP (B8/B8b), masking PLAIN (B9), mode
   LoRA (PFT1), data raw route (TU2), LR 2e-4 held (4-arm sweep).
3. **W16 pipeline adaptation** (tokenizer/PSM/eval harnesses on the
   pick) — PENDING; the live head of the track.
4. Comparable recipe run (3000 steps, sft_v7) → full battery incl. v2
   columns = the winner-arm baseline for every future gate.
5. Recipe deltas (steps dose, B11 data, B10 blends) — one at a time,
   each with the McNemar line per the audit rule.
6. RL phase (2c) decision AFTER production SFT #1 — W17 constraints
   banked.

## 5. Risk register (v1 additions — the negatives that carry forward)

- **Domain-asymmetric forgetting (PFT1 secondary)**: b4's LoRA degraded
  base general-text BPB by +20.3% vs base (0.0893→0.1074) where the
  properly-lr'd full update held it (0.0896 ≈ base 0.0893). The rank-32
  "forgetting insurance" cuts both ways: it freezes 98% of the weights
  yet perturbs out-of-domain text MORE than full FT at this scale. No
  action while the product axes stay in-domain; flagged the moment
  out-of-domain quality becomes an axis.
- **doc_sync = construction, at max power**: 0.0 with EVERY parameter
  trained (FULL-FT) AND at every size AND for the teacher (targets
  verbatim-pinned — glm-5.3 produces close paraphrases that score 0).
  The data fix lives in TU3's construction lane, not in training.
- **noopFP residual is structural**: production field 58.8-59.8 scored
  / 67.4-68.2 all-cases across every base/size/recipe/mode tried;
  short single-mixture adaptations run 93-99% (no restraint families
  in the mix — TU2's structural note). No capacity signature — it
  moved the WRONG way under full FT (58.8→63.7). The bankable
  mitigation instrument is **X5's residual-gated abstain: 44.5% of
  incorrect rows suppressed at 0-correct-lost @64, 80.3% @32d8**
  (S0 replay baseline 28.5%) — flagged as an H-series/product lever
  (an inference-time gate, orthogonal to the training recipe).

## 6. Infra (v1)

- **Serving path: W16 pending** (tokenizer/PSM render formats/eval
  harnesses on the pick) — the track's live head; everything upstream
  of it is banked.
- **Abstain-gate option**: X5's instrument (§5) is the named lever if
  noopFP must go below the structural floor; serving-side, orthogonal
  to the recipe.
- **Cloud economics (Anyscale, proven)**: measured $0.57/arm (LR
  sweep, g5.xlarge, incl. 2 eval passes); TU2's short arms ~$1.1 the
  batch; runbook estimates B7-class ≈$2.7, GRPO arm ≈$2.5-3,
  B13-class ≈$6.5-8 — i.e. ~$1-3 per SFT/RL arm at the 1-2B class.
  Ledger: ~$3.9 of the $100 org credit spent (smoke $0.55 + TU2
  ~$1.1 + sweep $2.26) → **~$95 remaining, reserve posture** — W29
  policy holds ≈$85 for the production fine-tune; opportunistic
  de-serialization spends only.
- **Kaggle**: the GPU fleet CANNOT train the b4 GDN recipe (all
  pre-Ampere, no bf16; unsloth's no-bf16 path broken — the
  9-iteration ladder); the CPU pipeline is proven (byte-exact staging
  + attachment guard). The $100/mo benchmark credits are
  model-inference spend, NOT training compute — earmarked for the
  benchmark-scan thread (R-edit benchmark proposal, model B monthly
  external-model scan).

## 7. What v1 does NOT decide (open, each with its queue home)

- **W16 serving-path adaptation** — queue §3f W16 (the live head).
- **Benchmark carve/publication** (user) — R-edit benchmark proposal
  (staged B-first recommendation); user GO pending.
- **LOC1-S1 paid pilot** (user) — recipe-pilot GO only if the user
  funds ~$50-100 API + GPU half-day (S0 passed but lexical ties/wins).
- **S3 gate** (Uno diffusion-draft fork warrant) — PROPOSED, awaiting
  GO; S0 prize math lands after S1-CPU legs.
- **H3-S0** (render-format paper gate) — CPU-class, chain-safe, per
  H1's skip-to.
- **TU3 build** (cross-file reference family incl. the doc_sync rescue
  lane) — CPU/API build; its GPU arm was behind TU2, now cleared.

## Amendment log

- 2026-09-04 v0: skeleton from gate B-β; §1 recommended; §2 slots open.
- 2026-09-06 v1: recipe RESOLVED — base by elimination (B13), mode
  LoRA (PFT1), LR 2e-4 held (4-arm sweep), data sft_v7 raw route
  (TU2), midtrain DROP (B8/B8b), masking PLAIN (B9); §1 GO superseded
  by elimination (user try-first honored: LFM run, Ouro/K2
  disqualified, MiniCPM prior); risk register, infra, and open-items
  sections added. Sources: base_bakeoff RESULTS.md §B-β/§B13/§B8/§B8b/
  §B9; lora-vs-fullft §6; TU2_RESULTS; queue §3 PFT1/X5 rows;
  kaggle-compute-integration LR-sweep appendix + the board's Anyscale
  ledger.
