# B-series RESULTS — base bake-off × param-floor ladder

Canon per runbook `docs/research/2026-08-31-base-bakeoff-plan.md`. One section
per rung, gates as their own sections. Owner: zcode-queue-mgr (chain
`scripts/run_b_series_chain.sh`, battery fixer + watcher daemons, 2026-09-02).

## Battery conventions (this series)

- Scenarios/noop run via `.venv` (tree_sitter_r); midtyping raw+suffix via
  `.venv-sft` run_eval, `--limit 18`, first-18 of the CURRENT
  edit_pairs_v1/eval.jsonl — which has DRIFTED since v7 (banked rows
  cli/data.table do not join; caveat on the board 2026-09-02T04:0x).
  Within-series comparability holds (identical command/file across arms);
  never quote a B-stem against v7 numbers without the a0 delta.
- Full raw completions persisted per row (`raw` field) — eval-v2 re-scorable.
- Decode rows: llama-bench t8 CPU (b10453), contended-box convention.
- v2 columns (eval-strategy `docs/research/2026-09-02-eval-strategy-v2.md`):
  ast_equiv via `experiments/eval/ast_equiv.py`, additive-only.

## b12 pre-gate 2 — trainer-stack calibration anchor (b12_cal_minicpm5)

PASSED 2026-09-02 (both B12 pre-gates now green; spark rung joins gate B-α).
- Train: TRL/PEFT path (train_sft_trl.py), 3000 steps, sft_v7 data —
  COMPLETED (train_loss 1.2405/eval 1.2723); smoke-gen crashed on
  token_type_ids AFTER training (script fixed); final_lora recovered from
  checkpoint-3000, exported standard-path (b12_cal_minicpm5-Q8_0.gguf).
- The measured unsloth↔PEFT delta vs b1_ref24 (same base/data/steps,
  n=255 paired rows, exact McNemar):
  | metric | a0 (unsloth) | anchor (PEFT) | delta | p |
  |---|---|---|---|---|
  | scenario valid | 76.9% | 73.7% | **-3.1pp** | 0.039 |
  | scenario exact | 65.9% | 63.5% | -2.4pp | 0.180 |
- Application rule: read b12_spark17b with a **+2..3pp** handicap against
  unsloth-trained rungs (valid leg barely significant, exact not; the band
  is the honest quote). Per-family delta concentrates in
  format_propagation (-7.5pp) and pipe_rewrite (-5.6pp); rename -0.7pp;
  na_rm n=5 (noise).

## B1 a0 — reference 24L (b1_ref24)

- Train+export clean (unsloth, 1.5h). Decode: pp512 288.5, **tg128 34.4 t/s**
  (t8 CPU, contended). Midtyping raw/suffix: 18 rows, 0 exact / 0 first_line
  (matches the banked convention — v7's own rows were also all-zero; weak
  signal leg). Noop FP: pending fixer completion. doc_sync 0/15 (the known
  open thread; B5's disambiguation stands).
- Scenario battery: **76.9% valid / 65.9% exact** (255 rows; per-family in
  the jsonl; rename 86.7, pipe 94.4, format 67.2, na_rm 80.0, doc_sync 0.0).

## B1 a1-a3, B2, B3 — pending (chain running)


## B3 — LFM2.5-350M (conv+GQA at the latency floor) — rerun verdict

- v1 INVALID (LoRA under-attached: 983K/355M params, q/k/v only — see board
  06:5x). Rerun with full all-projection targets (10.0M/364M ≈ 2.75%,
  incl. conv in/out_proj + w1-3): **15.7% valid / 95.0% no-op FP**
  (255+258 rows) vs v1's 26.7%. MORE adapter capacity made it WORSE —
  the conv trunk is fragile under LoRA at lr 2e-4, or 350M simply lacks
  the capacity for the zeta2 format+task. Per-family: rename 21, format
  12, pipe 6, na_rm 0, doc_sync 0.
- Decode is the class's one strength: **tg128 94.1 t/s** (2.7x the 1B
  dense reference's 34.4) — fastest arm in the series.
- GATE READING: conv+GQA at 350M is ELIMINATED on the product axis
  (15.7% vs leader 82.7%, far outside the 2.0pp survival rule). B7's
  rescue condition (LFM2.5-1.2B only if B3 collapses on edit accuracy)
  is TRIGGERED but stays conditional — not auto-queued; the gate note
  carries it.

## GATE B-α (2026-09-02, 18:4x) — the architecture verdict

All arms, uniform recipe (3000 steps, sft_v7, LoRA r32α64), series-internal
comparability. Decode = llama-bench t8 CPU (contended-box; spark's row
re-measured -ngl 0 — its PR-build default offloads GPU). ast_equiv v2
columns in preparation (raw outputs persisted per row).

| stem | class | params | valid% | exact% | noopFP% | tg128 |
|---|---|---|---|---|---|---|
| b12_spark17b | SWA 3:1 | 1.71B | **85.1** | **77.3** | **67.8** | 17.4 |
| b2_qwen35_08b | GDN 3:1 | 752M | 82.7 | 73.3 | 71.3 | 36.2 |
| b1_ref24 (a0) | dense | 1.08B | 76.9 | 65.9 | 79.5 | 34.4 |
| b12_cal (anchor) | dense, TRL path | 1.08B | 73.7 | 63.5 | 82.9 | 28.8 |
| b1_l20 | dense | 967M | 72.2 | 63.5 | 81.0 | 28.0 |
| b1_l16 | dense | 854M | 56.9 | 51.4 | 79.8 | 46.1 |
| b3_lfm25_350m | conv+GQA | 355M | 15.7 | 10.2 | 95.0 | 94.1 |
| b1_l12 | dense | 741M | 12.5 | 9.4 | 99.6 | 56.1 |

Curves: accuracy-vs-params — GDN@752M (82.7) > dense@1.08B (76.9);
SWA@1.71B tops. accuracy-vs-decode — GDN is the sweet spot (82.7 at
36 t/s); SWA buys +2.4pp for 2x decode cost; conv's 94 t/s buys nothing.

**VERDICT (pre-registered rule: top-2 classes on the product axis;
survival = within 2.0pp validator of the leader):**
1. **SWA 3:1 hybrid (Spark) = LEADER.** 85.1% raw on the HANDICAPPED
   (TRL) path — with the measured trainer delta (-3.1pp valid, pre-gate
   2) its unsloth-equivalent is ≈88%. Best validator, best exact, best
   no-op FP, first arm to max a family (na_rm 100%). Decode price: 17.4
   t/s CPU — quote it with every number. Verdict-grade per runbook
   (parity 3/3 + trainer anchor both passed before the gate).
2. **GDN 3:1 (Qwen) = conditional #2.** 2.4pp behind the raw leader —
   outside the strict 2.0pp bar, but the best validator at mainstream
   decode and top-2 by elimination. **B4 (Qwen3.5-2B) is now the
   decisive rung for the class**: GDN@2B within 2pp of spark-adjusted
   reinstates it outright.
3. **Dense (MiniCPM lineage) does not survive** as the next-build
   substrate: 8.2pp behind at decode parity with GDN. The serving
   lineage (v7/v8_2) is unaffected — this gates the NEXT base choice.
4. **conv+GQA eliminated** (15.7%; more-LoRA-worse fragility, §B3).
5. **doc_sync = 0/15 on every arm incl. 1.71B** — the
   capacity-vs-construction question tilts construction/data; B5
   (granite-4.1-3b) is the disambiguation's final word and stays
   scheduled (its native-FIM read-out also stands).
6. Sequencing (card budget): D-grid overnight (user's undertraining
   question), then B4 → B5 tomorrow; B8/B9/B10 fold onto the winner.

## Gate B-α amendments (user review 2026-09-02 evening)

1. SIZE CONFOUND (explicit): Spark's lead is at 1.71B with NO matched-size
   competitor in the grid — "SWA hybrid leads" is provisional on size
   controls. The clean matched-params readings so far: GDN@752M beats
   dense@1.08B (82.7 vs 76.9, -30% params) — the GDN-over-dense claim is
   size-controlled; the SWA-over-GDN claim is NOT (1.71B vs 752M).
   B4 (GDN@2B) and B5 (dense@3B) bracket Spark from both sides and are
   the controls. Do not attribute Spark's +2.4pp to the SWA mechanism
   before they land.
2. DECODE-OPTIMIZATION SCOPING (W36): compute-scaled expectation for
   1.71B at t8 from the dense 1.08B row (34.4 × 1.08/1.71) ≈ 21.7 t/s;
   observed 17.4 = ~80% of curve — a ~20-25% gap, plausibly draft-PR
   overhead (parity-verified, not perf-tuned). Bigger lever: the PR build
   on GPU measured 254 t/s tg128 — if the product path ever allows GPU
   offload, Spark's decode objection mostly dissolves. W36 = the 1-2 day
   probe, conditional on Spark surviving B4/B5 as the pick.

## GATE B-β (2026-09-04, 21:5x) — the final architecture verdict

| stem | class | params | valid% | exact% | noopFP% | tg128 | trainer |
|---|---|---|---|---|---|---|---|
| b5_granite3b | dense-3B (native FIM) | 3B | **87.8** | **78.0** | 68.2 | **10.65** | unsloth |
| b12_spark17b | SWA 3:1 | 1.71B | 85.1 | 77.3 | 67.8 | 17.4 | TRL* |
| b4_qwen35_2b | GDN 3:1 | 1.88B | 85.1 | 76.5 | 67.4 | 19.2 | unsloth |
| b2_qwen35_08b | GDN 3:1 | 752M | 82.7 | 73.3 | 71.3 | 36.2 | unsloth |
| b1_ref24 | dense | 1.08B | 76.9 | 65.9 | 79.5 | 34.4 | unsloth |

McNemar (valid, n=255 paired): spark-vs-b4 discord 12/12 **p=1.0000 —
exact tie**; b5-vs-spark 11/4 p=0.119; b5-vs-b4 13/6 p=0.167 — b5's
+2.7pp is NOT significant (TIE-UNDERPOWERED per the audit vocabulary).

**VERDICT: three-way statistical tie at the top (b5/spark/b4); the
classes separate on the product axis, size, and secondary readouts.**
1. **Quality-at-decode (the pre-registered product axis): GDN wins.**
   b4 ties the field at 19.2 t/s (fastest of the top-3); b5 buys an
   insignificant +2.7pp for 1.8x decode cost (10.65). The GDN class
   holds the best validator-at-decode at BOTH sizes (b2 82.7@36.2,
   b4 85.1@19.2).
2. **Size-confound RESOLVED (user's amendment)**: at ~1.8-1.9B, GDN =
   SWA exactly (p=1.0). Spark's gate-α lead was size, not SWA. The
   earlier size-controlled claim (GDN > dense at 0.75-vs-1.08B) stands;
   at 3B dense catches up on quality while losing decode.
3. **spark's asterisk**: TRL-trained — unsloth-equivalent ≈88 by the
   b12_cal delta (would lead nominally); W36's GPU-offload path (254
   t/s measured) dissolves its decode objection IF the deployment
   allows. Quoted both ways; the measured number is 85.1.
4. **granite's native-FIM signal**: format_propagation 79.1% = best of
   ALL arms (b4 71.6, spark 73, a0 67.2) — consistent with native-FIM
   pretraining helping format-shaped edits; B8's midtrain cross-check
   on the winner will test whether that transfers to the GDN base.
5. **doc_sync = 0.0% at EVERY size (350M→3B)** — capacity-vs-construction
   DISAMBIGUATED: construction/data. B8 + the data program (W6/W8)
   own it; no bigger base fixes it.
6. **PRODUCTION RECOMMENDATION (owner-confirmed pick pending)**: the
   **GDN class (Qwen3.5 lineage) as the production fine-tune base** —
   b4's config specifically (1.88B): tied-best quality, best decode
   of the leaders, two-point size curve, mainstream ecosystem. Spark =
   challenger pending W36; granite = the native-FIM control via B8.
   Infra note: this base trains ONLY via unsloth-with-knobs on this
   box (5-attempt B4 saga documented in gpu.md; HF path = 25s/it).

## B13 — LFM2.5-2.6B-Base (conv+GQA at the family ceiling, TRUE full attachment) — 2026-09-05

Stem `b13_lfm25_26b`. Post-gate user-named rung (queue §2b; recon
`2026-09-04-base-candidate-recon-ouro-k2-lfm.md`). Uniform recipe: 3000
steps, sft_v7, LoRA r32α64, lr 2e-4 cosine, seq 2048, seed 3407 —
identical to B2/B4. Weights were already local (`experiments/models/
lfm25-2b-base-hf`, 2026-08-20 survey pull) — verified at rung start:
sha256 `3331a7db…402e0551` == HF LFS oid, 5,394,427,448 B exact.

### B3 forensic correction (load-bearing for this rung)

B3-rerun's "full attachment" was NOT full: its saved adapter has 72
modules (self_attn q/k/v/out + feed_forward w1-3), ZERO `conv.*` —
unsloth_zoo `get_peft_regex` converts an explicit leaf list into a
parent-tag regex that does not know the parent name `conv`, so
`conv.in_proj`/`conv.out_proj` silently froze (arithmetic: 72-module
expectation = 10,027,008 = exactly B3's printed count). B3 therefore
never LoRA-trained the conv trunk; the gate-α reading "conv trunk
fragile under aggressive LoRA" was untested — capacity + conv-freeze
were confounded. B13 fixes this via a RAW regex target (train_sft.py
`SFT_TARGETS=regex:` passthrough, added 2026-09-05, default-off):
166 modules = 8×(q,k,v) + 30×out_proj (22 conv + 8 attn) + 22×in_proj
+ 90×(w1-3). **Attachment verified on every load: "Trainable
parameters = 48,922,624 of 2,746,121,216 (1.78%)" — 4/4 loads
identical (meta-device pre-computation + 3 training legs).** A
plain-list run would have silently under-attached to 40,271,872.

### Train / ops ledger (3 launches, 2 reaps, 1 sysmem fix)

- L1 (bs4, fresh): healthy 2.6 s/it to step 1394, then killed at ~1h by
  the session task reaper (tracked bg task; cascade-killed the trainer).
- L2 (bs4, resume ckpt-1000): VRAM crept 23.2→31.9GB into eval-1500 and
  collapsed to 45-49 s/it — the B4-TRL sysmem-fallback signature
  (128k-vocab unfused-CE logits at bs4; lfm2 is not in unsloth's
  fused-CE family). Killed per decision rule at step ~1497.
- L3 (bs2×ga8, resume ckpt-1000, **setsid-detached** — the reap fix):
  VRAM 14.9-16.8GB flat through eval-1500/2500, finished 3000/3000,
  train_loss 0.6765. Pace 2.75→~6.5 s/it under co-runner CPU/bandwidth
  contention (two foreign llama-server eval processes at ~790% CPU each
  for most of the window). Wall 23:49→06:14 (~3.4h net stepping).
- Memory delta vs plan (for future rungs): workload-side ~26.7GB peak
  at bs4-resume (expected ~20) vs 16.8GB at bs2 — the logits block
  (bs×2048×128k vocab) dominates; on this box train LFM2.5-2.6B at
  bs2×ga8. Ops lessons on the board (2026-09-05 02:4x): workload
  detached + watcher tracked; ~1h reap kills tracked tasks.

### Battery (Q8_0, b10453 CPU convention; raw outputs persisted)

| metric | b13_lfm25_26b | b4 (bar) | spark | granite (b5) | b3 (350M) |
|---|---|---|---|---|---|
| valid % | **87.1** | 85.1 | 85.1 | 87.8 | 15.7 |
| exact % | 76.9 | 76.5 | 77.3 | 78.0 | 10.2 |
| noopFP % (all-cases / scored-only) | **99.2 / 99.0** | 67.4 / 58.8 | 67.8 / 59.3 | 68.2 / 59.8 | 95.0 / 93.6 |
| midtyping raw/suffix (18 rows) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| tg128 t/s (Q8, t8 CPU) | **6.60 ± 0.43** | 19.21 | 17.4 | 10.65 | 94.1 |

- **Quality: PASSES the bar and joins the top-of-table tie.** McNemar
  (n=255 paired): vs b4 valid 11/6 p=0.33, exact 10/9 p=1.0; vs spark
  valid 10/5 p=0.30; vs granite valid 6/8 p=0.79 — statistically
  indistinguishable from all three leaders. Per-family: rename 98.7/
  90.0, pipe 88.9/88.9, format 79.1/61.2 (ties granite's best-in-field
  format valid), na_rm 100/80 (n=5), doc_sync 0/15 (universal
  construction problem). B3's 15.7% collapse is REFUTED as a class
  property: it was 350M capacity + the conv-freeze artifact.
- **Restraint: FAILS, B3-class.** noopFP 99.0% on scored no-op classes
  (n=204; every class ≥0.96, incl. both temptation classes at 1.00) vs
  the 58.8-59.8% field. The class's failure mode (propose-always ghost
  text) persists at the ceiling with the conv trunk trained — this is
  the pre-registered format-fragility arm and it is load-bearing:
  quality ties the field, restraint disqualifies.
- **Decode: FAILS at the standard quant (contended reading).** 6.60
  t/s tg128 measured under load ~20 (two foreign eval servers active
  through the bench — documented above). Param-scaled clean estimate
  ~11-14 t/s (b4 19.2 × 1.88/2.69; granite 10.65 × 3.40/2.69) — under
  the 19.2 bar either way at Q8_0. pp512 41.4 ± 1.5 same conditions.
- Midtyping join-check: PASS 18/18 (i,sha) keys identical to banked b4
  rows, same order, raw+suffix (the chain's in-script check used the
  v7-era `id` field — these files key on `(i, sha)`; corrected offline,
  script comment updated by hand in RESULTS only).
- Artifacts: `experiments/models/b13_lfm25_26b-Q8_0.gguf` (2.87 GB),
  runs/logs under `/mnt/h/sepalith/runs/b13_lfm25_26b*`, eval rows
  `experiments/eval/results_{scenarios,noop_fp}_b13_lfm25_26b.jsonl` +
  `results_b13_lfm25_26b_midtyping{,_suffix}.jsonl` (per-example rows
  persisted for eval-v2 / McNemar re-runs).

### Verdict + B7 order rule

**conv+GQA at 2.6B: quality-capable, product-eliminated.** The class
reaches the top-table tie at 87.1/76.9 (B3's elimination at 350M was
real but its cause is now split: capacity yes, conv-LoRA-fragility
unproven — the trunk was never trained there). On the pre-registered
product axes it fails twice: restraint (99% noopFP vs ~59-68 field)
and decode at the standard quant (6.6 measured / ~11-14 clean vs 19.2
bar). The B-β production recommendation (GDN/Qwen3.5 b4-config) is
UNCHANGED; LFM remains license-constrained (lfm1.0 $10M revenue cap)
for any production rec.

**B7 ORDER RULE: B13 PASS on quality ⇒ B7 (LFM2.5-1.2B-Base) stays
OPTIONAL, not auto-retired.** Recommendation: do not spend it now —
the two product-axis failures (restraint, decode-at-quant) are
class-shaped, and a 1.2B rung would mainly re-test them at a size
where capacity is lower; if a future CPU-latency tier ever wants the
94-t/s-class decode family, B7 is the probe to run (with this rung's
regex target set + bs2 knobs + detached-launch pattern).

## B8 — AST-FIM midtrain re-probe, FIXED instrument, on the gate winner (b8_midtrain_qwen35_2b) — 2026-09-05

Arm = the banked b4_qwen35_2b recipe VERBATIM (3000 steps, LoRA r32/a64
lr2e-4 cosine, seq 2048, seed 3407, shuffle(42)+48k cap, the qwen3.5 GDN
target list) but with the dataset REPLACED by astfim_v1 (276,206 rows) and
MIDTRAIN_MASK=1: completion-only loss masking (prefix-route LCP) + the
conservative bucket packing (group_by_length; MIDTRAIN_PACK=seq is refused
for GDN by the instrument). Paired control = the BANKED b4 rung (same base,
sft_v7, no midtrain). Base = experiments/models/qwen3.5-2b-base-text-hf
(the resolved GDN pick). Runner zcode-b8-run; chain script
`scripts/run_b8_midtrain.sh` (gates A/B/C in-script); McNemar helper
`experiments/eval/b8_mcnemar.py` (validated: reproduces the banked b4
85.1/76.5 and the B-β discord counts on self-pairing).

### Instrument health — the thing B8 existed to un-confound (ALL PASS)
Pre-registered signature vs measured (train_sft [midtrain:*] telemetry):
- attachment: Trainable 21,823,488 of 1,903,648,576 (1.15%) — the exact b4
  line (gate A; B3 incident rule).
- prefix-route: 47,381/47,381 routed = 100% (pre-registered ~100%);
  619 rows >2048 tok dropped, 0 seam-dirty.
- **token-seam exact: 48000/48000** (pre-registered; the broken 08-19
  instrument had 0% exact masking — this is the fix, measured on the full
  48k selection). eval split 500/500, 12.0% completion.
- completion share 12.2% (pre-registered ~13-14%; b8-patch sample 13.5%).
- finite losses throughout: first 0.856 (step 20), eval_loss
  0.773→0.761→0.743→0.720→0.709→0.710 (plateau), final train 0.604;
  zero non-finite entries (gate C). Runs ABOVE b4's ~0.68 final as expected
  — completion-only masking scores only target tokens.
- ops: 3h01m wall, ~3.6s/it avg (astfim rows are longer than sft_v7),
  VRAM flat 16.2-16.3GB (b4 peak 21.6) — no creep, no SFT_PD_BATCH fallback.

### Battery (Q8_0, b10453 CPU convention; contended box — two foreign
llama-servers through most legs; raw outputs persisted)

| metric | b8_midtrain | b4 (paired control) | granite (b5) |
|---|---|---|---|
| valid % | **0.8** | 85.1 | 87.8 |
| exact % | **0.0** | 76.5 | 78.0 |
| noopFP % (all / scored n=204) | **93.4 / 91.7** | 67.4 / 58.8 | 68.2 / 59.8 |
| format_propagation valid % | 3.0 (2/67) | 71.6 | 79.1 |
| midtyping raw / suffix (18, join PASS) | 0 / 0 | 0 / 0 | 0 / 0 |
| tg128 t/s (Q8, t8 CPU) | 17.99 ± 1.67 | 19.21 | 10.65 |

McNemar vs b4 (n=255 paired, exact binomial): valid discord 215/0
(ctl+/arm- 215, arm+ 0) **p≈1e-48 → significant DECISIVE loss**; exact
discord 195/0, p≈1e-44. Per-family valid: rename 0/150 (b4 97.3), pipe
0/18 (100), na_rm 0/5 (100), format 2/67 (71.6), doc_sync 0/15 (0 — tied,
universal construction problem). Midtyping join-check: PASS 18/18 (i,sha)
keys identical to banked b4 rows, same order, both alignments; line_f1
0.009/0.007 vs b4 0.006/0.033 (both ~floor).

### Failure mode (read the predictions, not just the rates)
Output is FLUENT R but in AST-FIM stream format: `<filename>` chains,
`<[fim-prefix]>/<[fim-suffix]>` marker echo, `=======`/`>>>>>>> REPLACE`
diff-marker loops, degenerate repetition (e.g. `vignettes2 <- FALSE` x N).
The scenario validator rejects on shape ("single-line region family, got
92 lines"). The model learned span-completion, not the zeta2 edit-block
contract — it never saw a product-format row (astfim_v1 replaced sft_v7
for the full 3000-step budget). The pre-registered RAW-PSM/format-transfer
caveat applies in full: these numbers measure ZERO-SHOT format transfer of
the zeta2 battery onto an astfim-only model, not absence of edit-span
ability per se. Restraint collapsed with it (91.7% scored noopFP —
B13-class propose-always; every no-op class ≥0.78, temptation classes
1.00/0.83).

### Verdict — instrument VALIDATED, arm DECISIVE NEGATIVE (p≈1e-48)
1. **The 08-19 confound is resolved**: with completion-only masking +
   packing verified end-to-end (seam exact 48000/48000 on the real corpus),
   the midtrain instrument works. Any future astfim-class result is now
   interpretable.
2. **Midtrain-only does not subsume product SFT** — it is a catastrophic
   product regression on every axis (quality p≈1e-48, restraint 58.8→91.7,
   midtyping flat). AST-FIM span training alone does not induce the
   product edit format on the GDN base.
3. **Granite cross-check (the B5 question)**: granite's best-in-field
   format_propagation 79.1 sits ON TOP of product SFT (native-FIM
   pretraining + sft_v7). B8 shows the granite class of gain does NOT
   transfer to GDN via midtrain-only: format 71.6→3.0. The runbook's
   "then the standard loop" stacked arm (midtrain THEN sft_v7) remains
   UNTESTED — as staged by the queue manager (single 3000-step astfim run
   vs banked b4), this run cannot attribute any gain to the midtrain stage.
4. **Production plan §2 midtrain slot: recommendation DROP** (as
   currently constituted — full-replacement astfim stage). Remaining
   live options per the 08-19 close-out: midtrain→SFT stacking (~3h GPU,
   queue-mgr call), shorter spans, or the midtrain-native base (granite).
   Nothing here revives doc_sync (0/15, tied with everything —
   construction problem, gate B-β).
- Artifacts: `experiments/models/b8_midtrain_qwen35_2b-Q8_0.gguf` (2.01GB),
  runs/logs `/mnt/h/sepalith/runs/b8_midtrain_qwen35_2b*` + `b8_chain.log`,
  eval rows `experiments/eval/results_{scenarios,noop_fp}_b8_midtrain_qwen35_2b.jsonl`
  + `results_b8_midtrain_qwen35_2b_midtyping{,_suffix}.jsonl` (mirrored to
  `/mnt/h/sepalith/runs/b8_midtrain_qwen35_2b/eval_rows/`).
