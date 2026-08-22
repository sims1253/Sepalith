# FIM Dose Ladder — results (POC scale, night of 2026-08-22)

The dispute: A2 said 35-40% of pretraining should be PSM-format span-FIM;
B2 said 18%. Red-team adjudication (redteam-2 §4, A2 §5.3): neither reading
is evidenced — settled here with data.

Instrument: 4 arms x 350.2M tokens (668 steps x 524,288) on the 206.5M
TinyGQA, adopted twin-POC recipe (Muon 0.01 / embed 4e-3 / WSD / QK-Clip
tau=100 / seed 1273 / torch.compile), IDENTICAL everything except the
PSM-FIM token share. Paired data: both streams tile the SAME astfim_v1
fixed-corpus rows — FIM stream = row text as-is (PSM surround format;
byte-identical to the twin-POC stream), causal stream = the same rows
re-rendered as plain documents (markers stripped, span restored to natural
position). Nested Bernoulli slot mask (one fixed U[0,1) stream, MASK_SEED
90210; slot is FIM iff u < dose) → dose 0.35's FIM slots ⊃ 0.20's ⊃ 0.10's;
realized FIM shares 0.0 / 0.102 / 0.203 / 0.355. Arms ran sequentially,
~81-84 min each, zero GPU yields, 68-70k tok/s shared with three other
agents' llama-servers. Full-loss training on both streams at POC scale (the
dose axis is the FORMAT mix; masked-span loss is the 0.5BT A2-prime arm and
is untouched by this ladder).

## Dose-response table (all at 350.2M tokens)

| arm | FIM dose | train loss @350M | causal BPB (floor) | floor Δ vs 0% | PSM-slice BPB | stop-acc (TF) | served line-F1 | served stop-rate | canary: in-corpus mean span (held-out floor) |
|---|---|---|---|---|---|---|---|---|---|
| ladder_fim0 | 0% | 1.449 | 0.7621 | — | 0.8458 | 3.6% | 0.0027 | 18.8%* | 40.5 (2.6) |
| ladder_fim10 | 10% | 1.430 | 0.7608 | −0.18% | 0.7476 | 16.1% | 0.0026 | 8.1% | 30.2 (1.8) |
| ladder_fim20 | 20% | 1.433 | 0.7590 | −0.41% | 0.7446 | 14.8% | 0.0039 | 8.1% | 32.8 (2.3) |
| ladder_fim35 | 35% | 1.443 | 0.7561 | −0.79% | 0.7409 | 15.3% | 0.0005 | 0.0% | 21.0 (2.4) |
| (anchor muon_final, 681M, 100% PSM diet) | — | — | 0.7565 | −0.74% | 0.7032 | 16.1% | — | — | — |

*fim0's "stops" are incidental — it never learned the terminator; its
18.8% stop-rate is literal-token coincidence in code context.

Eval sets: causal floor = 500 held-out rows plain-rendered (418k tok /
1.283MB); PSM slice = the parent 500-row PSM eval stream (430k tok /
1.309MB); served = 223 held-out PSM rows (prompt ≤640 tok, target ≤320 tok,
prompt+gen inside the trained 1024 positions), greedy, stop `<|end|>`,
llama-server CUDA --parallel 4 port 18107 (4 client threads); canary =
12-gram verbatim-span instrument (arms/regurgitation.py), causal + FIM
renderings, 10 in-corpus + 10 held-out prompts each.

## 1. Causal floor (the pre-registered hard gate)

NO dose pays a causal penalty. The whole ladder sits within −0.79% to
−0.18% of the 0% control — i.e. dose monotonically IMPROVES held-out
plain-text BPB slightly (treat as "no detectable penalty"; the deltas are
inside paired-eval noise). B2's implicit worry — that ~35% FIM-format data
taxes causal quality — does not materialize at 350M tokens. The A2 rule
"highest dose whose causal-floor penalty is ≤1%" therefore points at the
TOP of the ladder: 35% passes with margin.

## 2. Span-objective response (teacher-forced)

PSM-slice BPB: 0.8458 → 0.7476 (−11.6% at dose 10) → 0.7446 → 0.7409.
Strongly saturating: dose 10 captures 93.6% of the total 0→35 improvement,
dose 20 captures 96.5%; the last 15 points of dose buy ~0.5% more.
Teacher-forced stop-accuracy saturates the same way: 3.6% → 16.1% → 14.8%
→ 15.3% (POC stop-density is ~1000x below the 0.5BT design — 350M tokens
vs A2's 6B FIM tokens / 7-17M stop events — so the 90% advisory bar is not
reachable here by construction; the anchor at 2x budget also sits at 16%).

## 3. Served free-running span completion (probe2-style)

Exact = 0 everywhere; line-F1 floored at 0.0005-0.0039 (package-held-out
spans, 206M params, zero-shot — absolute values advisory at POC scale per
A2 §5.3). The informative part is the DOSE CONTRAST and the failure mode:
dose 20 is the best arm (F1 0.0039, top rows to 0.18), while dose 35
COLLAPSES in free-running mode — 0% of generations stop, median completion
runs to the 384 cap, and the text degenerates into repetition loops
("##\n    ##\n    ##..."). The 10-20% arms continue code-plausibly but
repetitively; none reproduce held-out spans. This collapse is a NEW
failure mode neither design anticipated: at 35% share with FULL loss, the
model over-commits to the PSM rendering's surface statistics and its
free-running generation destabilizes. (muon_final — 681M tokens, 100% PSM
diet — did not collapse in the twin POC's 20 greedy samples, so budget
also matters; but at matched budget the collapse appears exactly at the
top dose.)

## 4. Memorization canary (2605.22981 prediction)

No amplification with dose. In-corpus mean verbatim span: 40.5 / 30.2 /
32.8 / 21.0 tokens (causal prompts) — flat-to-DECREASING; held-out floor
2-3 tokens for every arm; prompt-continuation match ~1-2 tokens everywhere.
The 30-80-token in-corpus spans common to ALL arms are 12-gram echoes of
boilerplate R idioms, not dose-driven memorization. The prediction "higher
FIM dose amplifies verbatim regurgitation" does NOT fire at this scale;
what dose 35 does amplify is repetition DEGENERATION (§3), a behavior
cousin, not a memorization effect.

## Verdict (pre-registered rule + honest scope)

- HARD GATE (≤1% causal-floor penalty vs 0%): passes at 10%, 20%, AND 35%
  (all deltas negative). The gate does not bind anywhere on the ladder.
- Span-F1-per-causal-BPB-cost: the reliable span-objective signal (PSM-slice
  BPB) improves monotonically to 35% but is 93-97% saturated by dose 10-20;
  the served free-running signal peaks at 20% and collapses at 35%.

**A2-prime mapping: adopt a 20-35% PSM-FIM doc-rate band, center ~25-30%
— keep A2's masked-loss + span-policy + stop-density countermeasures as
the enabling conditions.** The ladder REFUTES B2's dose worry (no causal
cost anywhere ≤35%) and simultaneously declines to hand A2's 35-40% top a
clean win: at matched budget the top dose's free-running generation
collapses under full loss, and its marginal teacher-forced gain over 20%
is ~0.5% BPB. The 0.5BT A2-prime run should carry masked loss (already
specified) — which trains no non-span free-running behavior in FIM docs
and is the direct countermeasure to the observed collapse — and keep the
anneal-stage lift toward 45% gated on the stop-accuracy readout.

NOT testable here (noted, not silently dropped): A2's masked/unmasked ≥2x
line-F1 gate needs the unmasked-FIM probe2-replica arm (the tasking's
4-arm design has no unmasked-FIM arm at fixed dose); POC-scale absolute
line-F1/exact are advisory by pre-registration.

## Ops notes

- Serving path: TinyGQA → GGUF (llama arch; attention.key_length/value_length
  64 with head_count 16 — head size ≠ d_model/n_heads needs the explicit
  KVs; untied duplicated output head; F32; tokenizer copied from the
  sft_v7_minicpm5 GGUF, add_bos_token=False) — llama-server CUDA loads and
  serves it; tokenizer verified byte-identical to the HF MiniCPM5
  tokenizer on eval prompts; perplexity cross-checked vs torch within
  chunk-noise. GOTCHA fixed mid-run: the fixed-corpus `prompt` field ends
  with a `<|end|>\n` terminator that the TRAINING text does not have at
  that position — served prompts must strip it (else every model drifts
  into next-document mode and never stops; the first fim0/fim10 eval
  pass was re-run after the fix).
- All artifacts: /tmp/poc_twin/ladder/ (two packed streams, eval rows,
  tokenizer json, ckpt_*/final.pt per arm, arm GGUFs); telemetry in
  ladder/logs/*.jsonl; this directory holds code + configs + results.

## Files

- configs/ladder_arms.json (pre-registered design)
- data_prep_ladder.py (causal stream + eval instruments)
- train_ladder.py (mixed-stream trainer, nested dose mask)
- convert_gguf.py, eval_fim_served.py, regurgitation_ladder.py,
  bpb_eval.py, run_ladder.sh
- logs/: ladder_fim{0,10,20,35}.jsonl, fim_eval_*.{jsonl,summary.json},
  canary.json, bpb_eval.json, per-arm stdout/server logs, orchestrator.log
