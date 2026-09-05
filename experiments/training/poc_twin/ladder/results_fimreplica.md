# FIM-Replica — the masked-loss @35% control arm (2026-09-05, zcode-gpushorts)

The last untested leg of the adopted 20-35% FIM-dose verdict (2026-08-22
ladder; `results.md`): design-A2 §5.3's pre-registered **masked/unmasked ≥2x
line-F1 gate** (RT2-A6's relative-gate fix). This file closes it.

## Polarity correction (documented anomaly)

The queue row and tasking label the pending arm "unmasked-FIM @35%
probe2-replica". Repo state says the opposite and the code agrees: the
banked `ladder_fim35` IS the full-loss/unmasked arm — `train_ladder.py`
computed `chunked_ce` = mean CE over ALL tokens of both streams
(commit 1612bce; the 2026-08-22 stdout logs show no loss-mask flag), and the
night-ladder's own verdict attributes the 35% free-running collapse to
exactly that full loss, calling masked loss "the direct countermeasure …
already in A2-prime". No masked-loss arm existed anywhere (bpb_eval.json,
/mnt/h/sepalith/runs, poc_stab checked). The untested side of the ≥2x gate
is therefore the MASKED arm — which is also the A2-prime objective as
specified (§5.2: "loss on span + <|end|> only"). That is the arm run here;
the board anomaly note was posted pre-run (2026-09-05 17:37).

## Instrument (paired to the banked arm, one variable)

- Same 206.5M TinyGQA, same recipe: Muon 0.01 / embed 4e-3 / wd 0.1 / WSD /
  QK-Clip tau=100 alpha=.5 / clip 1.0 / seed 1273 / torch.compile / 668
  steps × 524,288 tok = 350.2M tokens.
- Same nested-Bernoulli slot mask (MASK_SEED 90210, dose 0.35) over the
  same rebuilt streams — determinism verified: FIM stream 245,416,745 tok /
  239,430 blocks, causal 238,380,813 tok / 232,566 blocks, eval 408 blocks /
  1,282,658 bytes, 223 served-eval rows — all byte-equal to the banked
  2026-08-22 metadata. The token stream and slot order are IDENTICAL to
  ladder_fim35's; only the loss discipline differs.
- Masked loss: CE restricted to the trailing target tokens (span +
  `\n<|end|>`) of each FIM doc; causal docs keep full loss; the
  doc-separator EOS carries no loss in FIM docs. Mask built by token-suffix
  alignment (`data_prep_mask.py`): 276,206/276,206 docs exact-aligned, 0
  fallbacks; 29,706,996 masked tokens = 12.1% of FIM-stream tokens.
- Readouts = the banked instruments unchanged: causal-floor BPB + PSM-slice
  BPB + TF stop-accuracy (`bpb_eval.py`, appended to `logs/bpb_eval.json`)
  and the probe2-style served eval (`convert_gguf.py` → llama-server CUDA
  b10453, port 18107, --parallel 4 → `eval_fim_served.py`, same 223 rows).

## Results (run closed 2026-09-05T19:33; train 96 min, 57-65k tok/s, zero yields, zero NaN)

| arm (350.2M tok) | causal BPB (floor) | PSM-slice BPB* | stop-acc (TF) | served line-F1 | served stop-rate | served exact |
|---|---|---|---|---|---|---|
| ladder_fim0 (banked control) | 0.7621 | 0.8458 | 3.6% | 0.0027 | 18.8% (incidental) | 0.0 |
| ladder_fim20 (banked, unmasked) | 0.7590 | 0.7446 | 14.8% | 0.0039 | 8.1% | 0.0 |
| ladder_fim35 (banked, UNMASKED full loss) | 0.7561 | 0.7409 | 15.3% | 0.0005 | 0.0% (COLLAPSE) | 0.0 |
| **ladder_fim35m (this run, MASKED span+end loss)** | **0.7793** | **0.8785*** | **15.7%** | **0.0019** | **4.9%** | 0.0 |

*The banked PSM-slice instrument is FULL-CE over the whole PSM doc
(context+suffix+span). A masked arm trains no loss on FIM context/suffix
tokens, so this column is structurally unfair to it by construction —
reported for completeness, not compared as a quality readout. The
span-behavior readouts (line-F1, stop-acc, stop-rate) are the pre-registered
gate metrics.

Served-eval per-example signature (223 rows, greedy, stop `<|end|>`, cap 384):
masked arm stops 11/223 (4.9%), median completion still at the 384 cap, 6
rows with line-F1 > 0 (best 1 row > 0.1). Banked fim35: 0 stops, 0.0005.

## Verdict (2026-09-05T19:35, zcode-gpushorts)

**The pre-registered gate PASSES: masked/unmasked line-F1 = 0.0019/0.0005 =
3.8x ≥ 2x. The adopted 20-35% FIM-dose verdict SURVIVES its falsifier** —
masked loss, the enabling condition the 2026-08-22 verdict leaned on without
testing, is now evidence-backed at POC scale:

1. **The 35% free-running collapse is a full-loss artifact.** Under masked
   loss at the SAME dose, stopping returns (0.0% → 4.9% of generations stop;
   the dose-10/20 class sits at 8.1%) and served line-F1 improves 3.8x.
   "Masked loss … no non-span free-running is trained in FIM docs" — the
   night-ladder's mechanism claim — is directly confirmed.
2. **Stop supervision is concentrated, not diluted**: TF stop-accuracy
   15.7% ≈ family best (15.3-16.1%) while only 12.1% of FIM-doc tokens
   carry loss.
3. **NEW measured cost (first masked-arm datapoint)**: causal-floor BPB
   0.7793 = **+3.07% vs the unmasked twin** (0.7561) and **+2.25% vs the 0%
   control** — outside the dose design's ≤1% band. Partly an
   effective-loss-budget artifact: at matched data budget a masked step
   carries ~31% fewer gradient tokens (all causal tokens + 12.1% of FIM
   tokens) — the unmasked twin effectively gets extra causal-like
   supervision from FIM context/suffix positions. Whether the floor cost
   persists at the 0.5BT A2-prime scale (where stop density, not token
   efficiency, dominates) is precisely what the design-§5.3 masked arms
   measure; the POC datapoint says: watch the causal floor there.
4. **Caveats carried**: POC absolute line-F1/exact advisory (both arms
   floored; masked@35's 0.0019 is still below the banked dose-20 peak
   0.0039); exact = 0.0 across all arms at this scale; n=223 rows.

Net for A2-prime: the adopted recipe (20-35% band, center 25-30%, masked
loss + span policy + stop density) stands — now with its masked-loss leg
measured rather than assumed — and the causal-floor cost joins stop-accuracy
as the two readouts the 0.5BT ladder must clear.

## Files

- `data_prep_mask.py` (mask builder, 276,206/276,206 exact token-suffix
  aligned), `train_ladder.py --loss-mask` (masked CE, same compile path)
- `run_fimreplica.sh` (this arm's chain: train → convert → serve 18107 →
  served eval → bpb_eval), `extract_tokenizer.py` (tokenizer JSON rebuild;
  merges pair-format fix)
- `logs/ladder_fim35m.jsonl` + `logs/ladder_fim35m_stdout.log` (telemetry;
  slot counts token-identical to banked fim35), `logs/bpb_eval.json` (tag
  `ladder_fim35m`), `logs/fim_eval_ladder_fim35m.jsonl` (223 per-example
  rows), `logs/fim_eval_summary.json`, `logs/ladder_fim35m_{convert,server,
  fimeval,bpb}.log`
- ckpt persisted: `/mnt/h/sepalith/runs/ladder_fim35m/final.pt` (1.64GB)
- Bug fixed en route: `bpb_eval.py`'s dose-table printer crashed on
  non-numeric tag suffixes ('35m'); tokenizer re-extract needed a merges
  pair→"a b" format fix (tokenizers>=0.21 serialization).
