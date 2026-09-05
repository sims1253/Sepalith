# S1 rig runbook — spec-decode wall-clock on the S0 frozen traces

Queue row: EXPERIMENT-QUEUE.md §3 S-series S1 (verdict pre-registered
there). Trace set: `/mnt/h/sepalith/datasets/spec_traces/` (MANIFEST.md —
its "For S1's rig" notes are binding and baked into the rig: `-c 10240`,
stop `>>>>>>> UPDATED`, `max_tokens 64`, Qwen-family counts).

Rig: `experiments/eval/spec_bench.py` (+ `test_spec_bench.py`, 25 tests).
Patterns: `cache_bench.py` / `keystroke_sim.py` / `harness_search/server.py`
(tracked-PID CPU llama-server, readiness = real POST, ports 18xxx).

## What exists (built 2026-09-05)

| artifact | what |
|---|---|
| `experiments/eval/spec_bench.py` | the bench: trace loader, 4 single-mode arms, depth sweeps, streaming TTFT, acceptance math, summaries |
| `experiments/eval/test_spec_bench.py` | 25 pytest cases (loader invariants, acceptance semantics, TTFT, arm flags) |
| `experiments/eval/results_specbench/<run_id>/` | per_request.jsonl + summary.json per run |
| `experiments/models/mtp-b4_qwen35_2b-Q8_0.gguf` | b4 + upstream MTP head, nextn embedded (2.08 GB) — the draft-mtp serve artifact (W13) |
| `experiments/models/mtp-b4_qwen35_2b-head-Q8_0.gguf` | MTP-head-only draft file (616 MB) — the two-file `-md` variant |
| `experiments/models/qwen3.5-2b-mtp-head.safetensors` | cached upstream MTP tensors (121.7 MB, 15 tensors) |
| `experiments/training/export_gguf_mtp.py` | the MTP-preserving export path (fetch/graft/convert/quant/verify) |

## The MTP-preserving export (how the mtp GGUFs were made)

`export_gguf.py` passes `--no-nextn` because the LOCAL base
(`models/qwen3.5-2b-base-text-hf`, the 2026-08-20 text-only strip) has no
`mtp.*` tensors. The UPSTREAM `Qwen/Qwen3.5-2B-Base` ships them (15
tensors, bf16, contiguous 121.7 MB at the safetensors tail — header
range-verified). Export path:

1. **merge** b4 LoRA (`/mnt/h/sepalith/runs/b4_qwen35_2b/final_lora`) onto
   the base via the export_gguf.py flow. unsloth REFUSES CPU-only
   ("cannot find any torch accelerator"), so the PEFT branch
   (MERGE_VIA_PEFT equivalent) is the working path on this box
   (`AutoModelForCausalLM` + `PeftModel.merge_and_unload`, `.venv-sft`).
2. **fetch** the 15 upstream `mtp.*` tensors by HTTP range request
   (cached at `models/qwen3.5-2b-mtp-head.safetensors`).
3. **graft** them into the merged checkpoint (single safetensors +
   `mtp_num_hidden_layers: 1` config).
4. **convert** with the b10453 converter (`/tmp/llamacpp-convert-b10453`,
   clone of tag b10453) WITHOUT `--no-nextn` → nextn tensors embedded;
   second run with `--mtp` → head-only file.
5. **quantize** Q8_0 with `bin/llama/llama-b10453/llama-quantize`.
6. **verify** (automated in the script): tensor names + KVs.

Verified evidence: `mtp-b4_qwen35_2b-Q8_0.gguf` has `block_count = 25`
(24 body + 1 MTP), `qwen35.nextn_predict_layers = 1`, and nextn tensors
`blk.24.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight` (+ the MTP
layer's attn/mlp as regular blk.24.* tensors) — survived quantize.
**Serve-verified on the CPU b10453 llama-server, both modes**:
(a) embedded: `-m mtp-b4_qwen35_2b-Q8_0.gguf --spec-type draft-mtp`
(no `-md`); (b) two-file: `-m b4_qwen35_2b-Q8_0.gguf -md
mtp-b4_qwen35_2b-head-Q8_0.gguf --spec-type draft-mtp`. Both draft and
verify (per-request `draft_n` / `draft_n_accepted` in the final stream
chunk).

HONEST CAVEAT: the grafted MTP head is the UPSTREAM BASE's head — the b4
LoRA never trained it (it wasn't in the training base). draft-mtp
acceptance measured on this artifact is a base-MTP-head-on-b4-body
datapoint, a LOWER bound for a b4-trained MTP head (A2's freeze decision
gets the mechanism + a conservative number). Smoke measured it: wired
(90 drafts/trace produced and verified) but **0 accepted** on the probe
traces — the base head's t+2 predictions diverge from the b4 body's
greedy path. The full legs quantify; if it stays ~0, the arm reports
mechanism-works/acceptance-dead and the verdict leans on
ngram/model-draft.

## Arms (single-mode only — A2 §4.5, no stacked claims)

| arm | serve flags (beyond the convention `-t 8 --parallel 1 -c 10240 -ngl 0`) | depth knob |
|---|---|---|
| `baseline` | — | — |
| `ngram-simple` | `--spec-type ngram-simple` | `--spec-ngram-simple-size-m M` (drafted tokens/step; default 48) |
| `draft-mtp` | model = `mtp-b4_qwen35_2b-Q8_0.gguf`, `--spec-type draft-mtp` | `--spec-draft-n-max N` |
| `model-draft` | `--spec-type draft-simple --spec-draft-model b2_qwen35_08b-Q8_0.gguf` | `--spec-draft-n-max N` |

`model-draft` note: the pre-registered arm is "Matryoshka S-tier
`--model-draft` (zero-PR fallback)". A2's 8L-prefix tier doesn't exist
yet; the rig uses the same-tokenizer-family 0.8b (b2) as the stand-in
draft. Swap `MODEL_DRAFT` in spec_bench.py when the real S-tier lands.
Depth grammar: `arm@depth` in `--arms` (e.g. `draft-mtp@2`).

## Requests (all arms identical)

Native `/completion`, `stream: true`, `temperature: 0`, `n_predict: 64`,
`stop: [">>>>>>> UPDATED"]`, `cache_prompt: false` (cold = the
fresh-edit scenario) + one warm pass (`cache_prompt: true`, the
keystroke/cache-hit case; H1 tie-in) per trace.

## Metrics

- `ttft_ms` — request-send → first CONTENT chunk (true streamed TTFT,
  includes prefill). `warm_ttft_ms` for the cache-hit pass.
- `gen_tps` — `predicted_n / predicted_ms` (decode-phase wall tok/s from
  server timings).
- `accept_tps` — tok/step = `predicted_n / (predicted_n -
  draft_n_accepted)` from the final chunk (== the server's logged
  "mean len"); `accept_rate` = `draft_n_accepted / draft_n`. `(None,
  None)` when nothing was drafted — baseline is NOT 1.0-by-default.
- `matches_baseline` — greedy text identical to the baseline arm's
  output on the same trace (spec is distribution-lossless; mismatches
  flag wiring trouble, and the lossless check doubles as the
  crash-under-churn guard).
- Per-request rows in `per_request.jsonl`; medians + speedup-vs-baseline
  per (arm, ctx_class) in `summary.json`.

## Run

Smoke (1 paired trace/class, baseline + ngram, 1 rep; ~10 min on a quiet
box):

    cd /home/m0hawk/Documents/Sepalith
    .venv/bin/python experiments/eval/spec_bench.py --smoke --port 18401

Full CPU legs (pre-registered: 3 reps, both ctx classes, all arms +
depth curves; expect n≈2 optimum per the Unsloth receipt). RUN IN A
QUIET WINDOW ONLY (see board; latency numbers need an uncontended box —
H1's two servers + any trainer invalidate wall-clock):

    .venv/bin/python -u experiments/eval/spec_bench.py --port 18401 \
      --ctx-classes 2k,8k --n-traces 100 --reps 3 \
      --arms baseline,ngram-simple@2,ngram-simple@3,ngram-simple@4,ngram-simple@8,ngram-simple@16,ngram-simple@48,draft-mtp@1,draft-mtp@2,draft-mtp@3,draft-mtp@4,draft-mtp@5,model-draft@1,model-draft@2,model-draft@3,model-draft@4,model-draft@5 \
      > /tmp/s1_spec/full_legs.log 2>&1

- `--n-traces` is a deterministic (seed 20260905) per-class sample;
  `--n-traces 550` = the full set. 100/class × 17 arm-configs × 3 reps ×
  2 classes ≈ 10k requests; on a quiet box (8k prefill ~2-4 s, 36-60 tok
  decode) budget roughly 15-24 h — raise/reduce `--n-traces` to fit the
  window, or split: run arms in 2-3 invocations (each writes its own
  run dir; concatenate summaries).
- 5090-offload legs: swap SERVER for the CUDA b10453 build + `-ngl 99`
  (+ `-ngld 99` for the draft) — IDLE-CARD windows only (W37: never
  beside a trainer). The CUDA build must be rebuilt (the /tmp tree was
  wiped 2026-09-05).

## Smoke results (2026-09-05, CONTENDED box load 17-25 — wiring proof only)

`results_specbench/smoke-20260905T030823` (baseline + ngram, 1 paired
trace/class) + `results_specbench/run-20260905T032607-n1-r1`
(draft-mtp@2 + model-draft@2, one 2k trace):

- rig mechanics: server prompt count == stored trace count (2240 == 2240,
  Qwen token parity); stop marker hit on every request; 0 errors;
  baseline draft_n = null (acceptance correctly None, not 1.0);
  warm cache-hit pass works (8k cold prompt 171.6s -> warm 3.9s).
- ngram-simple (default m=48): **accept_rate 0.556, 2.47 tok/step**,
  matches_baseline 1/1 (lossless check passes), wall speedup 1.62-1.79x
  EVEN under load-25 contention (n=1 — indicative only).
- draft-mtp@2: wired (90 drafts produced+verified) but **0 accepted** —
  the base-MTP-head-on-b4-body caveat realized. Costs ~2x decode time at
  0% acceptance (0.27 vs 0.47 tok/s) as expected for all-draft-no-payoff.
- model-draft@2 (b2-0.8b): **accept_rate 0.967 (29/30), 2.71 tok/step**
  — same-family same-data draft agrees with b4 nearly perfectly.
  gen_tps still below baseline under contention (draft+target share t8
  on a load-25 box) — the quiet window decides the real wall numbers.
  (matches_baseline is only computed in runs that include the baseline
  arm — the mini-leg didn't, so it's absent there, not 0.)

## Verdict mapping (pre-registered, EXPERIMENT-QUEUE §3 S1)

WINNER-SPEC iff a single-mode arm ≥1.4x wall tok/s (gen_tps) vs baseline
on either tier with no crash under suffix churn; nothing ≥1.3x → close
the spec line. Acceptance (tok/step) feeds W13 + the A2 MTP-head freeze
decision. Q7's fold: this rig's acceptance number IS Q7's answer on the
frozen set.

## Ports / protocol

18.4xx (S1 rig; H1 owns 1831x). Tracked-PID servers only; kill by PID
(the rig does). `/tmp/b_battery.lock` is held across server bind. Long
runs: setsid-detached + small watcher; heartbeats per comms.md.
