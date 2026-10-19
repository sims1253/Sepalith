# llama.cpp upgrade evaluation — master (f3fb2fc) vs pinned b10453

**Question:** should Sepalith move its pinned llama.cpp (tag b10453, commit
`3cb7ffb1a`, 2026-08-16) to current master (`f3fb2fc`, 2026-10-19 =
tag b11727 + 7 commits, 1281 commits after b10453)? Which pins (product
serving bundle vs campaign quant/eval toolchain), and what must be re-run?

Everything is CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 10`, 8 threads),
paired b10453/master in the same session, TRAIN data only. Raw outputs:
`/mnt/e/sepalith/campaign-20260915/serving-work/llama-upgrade-20261019/`
(`runs/`, `quant/`, `logs/`, `scripts/`). Machine-readable numbers:
`2026-10-19-llama-upgrade-eval.json`.

## Headline

| # | Question | b10453 | master f3fb2fc | Verdict |
|---|---|---|---|---|
| 1a | Spec-dec near-tie flips (ngram-simple) | n1m1 5/60, n2m4 8/60 flip | identical counts (PR #73); `parity_repro.py` reproduces the same "ling" −0.6881 / "led" −0.7010 flip with bit-identical logprobs | **Not fixed upstream** |
| 1b | No-spec greedy tokens, master vs b10453 (TRAIN, 60-trace subset, cap 256) | — | **60/60 token-identical** (warm-repeat 1.000 on both) | **No behaviour change** |
| 2 | Client abort cancels in-flight prefill? | yes, but only at the next `n_batch` boundary (2048 tokens by default) | identical | **Unchanged**; fix is `-b` (both builds) |
| 3 | `--tensor-type` partial ffn_down override shifts the q6_K mix | shifted | shifted; outputs **byte-identical** to b10453 | **Hazard persists**; quantize output unchanged |
| 4a | llama-bench pp512/pp2048, 8T | — | Q8_0 ≈1.00×; Q4_K_M **1.07–1.08×** (3/3 rounds) | small Q4 prefill win |
| 4b | Decode, 8T | — | llama-bench tg32: Q8_0 noisy (median 1.085, 0.98–1.43), Q4_K_M 0.99×; server Q8_0 paired decode 0.998× | **no decode change** |
| 4c | Warm e2e no-spec (server, 60 traces) | — | paired per-trace decode ratio **0.998×** (CI95 0.964–1.019); TTFT 81 vs 81 ms | **equal** |
| 4d | Spec-dec champion e2e (n2m4) | 1.40–1.44× (60-sub) | 1.548× (PR #73) | equal or slightly better |
| 5 | Server flags / API used by harnesses + extension | — | all present; `/v1/completions`, `/completion` response shapes unchanged | **No breakage** |

**Recommendation in one line:** no forcing reason to bump either pin now.
The two defects that motivated this evaluation (verify-batch determinism,
prefill abort) are *not* fixed on master, quantize output is byte-identical,
and the measurable speed delta is a ~8 % Q4_K_M prefill gain. Keep the
campaign toolchain on b10453 for the rest of the campaign; bump the product
serving bundle opportunistically to a tagged release ≥ the evaluated commit
when a release cycle needs it (details and re-run list in §7).

## 0. Builds

| | b10453 (pinned) | master |
|---|---|---|
| commit | `3cb7ffb1a` (2026-08-16) | `f3fb2fc` (2026-10-19), b11727+7 |
| llama-server | `~/.local/opt/llama.cpp-b10453/build-cpu` (sha256 `984253bd…`) | `…/specdec-20261017/llama-master/build-cpu` (sha256 `117f915d…`, provided build, used for all server runs) |
| bench/quantize/perplexity/imatrix | same install | new build of the same commit: `…/llama-upgrade-20261019/build-cpu` (rsync of the master source, `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF`, `-j 8`, nice 10) |

Neither install was modified. The 7 commits after b11727 include one CPU
numerics change (`42b70032b ggml-cpu: vectorize ggml_v_silu and
ggml_v_expf`), so results here hold for `f3fb2fc`, not automatically for
b11727; pin a tag that **contains** `f3fb2fc` (none existed at measurement
time) or re-run §1b/§4 on the tag actually pinned.

Host: Ryzen 9 5900X 12C/24T, WSL2, shared — the specdec loop
(`rzcode-specdec`, 8T) and a warm-pass proof server (8T) ran concurrently;
recorded 1-min load 7–24. All speed numbers are paired/interleaved ratios.
One incident: around 12:25–12:32 someone SIGTERMed my servers (and the
specdec loop's); affected runs were discarded and re-run (board 12:34).

## 1. Determinism

### 1a. Spec-dec verify-batch flips (not re-measured — PR #73)

PR #73 (merged) measured the same master build: flip counts identical to
b10453 (n1m1 5/60, n2m4 8/60), acceptance counters bit-identical, n-gram
draft code unchanged. Before stopping my own redundant run,
`parity_repro.py --server-bin <master>` reproduced the canonical cases on
master with **the same logprobs to 1e-7** as the b10453 receipt:
`0d318a5c…` diverges @61, baseline top-2 "ling" −0.6880879 / "led"
−0.7009912; `37c1d27e…` diverges @1, " co" −0.9155914 / " c" −0.9192249.
The verify-batch (k+1 rows) GEMM path still differs from the single-row
path; an upgrade does not unlock the 100 % identity gate.

### 1b. No-spec master vs b10453 (behaviour change vs pinned receipts)

`experiments/serving/specdec/cross_build.py --no-spec`: b10453 and master
llama-server (Q8_0 arm A, 8T, `--parallel 1`, `-c 16384`) side by side,
arm order rotated per trace, the specdec warm flow per request (warm-up
`n_predict 0`, then timed greedy request with `return_tokens`; plus an
identical repeat as the determinism control). The deterministic 60-trace
family-spread TRAIN subset of the specdec mission (`traces.jsonl`, sha256
`bad2ab2e…`, ids = `manifest.json`), cap 256; load 9–16.

| comparison | identical | parity |
|---|---|---|
| master vs b10453, same request, same session | 60/60 | **1.000** |
| b10453 warm-repeat | 60/60 | 1.000 |
| master warm-repeat | 60/60 | 1.000 |
| b10453 vs pinned receipt `reference_outputs.json` (cold, cap 1024, prefix-aware) | 52/60 | 0.867 |
| master vs pinned receipt | 52/60 | 0.867 (same 8 traces) |

The receipt mismatch is the known warm-vs-cold KV drift (specdec report
§5.1, 0.851 on 221) — identical on both builds, i.e. not a build effect.
Two traces of an abandoned full-set cap-1024 pass were also identical at
1024 tokens. **For the arm-A Q8_0 CPU path, master reproduces b10453's
greedy outputs exactly**: the 1281 intervening commits (incl. the silu/expf
vectorisation) did not move any argmax on this subset. Caveat: 60 traces at
cap 256 bounds a hidden flip rate only loosely (0/60 → ≤ ~5 % at 95 %);
Q4_K_M/Q6_K tiers and 4 threads were not checked.

## 2. Prefill abort

Probe: `experiments/serving/prefill/probe_abort_cancel.py` (new), the
largest frozen TRAIN prefill session event (2438 tokens), Q8_0, 8T,
`--parallel 1`, fresh server per scenario, client socket closed 3 s
(nonstream / SSE stream) after sending an `n_predict 0` warm-up; evidence is
the server log (`cancel task`, and the warm-up's `stop processing: n_tokens`
at release). `probe_interrupt.py` (prefill mission) re-run on both builds
agrees.

| build | n_batch | cancel posted | tokens prefilled at release | release after abort | follow-up cache_n / processed |
|---|---|---|---|---|---|
| b10453 | 2048 (default) | immediately | 2048 / 2438 | 28.8–29.5 s | 2048 / 390 |
| master | 2048 (default) | immediately | 2048 / 2438 | 27.2–27.3 s | 2048 / 390 |
| b10453 | 256 | immediately | 256 / 2438 | 0.9 s | 256 / 2182 |
| master | 256 | immediately | 256 / 2438 | 0.6–0.9 s | 256 / 2182 |

(Uninterrupted warm-up of the same prompt: 36–39 s under load ~10–24, both
builds, both batch sizes.)

Findings:

- **Behaviour is identical on both builds.** A client disconnect *is*
  detected (the server posts a cancel within its 1 s poll), but the cancel
  only takes effect between `llama_decode` calls, i.e. at the next
  `n_batch` boundary. With the default `n_batch 2048` any prompt ≤ 2048
  tokens therefore runs its whole prefill; for longer prompts it stops at
  2048. The prefill report's "abort does not cancel" is right in effect for
  PRM03-sized prompts, but the mechanism is batch granularity, not
  missing cancellation. Partial KV is kept (follow-up `cache_n` = released
  tokens).
- **Lever (works on b10453 already):** `-b 256 -ub 256` makes an abort free
  the slot in < 1 s. Uninterrupted prefill wall was unchanged within noise
  (37.7 vs 36.3 s master, 36.5 vs 39.1 s b10453) — not a controlled
  throughput measurement; a smaller `-b` with the default `-ub 512`
  (`-b 512`) should keep the compute path identical and is the safer
  candidate. Worth a prefill-mission follow-up if aborting warm-ups is
  wanted (it is not in the current PR #66 flow).
- Side finding: polling `GET /slots` while a request is in flight starves
  the disconnect check (each /slots result wakes the handler's 1 s wait, so
  it never times out and never checks `should_stop`); with 50 ms polling no
  cancel was ever posted on either build (archived under
  `runs/abort/polling-artifact/`). Clients that abort must not poll the
  server concurrently.

## 3. llama-quantize `--tensor-type` counter shift

Re-ran the exact receipt commands (`arm-a-imat-Q4_K_M` and the night-run
`arm-a-q4plus-Q4_K_M`: `--output-tensor-type q8_0 --token-embedding-type q8_0
--tensor-type attn_k\.weight=q8_0 --tensor-type attn_v\.weight=q8_0
--tensor-type blk\.(0|1|40|41)\.ffn_down\.weight=q6_K --imatrix …`) with both
quantizers; ffn_down types read with gguf-py `GGUFReader`
(`experiments/serving/llama_upgrade/ffn_down_mix.py`).

| file | ffn_down layers 0→41 (6 = q6_K, 4 = q4_K) | matches intent |
|---|---|---|
| plain imat Q4_K_M (both builds) | `666664464464464464464464464464464464666666` | — |
| intended q4plus (plain + 0,1,40,41 → q6_K) | `666664464464464464464464464464464464666666` | — |
| q4plus, b10453 | `666666644644644644644644644644644644646666` | no (22 layers differ) |
| q4plus, **master** | `666666644644644644644644644644644644646666` | **no (same 22 layers)** |
| pinned `arm-a-q4plus-Q4_K_M.gguf` | same shifted string | no |

All four outputs are **byte-identical across builds** and to the campaign
files (sha256 `19bf6c67…` plain = `arm-a-imat-Q4_K_M.gguf`, `4cb7e4f8…`
q4plus = `arm-a-q4plus-Q4_K_M.gguf`). So: the counter-shift hazard persists
on master (keep the "override a category fully or not at all" rule), and the
quantizer is not a source of drift for an upgrade — existing GGUFs need not
be regenerated. Note the intended override is a no-op (Q4_K_M already puts
q6_K on layers 0,1,40,41); it still shifts the every-3rd mix.

## 4. Speed

### 4a/b. llama-bench (8 threads, defaults n_batch 2048 / n_ubatch 512 / fa auto on both)

Three interleaved rounds (A/B order alternating), `-r 3` each.
Per-round paired ratio master/b10453:

| model | test | b10453 t/s (med) | master t/s (med) | ratio per round | read |
|---|---|---:|---:|---|---|
| Q8_0 | pp512 | 91.4 | 91.0 | 0.996, 1.014, 1.000 | equal |
| Q8_0 | pp2048 | 83.8 | 82.0 | 1.008, 1.029, (0.583*) | equal |
| Q8_0 | tg32 | 14.2 | 12.0 | 1.146, 0.845, (0.545*); rerun median 1.085 (0.98–1.43) | inconclusive → see §4c |
| imat-Q4_K_M | pp512 | 114.9 | 122.4 | 1.065, 1.084, 1.150 | **+7–8 %** |
| imat-Q4_K_M | pp2048 | 103.8 | 111.4 | 1.073, 1.080, 1.113 | **+7–8 %** |
| imat-Q4_K_M | tg32 | 22.4 | 21.1 | 0.944, 1.000, 0.992 | equal |

\* round 3 Q8_0: b10453 ran in a quiet window (load 8→7), master as other
missions' load returned (7→14); discarded. Q8 decode on this shared box is
contention-dominated (same caveat as the quant report's withdrawn tg32
claim). A dedicated Q8_0 tg32 rerun (6 interleaved rounds, `-r 5`,
load 8–14) gave per-round ratios 1.126, 1.314, 1.425, 1.044, 0.982, 0.988
(median 1.085): still contention-dominated. The decisive decode number is
the server-side paired per-trace comparison in §4c (identical tokens,
0.998×) — **no Q8_0 decode change**.

### 4c. Server warm e2e (no-spec)

Same run as §1b (warm flow, 60 traces, cap 256). Because the two builds
emit identical tokens, each trace's decode-time ratio is a clean paired
speed ratio:

| metric | b10453 | master | ratio |
|---|---:|---:|---|
| paired per-trace decode speed (median of 60 ratios) | — | — | **0.998×** (CI95 0.964–1.019; master faster on 29/60) |
| paired per-trace warm e2e (median of ratios) | — | — | 0.998× |
| decode tok/s (median) | 13.42 | 13.06 | — |
| TTFT warm (median) | 81.0 ms | 81.0 ms | 1.00× |
| warm e2e pooled (Σt) | — | — | 0.971× |

(The unpaired median-of-medians, 1828 vs 2127 ms, is a load artifact: the
per-trace pairing shows no difference.) Server-side decode speed for Q8_0 at
8T is unchanged.

### 4d. Spec-dec champion

Not re-measured (PR #73: n2m4 on master 1.548× median, CI95 lo 1.32,
60-trace subset at load 17 vs 1.40–1.44× on b10453; acceptance identical).

## 5. Flag and API compatibility

- `llama-server --help` diff (alias-aware): **removed** `--mmap`,
  `--no-mmap`, `--mlock`, `-dio/--direct-io`, `-ndio/--no-direct-io` (all
  already marked DEPRECATED in b10453 in favour of `--load-mode`); also
  removed from llama-perplexity / llama-imatrix / llama-completion;
  llama-bench lost `-mmp/--mmap` and `-dio` (use `-lm/--load-mode`).
  **Added** (server): `--kv-unified-per-slot`, `--lazy-mode/-lzm`,
  `--n-cpu-ffn`, `--moe-cache-mib`, `--rpc`, `--spec-draft-n-ubatch`,
  `--spec-draft-sampling`, `--spec-synth-*`, `--log-jsonl`, video/mmproj
  flags. llama-quantize: only `--max-buffer-size` added.
- Every flag spelled in `experiments/`, `scripts/`, `packages/`,
  `extensions/` next to a llama-server / quant-tool invocation exists in
  master (scripted scan; converter flags `--outfile/--outtype/--no-nextn`
  also still exist in `convert_hf_to_gguf.py`). The only hit for a removed
  flag is a historical receipt
  (`docs/training/plan-20260927/receipts/C09-prep-gguf-tooling.json`,
  llama-completion `--no-mmap`) — re-running that recipe on master needs
  `--load-mode none`.
- VS Code extension spawn args (`-m --alias --temp --port --host -c
  --parallel -t -ngl`) all exist. Spawned verbatim on both builds:
  `/health` → `{"status":"ok"}`; `/v1/completions` with the extension's body
  (prompt, max_tokens, temperature 0, stop list, stream false) returns the
  same keys, `choices[0].text` and `usage.completion_tokens` (identical
  text on both builds). Harness `/completion` (token-id prompt,
  `return_tokens`, `n_probs`, `samplers`) returns identical keys, tokens,
  `timings` fields (incl. `cache_n`) and `completion_probabilities[].top_logprobs`.
- Defaults that differ and matter: none found for our flags (n_batch,
  n_ubatch, fa auto, f16 KV, `--slots` enabled, `--cache-ram 8192` unchanged).
  Master logs a new warning ("chat template supports preserving reasoning")
  — harmless for raw `/completion`.

## 6. What an upgrade would and would not buy

Would buy: ~8 % Q4_K_M prefill on 8T CPU (relevant for the laptop Q4 tier's
cold path), PR #73's possibly-slightly-better spec-dec wall, upstream
fixes we do not use (MTP/drafter models, GPU backends, MoE, UI).
Would not buy: spec-dec bitwise parity, abortable prefill (needs `-b`, works
today), a quantizer fix, a faster Q8_0.

## 7. Recommendation

1. **Campaign quant/eval toolchain (`build_llama_cpp.sh`, export_gguf,
   llama-perplexity/imatrix/quantize, eval llama-server): stay on b10453
   until the campaign closes.** Quantize is byte-identical, so nothing is
   gained on the artifact side, while every eval receipt (DEV250 gate,
   KLD, token-identity, specdec/prefill latency) is pinned to b10453
   numerics and speed; a mid-campaign bump breaks receipt comparability for
   no measured benefit.
2. **Product serving bundle (`bundle_runtime.py` PIN, manifest `build`,
   runtime-bundles workflow, extension smoke/check-runtime fixtures):
   optional, not urgent.** API and spawn args are compatible, so a bump is
   low-risk mechanically. Do it only as a deliberate release step, to a
   tagged build ≥ `f3fb2fc` (re-run §1b/§4 if the tag differs), and only
   together with the re-runs below. If the Q4_K_M laptop tier ships, the
   ~8 % prefill gain is the one concrete reason to prefer it.
3. **Independent of the upgrade:** adopt the `n_batch` lever if warm-up
   aborts are ever wanted (§2), and keep the full-category `--tensor-type`
   rule (§3).

**If the serving bundle is bumped, re-run:**
- the no-spec cross-build identity on the full 221-trace TRAIN set at cap
  1024 (this report: 60-trace subset at cap 256), plus regenerate
  `experiments/serving/specdec/reference_outputs.json` /
  `experiments/serving/prefill/reference_outputs*.json` on the new build;
- the C09 quality gate (exact / no-op FP / McNemar) on DEV250 for the
  shipping tier on the new build (one recorded DEV use) — greedy tokens are
  the product contract;
- the prefill flow bench (`bench.py`, warm-up-full-eager) and the specdec
  champion bench (`bench_spec.py --server-bin`) at 8T and 4T, paired;
- extension smoke (`scripts/smoke.ts`), `check-runtime.ts` fixtures and the
  runtime bundle tests with the new `build` string; CUDA/Vulkan bundle
  builds (not measured here: CPU only).

**If the campaign toolchain is ever bumped (after the campaign):** in
addition, re-baseline KLD (`kld/bf16-base-cpu.kld`) and every tier's gate
receipt on the new build; GGUF files themselves can be kept (quantize
output byte-identical for the recipes tested, re-verify sha256 on the new
build for any new recipe).

## 8. Not measured / limits

- Spec-dec parity and champion e2e on master: taken from PR #73, not
  re-measured (coordinator instruction).
- Full 221-trace × cap-1024 cross-build identity: started, ~3 min/trace on
  the shared host (≈11 h) — abandoned after 2 traces (both identical at
  1024 tokens) in favour of the 60-trace cap-256 subset.
- llama-bench tg32 stayed contention-noisy on the shared host; the decode
  verdict rests on the server-side paired comparison (§4c). Q4_K_M server
  e2e and the 4-thread band were not run.
- 4-thread band, GPU/CUDA/Vulkan builds, llama-perplexity KLD numerics
  across builds, and imatrix generation: not measured (CPU-only brief;
  quantize identity used as the toolchain proxy).
- b11727 (latest tag at measurement time) was not built; it lacks the
  `ggml-cpu` silu/expf vectorisation commit included in `f3fb2fc`.

## Harness changes in this PR

- `experiments/serving/specdec/cross_build.py` — new: paired two-build ×
  {no-spec, spec} bench (or `--no-spec`), rotation-interleaved arms,
  cross-build parity matrix, reference (prefix-aware) parity, paired speed
  ratios with bootstrap CIs.
- `experiments/serving/specdec/parity_repro.py` — `--server-bin`,
  `--out-root`, `--port-base`, `--max-cases` (defaults unchanged).
- `experiments/serving/prefill/probe_abort_cancel.py` — new: server-log
  based abort/cancel probe with `--batch` variants.
- `experiments/serving/llama_upgrade/ffn_down_mix.py`,
  `summarize_llama_bench.py` — GGUFReader mix check and interleaved
  llama-bench aggregation.
