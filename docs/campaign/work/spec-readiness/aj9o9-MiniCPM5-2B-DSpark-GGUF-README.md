---
base_model: openbmb/MiniCPM5-2B-DSpark
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
license: apache-2.0
quantized_by: aj9o9
tags:
  - gguf
  - minicpm
  - minicpm5
  - dspark
  - speculative-decoding
  - llama.cpp
---

# MiniCPM5-2B-DSpark GGUF

GGUF of [openbmb/MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark)
for [llama.cpp](https://github.com/ggml-org/llama.cpp) speculative decoding.

This is the **draft** model only. Pair it with a MiniCPM5-2B target GGUF
(official [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
or your own convert). It will not run as a standalone chat model.

## A note from me

Same deal as my other uploads: I convert these to run on my own box, then share them.
OpenBMB shipped the DSpark weights as BF16 safetensors. There was no GGUF on the
official page, so I built these.

**Please report anything you find.** Bad accept length, crashes, wrong metadata,
a better flag for a particular card — open a discussion here or reach me at
[@ItsmeAjayKV](https://x.com/ItsmeAjayKV).

## Files

| File | Size | What it is |
|---|---|---|
| `MiniCPM5-2B-DSpark-F16.gguf` | ~623 MB | F16 draft. Closest to the BF16 source. |
| `MiniCPM5-2B-DSpark-Q8_0.gguf` | ~334 MB | Q8_0 of the same draft. Less VRAM. |

Tokenizer came from [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
via `--target-model-dir` during convert. Vocab size is 130560. Block size is 7.
Arch in the GGUF is `dflash` (DSpark = DFlash + Markov head).

## Download (CLI)

```bash
hf download hf://aj9o9/MiniCPM5-2B-DSpark-GGUF/MiniCPM5-2B-DSpark-F16.gguf
```

Or the smaller one:

```bash
hf download hf://aj9o9/MiniCPM5-2B-DSpark-GGUF/MiniCPM5-2B-DSpark-Q8_0.gguf
```

## How I run it

Needs a recent llama.cpp with `draft-dspark`. Pair with the target.
KV cache as Q8_0 (`-ctk` / `-ctv`) on my box:

```bash
llama-server \
  --model MiniCPM5-2B-F16.gguf \
  -md MiniCPM5-2B-DSpark-F16.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 \
  -ngl 999 -ngld 999 -fa on -np 1 -t 12 --jinja \
  -ctk q8_0 -ctv q8_0
```

Local `-m` / `-md` is the reliable path. Hub download flags vary by llama.cpp build.

Official sampling from the MiniCPM5-2B card: `temperature=1.0`, `top_p=0.95`.
For checking draft accept length, greedy / `t=0` is cleaner.

## Numbers I actually measured

Same box, same target (`MiniCPM5-2B-F16.gguf`), same llama.cpp, Q8_0 KV.
`llama-benchy`, `pp2048` + `tg256` at several depths. **Not MTP** — MiniCPM5-2B
has no MTP heads. Spec run is **DSpark** (`--spec-type draft-dspark`, `n-max 7`).

| | GPU VRAM |
|---|---|
| Baseline (no draft) | **8317 MB** |
| + DSpark F16 draft | **10.1 GB** |

### Baseline (no speculative)

| test | t/s | peak t/s | ttfr (ms) |
|---|---:|---:|---:|
| pp2048 @ d8192 | 8383.10 | | 1221.89 |
| tg256 @ d8192 | **109.54** | 111.00 | |
| pp2048 @ d16384 | 6825.90 | | 2700.69 |
| tg256 @ d16384 | **90.62** | 94.00 | |
| pp2048 @ d32766 | 5094.42 | | 6834.13 |
| tg256 @ d32766 | **70.18** | 71.00 | |
| pp2048 @ d65536 | 3377.05 | | 20013.13 |
| tg256 @ d65536 | **46.22** | 47.00 | |

### With DSpark (`n-max 7`)

| test | t/s | peak t/s | ttfr (ms) | TG vs baseline |
|---|---:|---:|---:|---:|
| pp2048 @ d8192 | 7751.67 | | 1321.36 | |
| tg256 @ d8192 | **181.78** | 198.00 | | **~1.66×** |
| pp2048 @ d16384 | 6483.51 | | 2843.11 | |
| tg256 @ d16384 | **134.47** | 155.00 | | **~1.48×** |
| pp2048 @ d32766 | 4887.29 | | 7123.73 | |
| tg256 @ d32766 | **85.03** | 93.00 | | **~1.21×** |
| pp2048 @ d65536 | 3255.84 | | 20758.13 | |
| tg256 @ d65536 | **69.96** | 84.00 | | **~1.51×** |

Read it like this:

- At **8k depth**, DSpark is the point — ~110 → ~182 tg t/s on a 3090.
- PP is a bit slower with the draft loaded (expected; more weight traffic).
- Speedup shrinks as depth grows, but TG still beats baseline out to 64k in this run.
- Extra VRAM is roughly the draft (~0.6G F16) plus overhead → ~8.3G → ~10.1G.

## A note on accept length

Official DSpark card reports aggregate accept length around **~5.5 at T=0** and
**~4.1 at T=1.0**, on in-distribution prompts.

The draft was trained with **max sequence length 12,288**. At long context
(32k / 64k+) I have seen accept length collapse toward ~1.6. That is not a
quant bug — you are outside what the draft saw. Keep `--depth` / context in
the training range if you care about the speedup. The TG table above still
shows a win at 64k on this box; accept length and wall-clock speedup are
related but not the same number.

## Convert recipe I used

```bash
# draft BF16 from OpenBMB + tokenizer from target MiniCPM5-2B
python convert_hf_to_gguf.py MiniCPM5-2B-DSpark \
  --target-model-dir MiniCPM5-2B \
  --outtype f16 \
  --outfile MiniCPM5-2B-DSpark-F16.gguf

llama-quantize MiniCPM5-2B-DSpark-F16.gguf MiniCPM5-2B-DSpark-Q8_0.gguf Q8_0
```

Draft `config.json` architecture is `Qwen3DSparkModel` even though the target
is `LlamaForCausalLM`. That is how OpenBMB shipped it; llama.cpp maps it.

## Hardware this was made on

| | |
|---|---|
| GPU | RTX 3090 24 GB |
| System RAM | 64 GB |
| Runtime | llama.cpp `master`, `--spec-type draft-dspark` |

## About the draft

- ~324M params, 5 layers, block size 7
- Target hidden layers used: `[1, 10, 20, 30, 39]` (GGUF stores extract indices as `[2, 11, 21, 31, 40]`)
- Trained for exact pairing with MiniCPM5-2B

See the original card for training details and their acceptance table:
[openbmb/MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark).

## Links

- Original DSpark (BF16): [openbmb/MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark)
- Target model: [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
- Target GGUF: [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
- Runtime: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (`docs/speculative.md`, `draft-dspark`)

## License

Apache 2.0, same as [openbmb/MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark).

## Disclaimer

Not affiliated with OpenBMB. Provided as-is.
Use the official cards for intended use, safety, and limitations.
