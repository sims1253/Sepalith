# REL-07 R2 quant recipe research

Status: read-only recipe research complete; root launch and promotion review pending.
Scope: TRAIN calibration only. This worker did not launch a conversion or inference
server, did not use CUDA, SSH, network, or paid services, and did not read model
tensor payloads.

## Inputs and identity

The selected input is the R2 model `SFT11-task-global-b-500` from the
MiniCPM5-2B-Midtrain -> CPT -> SFT11-task-global-b-500 path.

| Item | Value |
| --- | --- |
| F16 source | `/mnt/e/sepalith/campaign-20260915/intermediate-f16/SFT11-task-global-b-500-quant/model-F16.gguf` |
| F16 bytes / SHA-256 | `5039006496` / `fe1c38a2b53519fb15eeb4ac36efdd7b6f60ad5a58451475b51308a93cb8a8ee` |
| Native Q8 reference | `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf` |
| Q8 bytes / SHA-256 | `2679710496` / `d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db` |
| Model tensor inventory | 381 tensors in the matching Q8 integrity receipt: native Q8 has 296 `Q8_0`, 85 `F32` |
| Model shape | 42 layers, hidden size 2048, FFN size 6144, 16 attention heads, 2 KV heads, GQA 8 |
| Tied embeddings | false; `output.weight` and `token_embd.weight` are separate tensors |
| TRAIN imatrix | `/home/m0hawk/.local/state/sepalith/campaign-20260915/benchmarks/r2-imatrix-v1/imatrix.gguf` |
| Imatrix SHA-256 | `0c2f8dcbdbc55dd3f35953fc735b997190559c3ad91cfbd5a4d58f1a318607bf` |

The matrix acceptance receipt records 295 weight entries (42 layers x 7 linear
families plus `output.weight`), 51 chunks of size 2048, 3,150,208 bytes, and a
positive count of 104,448 for every entry. The matrix contains paired F32
`.in_sum2` and `.counts` tensors. The token embedding has no matrix entry, which
is safe here because it is explicitly sent to `Q8_0` and is excluded from the
quantizer's required-imatrix check.

## Exact root launch arrays

These are the three proposed native calibrated Q8IO candidates. They use the
pinned installed CPU quantizer, direct F16 input, the complete TRAIN matrix,
explicit `Q8_0` output and embedding tensors, and two threads. The output names
include `-q8io` so they cannot overwrite or be confused with the already
completed default mixed-precision artifacts named `model-Q4_K_M-imatrix.gguf`,
`model-IQ3_M-imatrix.gguf`, and `model-IQ2_M-imatrix.gguf`. Do not add
`--allow-requantize`:
the input is F16 and direct conversion preserves the intended calibration path.

```json
[
  "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-quantize",
  "--imatrix", "/home/m0hawk/.local/state/sepalith/campaign-20260915/benchmarks/r2-imatrix-v1/imatrix.gguf",
  "--output-tensor-type", "q8_0",
  "--token-embedding-type", "q8_0",
  "/mnt/e/sepalith/campaign-20260915/intermediate-f16/SFT11-task-global-b-500-quant/model-F16.gguf",
  "/mnt/e/sepalith/campaign-20260915/quant-candidates/SFT11-task-global-b-500/model-Q4_K_M-imatrix-q8io.gguf",
  "Q4_K_M", "2"
]
```

```json
[
  "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-quantize",
  "--imatrix", "/home/m0hawk/.local/state/sepalith/campaign-20260915/benchmarks/r2-imatrix-v1/imatrix.gguf",
  "--output-tensor-type", "q8_0",
  "--token-embedding-type", "q8_0",
  "/mnt/e/sepalith/campaign-20260915/intermediate-f16/SFT11-task-global-b-500-quant/model-F16.gguf",
  "/mnt/e/sepalith/campaign-20260915/quant-candidates/SFT11-task-global-b-500/model-IQ3_M-imatrix-q8io.gguf",
  "IQ3_M", "2"
]
```

```json
[
  "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-quantize",
  "--imatrix", "/home/m0hawk/.local/state/sepalith/campaign-20260915/benchmarks/r2-imatrix-v1/imatrix.gguf",
  "--output-tensor-type", "q8_0",
  "--token-embedding-type", "q8_0",
  "/mnt/e/sepalith/campaign-20260915/intermediate-f16/SFT11-task-global-b-500-quant/model-F16.gguf",
  "/mnt/e/sepalith/campaign-20260915/quant-candidates/SFT11-task-global-b-500/model-IQ2_M-imatrix-q8io.gguf",
  "IQ2_M", "2"
]
```

The installed executable is the pinned CPU binary at commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` (b10453), SHA-256
`28cd0f04614c2ca5a6de32feb4b2196625fc792052682749da64cc5879035365`.

## Expected actual GGUF types

The file label is a mixture preset. The actual GGUF inventory is derived from
`llama_tensor_get_type_impl`, so `IQ2_M` produces `IQ2_S` tensors and `IQ3_M`
produces `IQ3_S` tensors. Counts below include all 381 tensor records and are
predictions that root must check against the output headers.

| Preset | Expected actual type counts | Native rule behind the mixture |
| --- | --- | --- |
| `Q4_K_M` | `F32`: 85; `Q8_0`: 2; `Q4_K`: 252; `Q6_K`: 42 | `Q6_K` on 21 `attn_v` and 21 `ffn_down` tensors; all other quantized linears `Q4_K` |
| `IQ3_M` | `F32`: 85; `Q8_0`: 2; `IQ3_S`: 205; `Q4_K`: 89 | `Q4_K` on all 42 `attn_v`, all 42 `attn_output`, and `blk.0` through `blk.4` `ffn_down` |
| `IQ2_M` | `F32`: 85; `Q8_0`: 2; `IQ2_S`: 205; `IQ3_S`: 47; `Q4_K`: 42 | `Q4_K` on all 42 `attn_v`; `IQ3_S` on all 42 `attn_output` and first 5 `ffn_down`; other quantized linears `IQ2_S` |

For 42 layers, the Q4 high-bit layer schedule is
`{0,1,2,3,4,7,10,13,16,19,22,25,28,31,34,36,37,38,39,40,41}`.
This comes from the pinned `use_more_bits` rule and is a positional heuristic;
it is not selected from the measured matrix. The IQ2/IQ3 promotion of
attention V is a GQA rule (`n_gqa() >= 4`), and the first-five FFN-down rule is
also source logic.

All relevant first dimensions are 2048 or 6144, divisible by the 256-element
IQ/K block sizes. Therefore the expected fallback count is zero. A warning or
any fallback changes the inventory and rejects the candidate pending root
review.

## What the TRAIN matrix measured

The values below are read-only reductions of the verified F32 matrix arrays:
each `.in_sum2` vector was divided by its `.counts` value, as the pinned loader
does. `sum over layers` is the sum of the per-tensor `Sigma(Act^2)` values;
`max channel` is the largest normalized squared-activation value in the
family. These are importance scores, not raw activation magnitudes.

| Family | Entries and input width | Sum over layers | Largest per-layer sum | Largest channel value | Highest ZD |
| --- | --- | ---: | --- | --- | --- |
| `attn_q`, `attn_k`, `attn_v` | 42 x 2048 each; arrays equal in this matrix | 57,048.83 each | 2,320.41 at layer 41 | 266.47 at layer 2 | 2.83% at layer 29 |
| `attn_output` | 42 x 2048 | 7,478.63 | 719.92 at layer 39 | 5.27 at layer 39 | 19.43% at layer 30 |
| `ffn_down` | 42 x 6144 | 29,525.57 | 14,541.67 at layer 41 | 4,054.25 at layer 7 | 9.07% at layer 15 |
| `ffn_gate`, `ffn_up` | 42 x 2048 each; arrays equal in this matrix | 35,224.54 each | 1,778.70 at layer 41 | 115.46 at layer 41 | 11.52% at layer 30 |
| `output.weight` | 1 x 2048 | 44,835.31 | 44,835.31 | 12,814.05 | 0.59% |

The measured peaks support the broad intent of the native mixtures but do not
prove their layer schedule. In particular, the sparse `ffn_down` layer-7
channel peak and the aggregate layer-41 `ffn_down` score deserve focused
quality checks. The layer-2 attention peak is a single-channel signal, while
the layer-39/41 `attn_output` scores are broad enough to matter for IQ2/IQ3
tests. Explicit Q8 output and embeddings cover the largest output-side score.

The pinned imatrix quantizers use these values as column weights during scale,
codebook, and lattice selection. Q4 K uses weighted squared error; IQ2 and IQ3
use weighted grid-neighbor and scale searches. The source adds a factor based
on the local weight value and `sqrt(sigma2 + x*x)`. There is no separate
outlier clipping pass, FP16 outlier spill, or per-channel preservation in
these commands. `Q6_K`/`Q4_K` promotions and Q8 output/embedding are the only
precision treatment in the three native arrays above.

## Matrix requirements and test priority

Pass the complete matrix for every preset. The loader requires the GGUF
`imatrix.datasets`, `imatrix.chunk_count`, and `imatrix.chunk_size` metadata,
normalizes by counts, and rejects non-F32 or mismatched sums/counts. The
quantizer then looks up a matrix vector by exact tensor name and rejects a
size mismatch for non-embedding tensors.

`Q4_K_M` and `IQ3_M` can run without a hard high-level matrix requirement for
their resulting K/IQ3 types, but the matrix changes their weighted codebook
fit and is required for this calibrated comparison. `IQ2_M` has 205 actual
`IQ2_S` tensors; the high-level b10453 type check marks `IQ2_S` as requiring
an imatrix. A missing or mismatched matrix must be a hard failure, never a
successful uncalibrated fallback. The lower-level `ggml_quantize_requires_imatrix`
predicate omits `IQ2_S`, so acceptance must rely on the higher-level
`llama-quant.cpp` check and on the conversion log as well as the output result.

Root should validate candidates in this order:

1. Check source F16 hash, pinned executable hash, terminal success, no
   `Missing importance matrix`, no size-mismatch message, and zero shape
   fallbacks. Confirm the output metadata records the matrix file, dataset,
   entry count, and chunk count.
2. Parse all 381 output tensor headers and compare the exact type inventory
   above. Confirm the two named tensors `output.weight` and
   `token_embd.weight` are `Q8_0`, while the 85 intentionally unquantized
   records remain `F32`.
3. Compare each candidate with native Q8 on the bounded TRAIN panel using the
   existing no-op, edit, cap, parser, and latency checks. Include focused
   inspection of the measured layer-7/layer-41 `ffn_down`, layer-2 attention,
   and layer-39/41 `attn_output` risks. Run the CPU serving check with two
   threads and record memory and latency.
4. Treat `IQ2_M` as the highest-risk candidate because its 205 `IQ2_S`
   tensors depend on complete calibration and its layer-41 `ffn_down` remains
   low-bit despite the measured aggregate peak. Treat `IQ3_M` as the
   intermediate risk candidate because it raises attention output but leaves
   most FFN down tensors at `IQ3_S`. Use `Q4_K_M` as the expected robust
   calibrated baseline because it gives 21 positional high-bit tensors to
   both attention V and FFN down.

Only TRAIN calibration and the authorized bounded quality panel belong in this
recipe review. No DEV or final content is used here, and no recipe is a
promotion decision without root's scientific acceptance thresholds.

## Primary source record

All source files below are from the pinned b10453 tree and were read without
editing it.

| Source | Relevant lines | SHA-256 |
| --- | --- | --- |
| `tools/quantize/quantize.cpp` | 121-165 (CLI), 180-257 (matrix loading), 302-344 (type parsing), 407-527 (options and metadata) | `5a3e413ca1022d345c8b3a100fbc957f73e461bb0f24eb7b42addf107c926903` |
| `src/llama-quant.cpp` | 116-158 (categories), 289-366 (quantizable tensors), 373-421 (fallback), 430-432 (layer schedule), 450-668 (native mixtures), 673-713 (override/fallback), 780-845 (requirements/defaults), 915-934 and 1035-1067 (matrix checks), 1194-1275 (lookup and conversion) | `41a5bce8f9ee9d4fd27053fee30357c4b3724ae7c612bd108161b7e0ddc42244` |
| `common/imatrix-loader.cpp` | 82-168 (GGUF metadata, paired F32 sums/counts) | `fcd471a2c502f5d78dd26dec04b6129a755e7eaa5566db6161ba646ac046b373` |
| `tools/imatrix/imatrix.cpp` | 125-198 (statistics), 234-251 and 356-420 (squared activation collection), 968-1074 (statistics display) | `a56c35fe501a2b4dc4b10d55dcf0468e22b8bfd0e82987f0cec8121201d2bf5a` |
| `ggml/src/ggml.c` | 7917-8000 (imatrix predicate and quantize dispatch) | `fb177479234cc6e5d15105f5cac630d9321827de6a71ead648ac96862ec6367f` |
| `ggml/src/ggml-quants.c` | 993-1140 (weighted fits), 1553-1640 (Q4 K), 4169-4378 (IQ3 S), 5142-5326 (IQ2 S) | `07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d` |
| `tools/imatrix/README.md` | 3, 25, 43-45, 77-98 (purpose, output handling, calibrated example, squared-activation caveat) | `35223ad9d781b078bf8b33295e35a3f331e8d8c5ad1e12ded3bffa08d5a17995` |

The matrix identity and coverage are recorded in
`docs/campaign/receipts/REL-07-r2-imatrix-root-acceptance.json`; selected
model identity and F16/Q8 hashes are recorded in
`docs/campaign/receipts/REL-07-r2-quant-preparation.json` and
`docs/campaign/work/lead/r2-task-global500-native-selection/q8-integrity.json`.
