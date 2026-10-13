# REL-07 selected-R2 QAT feasibility

This packet answers whether the selected R2 MiniCPM5 draft can be trained with
quantization-aware training and then shipped as GGUF Q4_K_M (or a compatible
lower-bit format). The bounded check used source, CLI, package, and synthetic
module metadata only. It did not read target model payloads, launch a
conversion, use CUDA, use a provider, or open final/DEV data.

The deployable route that is already supported is post-training K-quantization:

1. Use the root-verified selected R2 F16 GGUF as the source. It must be the
   selected SFT11-task-global-b-500 artifact, never the selected Q8 artifact.
2. Optionally create an importance matrix from the exact TRAIN-only corpus and
   pass it to the pinned b10453 quantizer.
3. Run b10453 llama-quantize with Q4_K_M, preserving the campaign Q8_0 output
   and token-embedding overrides.
4. Inspect the resulting GGUF header, then run the existing fixed native panel
   against F16/Q8/Q4 with the same tokenizer, context, cap, parser, and
   ordinary decoding settings.

The root-owned command template is:

~~~text
QUANT=/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-quantize
F16=/mnt/e/sepalith/campaign-20260915/intermediate-f16/SFT11-task-global-b-500-quant/model-F16.gguf
IMATRIX=/home/m0hawk/.local/state/sepalith/campaign-20260915/benchmarks/r2-imatrix-v1/imatrix.gguf
OUT=/mnt/e/sepalith/campaign-20260915/quant-candidates/SFT11-task-global-b-500/model-Q4_K_M.gguf
"$QUANT" --output-tensor-type q8_0 --token-embedding-type q8_0 \
  --imatrix "$IMATRIX" "$F16" "$OUT" Q4_K_M 4
~~~

The no-imatrix control removes the --imatrix argument. The existing
export_gguf.py wrapper also supports --f16, --imatrix, --uncalibrated, and
--tiers Q4_K_M; its Q4 policy requires an explicit calibration matrix or an
explicit uncalibrated control. Root already owns the active Q4 and imatrix
guards, so this packet does not run either.

This is a real llama.cpp K-quant route. The pinned source defines QK_K as 256
and block_q4_K as eight 32-element blocks with packed four-bit values and
quantized scale/min fields (ggml-common.h:87-89,328-338). The quantizer
accepts Q4_K_M and aliases Q4_K to it (tools/quantize/quantize.cpp:62-68).
The selection code maps Q4_K_M to GGML_TYPE_Q4_K and can promote selected
layers/tensors to Q6_K (src/llama-quant.cpp:603-609,827-830); --pure disables
those K-quant mixtures. Therefore a generic per-group int4 tensor does not
establish a Q4_K_M deployment. Header inspection is required after each
conversion.

The installed TorchAO API does expose a QAT workflow. Its QATConfig source
says prepare applies fake quantization and convert applies the base PTQ
configuration; the only documented base configuration in this installation is
Int4WeightOnlyConfig (torchao/quantization/qat/api.py:43-80). That config uses
group sizes 256, 128, 64, or 32, defaults to 128, and stores one of TorchAO
PLAIN, PRESHUFFLED, PLAIN_INT32, or TILE_PACKED_TO_4D tensor formats
(torchao/quantization/quant_api.py:529-620). Its Int4Tensor conversion requires
mslk >= 1.0.0; mslk is absent from the canonical environment. A tiny
base-config prepare/convert probe consequently stopped at int4_tensor.py:140
with ImportError: Requires mslk >= 1.0.0. A separate custom fake-quant
prepare/backward probe passed with two fake-quantized linear layers, but custom
fake-quant configurations cannot use the QAT convert step. The XPU/NPU-only
Int4PlainInt32Tensor and CUDA tinygemm Int4TilePackedTo4dTensor formats are also
not GGML block_q4_K storage.

The project has no MiniCPM5/DSpark QAT adapter or QAT training entry point in
the inspected training source. train_sft.py uses ordinary full/adapter training
and load_in_4bit=False; export_gguf.py converts floating HF/F16 weights and
then invokes the external llama.cpp quantizer. Feeding a TorchAO packed tensor
to that exporter is therefore unsupported. Dequantizing a TorchAO checkpoint
back to BF16/F16 before export would cause a second, different Q4_K_M
quantization and would be a QAT-informed PTQ experiment, not training against
the deployed K-quant numerics.

BitsAndBytes is installed, but its quantize_4bit API accepts NF4 or FP4 with a
blocksize (default 64) and a BitsAndBytes QuantState. That is
weight-only/QLoRA-style quantization, not QAT, and it has no direct GGUF
Q4_K_M export path. It must not be reported as QAT evidence.

The practical decision is to continue the root-owned direct F16 -> Q4_K_M
path, with the active TRAIN imatrix as the calibrated candidate and an
uncalibrated Q4 control. If Q4 quality is weak, Q5_K_M or Q6_K are supported
by the same pinned binary; Q6_K is also exposed by the project wrapper. The
existing Q8 remains the rollback. A true QAT route would require all of the
following before admission:

- a MiniCPM5 DSpark-safe fake-quant module that reproduces Q4_K's 256-value
  superblocks, eight 32-value subblocks, scale/min quantization, and the
  Q4_K_M per-tensor mixed-type policy;
- a CUDA-capable training/export environment (the current TorchAO base path
  also needs mslk) and a bounded training run;
- either an exact GGUF K-block exporter or a proof that dequantize-then-export
  does not discard the learned quantization behavior; and
- native header, parser/no-op, accepted-token, and latency measurements.

No such adapter or format bridge exists in the inspected local source, so true
QAT is not launch-ready. The schedule extension permits a later custom
experiment, but it does not turn TorchAO generic int4 or BitsAndBytes NF4
into an exact deployable Q4_K_M route.

The bounded timing estimate for the supported PTQ route is 5 minutes for
provenance/setup, 10-30 minutes for F16 -> Q4_K_M quantization, and 10-30
minutes for the fixed native panel, with I/O and the active matrix determining
the actual time. These are estimates, not measurements from this worker.
The custom QAT route is expected to require hours for adapter, environment,
training, export, and evaluation, and has no admission command in this
packet.

Validation completed here:

- canonical .venv-sft package/import probe: Torch 2.11.0+cu130,
  Transformers 5.5.0, TorchAO 0.18.0, BitsAndBytes 0.50.1, GGUF 0.19.0,
  PEFT 0.20.0; no mslk, quanto, optimum, AutoGPTQ, AWQ, llama-cpp
  Python binding, LLM Compressor, or compressed-tensors package;
- custom TorchAO fake-quant prepare/backward: pass, two fake-quantized
  linear layers, finite loss and gradients;
- TorchAO base Int4WeightOnlyConfig prepare/convert: blocked by the
  reproducible missing-mslk error above;
- project test_quant_export.py: 5 tests pass using synthetic files only;
- exporter help: Q4_K_M, --imatrix, and --uncalibrated are exposed;
- quantizer help: Q4_K_M/Q5_K_M/Q6_K and --imatrix,
  --output-tensor-type, and --token-embedding-type are exposed. Its
  deliberate help path exits 1 after printing usage.

Root-owned gates remain output terminal status/hash, GGUF header/tensor-type
inspection, fixed TRAIN panel comparison, and the final quality decision.
This packet records feasibility only and makes no Q4 quality, QAT recovery,
or promotion claim.
