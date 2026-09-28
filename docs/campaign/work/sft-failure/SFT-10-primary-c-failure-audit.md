# SFT-10 primary-c failure audit

Audit scope: read-only CPU inspection of the preserved `SFT-primary-3000-c`
attempt. No model, tokenizer, tensor, CUDA device, server, launch, SSH, or
cloud work was performed by this audit. The input inspection was limited to
admission metadata and token-row structural fields; target and prompt text was
not copied into this note.

## Finding

The attempt completed optimizer step 2968 and then aborted before a recorded
step 2969. The first surfaced training exception is a BF16 CUBLAS internal
error while the installed Unsloth manual LoRA backward is executing
`fast_lora.py:498`. Python then terminates with `c10::AcceleratorError` for an
illegal CUDA memory access during object destruction. The log itself says that
CUDA kernel errors may be reported at a later API call, so neither line 498 nor
the destructor stack establishes the original fault location.

The strongest reproducible code-level risk is the combination of two enabled
optimizations:

* `fast_lora.py` uses custom LoRA QKV and MLP autograd for the Llama model. Its
  default `inplace=True` path writes `dX` into the saved forward activation and
  then accumulates K/V or gate/up derivatives with in-place `addmm_` calls.
* `use_gradient_checkpointing="unsloth"` at `max_seq_length=4096` selects the
  smart offloaded, re-entrant checkpoint path. That path copies activations
  between pinned CPU buffers and GPU buffers on an extra CUDA stream, optionally
  with double-buffer events, before nested backward recomputation.

This is a plausible aliasing or stream/allocator interaction, especially after
a long run, but the preserved evidence cannot distinguish it from a CUDA
driver/device failure or host paging pressure. The final batch has no admitted
shape or token-bound violation and the scalar telemetry remains finite.

## Attempt and first-error evidence

The attempt is `1635a4a3d2c24d289b2de8ccbb2acaf0`, job
`sft03-minicpm-primary3000-c-20260912`, resumed from the full step-2000
checkpoint on the unchanged 3000-step schedule. The execution record reports
return code `-6` (`SIGABRT`). The supervised command was still well before its
deadline; this was not a deadline classification.

The preserved log is 93,812 bytes with SHA-256
`e983aaa058eabe32020b17682d760001b9918eb5b0c3b5d6da97e006032209af`.
Its final ordered markers are:

1. progress reaches `2968/3000`;
2. the first training exception is `RuntimeError: CUDA error:
   CUBLAS_STATUS_INTERNAL_ERROR` from `cublasGemmEx`, surfaced at
   `unsloth/kernels/fast_lora.py:498`:
   `torch.matmul(dQ, QW.t(), out = X if ctx.inplace else None)`;
3. process termination reports `CUDA error: an illegal memory access was
   encountered`, followed by `TensorImpl::~TensorImpl` and `Py_FinalizeEx`;
4. the same log states that CUDA kernel errors may be asynchronously reported
   at another API call and suggests `CUDA_LAUNCH_BLOCKING=1` for localization.

There is no `NaN`, `Inf`, out-of-memory, or earlier exception marker in the
normalized log. The two Unsloth messages, “Will smartly offload gradients” and
“Double buffering enabled,” are present before the final metrics. Their output
may be buffered, so their textual position does not prove that either feature
first changed state at step 2968.

## Finite telemetry and resource trend

`training/SFT-primary-3000-c/telemetry.jsonl` is 461,018 bytes with SHA-256
`84a0c5779cbe554bd556fb13e5751d69705d8f7f8bb813f1c8c75a80437ee654`. It has
1,017 valid JSON records:

* 968 `optimizer_step` records, steps 2001 through 2968, all with
  `stop_reason: null`;
* 48 `trainer_metrics` records, all finite;
* one `train_begin` record at step 2000.

The metric ranges are loss `0.1437404873..0.2479232549`, gradient norm
`0.2676498889..0.5601509213`, learning rate
`9.7944618e-08..5.1031944e-05`, and epoch `0.6733333..0.9866667`; the finite
count is 48/48 for every metric. The last optimizer record is step 2968 at
`2026-09-12T16:28:25.182480+00:00`, with 7,642.48 seconds remaining and
2.512 seconds for that update.

Across optimizer records, allocated CUDA memory is
5,800,727,552..5,826,275,328 bytes at the readout points, while reserved CUDA
memory reaches 28,632,416,256 bytes. The process-lifetime peak allocated and
reserved values are 10,490,742,784 and 28,632,416,256 bytes. This gives no
direct OOM diagnosis: reserved memory growth and a high peak can be relevant to
allocator or driver stability, but the log contains no OOM event and does not
measure host RAM, pinned-memory pressure, pagefile activity, or device health.

The parent supplied a stronger host-side timing correlation after this audit's
initial read: Windows TDR error at `16:28:27.216`, recovery action `PF FLR` at
`16:28:27.708`, `GpuRcReset` at `16:28:29.511`, and reset/restart activity from
`16:28:30` through `16:28:32`. These follow the telemetry's step-2968 timestamp
(`16:28:25.182480Z`) by roughly two seconds. The parent also reports no
resource-exhaustion event in the preceding hour, 28,723 MiB available Windows
memory with commit 57.858/137.353 GiB, and zero page input/output at its later
`16:46:41` check. This makes a driver/watchdog/device reset a credible
correlated failure mode, but it does not prove whether the training workload
triggered the reset or whether an earlier device fault surfaced during CUBLAS.
The later healthy eight-request CUDA graph baseline is a forward-only health
observation and is not training-backward evidence.

## Exact schedule and data check

The admitted files are:

* `train-token-rows.jsonl`, SHA-256
  `7641bbdc8f609aca1e0ad72177561edddfdbdae1b49a8444c15470652bf5ebb6`;
* `draws-3000.json`, SHA-256
  `a439fe8af68b4929a7747d4a82b9e0b26a05fcd3973f8a3c81148a84581d7f94`.

The draw manifest declares 48,000 draws, 3,000 optimizer steps, and effective
batch 16. The training entry point constructs a sequential dataset from the
predeclared draw indices. Since the resumed checkpoint has sampler
`consumed_draws=32,000` at step 2000, the next uncompleted optimizer update is
mapped by `draw_index=(step-1)*16` through `step*16-1`.

The 16-row candidate for step 2969 is draw indices 47488–47503. Its exact
admission summary is:

| quantity | value |
| --- | ---: |
| rows | 16 |
| prompt-loss tokens | 19,973 |
| target-loss tokens | 1,850 |
| target-body tokens | 1,754 |
| total input tokens | 21,839 |
| maximum sequence width | 2,283 |
| family counts | finish 4, format 2, na.rm 1, no-op 1, pipe 3, rename 3, roxygen 2 |

The row metadata (IDs and lengths only) is:

| draw | row ID | family | bucket | prompt | target | total |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 47488 | `00a5e2f12372396aaa7581cd` | roxygen_drafting | short | 869 | 166 | 1,035 |
| 47489 | `17ca705c586464dc2d7f6a26` | finish_block | short | 710 | 348 | 1,058 |
| 47490 | `34ac2e2a5c4800e2fddf3a1f` | rename_propagation | short | 1,804 | 14 | 1,818 |
| 47491 | `6a2336bdc6b74223573a3c4d` | pipe_rewrite | short | 1,950 | 24 | 1,974 |
| 47492 | `afc9ff07a7513900faedbafe` | format_propagation | long | 2,254 | 29 | 2,283 |
| 47493 | `192c4580d98de36a3d40d531` | finish_block | short | 195 | 164 | 359 |
| 47494 | `05a357a017725d0f2349c8c4` | no_op | short | 297 | 12 | 309 |
| 47495 | `77fba6227ade9cc6c0ed7c78` | na_rm_propagation | short | 1,969 | 29 | 1,998 |
| 47496 | `092f432c92a67501d7296630` | rename_propagation | short | 1,673 | 27 | 1,700 |
| 47497 | `61c710b6c5a3b2584008ad40` | pipe_rewrite | short | 1,790 | 10 | 1,800 |
| 47498 | `05ee30cc132741b2f2831ae7` | finish_block | short | 261 | 427 | 688 |
| 47499 | `6d942c665403bf2c1e5c0f54` | format_propagation | short | 1,814 | 38 | 1,852 |
| 47500 | `01afb8a82f12697594ad4f64` | roxygen_drafting | short | 566 | 181 | 747 |
| 47501 | `247eab8f0c7ffdb1b96591d1` | finish_block | short | 506 | 338 | 844 |
| 47502 | `328390efbd4caf2ec2974751` | rename_propagation | long | 2,034 | 18 | 2,052 |
| 47503 | `75f256e02ba957d06c828331` | pipe_rewrite | short | 1,297 | 25 | 1,322 |

The draw manifest and stored rows agree under the row format's explicit BOS/EOS
accounting: draw prompt/target counts include the leading BOS and terminal EOS,
while row `prompt_token_count`/`target_token_count` omit those boundary tokens.
All 11,764 distinct drawn row IDs were found exactly once in the token-row
file. Every selected row passed these structural checks: integer IDs in
`0..130559`, BOS 0, one terminal EOS 1, valid target boundary/body prefix,
train split, and pinned tokenizer hash. No short row exceeds 2,048 total tokens;
no long row exceeds 4,096; the global total-token maximum is 3,064. The
candidate step therefore contains two ordinary admitted long rows but no
sequence, tokenizer, or provenance violation.

## Installed implementation evidence

The attempt used Unsloth `2026.8.18`, Unsloth Zoo `2026.8.12`, Torch `2.11.0`,
Transformers `5.5.0`, TRL `0.24.0`, bitsandbytes `0.50.1`, and PEFT `0.20.0` from
`/home/m0hawk/Documents/Sepalith/.venv-sft`.

Relevant pinned source hashes and behavior:

* `unsloth/kernels/fast_lora.py`, SHA-256
  `429b95149a7987b71633aa6dd5bb9e2292e73503a1d96b0f6a549b2225b335c2`:
  `LoRA_QKV.forward` saves `X` and accepts `inplace=True` (lines 366–428);
  `LoRA_QKV.backward` flattens saved activations and, at lines 495–517,
  dequantizes Q/K/V weights, writes Q's `dX` into `X` at line 498, and mutates
  that result with in-place K/V `addmm_`; `apply_lora_qkv` defaults to
  `inplace=True` (lines 543–569). `LoRA_MLP.backward` has the same Q-like
  in-place write at line 194.
* `unsloth/models/llama.py`, SHA-256
  `8a79d94ab5f7f76c241d49da9bb1af9a90a9491977d3d08bbf356334dd007f2d`:
  Llama patch setup calls `prepare_model_for_kbit_training` with
  `use_reentrant=True` (lines 3629–3633), installs the default MLP fast path
  (lines 3676–3681), and assigns `apply_lora_qkv` to every eligible attention
  layer (lines 3721–3737). The log confirms 42 QKV, 42 O, and 42 MLP layers.
* `unsloth_zoo/gradient_checkpointing.py`, SHA-256
  `50e85fea0e5ee06401636acfbf57c0e38bbbb323116c1da4b8d8c8ee7e2d33fe`:
  `max_seq_length >= 512` selects the smart mode through the Unsloth helper;
  initialization allocates 200 pinned CPU buffers and GPU buffers (lines
  605–700), optional double-buffer events are enabled after a free-memory
  check (lines 784–805), and re-entrant backward uses an extra stream, waits
  for buffer transfers, recomputes under `torch.enable_grad`, and records an
  event after nested backward (lines 890–1031). The diagnostic override is
  `UNSLOTH_DISABLE_DOUBLE_BUFFER=1` (lines 76–86).
* The attempt snapshot of `campaign_sft.py`, SHA-256
  `671a1da97939b21962cfaedda0dc7c2c6f2f01369cb2834ef351fd26c38541f9`, loads
  with `max_seq_length=4096`, BF16, and `load_in_4bit=False` (lines 312–315),
  requests `use_gradient_checkpointing="unsloth"` (lines 323–326), and uses
  per-device batch 4, accumulation 4, `max_length=4096`, and `packing=False`
  (lines 367–373). The snapshot of `campaign_sft_data.py`, SHA-256
  `31e34c5a3a662724cf50dce1ea9e6a0d248c5f44bc0f5eb6b0a5b45894433104`,
  validates full stored sequences and maps the sequential 16-row exposure
  (lines 21–78).

## Assessment and minimal future checks

Data cause: **not supported**. The exact candidate is admitted, finite, and
inside the 4,096-token SFT bound. There is no evidence that an individual ID,
family, target length, or long bucket caused an invalid tensor shape.

Memory exhaustion: **not established**. No OOM event was logged; reserved CUDA
memory is a cache/allocator observation rather than proof of available or
unavailable memory. A driver/watchdog/device reset remains an open possibility.
Parent-supplied host telemetry reports no resource-exhaustion event and zero
page I/O at the later check, but this worker did not independently measure host
memory or CUDA health.

Kernel/stream interaction: **plausible, unlocalized**. The custom in-place
LoRA derivative and the smart re-entrant offload/double-buffer path are both
active code paths that deserve a controlled reproduction. The asynchronous
CUDA warning prevents assigning fault to `fast_lora.py:498` from this log alone.

Before any future long RL run, the minimal live diagnostic is a short controlled
backward from a verified full checkpoint with:

1. `CUDA_LAUNCH_BLOCKING=1` for the first localization attempt;
2. explicit per-step finite checks for loss, gradient norm, adapter gradients,
   and generated/policy logits, plus synchronized allocator and host-RAM/
   pinned-memory telemetry;
3. one-factor comparison of the current smart offload/double-buffer path with
   `UNSLOTH_DISABLE_DOUBLE_BUFFER=1`, and then with standard checkpointing or a
   safe non-in-place LoRA path if the first run localizes to those paths;
4. **NOT APPLIED in this audit:** if root's CUDA lane needs an allocator
   pressure diagnostic, set an explicit per-process allocator fraction of 0.75
   or 0.80 immediately after the single CUDA owner is established, record the
   selected fraction in the live identity, and treat any resulting OOM as a
   diagnostic result. This is supported only as a bounded comparison because
   the run ended with reserved memory near the reported device capacity while
   no OOM was logged; it is not a fix claim and is not part of this receipt's
   source or recipe.
5. an immediate clean-process CUDA health check after any failure, preserving
   the original supervision receipt and marking the checkpoint untrusted until
   full optimizer/RNG/sampler identity is verified.

These checks are diagnostic gates, not evidence that changing a flag will fix
the failure. RL should remain blocked on a real policy backward/generation
smoke that passes these finite, mode-restoration, and durable-checkpoint checks;
the CPU audit does not admit the live RL gate.
