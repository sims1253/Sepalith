# sm120 variable-length attention backend preparation

Status: **prepared, not GPU-tested, not admitted**. The canonical virtual
environment was not modified.

The prior failure is narrower than “xFormers does not support RTX 5090.” The
canonical `xformers 0.0.35` extension was built for Torch 2.10/cu128 and only
through `9.0a`, while the runtime is Torch 2.11/cu130 on sm120. Its extension
does not load, so Unsloth's real sm120 forward probe disables xFormers and
`select_attention_backend(use_varlen=True)` returns `sdpa`.

An official cu130 xFormers 0.0.35 wheel was downloaded to E and installed only
into an isolated E overlay. It includes `12.0a` and `12.1a` code objects and
loads without the extension ABI warning under the canonical Torch 2.11/cu130
runtime when CUDA is hidden. This makes it the first diagnostic candidate; it
does not establish that its FlashAttention-2 forward/backward kernels execute
correctly on sm120.

`commands.json` gives the bounded root-run order. First run the 256-token
kernel-only BF16 block-diagonal forward/backward and cross-document mutation
control. If that passes, run the already reviewed actual MiniCPM harness through
the overlay. The actual-model result must preserve exact cross-document
isolation and show finite standalone/packed loss and gradient comparisons.
Numerical thresholds should be based on repeatability and the prior standalone
baseline; this packet does not invent a tolerance or performance claim.

Native Torch FlexAttention is an available fallback kernel API and already has
a specialized Unsloth use for GRPO prefix grouping. It is not wired to ordinary
`packed_seq_lengths`. A Flex kernel pass therefore requires a fresh isolated
Unsloth attention integration, followed by the same actual-model parity,
memory, warm-throughput, and full-state resume tests before training use.

The current CPT denominator is source-defined as a **global supervised-token
mean over one 16-microbatch accumulation window**. Transformers prefetches the
window; Unsloth counts shifted non-masked labels (and shifted attention mask)
and passes the same aggregate `num_items_in_batch` to every forward. For packed
inputs the installed counter additionally subtracts `members-1`; because the
prepared CPT members already mask each first label, the reviewed parity harness
passes the original explicit denominator instead. Any integration must record
`model_accepts_loss_kwargs=true` and the exact first-update denominator before
admission.

CPU verification:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/sm120-varlen-v1 \
  /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B backend_contract.py
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/sm120-varlen-v1 \
  /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B -m unittest discover -s tests -v
```
