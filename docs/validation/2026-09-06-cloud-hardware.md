# Cloud hardware and offload limits, 2026-09-06

Kaggle can free the local RTX 5090 for other work when a task fits the tested T4
environment. It does not currently provide a verified increase in per-device
memory. This note separates account observations, published specifications and
unverified options. It records research only; no jobs were launched.

## Available evidence

| Resource | Evidence | Capacity implication |
|---|---|---|
| Kaggle T4 ×2 | Both devices ran tensor operations; a 135M dense model completed 20 LoRA updates | Two usable GPUs are established; larger recipes and distributed training are not |
| Local RTX 5090 | Project's local baseline; NVIDIA specifies one 32 GB GDDR7 Blackwell GPU | One model can use a single larger device, subject to runtime overhead |
| Anyscale A10G | Project queue records a successful 60-step LoRA smoke | Verified previous cloud execution; AWS specifies 24 GB per A10G |
| Kaggle TPU | Tested batch requests exposed no working TPU backend | No usable TPU capacity established for this account |
| Kaggle H100 / RTX Pro 6000 | CLI accepts listed shape names, some with restricted eligibility | Device allocation and account access remain unverified |
| Single 80 GB cloud GPU | Conditional plan in the project queue | A hypothetical allocation, not the verified Anyscale A10G |

Account observations come from the [Kaggle capability report](2026-09-06-kaggle-capabilities.md)
and its [artifact index](2026-09-06-kaggle-capabilities-artifacts.json). Prior Anyscale
execution and the conditional 80 GB plan appear under W29 and W14 in the
[experiment queue](../EXPERIMENT-QUEUE.md). Hardware specifications come from
[NVIDIA's RTX 5090 page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
and [AWS's G5 page](https://aws.amazon.com/ec2/instance-types/g5/).
The [Kaggle CLI shape list](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
is a request vocabulary, not evidence of an account entitlement.

## GPU memory and kernels

Each T4 has 16 GB GDDR6 and a PCIe Gen3 interface. Two cards provide separate
allocations; a single-device allocation cannot use their combined nominal 32 GB.
The actual free/usable bytes must come from the job's device report, not the
marketing capacity. [NVIDIA T4 specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/).

Ordinary one-process-per-GPU DDP keeps a model replica on each GPU and synchronizes
gradients. It does not solve a model that exceeds one card's memory. Explicit model
partitioning or sharding would be a separate implementation and validation task;
no such result is established here. [PyTorch DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).

T4 is Turing, compute capability 7.5. RTX 5090 is Blackwell, capability 12.0.
[NVIDIA capability table](https://developer.nvidia.com/cuda/gpus).
T4 lacks native BF16 support. PyTorch's current `is_bf16_supported()` defaults to
allowing emulation; successful BF16 tensor creation therefore does not prove
native BF16 execution. Where supported by the installed version, record
`is_bf16_supported(including_emulation=False)` as well as device capability.
[PyTorch implementation](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/cuda/__init__.py).

Upstream FlashAttention-2 lists Ampere, Ada and Hopper for its CUDA path. It points
Turing users to a separate implementation with a subset of features. The standard
FA2 installation is therefore not a verified T4 path. FA2 supports FP16 as well as BF16; its architecture support, rather than a
universal BF16 requirement, is the restriction here. These are backend-specific constraints, not a claim that every form
of efficient attention is impossible on T4. The cited FA2 support list alone also
does not establish compatibility for a particular RTX 5090 build.
[FlashAttention support matrix](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support).

The Kaggle probe used FP16 base weights and FP32 adapters. Its success does not
resolve the earlier Qwen3.5/Unsloth dtype failures or validate that model's GDN
kernels. Those observed limits are recorded in the
[capability report](2026-09-06-kaggle-capabilities.md).

## Host resources, persistence and time

Kaggle's published notebook profiles list:

| Profile | CPU cores | Host RAM |
|---|---:|---:|
| CPU | 4 | 30 GB |
| P100 | 4 | 29 GB |
| T4 ×2 | 4 | 29 GB |
| TPU 1VM | 96 | 330 GB |

The same documentation gives 12-hour CPU/GPU sessions, 9-hour TPU sessions, 20 GB
of saved output in `/kaggle/working`, and additional scratch space that does not
persist. It states that CPU platforms can vary and that interactive editing has
a 20-minute idle limit. These are published profiles and limits, not measurements
of this account's jobs. The TPU figures do not establish TPU access.
[Kaggle notebook specifications](https://www.kaggle.com/docs/notebooks#technical-specifications).

No measured host-RAM, CPU-quota or free-disk result is established in this note.
For a future authorized job, record visible CPU affinity and container quotas,
available RAM and any container memory cap, free working/scratch space, GPU free
memory and remaining account quota before selecting batch size. Preserve outputs
within the job's explicit wall-time cap. Do not infer a guaranteed recurring quota
from one day's remaining-hours reading.

## Placement decision

The evidence supports using Kaggle for separately specified small dense-model,
embedding or evaluation work that fits a T4 and its tested software path. This is
an inference from the hardware limits and completed probe, not a throughput
benchmark. Retain the RTX 5090 for work that needs its larger per-device memory
or the validated local kernel stack. Treat the verified Anyscale A10G as another
24 GB placement option; assess an 80 GB GPU only after its actual allocation,
price and recipe compatibility are established. No measured speed ratio between
T4 ×2 and RTX 5090 follows from these specifications.
