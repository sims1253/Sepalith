# Where to run the pending experiments

Decision review, 6 September 2026. This assigns preferred resources; it does not
launch experiments or change parked authorizations. The queue manager still owns
queue status and reservation of work. Existing local migration remains pending.

## Recommendation

Use Kaggle first for independent quality evaluation, then for small retrieval or
scoring models whose training fits one T4. Keep production GDN training, long
from-scratch runs and hardware-specific performance experiments on the local
5090 or an explicitly selected larger cloud GPU. Kaggle currently offers us
additional independent capacity, not a demonstrated upgrade in memory or speed.

The live account reports 29.23 GPU hours and 20 TPU hours remaining, with next
refresh on 12 September. There is no Sunday-night expiry to optimize around.
The separate inference allowance is $10/day and $100/month, unused; proxy token
access is blocked. Do not plan teacher calls against that allowance yet.

## Resource comparison

| Resource | Evidence | Best use and limitation |
|---|---|---|
| Local workstation | Live OS inspection: Ryzen 9 5900X, 24 logical CPUs, about 47 GiB visible RAM, about 23 GiB available at inspection. Repository records RTX 5090 with 32 GB VRAM. Visible RAM is the WSL allocation, not proof of installed physical RAM. | Main training and final CPU/5090 product measurements. W37 forbids simultaneous local CUDA workloads; even CPU preparation can disturb the active benchmark. |
| Kaggle T4 session | Two Tesla T4 devices executed tensor operations in our archived probe. Hardware specification is 16 GB per device. Kaggle publishes 4 CPU cores and 29 GB host RAM for this profile; the actual allocation, usable device memory and disk were not captured by our probe. | Separate model or evaluation shard per GPU is the simplest useful arrangement. Two devices do not form a single 32 GB allocation. DDP duplicates model state; larger-model sharding needs explicit implementation and measurement. |
| Kaggle TPU | Three completed jobs produced no usable TPU; account reports no TPU session ever started. | No scheduled scientific workload until device access and the actual training stack pass. GPU code is not automatically portable to TPU. |
| Anyscale A10G | Prior SFT throughput recorded at 2,746 tokens/s versus local 4,050–4,360; 24 GB card specification. Live reconciliation found all visible jobs terminal. | Working escape route for supported training when the workstation is occupied. Neither more VRAM nor faster in that measured SFT comparison. Preserve the existing production credit reserve. |
| A100/H100-class 80 GB or larger rental | A resource option, not verified Kaggle entitlement or a booked instance. | Prefer when a measured working set exceeds 32 GB or uninterrupted duration matters. Select the exact GPU for required kernels and precision: A100 supports native BF16; W1 targets H100/B200-class FP8. BF16 alone does not require leaving the 5090. Measure throughput and total cost, including transfer/setup. |

See the [hardware source review](2026-09-06-cloud-hardware.md) for primary-source
specifications and provider limits. An accelerator name accepted by an API is not
proof of access. We have not established that Kaggle provides more host RAM than
this workstation. More host RAM also does not substitute for VRAM without an
explicit offload design and its transfer cost.

## Ranked candidates

Ranks combine research value, independence and porting cost. Durations in the
queue describe the original machine; they are not Kaggle estimates.

| Priority / item | Placement decision | Required evidence before dispatch; scientific boundary |
|---|---|---|
| 1 — W33 v7 ablation repair | Best first real offload candidate: frozen GGUF inference and scoring, one T4 initially. Repair missing evidence without retraining. | Manager confirms no newer repair exists and reserves the exact row set. Freeze model, evaluation rows, renderer, runtime and decode settings. Compare a small known-good slice with the original backend; reject missing/error responses rather than treating them as model abstentions. Complete the damaged evaluation and preserve denominators. Cloud timing is not local product latency. |
| 2 — B10 WiSE-FT interpolation quality battery | Good parallel evaluation workload after adapter selection. CPU merge/preparation can happen outside the quiet workstation; one blend per T4 is plausible. | Needs two compatible adapters on the exact same base. Do not choose a scientifically rejected adapter merely to fill a slot. Freeze alpha grid 0.3/0.5/0.7, controls and held-out selection policy; adopt only under the original product-dominance rule. Final local latency remains local. Queue status and launch scope need reconciliation. |
| 3 — LOC1-S1 small retrieval pilot | Strong proposed training candidate, especially the 47M arm; 346M arm needs measured memory. Embedding batches and negative mining can also run remotely. | S0 is already complete: neural retrieval passed the transfer gate but did not beat BM25. Do not repeat S0 as a new discovery. S1 still requires its separately parked API/data authorization, disjoint-repository evaluation and a tested CUDA/FP16 path for custom model code. WINNER-LOC must beat both as-is neural and lexical baselines; CPU latency measured locally. |
| 4 — MLV feedforward verifier | Promising proposed T4 task once the scorer/backbone is specified. Prefer frozen-feature extraction and a small head if that matches the approved design. | Implementation and data sufficiency are open. V1d calibration has only 49 anchor and 148+148 judged comparison points, with shared trajectories; those are not automatically a training corpus. Split by trajectory/repository before training and define tie handling. Preserve AUROC kill below 0.65, preference accuracy, calibration and matched-FLOP routing readouts. Do not silently substitute an easier model for the proposed experiment. |
| 5 — O2 pre-screen / O4 novelty audit | Offload batched frozen-model inference or embedding extraction; keep API-only and arithmetic work off GPU. | Keep the frozen model, prompt and reward validators identical. O2's actual claim needs matched-rollout filtered/unfiltered RL plus held-out exact improvement; a successful pre-screen is not the verdict. Full GDN GRPO is not yet a viable T4 port. |
| 6 — H3-S1 short render adaptations | Conditional, after H3-S0 selects survivors and the actual training stack works. Independent survivors could use separate devices. | Keep the chosen base and equal data/optimization budgets. A small dense-model substitute only validates tooling. Final adoption needs exact-quality parity plus a serving benefit, measured on the intended local tier. |
| Later — P11 vocabulary LM legs; ML1/ML2 loop probes | Possible small-model workloads after their earlier gates, but not first-wave offloads. | Existing ladder/Muon code hardcodes BF16. An FP16 port changes numerical behavior and may require loss scaling and a fresh control. Preserve byte-based cross-vocabulary scoring for P11 and FLOP-matched controls/monotonicity for ML1. ML2 depends on ML1 winning. |

## Keep elsewhere, or do not repeat

- S1/S2, V1c latency-under-load, H5 iGPU and P12 roofline: run performance
  claims on the named deployment hardware. Kaggle can compute independent quality
  legs only under a separately reserved scope. It cannot replace the active CPU
  benchmark, and moving V1c would change the prepared migration candidate.
- MLK and W35 throughput tuning: golden correctness tests are portable, but kernel
  and microbatch speed conclusions belong to the target GPU. W1's H100/B200 FP8
  gate requires the matching hardware. T4 results cannot establish these claims.
- Production Qwen3.5 SFT and W17/O2/O4 full GRPO: retain the proven environment.
  Current `train_sft.py` documents Unsloth's GDN FP16 veto; its misleadingly named
  `SFT_FP16` option actually requests FP32. T4 has no native BF16. Our tiny dense
  SmolLM LoRA success proves neither GDN training compatibility nor memory fit.
- Q5 (~35 local GPU hours), ML3 (~26-hour pair), P4/P5 and large A2 runs: prefer
  local or durable larger rentals after their scientific gates. T4 runtimes may
  be much longer; checkpointing and exact resume must precede any session-bounded
  port. Free quota does not remove engineering or comparison costs.
- W6/W8/W9/W10 corpus work, SY1 scan, B11 authors, H3/P13 paper gates and W32
  archaeology: primarily CPU, storage, API or analysis work. Use a cloud CPU only
  if transfer, storage and isolation make it worthwhile; do not allocate GPU just
  to consume its quota. The 91 GB W6 corpus is a transfer constraint, not a reason
  to assume a notebook has more memory.
- B-series bake-off, B8/B8b/B9, PFT1, TU1/TU2, X5, M1 and P10 have later verdicts.
  Do not rerun them because older sections still list the initial proposal. E2 is
  closed and E3/E4 depend on gates that did not pass. H2 is moot until weights move.
  Q4 and remaining X/P/W options retain their existing parked/dependency status;
  none outranks restoring missing evaluation evidence on current information.

## First offload preparation

Reserve W33 with the queue manager, then prepare one complete evaluation recipe.
Capture reviewed working-tree code and exact model/input bytes into immutable
source and private input artifacts. Record hashes, license/access constraints,
interpreter, image/dependencies, runtime build and command arguments. No remote
command may depend on this mutable checkout or receive workstation credentials.

Start with a small calibration slice, not the entire battery. Its receipt must
include actual RAM/cgroup limit, CPU affinity, per-device total/free memory, disk
space, runtime versions, throughput and output identity. Measure peak memory with
the real context and batch size. Use those results to set the full-run timeout
and quota reservation, leaving time for collection and explicit recovery. There
is no defensible six-hour throughput estimate from the 135M smoke alone.

Use one GPU first. Add a second independently assigned shard only after model
memory, output row coverage and two-device isolation pass. Reconcile other active
account sessions before submission; a controller lock covers only its own state.
Validate output completeness, errors, duplicates, hashes and provenance; keep
command success separate from the scientific verdict. Archive raw evidence in the
private bucket and commit the summary/index. No automatic retries or follow-ons.

## Sources and reconciliation

Reviewed the current shared queue and append-only board, rather than the older
queue copy in this worktree. Shared HEAD at inspection was
`73d0bfef01adfe7288bd56d44f4a32d3d8f9e167`; current queue SHA-256 was
`16a5efa79d5271544922a2fa7e8a088f30a1e81104c7355f270c499f9acc11a5`.
These identify the review context, not a source capture for an executable run.

Primary project records: [queue](../EXPERIMENT-QUEUE.md),
[coordination board](../../comms/board.md),
the shared-checkout `experiments/data-mining/LOC1_S0_RESULTS.md`,
[V1d result](../../experiments/eval/V1D_RESULTS.md),
[ablation evaluator](../../experiments/eval/eval_ablation.py),
[training entry point](../../experiments/training/train_sft.py), and
[verified Kaggle probes](2026-09-06-kaggle-capabilities.md).
The local historical `docs/research/2026-08-31-base-bakeoff-plan.md` B10 section
and `docs/research/landscape-v7-vs-glm53.md` ablation warning were also inspected;
these research notes and the shared-only LOC1 result are not guaranteed to exist
in a fresh clone.
No pending experiment was launched or marked complete by this review.


## Azure credit added, 7 September

The user reports a new Azure account showing about $200 credit. Treat the balance,
subscription offer, expiry and GPU access as unverified account facts until an
authenticated read-only check. No Azure CLI is installed in this environment.

If this is the standard free-account offer, the $200 lasts 30 days, rather than
renewing weekly. [Azure offer](https://azure.microsoft.com/en-us/free/).
Free Trial subscriptions cannot request quota increases; credit does not imply
GPU-family quota or capacity in a region.
[Azure quota limits](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/azure-subscription-service-limits).

Azure could add actual memory capacity: `Standard_NC40ads_H100_v5` has one H100
NVL with 94 GB VRAM and 320 GiB host RAM. That is a possible resource for a
memory-constrained or H100-specific experiment, not an account entitlement.
[Microsoft hardware specification](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncadsh100v5-series).

Next checks: subscription offer and credit expiry, regional/family vCPU quota,
SKU restrictions and current hourly price including disk/storage. Prefer a
short, specified experiment that needs the extra capacity if access is available.
Otherwise consider independent CPU preparation/evaluation within the actual CPU
quota. Do not upgrade billing, remove spending protection or provision resources
as a consequence of this note. No Azure spend or resource creation occurred.
