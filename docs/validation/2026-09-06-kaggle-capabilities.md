# Kaggle capabilities tested on 2026-09-06

T4 training works for a small standard model. Our tested TPU batch paths did not
expose a working JAX TPU backend. Benchmark model-proxy authentication returned 403.
All four submitted jobs are complete, their reports verified, and no follow-on is
running. These are capability probes, not scientific model comparisons.

## What the account actually supplied

| Resource | Observation | What we can conclude |
|---|---|---|
| T4 GPU | Two Tesla T4s visible; tensor operations succeeded on both | Independent GPU use works; distributed training and pooled memory were not tested |
| Small model training | SmolLM2-135M, 20 LoRA updates, adapter save/reload and generation passed | The default GPU image supports this standard architecture without installing a training stack |
| Legacy TPU shape | JAX selected CPU; after installing its TPU plugin, initialization found no TPU device | This tested batch path is not ready for TPU work |
| Newer TPU shape | `TpuV6E8` also failed TPU initialization with the pinned plugin | Accepting the shape name does not establish usable TPU access |
| Benchmark model proxy | Credential endpoint returned HTTP 403 | No inference request was sent; account access needs investigation |

The [CLI lists](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) P100,
T4, L4, A100, H100 and RTX Pro 6000 shapes, among others. It explicitly limits some
to competitions or administrators. We have confirmed T4 access, not Blackwell
access. Two 16 GB T4 cards have separate memory pools; they are not one 32 GB card.
The [metadata guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
also warns that the default PyTorch image lacks kernels for the older P100.

## GPU result

Job `m0hawk/sepalith-gpu-fa842ce919df43ce/1` used public
`HuggingFaceTB/SmolLM2-135M` at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The source embeds four synthetic R snippets.
The report includes hashes of every downloaded model/tokenizer file and the input
text. The model-weight SHA-256 is
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.

The base weights used FP16. Small custom rank-4 adapters on 60 q/v projection
modules used 230,400 trainable FP32 parameters. Twenty AdamW updates took 3.10
seconds, with peak PyTorch allocated VRAM about 434 MiB. The full probe, including
imports/download/loading, took 61.64 seconds. These timings do not include every
part of Kaggle provisioning and billing.

In-sample loss fell from 2.05695 to 0.84378. Saving and restoring the adapter
preserved the loss exactly. Generation ran, but its R output was incorrect.
This establishes working training/inference plumbing, not useful edit quality.
It does not establish PEFT/Unsloth compatibility, large-model throughput, or a fix
for the production Qwen3.5 GDN recipe's earlier dtype failures.

The observed environment was Python 3.12.13, PyTorch 2.10.0+cu128 and Transformers
5.0.0. No package install was needed in the GPU job. The managed image was observed,
not pinned at submission; downloaded provider metadata is in the archive.

## TPU investigation and explicit retries

| Attempt | Shape | Job suffix | Code seconds | Outcome |
|---|---|---|---:|---|
| Stock image | Tpu1VmV38 | `sepalith-tpu-5a29289a9d9a4297/1` | 3.95 | Only `TFRT_CPU_0`; rejected CPU fallback |
| Add plugin | Tpu1VmV38 | `sepalith-tpu-runtime-6b2c76faa43246de/1` | 14.10 | Plugin installed; TPU initialization found no device |
| Newer shape | TpuV6E8 | `sepalith-tpu-v6-fc1512ff19134406/1` | 13.78 | Same TPU initialization error |

All IDs above have owner `m0hawk`. Each retry used a new ID after the preceding job
was confirmed complete and its report verified. The states retain `retry_of` and
the reason. There was no automatic retry or reset of an uncertain launch.

The stock image had JAX/JAXlib 0.7.2 but no libtpu. Follow-ups installed
`libtpu==0.0.23`, the dependency specified by the JAX 0.7.2 TPU extra, and set
`JAX_PLATFORMS=tpu`. Both then failed with `No ba16c7433 device found`.
The submitted/provider metadata says TPU enabled. This mismatch needs a working
interactive TPU notebook or a provider-side explanation before another training
attempt. It does not prove that Kaggle never supplies TPUs or identify the cause.
[JAX installation guidance](https://docs.jax.dev/en/latest/installation.html) describes
the separate TPU plugin requirement.

## Benchmark proxy

The local Kaggle 2.2.4 SDK could authenticate for normal jobs, but its
`create_default_model_proxy_token` call returned 403. An independent credential-only
diagnostic confirmed the SDK's translated HTTP status. No token values were saved
and no model calls were made. The original journal predates the error-code reporting
fix, so its `http_status: null` is preserved alongside the diagnostic evidence.

The prepared probe permits three short synthetic R prompts, with no publication,
private training data or permanent benchmark carve. It refuses an existing state
directory, journals before each request, and disables HTTP redirects so a bearer
token cannot be forwarded to another origin. The [Benchmarks library](https://github.com/Kaggle/kaggle-benchmarks)
notes that available models depend on the execution environment and proxy token.
A listed credit allowance does not establish working API access.

## Budget and evidence

The exploration allowance was 15 GPU minutes, 10 TPU minutes and at most three
short proxy model requests. Server limits were 900 seconds for GPU, then 600/300/180
seconds for the TPU attempts. Those TPU limits were not simultaneous: each later
attempt followed observed early completion of the earlier one. The first two TPU
jobs were confirmed complete within about 115 seconds of submission combined; the
final one within another 78 seconds. This bounds their combined observed duration
to about 193 seconds, below ten minutes. The GPU job was confirmed complete within
151 seconds of submission. These bounds include polling delay, not exact billing.
The controller limits individual jobs; it does not enforce an aggregate account
budget or discover jobs in other state directories.

Quota readings moved from 29.25 to 29.23 GPU hours remaining, while TPU remained at
20.00 hours. These rounded account readings can lag; do not interpret the TPU value
as proof of zero usage. Model-proxy requests sent: zero.

The [artifact index](2026-09-06-kaggle-capabilities-artifacts.json) references exact
uploaded source, state, reports, receipts, GPU adapter and diagnostic/test evidence
in the private `scholzmx/sepalith-raw` bucket. Objects use content-hash keys and are
verified after download. Bucket storage itself is mutable. Local originals are
under `~/.local/state/sepalith/kaggle-capabilities-20260906/`.

Fifteen offline tests cover the existing launch/recovery contracts, profile resource
selection, blocked versus supported outcomes, interrupted preparation, proxy status
reporting and redirect rejection. Two controller defects were fixed: an empty
ancestor directory named `.git` no longer falsely blocks external state, and a
partially prepared profile cannot dispatch the default smoke payload.

## Practical next use

### Account access follow-up, 15:25 UTC

Read-only SDK quota calls report 46 minutes 24.618 seconds of GPU use out of
30 hours, with no reserved time. TPU use and reserved time are zero; the provider
also reports `has_ever_run=false` for TPU. This supports treating the earlier jobs
as failed attempts to obtain TPU compute. It does not establish why allocation
failed. Inspection of the installed CLI confirms that it passes both
`enable_tpu` and the requested `machine_shape` to `save_kernel`; neither field was
dropped by our submission wrapper. Further package-install retries have no clear
basis until a session exposes a real TPU.

The model-proxy quota endpoint succeeds and reports **$10 daily and $100 monthly**,
both unused. These are inference spend allowances, not hours. Token creation
still has the previously observed HTTP 403. The installed SDK documents phone
and Persona verification as requirements for local proxy tokens; we have not
verified the account's verification state, so this is a possible prerequisite
to inspect, not an established diagnosis. Quota visibility does not grant token
access. The next access check belongs in the signed-in Kaggle account interface.

The allowlisted response is archived as `access-followup.json` in the artifact
index. This follow-up submitted no jobs and made no model inference requests.
The local S1 supervisor, benchmark and separate CPU server were all observed
alive at this check. Migration still requires completed evaluation, verdict and
archive work, plus the queue manager's follow-on hold and candidate-scope ACK.

Use the proven T4 path for a modest, separately specified dense-model pilot or
embedding/evaluation task. Do not spend another session repeating the already
completed Anyscale LR sweep. Keep TPU training pending real device access, and keep
Benchmark-model work pending proxy authorization. Production recipe decisions and
the local runner cutover remain unchanged.
