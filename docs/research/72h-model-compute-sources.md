# 72-hour model and compute source review

**As of:** 2026-09-11 (Europe/Berlin)
**Scope:** the best feasible base model, post-training substrate, serving path,
and accelerator allocation for the Tuesday morning release target.

This note separates upstream facts, measurements in this repository, estimates,
and account-specific observations. It does not start jobs, spend credits, create
accounts, or download weights.

## Decision

Keep **Qwen3.5-2B-Base, the text-only local GDN checkpoint used by B4**, as the
production substrate. The repository's B-series places B4 at the best quality /
CPU-decode tradeoff: 85.1% valid, 76.5% exact, 67.4% no-op false-positive rate,
and 19.2 tokens/s on the contended t8 CPU battery. It is statistically tied with
the Spark and Granite leaders, while decoding faster than both. The latest
O2-filtered GRPO arm is the strongest trained candidate (255/307 exact versus
242/307 for the B4 bank, paired p=0.0024), but its felt-preference column is
underpowered (78/200 valid comparisons, p=0.14), so it should remain a candidate
until that column is rerun or the product owner accepts the uncertainty.

The base choice should not change during this 72-hour window. A new base would
consume an entire SFT, export, and evaluation cycle, and none of the plausible
alternatives has a matched Sepalith result that beats B4 on the product axis.

## Base candidates

| Candidate | Upstream facts | Repository evidence and fit | Action |
|---|---|---|---|
| **Qwen3.5-2B-Base** | The official card describes a pre-training-only 2B model, Apache-2.0 licensed, with 24 layers and a hybrid Gated DeltaNet / gated-attention layout. It is intended for fine-tuning and lists Transformers, vLLM, and SGLang compatibility. [Model card](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | B4 is 85.1 valid / 76.5 exact / 67.4 no-opFP at 19.2 t/s. The local copy is text-stripped; the public card is multimodal, so do not silently substitute a fresh VL checkout for the tested local files. | **Use for the final SFT and RL chain.** Preserve the B4 target regex and attachment audit. |
| **Granite 4.1 3B Base** | IBM's card describes a decoder-only 3B Apache-2.0 model intended for broad text generation and FIM code completion, with 131,072 context and BF16 weights. [Model card](https://huggingface.co/ibm-granite/granite-4.1-3b-base) | B5 is 87.8 valid / 78.0 exact / 68.2 no-opFP, but only 10.65 t/s. Its +2.7pp valid lead over B4 is not significant (paired p=0.167). It has the best format-propagation readout (79.1%), which supports its use as a control for native-FIM effects. | **Do not switch.** It is the only scientifically motivated control if one extra local experiment is affordable; it has already answered the main base screen. |
| **Spark-X2.5-1.7B** | The repository's tested arm is a hybrid SWA model; no new upstream investigation is needed for this release decision. | 85.1 valid / 77.3 exact / 67.8 no-opFP at 17.4 t/s, trained through the TRL path. It ties B4 on paired validity (12/12 discordant rows, p=1.0), so the apparent early lead was a size/trainer confound. | **Challenger only.** The production path remains the better measured decode tradeoff. |
| **LFM2.5-2.6B-Base** | The repository's license review records a revenue cap; treat this as a licensing constraint requiring separate review. | 87.1 valid / 76.9 exact, but 99.0% scored no-opFP and 6.60 t/s. It is proposal-happy and too slow for the CPU product tier. | **Eliminate for this release.** Do not spend the cloud budget on its 4–8 hour arm. |
| **StarCoder2-3B** | BigCode reports 17 programming languages, 16,384 context with 4,096-token sliding attention, and pretraining with FIM on more than 3T tokens. The model is not instruction-tuned and uses the BigCode OpenRAIL-M license. [Model card](https://huggingface.co/bigcode/starcoder2-3b) | FIM pretraining is attractive, but there is no matched R-edit battery, no tested converter/marker path in this repository, and the license is less simple than B4's Apache-2.0 base. | **Do not screen in this window.** It is a future control, not evidence for a Tuesday switch. |
| **CodeGemma-2B** | Google's official documentation says the 2B pretrained model targets code completion and that CodeGemma uses four FIM control tokens: `fim_prefix`, `fim_suffix`, `fim_middle`, and `file_separator`. [FIM format](https://ai.google.dev/gemma/docs/codegemma/prompt-structure), [model overview](https://ai.google.dev/gemma/docs/codegemma) | The native FIM vocabulary is a good hypothesis, but the model uses the Gemma license/access path and has no Sepalith R-edit result. Changing from the trained Zeta-2 markers would require a new data/render and serving battery. | **Park.** The native-FIM advantage is already tested more cheaply by the Granite control. |
| **MiniCPM5-2B / NeoHorse** | These are local challenger probes rather than new external options. | The released MiniCPM5-2B checkpoint scored 13/255 valid and 0/255 exact, with 48.5% generation-limit rate and worse no-op restraint. NeoHorse was 7.87 t/s versus B4's 17.82 t/s and was killed at the CPU product gate. | **Closed for this release generation.** |

The B-series also shows that larger capacity does not repair the universal
`doc_sync` 0/15 result. That is a data/construction issue, so changing bases is
not the high-value response to that failure.

## FIM and serving compatibility

The local packaging contract is llama.cpp **b10453**, commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`, with a CUDA-enabled build for GPU
serving and the Q8_0 export as the quality-preserving default. The repository's
[serving and packaging contract](../SERVING-PACKAGING.md) is the authority for
the pinned converter, quantizer, readiness probe, and manifest checks.

Current upstream llama.cpp documents `POST /infill` with `input_prefix`,
`input_suffix`, `input_extra`, and `prompt`; it can assemble a native
`FIM_PRE / FIM_SUF / FIM_MID` sequence and has an SPM-infill option. [Server
infill documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
The current source also names Qwen3.5 in the model architecture and conversion
code ([architecture mapping](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp),
[Qwen converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py)).
That upstream fact does not prove compatibility with the pinned b10453 build.
The Sepalith protocol uses its Zeta-2 marker strings and repository-specific
context ordering; `/infill` must not be treated as a drop-in replacement for
the trained prompt format. Any new model would need the full held-out battery,
including marker compliance, no-op restraint, and latency, before promotion.

The latest serving evidence supports `-b 256` for the CPU tier: suffix-append
cache median was 89.6% at 256 versus 79.4% at 512 and 79.3% at 1024. The
imatrix-Q4 arm failed its non-inferiority bound, so keep Q8_0 while the release
model is chosen.

## Compute options

| Resource | Observed or documented facts | 72-hour implication |
|---|---|---|
| **Local RTX 5090** | 32 GB VRAM, 47 GB system RAM, WSL2. The B4 family runs through the tested Unsloth-with-knobs path. A 3,000-step B8b run took about 1h46; typical B4 peak was 21.6 GB, with a 32.1 GB transient seen on one run. `bs=2`, gradient accumulation 8 preserves effective-16 optimizer math and is the safer fallback. WSL2 allows one CUDA workload at a time. Root has only about 8.6 GB free; the coordinator reports about 1.8 TB free on `/mnt/e`. | **Primary final-model lane.** Keep trainer, evaluator, and export serial. Put checkpoints and temporary merged models on `/mnt/e` or the existing NAS path; do not duplicate multi-GB weights on root. |
| **CPU hosts / AMD notebook** | The SSH host has useful system RAM but no measured accelerator path for this recipe. The notebook's mobile AMD GPU has no validated CUDA or ROCm training path. CPU hosts remain useful for exports, server smoke tests, and eval orchestration. | Use the SSH host for storage/CPU work and keep the final training on the 5090. A bounded Vulkan smoke on the notebook is acceptable only if it does not take a 5090 window; do not choose a larger base merely because another host has more RAM. Serve one validated ~2B Q8_0 checkpoint from the local workspace. |
| **Kaggle T4x2** | Kaggle documents weekly accelerator quotas and reset behavior ([GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage), [notebooks](https://www.kaggle.com/docs/notebooks)). The live account observation is 22.77 GPU-hours used, 7.23 hours remaining of 30, and reset `2026-09-12T00:00` (timezone suffix not shown by the CLI). The account has T4x2/P100 access; those are pre-Ampere and have no BF16. The offload-disable patch now makes B4 fp32 training clean: a 300-step T4x2 run took about 98 minutes. At the matched 300-step budget, Kaggle scored 66/307 exact versus 161/307 on the local BF16 reference; this is a measured short-budget platform gap, not a universal conversion factor. | **Do not use for the final short arm.** The 30-hour pool is suitable for one full 3,000-step diagnostic/challenger after reset (about 16 hours estimated) or several smokes. A full-length run is required before treating a Kaggle adapter as a candidate; the current 7.23 hours cannot finish one. The earlier board entry saying the quota was exhausted is superseded by the latest CLI read. |
| **Anyscale A10G** | The repository proved a g5.xlarge A10G job: first step in about 3.5 minutes, 60 steps in about 8.5 minutes, steady 2.75–3.0 s/step, and about $0.55 against the then-current $10 account cap. The current public pricing page advertises pay-as-you-go, a $100 credit offer, and hosted rates of $0.5682/h T4, $0.9542/h L4, and $1.3635/h A10G. [Pricing](https://www.anyscale.com/pricing), [jobs](https://docs.anyscale.com/jobs) | A B4 3,000-step run is roughly 2.3–3.0 hours of training by extrapolation, plus setup/export; at the observed A10G rate this is roughly $3–$4. The user reports about $95 of Anyscale credit, but that balance, expiry, and current rate are **unverified**. The repository's current cloud-admission note records that provider adapters and a bounded lifecycle test are not implemented; raw paid jobs are not an admission path. Keep this as a contingency only if the owner separately verifies admission and credit. |
| **Azure GPU** | Eligible new Azure free accounts receive $200 for 30 days, with verification and a move to pay-as-you-go required beyond the free period or credit. [Azure account](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account) The NCasT4_v3 family offers one to four 16-GB T4 GPUs. [NCasT4_v3](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series) Azure checks subscription quota and regional capacity separately, and either can block deployment. [Quota documentation](https://learn.microsoft.com/en-us/azure/virtual-machines/quotas) | The user reports about €170 of Azure credit, but the balance, GPU quota, region capacity, and credit-only billing protection are **unverified**. A 16-GB T4 is below the observed B4 transient footprint; no B4 training recipe is validated there. The repository's cloud-admission note also records that the Azure adapter and PAYG protection are unverified, so do not use a raw VM command as a critical path. |
| **Other free services** | No account-specific entitlement or stable GPU quota has been verified. Free offers change and often require identity/card checks, quota requests, or environment work. | **Skip new signups.** They have lower expected value than finishing the tested local chain. |

The Anyscale public offer, the repository's old `$10` account cap, and the user's
reported `$95` balance refer to different dates/account states. Treat none as
available budget until the account and the repository's cloud-admission boundary
are checked. The same applies to the reported Azure balance. Kaggle's documented
quota policy does not override the live account counter.

## Recommended allocation through Tuesday

1. **Run the final candidate on the local 5090.** Start from the B4 local text-only
   base and preserve the proven data split, target audit, optimizer geometry, and
   CUDA serialization rules. The O2-filtered GRPO adapter is the first candidate
   to finish the remaining preference check; if it fails that check, retain the
   B4-derived arm.
2. **Spend the next local windows on decision columns and packaging.** Complete
   preference/intent evidence, forgetting BPB, the full 307-row scenario battery,
   no-op restraint, Q8_0 export, and the pinned llama.cpp CPU smoke. These tests
   have higher expected value than another base screen.
3. **After the Kaggle reset, use T4x2 only for a pre-registered full-length
   replication or a diagnostic that can tolerate fp32/platform loss.** Do not
   infer a production result from a 300-step adapter. Keep the live quota counter
   as the source of truth immediately before firing.
4. **Keep cloud execution off the critical path.** The historical Anyscale
   `git archive` job is a proven capability smoke, but the current repository
   has no real provider adapter or bounded cloud lifecycle admission. If the
   owner later verifies existing credit and explicitly admits one bounded job,
   A10G is the first candidate; otherwise do not submit a raw Anyscale or Azure
   command and do not spend the release window on account creation.
5. **Do not start StarCoder2, CodeGemma, MiniCPM, NeoHorse, or a fresh Granite
   SFT for this release.** Their hypotheses are either already covered by the
   B-series/native-FIM control or lack the matched battery and serving path needed
   to beat a known B4/O2 candidate.

### Source and status notes

- Upstream links above were checked on 2026-09-11. Model-card facts are not
  claims about Sepalith quality.
- Repository measurements come from `experiments/training/base_bakeoff/RESULTS.md`,
  `docs/SERVING-PACKAGING.md`, `docs/research/2026-09-06-kaggle-compute-integration.md`,
  `docs/research/2026-09-04-anyscale-sft-cloud-runbook.md`, and the latest
  queue-owner validation receipts (`o2f1800`, the LR/platform decomposition,
  challenger screens, and Q4 packaging). The queue-owner board is newer than
  the older Kaggle integration note; the offload-disable result supersedes its
  original “GDN blocked” conclusion.
- `docs/CLOUD-BUDGET-ADMISSION.md` is the current operational boundary when it
  is present in the coordinating worktree: provider adapters, credit evidence,
  and bounded lifecycle validation are prerequisites for paid cloud dispatch.
- Live account observations are volatile and were read-only. No signup, charge,
  cloud submission, weight download, or external write was performed for this
  review.
