# LoRA SFT on Kaggle T4s: feasibility probe, 2026-10-16

**Question.** Kaggle's free T4s (2 x 16 GB, no native bf16, 30 GPU-h per
week, 12-hour sessions) cannot run the full-weight SFT recipe. Can they run
LoRA SFT of the parent for experiments?

**Answer.** Yes. LoRA SFT of checkpoint 11,586 on the real SFT v2 rows
fits easily in fp16 and stays numerically stable. The fastest tested setup
(both T4s with DDP, micro-batch 4) needs about **0.8 h for a 268-update pilot**
and about **3.5 h for a full 1,071-update arm**. Use the results to rank
data and mixture choices, with the caveats below. Do not use them to choose
learning rates or optimizer settings, and do not use QLoRA.

## What ran

| Kernel (private) | Configs | Result | GPU-h |
| --- | --- | --- | ---: |
| `m0hawk/sepalith-lora-probe-20261016` | r16, r64, r16-gc2048, q4-r16, ddp:r16 | Only q4-r16 ran. The other configs hit the torchao blocker below. | 0.22 |
| `m0hawk/sepalith-lora-probe-20261016-r2` | r16, r64, r16-gc2048, r16-mb4, ddp:r16 | All ran except r16-gc2048 (out of memory) | 0.46 |
| `m0hawk/sepalith-lora-probe-20261016-r3` | ddp:r16-mb4 (40 updates) | Ran | 0.17 |

The probe used **0.85 GPU-h** of its 1.5 h budget. The quota went from 0.39
to 1.24 h used this week. All three runs are recorded in
`~/.local/state/sepalith/kaggle/quota-ledger.jsonl`. Per-update records are in
[2026-10-16-kaggle-lora-probe.json](2026-10-16-kaggle-lora-probe.json); the raw
outputs are under `~/.local/state/sepalith/kaggle/runs/probe-20261016*/`.

**Inputs.** All three are public on HF with their sha256 pinned and checked
in the kernel:

- the parent `scholzmx/sepalith-2b-cpt/checkpoint-11586`
  (`4bf617f9...`);
- the exact SFT v2 TRAIN rows,
  `campaign-20260915/sft/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl`
  (`ed1c7b4e...`, the bound recipe's `cohort.rows`);
- the SFT v2 draw schedule `schedule-v2/draw-schedule-v2.json` (`6820a847...`).

Every row used was checked to have `split == "train"`. Nothing new was
uploaded. No DEV data and nothing from the sealed final set was read.

**Recipe.** Each optimizer update is one schedule update: 16 rows, 13 edit
plus 3 no-op, bucketed by length as in SFT v2. The loss is prompt-masked
(labels from `target_start`, as in `campaign_edit_data`), summed over the
target tokens and divided by the update's target-token count. A CPU test
showed this equals HF's `labels` loss. The output head runs only on target
positions. No packing.

The probe sampled 40 updates evenly across the 1,071, with the update holding
the longest row (3,064 tokens) first so that out-of-memory errors show up
early. Hyperparameters:

- base weights: fp16, or NF4 for QLoRA;
- LoRA adapters: fp32 master weights on q, k, v, o, gate, up and down;
  alpha = r, no dropout;
- forward pass under fp16 autocast with `torch.amp.GradScaler`;
- non-reentrant gradient checkpointing;
- AdamW at 1e-4 with 5 warmup updates, gradient clipping at 1.0.

**Image.** Python 3.13, torch 2.11.0+cu128, transformers 5.5.0, peft 0.20.0,
bitsandbytes 0.50.2, 4 CPUs, 31 GiB host RAM, and 14.56 GiB usable per T4.

## Results

| Config | Peak alloc. GiB per GPU | Tokens/s | Pilot 268 (h) | Arm 1,071 (h) |
| --- | ---: | ---: | ---: | ---: |
| r16, 1 T4, micro-batch 1 | 7.67 | 457 | 1.8 to 2.1 | 7.7 to 8.9 |
| r64, 1 T4, micro-batch 1 | 7.96 | 432 | 1.9 to 2.2 | 8.2 to 9.4 |
| r16, 1 T4, micro-batch 4 | 8.74 | 640 | 1.3 to 1.5 | 5.6 to 6.7 |
| QLoRA NF4 r16, 1 T4 | 4.91 | 428 | 1.9 to 2.3 | 8.4 to 9.7 |
| r16, DDP 2 T4, micro-batch 1 | 7.67 / 6.98 | 818 | 1.16 | 4.9 to 5.05 |
| **r16, DDP 2 T4, micro-batch 4** | **8.74 / 8.30** | **1,174** | **0.81** | **3.45 to 3.48** |
| r16, no checkpointing for rows up to 2,048 tokens | out of memory at 13.86 | n/a | n/a | n/a |

How to read the table:

- **Ranges.** The low end comes from per-bucket mean seconds per update times
  the schedule's bucket counts. The high end comes from a least-squares fit
  of seconds against tokens and squared lengths, applied to every scheduled
  update. The single-T4 runs measured only 13 to 19 updates within their
  480 s cap and had no warm 4,096-bucket update, so trust their fit value
  more.
- **Timing basis.** Times are per optimizer update at effective batch 16 and
  exclude the warm-up update. Session setup (pip, 5 GB model download, plan)
  adds about 1 to 1.5 minutes.
- **Per bucket, best config** (DDP, micro-batch 4): 3.7 s for the 512 bucket,
  7.7 s for 1,024, 20.9 s for 2,048 and 29 s for 4,096.
- **PC reference.** The full-weight bf16 run on the 5090 takes about 12.5 s
  per update: 74 minutes to gate 268 including overheads, about 3.7 h for an
  arm (C07 status). The best T4 LoRA setup therefore runs at about the PC's
  wall-clock speed per arm.

**Numerical stability.** 128 measured updates across 6 configs gave these
results:

- zero non-finite micro-batch losses and zero GradScaler skips;
- the scale stayed at 65,536 throughout;
- gradient norms stayed between 0.25 and 3.1.

r16 on one T4, r16 with micro-batch 4, and both DDP variants produced the
same per-update losses to about 4 decimals on identical updates. DDP and
micro-batching therefore do not change the computation. On the same updates,
r64 reached lower loss than r16.

**QLoRA degrades the base.** Its first-update loss was 0.883, against 0.801
for the fp16 base on the same rows and weights. It stayed about 0.06 to
0.08 nats per target token higher throughout. It is not faster on a T4
either, because dequantization costs as much as the memory saving buys. The
fp16 base fits with about 6 GiB to spare, so QLoRA has no use here.

## Blockers hit, and the old Qwen3.5 crash

- **torchao (attempt 1).** Kaggle's image ships torchao 0.10.0. When peft
  0.20.0 wraps any plain `nn.Linear`, it calls `is_torchao_available()`,
  which *raises* `ImportError` for torchao below 0.16 instead of returning
  False. That killed every fp16-base config. QLoRA got through only because
  the bnb dispatcher matches first. Fix: `lora_probe_entry.py` uninstalls
  torchao in the session, since nothing here uses it.
- **No mixed-dtype crash.** The September Qwen3.5 attempts failed inside
  Unsloth's runtime patch path for the gated-DeltaNet hybrid. Unsloth
  refused fp16 for `qwen3_5`, and its patches left GDN tensors in bf16 while
  the rest ran in fp16, giving `BFloat16 != Half` at `q_proj`. Its pure-fp32
  fallback crashed the same way. None of that applies here: the model is a
  dense `LlamaForCausalLM`, and the probe uses plain transformers and peft
  without Unsloth. The `unsloth_version` field in the parent's config is
  inert.
- **bf16 trap.** `torch.cuda.is_bf16_supported()` returns True on the T4
  because it counts emulation. Do not let code select bf16 based on that
  call.

## Weekly capacity

The Kaggle quota counts session hours, not GPUs. The two efficient ways to
use a session cost about the same:

- **DDP, micro-batch 4:** about 0.85 GPU-h per pilot and about 3.5 GPU-h
  per arm. Fastest turnaround.
- **Two independent micro-batch-4 runs, one per T4:** two pilots in about
  1.5 h (about 0.78 GPU-h each), or two arms in about 6.5 h (about 3.3 GPU-h
  each). Slightly cheaper, and both fit a 12-hour session.

Experiments per week:

| Budget | Training only | Including gate evaluation |
| --- | --- | --- |
| 20 GPU-h (the R7 stop line) | about 23 pilots, or 5 to 6 arms | about 13 pilots, or about 3 arms |
| 30 GPU-h | about 35 pilots, or about 8 arms | |

The evaluation column assumes these costs:

- DEV75 in fp16 measured 0.21 GPU-h in C03; SFT models stop earlier than
  the parent, so this is a ceiling.
- DEV250 is estimated at about 0.5 GPU-h and was not measured.
- So a pilot plus its evaluation costs about 1.5 GPU-h.

The C12 census competes for the same quota: C03 projected it at 67 to 133
session-hours.

Untested speed-ups that are likely to help: a larger micro-batch for the 512
and 1,024 buckets, and turning off checkpointing only for rows up to 1,024
tokens. Turning it off up to 2,048 tokens ran out of memory.

## Usefulness for ranking, and risks

**Useful for** coarse, relative comparisons where every arm gets the same
LoRA treatment. Examples are data admission (roxygen arm B against arm A),
no-op share or exposure caps, and family mixture. The comparison is between
LoRA arms in fp16 on Kaggle, never against PC numbers.

**Not useful for** choosing learning rate, schedule or optimizer for the
full-weight recipe. The full-weight recipe uses aurora_mix at 1e-5 with
fixed-RMS updates. LoRA uses AdamW at about 1e-4 on a low-rank subspace.
Neither the optimal values nor the forgetting dynamics transfer.

Main risks:

1. **The LoRA-to-full-weight ranking is unproven.** Before trusting a
   ranking, calibrate once: reproduce a known PC comparison, such as R3
   (arm A against arm B) once both exist, or arm A's gate-268 result
   against the parent, with LoRA at the same data order.
2. **Precision drift.** In C03, fp16 inference was only 57% token-identical
   to the PC's bf16 on DEV75 (cap +3). Kaggle gate counts are therefore not
   comparable to the PC's R1 counts. Compare within Kaggle only.
3. **The DEV rules.** Ranking experiments on DEV75 or DEV250 would be a new
   DEV-based decision, which RULES.md does not define ("never tune on DEV
   beyond the decisions these rules define"). The user must pre-register
   such a rule. The alternative is to rank by loss on a held-out slice of
   TRAIN, which costs that slice as training data.
4. **Quota contention** with the C12 census and other inference, and the R7
   stop at 20 h.
5. **Operations.** The image is unpinned (the torchao break shows the risk).
   The probe has no checkpoint or resume, although a 3.5 h DDP arm fits one
   session. Gate evaluation needs the adapter merged into fp16 weights,
   which is not built yet.
6. **QLoRA perturbs the base** by about 0.07 nats per token; use the fp16
   base.

**Recommendation.** Adopt LoRA on Kaggle (fp16 base, fp32 adapters, r16 to
r64, DDP over both T4s with micro-batch 4) as a secondary experiment lane
for data and mixture questions. Before any ranking from it informs a PC
decision, the user should approve two things: one calibration pair against a
known full-weight result, and an explicit rule saying which panel or metric
ranks the LoRA arms.

## Files

- `scripts/kaggle/lora_probe_entry.py`: kernel entry. It handles the plan,
  per-GPU workers, DDP over torchrun, and extrapolation. `--selftest-tiny`
  runs the same logic on CPU with a tiny random Llama.
- `scripts/kaggle/lora_probe_push.py`: a thin wrapper over `infer_push.py`.
  It keeps the same bootstrap, code digest, secret check, R7 quota check and
  ledger, and uses the kernel slug `sepalith-lora-<run-id>`.

Example:

`python3 scripts/kaggle/lora_probe_push.py --run-id probe-20261016-r3 --estimate-hours 0.3 --timeout-hours 0.5 -- --phases ddp:r16-mb4 --updates 40 --ddp-updates 40 --config-seconds 600`

No CUDA work ran on the PC. The logic tests ran on CPU with
`CUDA_VISIBLE_DEVICES=""` while the night job ran.
