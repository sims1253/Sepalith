# Kaggle free weekly GPU pool — mechanics, payload, verdicts (2026-09-06)

Companion to `2026-09-04-anyscale-sft-cloud-runbook.md` (the proven cloud-SFT
pattern this reuses). Verdict up front, split by layer:

- **Pipeline: PROVEN** (CPU batch end-to-end: repo delivery, private-HF
  egress, byte-exact data staging, under-attach guard exact — kernel
  `m0hawk/sepalith-cpu-smoke` v6, clean exit at T+153s).
- **GPU for the b4 GDN recipe: BLOCKED UPSTREAM** — Kaggle's whole GPU fleet
  is pre-Ampere (T4×2 / P100, no bf16), and unsloth 2026.8.18 has no
  working no-bf16 path for the qwen3_5 GDN hybrid (mixed BFloat16/Half
  casts; full failure ladder in §gpu-blocker). 9 GPU iterations, 0.75h of
  the 30h quota spent to pin this down; 29.25h banked.
- **LR sweep**: spec below, **retarget to Anyscale** (~$0.6/arm — the
  proven path; only `SFT_LR` differs, and that knob is now wired).
- **TPU**: bank unused (§TPU). **Benchmark credits**: real, but
  model-inference spend, NOT training compute (§credits).

## Mechanics (verified 2026-09-06, live against the account)

| thing | value | source |
|---|---|---|
| GPU types | T4×2 (16GB, sm75) or P100 (16GB, sm60) — **both pre-Ampere, NO bf16 anywhere on the fleet** | diag kernel + docs |
| TPU | Tpu1VmV38 class | machine_shape enum + account |
| Weekly quota | GPU 30h / TPU 20h; **live: 0.75h used, 29.25h remaining**; refresh 2026-09-12T00:00:00Z (Saturday midnight UTC, fixed) | `kaggle quota` (CLI 2.x) |
| Session caps | interactive CPU12/GPU9/TPU9h; "Save & Run All" (batch) CPU12/GPU12/TPU9h | docs/community |
| Quota consumption | wall-clock per session with accelerator attached; T4×2 pair counts as ONE session (2-arms-per-session = half quota/arm, documented option) | observed |
| Phone verification | required for GPU/TPU + internet — **account IS verified** (internet + GPU sessions both ran) | proven live |
| Image (script kernels) | Ubuntu, python 3.12.13, 4 vCPU, 31GiB RAM, 1.1TB /tmp, 20GB saved-output cap, root, ephemeral; **no working `python -m venv`** (ensurepip broken → system pip); Kaggle's PyPI pipe: 3GB pins in ~3.5 min | smoke probes |
| Dataset upload | API create/version; **tar.gz auto-extracted** (dataset content = repo tree); version re-ingest flip = 15-20 min worst case; `datasets status` LIES during processing (the files-listing sentinel is the real signal — cost us 3 iterations) | measured |
| Dataset mount layouts | THREE observed: `/kaggle/input/<slug>`, `<slug>/<slug>` (sentinel at root), and GPU workers: `/kaggle/input/datasets/<owner>/<slug>[/<slug>]` — bootstrap probes all + walk fallback | measured |
| Kernel push | 1.x CLI: no accelerator field (default lands P100!); **2.x CLI (2.2.4, py≥3.11) has `kernels push --accelerator`** (`NvidiaTeslaT4` = the T4×2 shape, `NvidiaTeslaP100`, `Tpu1VmV38`) and `kaggle quota` — 2.x lives at /tmp/k2venv (uv venv py3.12) | proven |
| Egress | internet-on sessions reach HF hub fine (byte-exact 930MB sft_v7 pull; 3.6GB base pull; unauth probe returns HTTP 401 = full TLS path) | smoke v5/v6 |
| unsloth on Kaggle | official unsloth notebooks target Kaggle T4×2 for STANDARD archs (fp16) — but NOT for the GDN hybrid (§gpu-blocker) | measured + docs |

## Auth state (m0hawk)

`~/.kaggle/kaggle.json` + `KAGGLE_API_TOKEN` (KGAT_) in `~/.zshrc`; CLI
1.7.4.5 system + 2.2.4 at /tmp/k2venv. Dataset/kernel push, internet
sessions, GPU sessions, and quota reads all verified live. No blockers.

## The "$100/month benchmark credits" — what they actually are

Real program, wrong mental model if you think "training compute": **Kaggle
Benchmarks** (kaggle.com/benchmarks; Google "Community Benchmarks" blog)
grants ~$10/day + up to $100/month of **model-inference spend** through the
Model Proxy (Gemini/Claude/DeepSeek class). Consumed by calling models in
benchmark/notebook workflows; NOT convertible to GPU time (community thread
#727128 literally asks for that conversion — doesn't exist). Useful to us
only for a future LLM-judge eval leg at zero marginal cost. The training
pool is 30 GPU-h/week, full stop.

## Payload (repo machinery, committed through the session)

`scripts/cloud/kaggle_sft_entry.sh` — Kaggle entrypoint (Anyscale
`sft_entry.sh` adapted): node probe → **T4 gate** (P100 aborts with the fix
— sm60 vs cu130 pins) → system-pip pins (`PINS=audit` CPU subset, incl. the
torchao 0.18 pin: the image ships 0.10.0 and peft 0.20's dispatcher
hard-fails <0.16) → `pull_data.py` (SFT_DATA_REPO/FILES) → `audit_targets.py`
guard → `train_sft.py` (STEPS/SFT_LR/bs-knobs; SFT_FP16 no-bf16 channel) →
final_lora to kernel output AND HF push-back.

`scripts/cloud/kaggle_push.sh` — driver: `git archive HEAD` → private
dataset `m0hawk/sepalith-repo` (REPO_SHA sentinel; 45-min flip poll;
SIGPIPE-safe checks — `grep -q` + pipefail + the CLI's stream = a silent
never-true trap; `KAGGLE_REPO_SHA` override to fire a staged tree) →
bootstrap kernel (env block carries HF_TOKEN from ~/.zshrc — never in the
repo, the Anyscale yaml convention) → GPU pushes via CLI 2.x
`--accelerator NvidiaTeslaT4`. `mode=gpu` needs `--fire`.

Trainer wiring (all default-OFF = banked byte-identical): `SFT_LR` knob
(default 2e-4); `audit_targets.py` `regex:` passthrough (comma-split lists
exact-match 0 modules for regex recipes); multi-GPU-safe shared-machine
guard (first nvidia-smi line — T4×2 emits one line per device); `SFT_FP16`
no-bf16 channel (see §gpu-blocker for its limits).

### Guard value provenance

`EXPECT_TRAINABLE=21823488` = the **banked b4 attachment** (96 modules: MLP
gate/up/down×24 + dense q/k/v/o×6; the b4 REGEX deliberately excludes the
GDN conv in-projs). Reproduced EXACTLY on Kaggle hardware (plain peft, CPU):
`AUDIT trainable_params=21,823,488 of 1,881,825,088 (1.16%) — PASS`.
(The TU2 templates' 33638400 is the explicit union-LIST attach incl. conv
in-projs — different geometry; b4-recipe arms use the regex + 21823488.)

## CPU smoke — PASS (the proven baseline)

`m0hawk/sepalith-cpu-smoke` v6, session exit T+153s, zero GPU quota:
bootstrap (tree @225ba32 via `/kaggle/input/...`) → pins 46s → **byte-exact
sft_v7 pull** (941,550,070 / 32,913,557 B — identical to the NAS files) →
**AUDIT PASS 21,823,488 exact**. This validates: repo packaging, dataset
versioning, token path, egress, data staging, and the B3 under-attach rule
on Kaggle hardware. The five earlier versions were packaging races/bugs
(attach-before-ready; auto-extract layouts; quoting trap; ensurepip;
torchao pin) — each fixed in the committed driver.

## GPU blocker — the full ladder (why the b4 recipe cannot run today)

Setup-side all PROVEN on GPU sessions: T4 allocated via 2.x accelerator
flag; full 3GB pins install (~3.5 min); data + audit exact; tokenizer +
trainer reach step 0. Then, at every attempt, the same crash at q_proj:

```
RuntimeError: expected mat1 and mat2 to have the same dtype,
             but got: c10::BFloat16 != c10::Half
```

The ladder (9 iterations, 0.75h GPU):
1. T4 has no bf16 → transformers hard-rejects the banked `bf16=True`
   ("You need Ampere+ GPU").
2. fp16 config + fp16 load: unsloth VETOES fp16 for qwen3_5 ("Using float16
   precision for qwen3_5 won't work! Using float32") and its own veto'd
   state collides with the config flag → same mixed-dtype crash.
3. Both precision flags OFF (let unsloth auto-fp32): same crash — its
   "Switching to float32 training" does not produce dtype-consistent
   tensors.
4. Purged the repo's 5090-compiled `unsloth_compiled_cache/` (compile-
   disable does NOT prevent cache USE — a landmine worth knowing): same
   crash → the mixed cast is in unsloth's RUNTIME patch path, not the cache.
5. The sanctioned pure-fp32 flow (`dtype=torch.float32` load → unsloth
   `_unsloth_user_float32`, + `UNSLOTH_FORCE_FLOAT32=1`): same crash.

Prime suspect for a follow-up session: the "**Will smartly offload
gradients to save VRAM!**" patch (fires only on low-VRAM cards — the 16GB
T4; never triggered on 24GB A10G/32GB 5090) casting LoRA/base leaves to
half while GDN block tensors stay bf16. Candidate next probes (NOT run,
~10 min GPU each): disable the offload (unsloth_zoo env, e.g.
`UNSLOTH_DISABLE_...` gradient-offload knob — exact name TBD in
unsloth_zoo source) or model.float() post-attach. Also worth one check
whether a newer unsloth (2026.9.2 is out) fixes the no-bf16 path.

P100 is no escape (sm60, same no-bf16 + our cu130 pins don't support it at
all). Plain transformers/TRL (no unsloth) measured 25 s/it on the 5090 for
this GDN base → ~T4-fp32 would be far over the 12h session cap for 300
steps. **Conclusion: Kaggle's GPU fleet cannot train the b4 GDN recipe
today.** Kaggle GPU remains good for: fp16-safe STANDARD archs (llama-class
dense bases — unsloth's whole T4 notebook catalog), and the CPU pipeline is
fully usable for smokes/audits/data work for free.

## LR sweep spec (PARKED — fire on queue word; recommend ANYSCALE)

Science: 4-point LoRA-LR sharpening of the b4 recipe; the 2e-4 anchor arm
doubles as a cross-platform normalization line.

- Arms: `SFT_LR ∈ {5e-5, 1e-4, 2e-4 (anchor), 4e-4}` × STEPS=300, everything
  else = banked b4 recipe (sft_v7 shuffle(42)+48k, seed 3407, cosine,
  warmup 0.03, r32/α64, regex targets, EXPECT_TRAINABLE=21823488,
  UNSLOTH_COMPILE_DISABLE=1, UNSLOTH_DISABLE_AUTO_PADDING_FREE=1).
  bs2×ga8 (B13-sanctioned effective-16 geometry — also the 24GB-A10G-safe
  default). RUN_NAME=kaggle-lr<val>-300 → `scholzmx/sepalith-lora`.
- **On Anyscale** (recommended, proven): 4× `job_tu2_*.yaml.example`-style
  yaml with MODEL=the staged b4 base, STEPS=300, SFT_LR=arm value,
  EXPECT_TRAINABLE=21823488 + the regex SFT_TARGETS (TU2 templates show the
  union-list variant — swap to the regex), g5.xlarge ≈ 3.2 s/it → ~21 min
  train/arm + ~6 min setup ≈ **$0.35-0.45/arm, ~$1.6 total, ~30 min wall
  parallel**. No new machinery needed.
- **On Kaggle**: blocked by §gpu-blocker until unsloth's no-bf16 path is
  fixed; if fixed, per-arm = ~15-20 min setup + 300 × T4-s/it (unmeasured;
  fp32 est 10-25 s/it → 50-125 min) ≈ 1.2-2.5h/arm ≈ 5-10 GPU-h for 4 arms
  — fits the 29.25h week. Fire command per arm (if/when unblocked):
  `KAGGLE_REPO_SHA=<staged> bash scripts/cloud/kaggle_push.sh sepalith-sft-lr5e-5 300 5e-5 gpu --fire`
- Verdict rule (pre-register at fire): eval_loss@300 + the standard
  noopFP/format battery on GGUF exports (local export_gguf.py from pushed
  adapters); anchor arm calibrates against the b4 rung's first-300-step log.

## TPU verdict — bank the 20 h unused

Our stack is CUDA-shaped end to end: the b4 recipe's speed DEPENDS on
unsloth's patched GDN kernels + fused CE (plain transformers/TRL measured
25 s/it on the 5090), and unsloth does not support TPU (torch-xla) at all;
PEFT/TRL-on-XLA plus paged 8-bit AdamW unavailability plus XLA recompiles
on variable-length SFT batches is exactly the yak-shave class the B4 saga
paid once and fenced out. The only TPU-shaped leg on the board (LOC1
embedder training, 47M-346M bi-encoders) is minutes-scale on the 5090 or
CPU-tier — not worth a port. 20 TPU-h/week stays banked; revisit only for a
future large-batch embedding-backfill or TPU-native serving eval. (And note:
the same GDN-on-nonstandard-accelerator dtype saga we just paid on T4 would
likely repeat on TPU with less community precedent to lean on.)

## Limits / open items

- GPU path for GDN: blocked upstream (ladder above); follow-up = gradient-
  offload knob bisect + retry on unsloth ≥2026.9.2 (both cheap, ~10 min GPU
  each), NOT tried under this session's smoke budget (0.75h spent).
- `SFT_FP16` current semantics = pure-fp32 flow (GDN-safe attempt). For a
  future llama-arch arm on T4 the right semantics is fp16 load + fp16=True
  (unsloth permits fp16 for standard archs) — adjust the knob then.
- 15-20 min dataset re-ingest per HEAD change: fire from one staged SHA
  (`KAGGLE_REPO_SHA`) or eat the wait; driver automates it.
- Concurrent-GPU-session cap unmeasured (irrelevant while GDN-blocked).
- Benchmarks inference credits: unspent, unusable for training; flagged for
  a future LLM-judge eval leg.

## APPENDIX (2026-09-06 15:41) — LR sweep FIRED on Anyscale: DONE, 4/4 SUCCESS

Executed the queue FIRE (~15:00 local): 4 arms, g5.xlarge, package @83b43b2
(includes the SFT_EVAL_STEPS=150 readout knob). Jobs: prodjob_ttgi…(5e-5),
y44u…(1e-4), 2w4m…(2e-4), qckk…(4e-4). All four: audit exact 21,823,488,
train to 300 clean, adapters pushed + hub-verified at
`scholzmx/sepalith-lora/{lr_sweep_5e5,lr_sweep_1e4,lr_sweep_2e4,lr_sweep_4e4}/final_lora`.

| arm | eval@150 | eval@300 | train@300 | wall (entry) |
|-----|----------|----------|-----------|--------------|
| 5e-5  | 1.255 | 1.249 | 1.114 | T+1904s |
| 1e-4  | 1.244 | 1.237 | 1.098 | T+1880s |
| 2e-4  | 1.238 | 1.226 | 1.084 | T+1877s |
| 4e-4  | 1.244 | 1.222 | 1.076 | T+1902s |

No divergence anywhere (kill rule never fired). Cost $2.26 total (~$0.57/arm
— above the $1.6 estimate: 2 eval passes + setup; ~$2.8 of the $10 trial
cap now used). Verdict sketch: banked 2e-4 anchor confirmed (4e-4 eval-ties
it at this horizon, 5e-5 clearly under-learns); production LR unchanged.
Ops traps hit + fixed: zsh no-word-split in the monitor loop (use ${=var} or
python); YAML double-quoted scalars eat `\b`/`\d` — single-quote regex env
values in anyscale yamls.
