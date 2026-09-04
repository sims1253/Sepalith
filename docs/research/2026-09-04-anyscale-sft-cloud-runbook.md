# Anyscale cloud SFT capability — proven runbook (2026-09-04)

Companion to `2026-08-30-anyscale-setup-and-smoke.md` (auth + CPU smoke #1).
This note closes that doc's follow-ups #1 (GPU ladder smoke, tok/s) and #2
(repo packaging decision), and records the working recipe for running a
B-series LoRA SFT rung in the cloud. Status: **capability PROVEN** (job
`prodjob_a95nwxqrq51zpyh8u1zyln2eec`, 60-step LoRA smoke trained to
completion on an A10G; total trial spend ≈ $0.55 of the $10 cap).

## Packaging decision (follow-up #2) — CLOSED: git-archive working_dir

Neither of the two candidate options; the third option the smoke-#1 loop
already proved, scaled up:

- The repo's **tracked** content is 323 files / 12 MB (`git ls-files | wc -l`;
  `git archive` zip). The 90 GB working tree is untracked local artifacts —
  which made "PAT clone vs baked image" look harder than it is.
- **Chosen: `git archive HEAD` → Anyscale `working_dir`** (12 MB upload,
  ~3 s). Zero new auth machinery, zero image-build time, secrets stay out.
  The GitHub repo being public (coordinator update) removes any residual
  argument for PAT plumbing; a plain `git clone https://github.com/sims1253/
  Sepalith.git` on the node is the documented fallback for ad-hoc debugging
  sessions, not needed for jobs.
- Baked image (the 2026-08-30 recommendation) is the RIGHT upgrade once
  there are many jobs (kills the ~20 s env install and ~30 MB uv-python
  download per job) — not worth build time at the current job volume.
  Cost of the per-job setup it would remove: ≈ 25 s ≈ $0.007.

Repo machinery (committed): `scripts/cloud/` —
`requirements-cloud-sft.txt` (exact pins), `sft_entry.sh` (timed
entrypoint), `pull_data.py` / `push_sft_v7.py` (HF data stage),
`report_toks.py` (post-run tok/s accounting).

## Data staging — sft_v7 on the private HF hub

`/mnt/h/sepalith/datasets/sft_v7` (930 MB: train.jsonl 941,550,070 B /
264,516 rows, eval.jsonl 32,913,557 B / 9,252 rows) →
`huggingface.co/datasets/scholzmx/sepalith-sft-v7` (private), gzipped
(153 MB + 3.8 MB; pigz locally, upload ≈ 3 min at ~10 MB/s up).

The trainer's row selection (shuffle seed 42 → first 48k) runs over the
full file, so the whole file must be staged — gzip keeps it cheap and
gunzip restores byte-identical jsonl (row selection therefore matches
local runs exactly). Cloud-side pull + gunzip: **~9 s** (HF Xet CDN on AWS).

## Environment (cloud .venv-sft equivalent)

Default Anyscale image (python 3.11.15, ray 2.58) + uv:
`uv python install 3.10` (standalone CPython with headers — see gotcha) →
venv → `uv pip install -r scripts/cloud/requirements-cloud-sft.txt`
(pins lifted from the live `.venv-sft`: torch 2.11.0+cu130, unsloth
2026.8.18, trl 0.24.0, transformers 5.5.0, datasets 4.3.0, peft 0.20.0,
accelerate 1.14.0, xformers 0.0.35, bitsandbytes 0.50.1, triton 3.6.0,
numpy 2.2.6, huggingface_hub 1.27.0, gguf 0.19.0).
Install: **~15 s** (AWS pipe to PyPI; the wheels' CUDA runtime ships as
nvidia-* deps — no extra index). WSL2-only knobs (allocator expandable
segments) dropped; the B4-safe recipe env kept verbatim
(`UNSLOTH_COMPILE_DISABLE=1`, `UNSLOTH_DISABLE_AUTO_PADDING_FREE=1`,
union `SFT_TARGETS`).

### Gotchas found (each cost one ~7-min attempt, see §burn)

1. Job user is `ray`, not root — nothing writable under `/root`.
2. The image has NO python3.10 dev headers AND restricted sudo; triton
   JIT-compiles `cuda_utils` (gcc `-I/usr/include/python3.10`) at first
   kernel launch → CalledProcessError on step 1. `uv venv --python 3.10`
   silently binds the header-less system 3.10. Fix (in `sft_entry.sh`):
   `uv python install 3.10` (standalone build bundles headers) + venv from
   that explicit interpreter + `C_INCLUDE_PATH` export + gcc syntax gate.
3. `Qwen/Qwen3.5-0.8B-Base` from the hub is the VL-capable class
   (Qwen3_5ForConditionalGeneration + qwen3_vl processor). Training is
   unaffected (union SFT_TARGETS match), but the post-train *diagnostic
   generation* misroutes FIM markers (`<[fim-suffix]>`) into the image
   processor → PIL ValueError. train_sft.py now guards the diagnostic
   (try/except, commit ced5909); training bytes unchanged. The repo's
   local B2 copy was text-stripped, so local runs never saw this.

## The smoke result (60 steps, Qwen3.5-0.8B-Base, sft_v7, banked recipe)

g5.xlarge (A10G 24 GB) on-demand ≈ **$1.006/h**:

| phase | wall | cum |
|---|---|---|
| submit → entrypoint start (node provision) | ~2 min | 2 min |
| venv + 3 GB wheels (uv) | 24 s | T+24s |
| data pull+gunzip 930 MB | 11 s | T+35s |
| model load + triton JIT → first optimizer step | ~60 s | ~3.5 min |
| 60 train steps (avg 4.75 s/it incl. warmup; steady 2.75-3.0 s/it) | 285 s | ~8 min |
| node auto-terminate after job | immediate | ~8.5 min total |

- **Time-to-ready (submit → first step): ≈ 3.5 min.**
- **Loss curve sanity**: 1.495 (step 20) → 1.32 (40) → 1.326 (60),
  grad_norm 0.92 → 0.67 — same shape/level as local B-rung early steps.
- **Speed vs 5090** (same script/data/model): 5090 steady 1.6-2.0 s/it
  (B2), A10G steady 2.75-3.0 s/it → **A10G ≈ 0.6-0.65× the 5090**.
  Content tok/s over the exact shuffle(42)+48k selection (mean 490.9
  tok/row × 16 rows = 7,854 tok/step): **A10G 2,746 tok/s steady**
  (1,655 incl. JIT warmup) vs **5090 ≈ 4,050-4,360 tok/s** (B2 log).
  (The 48k tok/s figure in the 08-30 doc is the A2-ladder packed-throughput
  reference, a different geometry — not directly comparable.)
- The FLA "Triton is not supported" warning seen when headers were broken
  disappears once triton can compile; no naive-GDN-kernel collapse
  (that would show ~25 s/it; we saw 2.75-3.0).

## Repeat a real rung (exact commands)

One-time (done): data staged to `scholzmx/sepalith-sft-v7`; cloud scripts
committed under `scripts/cloud/`.

```bash
# 1) package (from repo root, after committing any rung-specific changes)
rm -rf /tmp/anyscale_sft/pkg && mkdir -p /tmp/anyscale_sft/pkg
git archive HEAD | tar -x -C /tmp/anyscale_sft/pkg

# 2) job yaml (token ONLY here, never in the repo)
eval "$(grep -E '^export HF_TOKEN=' ~/.zshrc)"
cat > /tmp/anyscale_sft/job_rung.yaml <<EOF
name: sepalith-sft-b13
entrypoint: bash scripts/cloud/sft_entry.sh
working_dir: /tmp/anyscale_sft/pkg
max_retries: 0
env_vars:
  HF_TOKEN: "${HF_TOKEN}"
  MODEL: "LiquidAI/LFM2.5-2.6B-Base"     # or any hub base
  STEPS: "3000"
  SFT_TARGETS: "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_a,in_proj_b,in_proj_z,out_proj,gate_proj,up_proj,down_proj"
  UNSLOTH_COMPILE_DISABLE: "1"
  UNSLOTH_DISABLE_AUTO_PADDING_FREE: "1"
compute_config:
  head_node:
    instance_type: g5.xlarge            # A10G 24GB $1.006/h
  worker_nodes: []
EOF
# NOTE: SFT_TARGETS above is the union set; for llama-arch bases use the
# default (omit), for LFM2 use the LFM set (out_proj/in_proj/w1-3 per the
# base-candidate recon). SFT_PD_BATCH/SFT_GRAD_ACCUM env knobs work too.

# 3) submit + watch (full driver log is the source of truth)
cd /tmp/anyscale_sft && anyscale job submit -f job_rung.yaml
anyscale job logs --id <prodjob_...>    # CLOUD T+ stamps, loss, runtime

# 4) artifacts come back manually if needed (final_lora stays on the dead
#    node's /tmp) — for rungs that must export GGUF, add the export to the
#    entrypoint and `anyscale job logs`/result-download; or run the export
#    locally from the pushed adapter. (Open item, see §limits.)
```

## Burn-rate table (Anyscale on-demand estimates)

| use class | local 5090 wall | cloud est. | Anyscale cost | note |
|---|---|---|---|---|
| 60-step smoke (this run) | — | 8.5 min node | **≈ $0.15** | measured |
| B7-class rung (1.2B, 3000 steps) | ~1.7 h | ~2.7 h | **≈ $2.7** | 1.94 s/it ÷ 0.63 |
| B13-class rung (LFM2.5-2.6B, 3000 steps) | 4-5 h | ~6.5-8 h | **≈ $6.5-8** | board estimate 4-5h ÷ 0.63; setup +$0.01 |
| GRPO 300-step arm (rl_smoke.py class) | ~1.7 h | ~2.5-3 h | **≈ $2.5-3** | rl_grpo_v3 anchor: 50 steps/16 min; gen-bound, 0.6× assumed |
| H100 FP8 W1 cell (~0.5 BT) | n/a | ~4.5 h @30k tok/s | **$57 on Anyscale H100 ($12.3/GPU-h) — use the A2 rental at $1.47/h ≈ $7 instead** | Anyscale per-H100 pricing unverified; W1 is the A2 runbook's own checklist item |

Where the $100 goes furthest: rungs + GRPO arms on A10G/L4 consume <$15
combined; W1-class FP8 work does NOT belong on Anyscale credits (8× the
A2 rental price); keep ≈ $85 reserved for the production fine-tune
(≈ 85 A10G-hours ≈ 12× B13-class rungs, or fewer on bigger nodes).

## Limits / open items

- final_lora artifacts on the node die with it — jobs needing exports must
  push the adapter (HF hub, same pattern as the dataset) from the
  entrypoint before exit. Not wired yet; trivial add.
- L4 (g6.xlarge ≈ $0.80/h) untested — likely ~15-20% cheaper per token
  than A10G; one 60-step smoke would pin it.
- nproc reads 1 in the job cgroup (cosmetic, matches smoke-#1 note).
- Anyscale H100 availability/pricing on this cloud not verified.
