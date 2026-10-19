# Kaggle inference (card C03)

Batch generation for Sepalith checkpoints on Kaggle's free T4s, so the PC's
GPU stays free for training. The census (C12) and the RFT/DPO sample
generation (C13) use it. Kaggle has no bf16, so runs use fp16 (or fp32); the
parity check in `docs/training/plan-20260927/c03/` compares them with the PC.

| File | Runs on | Purpose |
| --- | --- | --- |
| `infer_push.py` | PC | Builds and pushes the kernel (T4 accelerator), polls it, fetches the outputs, records quota (R7), optionally uploads the outputs to HF |
| `infer_entry.py` | Kaggle (or any CUDA host) | Downloads the model and inputs, splits the work across the GPUs, generates, merges, writes the outputs |
| `lora_probe_push.py`, `lora_probe_entry.py` | PC, Kaggle | LoRA-SFT feasibility probe on T4s (research, not a plan card); see `docs/validation/2026-10-16-kaggle-lora-probe.md` |

## Engine

`transformers` generation through the evaluator helpers in
`sepalith.training.eval.campaign_eval`. vLLM was not used: the PC gates and the
baseline are bound to the evaluator's batching (batch 32, prompts sorted by
prompt length descending then case id, left padding with pad id 1, dynamic KV
cache), and the night-1 measurements showed that batch composition alone moves
bf16 outputs (batch 1 vs 32: 54% identical on the parent). Only the same
batching on both sides gives a like-for-like parity check, and vLLM's
continuous batching and paged attention cannot reproduce it. If C12's
throughput turns out to be the bottleneck, vLLM can be evaluated for the
census separately (it would need its own parity check).

## Inputs

Sources use a scheme prefix:

- `hf:<owner>/<repo>/<folder>[@rev]`: a model folder in a public HF model
  repository, e.g. `hf:scholzmx/sepalith-2b-cpt/checkpoint-11586`.
- `hf-dataset:<owner>/<repo>/<path>[@rev]`: a file in a public HF dataset.
- `kaggle:<owner>/<slug>[/<subdir>]`: an attached Kaggle dataset; the driver
  adds it to the kernel's `dataset_sources`. Use this for weights that are
  not public on HF (private Kaggle datasets stay inside the account).
- `path:/abs/path`: a local path (PC tests, or Kaggle paths you know).

Work is either a DEV panel (`--panel dev75|dev250`, through
`development_evaluator`, so the outputs carry the evaluator's classification
and the R1 counts; the panel's sha256 is pinned) or a pre-tokenized prompts
file (`--prompts`, JSONL with `id`, `prompt_token_ids` and optional `split`,
`family`, `package_id`, `metadata`; rows with a split other than `train` or
`dev` are refused). Never pass the sealed final set (`DAT-07-final-*`) or
anything derived from it.

## Work split

The global batches are formed once, exactly as the PC evaluator forms them.
Batch `b` goes to session shard `b % --shard-count`, and within a shard its
`j`-th batch goes to worker `j % --workers`. Each batch therefore has the same
cases and padding as on the PC, whatever the split. `--workers 2` runs both
T4s as two independent processes (one GPU and one model copy each); the
quota counts the session, not the GPUs. `--shard-index/--shard-count` split
long jobs across sessions (12-hour limit).

## Outputs

In `/kaggle/working/<run-id>/`, fetched to
`~/.local/state/sepalith/kaggle/runs/<run-id>/<run-id>/`:

- `generations.jsonl`: one record per prompt and sample with `generated_ids`,
  `text` (decoded without the stop token), `finish_reason` (`stop` or
  `length`), `stop_token_id`, `cap_hit`, `prompt_tokens`, `prompt_sha256`,
  `batch_index`, `worker`.
- `summary.json`: inputs and their hashes, model sha256, runtime versions,
  generation settings, per-worker load and generation time, peak memory,
  tokens per second, the count of output-head calls with NaN or inf logits,
  and for panels the evaluator counts, NLL and (DEV75) the R1 counts.
- `cases.json` (panels): the evaluator's per-case records, merged.
- `bootstrap.json`, `plan.json`, `worker-*.log`, `worker-*/`.

## Usage

```bash
# The commit must be pushed: the kernel downloads it from GitHub and checks a
# digest of packages/sepalith/src and infer_entry.py against the local git.
python3 scripts/kaggle/infer_push.py --run-id c03-smoke-20261012 --estimate-hours 0.5 --timeout-hours 1 -- \
    --model hf:scholzmx/sepalith-2b-cpt/checkpoint-11586 \
    --model-sha256 4bf617f9ea66a31e547397b93f0998372686e076d03ba22961780abd285a3c51 \
    --panel dev75 --case-limit 10 --max-new-tokens 128 --dtype fp16 --batch-size 5 --workers 2

# Parity run at the bound settings (batch 32, cap 1,024), both T4s:
python3 scripts/kaggle/infer_push.py --run-id <id> --estimate-hours 1 -- \
    --model <source> --panel dev75 --dtype fp16 --batch-size 32 --workers 2
python3 scripts/kaggle/infer_push.py --run-id <id> --estimate-hours 1 -- \
    --model <source> --prompts hf-dataset:scholzmx/sepalith/<path>/c03-train200-seed3407.jsonl \
    --prompts-sha256 3a99e2ad6a6b04995cd52a2614d67e13e31ba8e30eae2d2fc4a6d502ff250dc2 \
    --dtype fp16 --batch-size 32 --workers 2

# Census-style sampling (C12): k samples per prompt, seeded per batch.
... -- --model <source> --prompts <source> --temperature 1.0 --samples 8 --batch-size 4 --workers 2 \
       --shard-index 0 --shard-count 4
```

Driver options: `--dry-run` writes the kernel folder without pushing;
`--no-wait` pushes and returns, and `--fetch-only` later polls, fetches and
records; `--cpu` runs without an accelerator (no GPU quota); `--upload-hf`
uploads the fetched JSON/JSONL/log files to
`datasets/scholzmx/sepalith` under `campaign-20260915/inference/<run-id>/` in
one commit with the PC's `hf` login.

## Parity procedure (C03 step 4)

1. PC references (night jobs `docs/training/plan-20260927/c03/jobs/C03-parity-references.json`
   and `C03-parity-references-arm-a.json`, script `c03/parity_references.py`):
   bf16 at the bound batching for DEV75 and the 200 TRAIN prompts of
   `c03/train_prompts.py` (seed 3407, stratified by family), on parent 11,586
   plus lr1e-5 gate 268 (2026-10-14) and plus arm A, checkpoint 536.
2. The prompts file is public at
   `hf-dataset:scholzmx/sepalith/campaign-20260915/C03/c03-train200-seed3407.jsonl`;
   references go to `campaign-20260915/C03/parity-references/<attempt>/`
   (one commit per batch). Generations are outputs, not weights, so they may
   be published. Arm A's weights are public at
   `hf:scholzmx/sepalith-2b-edit-sft/arm-a` (sha256 `befe587a...`), so Kaggle
   runs the representative SFT check on it.
3. Kaggle: the same prompts in fp16 with `--batch-size 32 --kv-cache dynamic`
   (the default cap of 1,024), one run for `--panel dev75` and one for
   `--prompts`, per model whose weights Kaggle can read.
4. `c03/compare_parity.py --reference-dir <attempt>/<label> --reference-receipt <attempt>/references.json --dev75 <kaggle dev75 run> --train200 <kaggle train run> --output <receipt>`.
   Pass: no NaN or inf, the DEV75 counts `valid`, `edit`, `nofp` and `cap`
   within 1, and at least 90% of all cases token-identical. A run with
   different batching is reported as not comparable. If fp16 fails, repeat in
   fp32 at the same batching (the T4 memory at batch 32 in fp32 is unmeasured).

## Secrets and quota

- The kernel source contains no secret; the driver refuses to push a source
  containing a token-like string. Public HF reads need no token; uploads run
  on the PC.
- The Kaggle CLI 2.x authenticates with `KAGGLE_API_TOKEN` (a `KGAT_` access
  token, no `kaggle.json` needed). The driver uses the environment, or reads
  the line from `~/.config/sepalith/secrets.env` into the CLI's environment
  only.
- R7: the driver reads `kaggle quota` before the push and after the run,
  refuses (exit 3) when the week's used GPU hours plus `--estimate-hours`
  would exceed 20, and appends every run to
  `~/.local/state/sepalith/kaggle/quota-ledger.jsonl`. The Kaggle quota week
  resets on Saturday 00:00 UTC.

## Kaggle facts (verified 2026-10-12)

Image: Python 3.13, torch 2.11.0+cu128, two Tesla T4 (15,360 MiB, driver
580.178.04). The bootstrap installs transformers 5.5.0, tokenizers 0.22.2 and
accelerate 1.14.0 (the PC's pins) in about 20 s. Internet sessions download the
5 GB parent from HF in well under a minute. The older mechanics notes are in
`docs/research/2026-09-06-kaggle-compute-integration.md`; its Qwen3.5 and
unsloth parts do not apply to this model.
