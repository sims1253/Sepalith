# C03. Kaggle inference harness and fp16 parity

- **Where:** PC during the day, driving Kaggle T4 kernels (free quota)
- **Needs:** C01 for code. The parity step needs the bf16 references from C07.
- **Produces:** a reusable Kaggle inference entry point, driver and parity receipt
- **Effort:** one to two agent-days, plus about 2 to 4 Kaggle GPU-hours

## Goal

Run batch generation for this checkpoint on Kaggle T4s so the PC stays free.
The census (C12) and sample generation for RFT and DPO (C13) depend on it. The
T4 has no bf16, so results must be shown to match the PC's bf16 generation
closely enough for those uses.

## Read first

- CONTEXT.md, sections Model, Environments, Public repositories
- `docs/research/2026-09-06-kaggle-compute-integration.md` for the Kaggle
  mechanics that still hold: dataset mount layouts, CLI 2.x
  `--accelerator NvidiaTeslaT4`, quota reads, egress. Ignore its
  Qwen3.5/unsloth training parts.
- `scripts/cloud/kaggle_push.sh` and `kaggle_sft_entry.sh`. Reuse the
  bootstrap mechanics; replace the LoRA training payload.

## Steps

1. **Engine choice.** Try vLLM first, pinned to a version that supports
   compute capability 7.5, in fp16. Pass prompts as token ids (`prompt_token_ids`)
   taken from the TRAIN rows or the DEV panels, and stop on ids 1 and 130073.
   If vLLM cannot run on T4, use `transformers` generation with static
   batching as the fallback. Record the choice and the reason.
2. **Entry point.** Add `scripts/kaggle/infer_entry.py`. Inputs are a public HF
   model path (repository and folder) and a public HF prompts file with
   ids and metadata. It also takes sampling settings (greedy, or temperature
   with k samples), `max_new_tokens`, and a shard index and count, so work can
   split across the two T4s or across sessions. Outputs are JSONL with the
   generated ids, decoded text, finish reason and the cap-hit flag. Push the
   outputs to the HF dataset under `campaign-20260915/inference/<run-id>/`.
   No secret goes into kernel source. Pass the HF token as a Kaggle secret or
   through the environment block, as the existing driver does.
3. **Driver.** Add `scripts/kaggle/infer_push.sh`, or rewrite `kaggle_push.sh`.
   It packages the repository at a commit, pushes the kernel with the T4
   accelerator, polls status, and records `kaggle quota` before and after.
4. **fp16 parity.** C07 writes bf16 greedy references for DEV75 plus 200
   TRAIN prompts. Run the same prompts on Kaggle in fp16 with greedy decoding
   and compare:
   - exact token-id match rate per case
   - agreement on the evaluator's `valid`, `edit`, `nofp` and `cap`
   - any NaN or inf

   **Pass criteria:** no NaN, the four counts within 1 of the bf16 run, and at
   least 90% of cases token-identical. If fp16 fails, repeat in fp32 (the
   model is about 10 GB in fp32 and fits one T4) and apply the same criteria.
5. **Throughput.** Measure tokens per second and projected hours for 15,006
   prompts × 8 samples × up to 1,024 tokens. Store this in the status file
   for C12.
6. Mark `scripts/cloud/*` as historical (Qwen3.5 LoRA era) in a short README.
   Keep the reusable parts referenced from the new scripts.

## Acceptance

- The parity receipt passes, in fp16 or fp32.
- The entry point and driver are documented.
- The quota used is recorded.

## Stop and ask if

- Neither fp16 nor fp32 passes parity.
- Kaggle's quota or session limits make the census projection exceed 20
  GPU-hours per week. Colab free is the fallback to propose.
