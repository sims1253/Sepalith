# scripts/cloud (historical)

These scripts belong to the Qwen3.5 LoRA era (September 2026): Anyscale and
Kaggle SFT jobs for the gated-DeltaNet base, which is not the current model.
They are kept for reference and are not used by the post-CPT plan.

For Kaggle inference with the current dense MiniCPM-lineage model, use
`scripts/kaggle/` (card C03). It reuses these mechanics:

- the CLI 2.x `--accelerator NvidiaTeslaT4` push and `kaggle quota` reads
  (`kaggle_push.sh`);
- the Kaggle dataset mount layouts (`/kaggle/input/<slug>`, nested, and
  `/kaggle/input/datasets/<owner>/<slug>`), probed by `infer_entry.py` for
  `kaggle:` sources;
- the GPU probe and system-pip install in the session (`kaggle_sft_entry.sh`).

It drops what no longer fits: the repository is fetched from GitHub at a
pinned commit and digest instead of being re-uploaded as a Kaggle dataset
(15 to 20 minute re-ingest), and no token is written into the kernel source
(`kaggle_push.sh` put `HF_TOKEN` there; public HF reads need none, and uploads
now happen on the PC).
