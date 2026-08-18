# Training

LoRA SFT on the assembled mixtures, then merge and GGUF export for
llama.cpp. Checks `nvidia-smi` before launch and aborts if the GPU is busy.

| Script | What it does |
|---|---|
| `train_sft.py` | LoRA fine-tune with unsloth. Takes a model id, step count, and data directory. Saves checkpoints to the NAS. |
| `export_gguf.py` | Merges a trained LoRA, converts to GGUF, quantizes to Q8_0 for llama.cpp serving and eval. |
| `rl/` | Verifiable-reward training scaffold. See its README. |
