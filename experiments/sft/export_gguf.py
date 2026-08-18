#!/usr/bin/env python3
"""Merge a trained LoRA and export GGUF for llama.cpp serving + eval.

Usage: export_gguf.py <model_id> <lora_dir> <out_stem>
Run inside .venv-sft. Produces <out_stem>-Q8_0.gguf in experiments/models.
"""
import subprocess, sys
from pathlib import Path

MODEL, LORA, STEM = sys.argv[1], sys.argv[2], sys.argv[3]
MERGED = Path("/tmp") / f"merged_{STEM}"
MODELS = Path("/home/m0hawk/Documents/Sepalith/experiments/models")
QUANT = MODELS.parent / "bin" / "llama" / "llama-b10453" / "llama-quantize"
CONVERT = "/tmp/llamacpp-src/convert_hf_to_gguf.py"

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA,  # unsloth loads base + adapter from the saved dir
    max_seq_length=2048, dtype=None, load_in_4bit=False)
model.save_pretrained_merged(str(MERGED), tokenizer, save_method="merged_16bit")

py = sys.executable
subprocess.run([py, CONVERT, str(MERGED), "--outfile", str(MODELS / f"{STEM}-f16.gguf"),
                "--outtype", "f16"], check=True)
subprocess.run([str(QUANT), str(MODELS / f"{STEM}-f16.gguf"),
                str(MODELS / f"{STEM}-Q8_0.gguf"), "Q8_0"], check=True)
(MODELS / f"{STEM}-f16.gguf").unlink()
print(f"OK -> {MODELS / (STEM + '-Q8_0.gguf')}")
