#!/usr/bin/env python3
"""Merge a trained LoRA and export GGUF for llama.cpp serving + eval.

Usage: export_gguf.py <model_id> <lora_dir> <out_stem>
Run inside .venv-sft. Produces <out_stem>-Q8_0.gguf in experiments/models.

Alt-build / remote-code overrides (see docs/research/2026-09-01-local-builds.md
for what's on disk): LLAMA_QUANT, LLAMA_CONVERT point at a different llama.cpp
build; MERGE_VIA_PEFT=1 merges via transformers+PEFT instead of unsloth
(remote-code archs like spark2_5 — run that under .venv-spark).

NO_LORA=1 (PFT1, 2026-09-06): <lora_dir> is already a FULL HF model dir (no
adapter_config.json — e.g. train_sft.py FULL_FT final_model). Skips every
merge path and converts that dir directly (base-model arg is unused).
"""
import json, os, subprocess, sys
from pathlib import Path

MODEL, LORA, STEM = sys.argv[1], sys.argv[2], sys.argv[3]
MERGED = Path("/tmp") / f"merged_{STEM}"
MODELS = Path("/home/m0hawk/Documents/Sepalith/experiments/models")
QUANT = os.environ.get(
    "LLAMA_QUANT", str(MODELS.parent / "bin" / "llama" / "llama-b10453" / "llama-quantize"))
CONVERT = os.environ.get("LLAMA_CONVERT", "/tmp/llamacpp-src/convert_hf_to_gguf.py")

if os.environ.get("NO_LORA"):
    # PFT1 full-FT export: the trained dir IS the model — convert in place.
    assert (Path(LORA) / "config.json").exists(), f"{LORA} is not an HF model dir"
    assert not (Path(LORA) / "adapter_config.json").exists(), \
        "NO_LORA=1 but an adapter_config.json is present — wrong mode"
    MERGED = Path(LORA)
elif os.environ.get("MERGE_VIA_PEFT"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = json.load(open(Path(LORA) / "adapter_config.json"))["base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, LORA).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model.save_pretrained(str(MERGED))
    tokenizer.save_pretrained(str(MERGED))
else:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA,  # unsloth loads base + adapter from the saved dir
        max_seq_length=2048, dtype=None, load_in_4bit=False)
    model.save_pretrained_merged(str(MERGED), tokenizer, save_method="merged_16bit")

py = sys.executable
if not Path(CONVERT).exists():
    if os.environ.get("LLAMA_CONVERT"):
        raise SystemExit(f"LLAMA_CONVERT={CONVERT} not found — explicit builds "
                         "are not auto-cloned; rebuild per local-builds.md")
    subprocess.run(["git", "clone", "-q", "--depth", "1",  # /tmp gets wiped between sessions
                    "https://github.com/ggml-org/llama.cpp",
                    str(Path(CONVERT).parent)], check=True)
# --no-nextn is the MTP-less qwen3_5 workaround; the converter REJECTS it
# for LlamaForCausalLM (MiniCPM5), so pass it only for qwen architectures
archs = json.load(open(MERGED / "config.json")).get("architectures", [])
nextn = ["--no-nextn"] if any("qwen" in a.lower() for a in archs) else []
subprocess.run([py, CONVERT, str(MERGED), "--outfile", str(MODELS / f"{STEM}-f16.gguf"),
                "--outtype", "f16", *nextn], check=True)
subprocess.run([str(QUANT), str(MODELS / f"{STEM}-f16.gguf"),
                str(MODELS / f"{STEM}-Q8_0.gguf"), "Q8_0"], check=True)
(MODELS / f"{STEM}-f16.gguf").unlink()
print(f"OK -> {MODELS / (STEM + '-Q8_0.gguf')}")
