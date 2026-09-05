#!/usr/bin/env python3
"""B8b stage 1: merge the BANKED B8 midtrain LoRA onto the GDN base (CPU).

Flow = export_gguf.py's documented MERGE_VIA_PEFT branch verbatim
(AutoModelForCausalLM bf16 + PeftModel.merge_and_unload), with two
differences required by the B8b arm:
  1. output goes to a PERSISTENT dir (/mnt/h/sepalith/runs/...), not
     /tmp — the merged base is the training input for the stacked SFT
     and must survive (SYSTEMS.md: /tmp is volatile by design);
  2. merge gates (B-series discipline):
     G1  adapter has 96 lora_A modules {down/gate/up_proj x24,
         q/k/v/o_proj x6} — the verified b4/b8 attachment profile;
     G2  a targeted weight CHANGED vs base (mlp.down_proj) and an
         untargeted one is BYTE-IDENTICAL (input_layernorm);
     G3  saved dir has config.json + model weights + tokenizer files,
         architectures == Qwen3_5ForCausalLM.
CPU-only (no CUDA context; card discipline). Run under .venv-sft.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/home/m0hawk/Documents/Sepalith")
LORA = Path("/mnt/h/sepalith/runs/b8_midtrain_qwen35_2b/final_lora")
OUT = Path("/mnt/h/sepalith/runs/b8b_stacked_base_merged")

# --- G1: adapter attachment profile before touching the base ---
with safe_open(LORA / "adapter_model.safetensors", "pt") as f:
    a_keys = [k for k in f.keys() if k.endswith("lora_A.weight")]
prof = Counter(k.split(".")[-3] for k in a_keys)
want = {"down_proj": 24, "gate_proj": 24, "up_proj": 24,
        "q_proj": 6, "k_proj": 6, "v_proj": 6, "o_proj": 6}
print(f"G1 adapter profile: {sum(prof.values())} modules {dict(prof)}", flush=True)
if dict(prof) != want:
    sys.exit(f"G1 FAIL: adapter profile {dict(prof)} != {want}")

cfg = json.load(open(LORA / "adapter_config.json"))
base = cfg["base_model_name_or_path"]
bp = Path(base)
if not bp.is_absolute():
    bp = REPO / base  # adapter stores the repo-relative launch path
print(f"base: {bp}", flush=True)
if not (bp / "config.json").exists():
    sys.exit(f"base not found at {bp}")

print("loading base (bf16, CPU)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    bp, dtype=torch.bfloat16, trust_remote_code=True)

# reference tensors for G2 (grab BEFORE the merge mutates the module)
tgt = model.model.layers[0].mlp.down_proj.weight.detach().clone()
ref = model.model.layers[0].input_layernorm.weight.detach().clone()

model = PeftModel.from_pretrained(model, LORA).merge_and_unload()

# --- G2: targeted moved, untargeted byte-identical ---
d_tgt = (model.model.layers[0].mlp.down_proj.weight.detach() - tgt)
delta = d_tgt.abs().float().mean().item()
same_ref = bool(
    torch.equal(model.model.layers[0].input_layernorm.weight.detach(), ref))
print(f"G2 targeted mean|delta|={delta:.3e} (must be >0); "
      f"untouched identical={same_ref}", flush=True)
if delta <= 0.0 or not same_ref:
    sys.exit("G2 FAIL: merge did not apply as expected")

print("saving merged model + tokenizer...", flush=True)
model.save_pretrained(OUT)
tok = AutoTokenizer.from_pretrained(bp, trust_remote_code=True)
tok.save_pretrained(OUT)

# --- G3: output inventory ---
mcfg = json.load(open(OUT / "config.json"))
files = sorted(p.name for p in OUT.iterdir())
print(f"G3 files: {files}", flush=True)
assert mcfg.get("architectures") == ["Qwen3_5ForCausalLM"], mcfg.get("architectures")
assert any(p.suffix == ".safetensors" for p in OUT.iterdir())
assert (OUT / "tokenizer.json").exists() and (OUT / "tokenizer_config.json").exists()
print(f"OK -> {OUT} (B8b stacked base ready)", flush=True)
