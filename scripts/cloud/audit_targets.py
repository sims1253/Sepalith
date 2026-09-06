#!/usr/bin/env python3
"""Pre-flight LoRA attachment audit — the B3 under-attach rule.

Board 2026-09-02: hybrid archs silently attach only the name-overlapping
subset of the default target list (B3-v1 trained 983K params instead of the
recipe's 22.4M — only q/k/v matched; LFM2 names its attention output
`out_proj` and its MLP `w1/w2/w3`).

Loads MODEL on CPU (plain transformers + peft — no unsloth, no GPU, no CUDA
context), applies the train_sft.py LoRA geometry (r=32, alpha=64, dropout=0,
bias=none) to TARGETS, prints per-family attachment counts + total trainable
params. Exit codes: 0 = pass; 1 = mismatch vs $EXPECT_TRAINABLE; 2 = sanity
floor tripped (trainable < 1% of base — classic under-attach signature).

Usage:  python audit_targets.py MODEL [TARGETS_CSV]   # default TARGETS from
        $SFT_TARGETS, else the llama-arch default set
"""
import os
import sys
from collections import Counter

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model_id = sys.argv[1]
# "regex:<pattern>" passthrough (2026-09-06, zcode-kaggle-intel): same
# convention as train_sft.py — pass the RAW regex string to PEFT (str
# target_modules => re.fullmatch semantics) instead of the exact-name list,
# so the guard can validate regex recipes (the banked b4 GDN target is a
# regex; a comma-split LIST would exact-match 0 modules and false-fail).
_raw = (sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "SFT_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"))
if _raw.startswith("regex:"):
    targets = _raw[6:]
else:
    targets = [t.strip() for t in _raw.split(",") if t.strip()]

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
total_base = sum(p.numel() for p in model.parameters())

model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0, bias="none", target_modules=targets))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

families = Counter(n.split(".")[-2] for n, _ in model.named_modules()
                   if n.endswith(".lora_A"))
print(f"AUDIT model={model_id} targets={targets}")
print(f"AUDIT attached_lora_modules={sum(families.values())} families={dict(families)}")
print(f"AUDIT trainable_params={trainable:,} of {total_base:,} "
      f"({100.0 * trainable / total_base:.2f}%)")

if trainable < 0.01 * total_base:
    print("AUDIT FAIL: trainable < 1% of base — under-attachment suspected")
    sys.exit(2)
expected = os.environ.get("EXPECT_TRAINABLE")
if expected and int(expected) != trainable:
    print(f"AUDIT FAIL: expected {int(expected):,}, got {trainable:,} — "
          "aborting before burn (B3 rule)")
    sys.exit(1)
print("AUDIT PASS")
