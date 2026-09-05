#!/usr/bin/env python3
"""Post-run tok/s accounting for the cloud SFT smoke.

Recomputes the trainer's row selection (load train.jsonl, shuffle(seed=42),
select 48k — the same ops train_sft.py does), tokenizes it with the run's
base-model tokenizer (CPU), and combines with TRL's train_runtime from
/root/train.log to report content tok/s (excl. padding) — the number
comparable to the 5090's 48k tok/s reference.
"""
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-0.8B-Base")
DATA = Path(os.environ.get("DATA_DIR", "/root/data/sft_v7"))
LOG = Path(os.environ.get("TRAIN_LOG", "/tmp/train.log")).read_text(errors="ignore")

m = re.search(r"train_runtime['\"]?\s*[:=]\s*['\"]?([\d.]+)", LOG)
runtime = float(m.group(1)) if m else None
m2 = re.search(r"train_steps_per_second['\"]?\s*[:=]\s*['\"]?([\d.]+)", LOG)
sps = float(m2.group(1)) if m2 else None
losses = re.findall(r"'loss': '?([\d.]+)", LOG)

ds = load_dataset("json", data_files=str(DATA / "train.jsonl"))["train"]
sel = ds.shuffle(seed=42).select(range(min(48000, len(ds))))
tok = AutoTokenizer.from_pretrained(MODEL)
total, n = 0, 0
texts = sel["text"]
for i in range(0, len(texts), 2000):
    enc = tok(texts[i:i + 2000], add_special_tokens=False)["input_ids"]
    total += sum(len(x) for x in enc)
    n += len(enc)
mean_row = total / n

# tok/s = tokens per step (mean_row*16) / seconds per step (runtime/STEPS)
steps = float(os.environ.get("STEPS", 60))
out = {
    "rows_tokenized": n,
    "mean_tokens_per_row": round(mean_row, 1),
    "train_runtime_s": runtime,
    "steps_per_s": sps,
    "effective_batch_rows": 16,
    "content_tok_s": round(mean_row * 16 * steps / runtime, 1) if runtime else None,
    "loss_curve": losses,
}
print("METRICS_JSON " + json.dumps(out, indent=2))
