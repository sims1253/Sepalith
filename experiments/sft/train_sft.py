#!/usr/bin/env python3
"""Stage-2 Route-B SFT v1: LoRA fine-tune on zeta2-format finish-block data.

Run inside .venv-sft: python train_sft.py <model_id> [steps]
Defaults: openbmb/MiniCPM5-1B, 1 epoch-equivalent capped at 3000 steps.
Abort rule (shared-machine policy): GPU must be free at launch; re-check via
nvidia-smi before starting.
"""
import json, subprocess, sys
from pathlib import Path

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openbmb/MiniCPM5-1B"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
DATA = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/mnt/h/sepalith/datasets/sft_v1")

# shared-machine guard
gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
util, mem = [x.strip().split()[0] for x in gpu.split(",")]
print(f"GPU check: util={util}% mem={mem}MiB", flush=True)
if float(mem) > 8000:
    raise SystemExit("GPU busy (>8GB used) — aborting per shared-machine policy")

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=2048, dtype=None, load_in_4bit=False)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=64, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", use_gradient_checkpointing="unsloth", random_state=3407)

from datasets import load_dataset
ds = load_dataset("json", data_files={"train": str(DATA / "train.jsonl"),
                                      "eval": str(DATA / "eval.jsonl")})
ds = ds.map(lambda x: {"text": x["text"]}, remove_columns=[
    c for c in ds["train"].column_names if c != "text"])

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=ds["train"].shuffle(seed=42).select(range(min(48000, len(ds["train"])))),
    eval_dataset=ds["eval"].select(range(500)),
    args=SFTConfig(
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=1, max_steps=STEPS,
        learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=20, eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=1000, save_total_limit=2,
        output_dir=f"/mnt/h/sepalith/runs/sft_v1_{MODEL.split('/')[-1]}",
        bf16=True, seed=3407, report_to="none", dataset_text_field="text",
        max_seq_length=2048),
)
trainer.train()

# smoke generations: does it learn the format?
FastLanguageModel.for_inference(model)
sample = json.loads(open(DATA / "eval.jsonl").readline())
gen = tokenizer(sample["prompt"], return_tensors="pt").to(model.device)
out = model.generate(**gen, max_new_tokens=200, do_sample=False)
print("=== SMOKE: prompt tail ===\n", sample["prompt"][-150:])
print("=== SMOKE: generation ===\n", tokenizer.decode(out[0][gen.input_ids.shape[1]:], skip_special_tokens=True)[:400])
model.save_pretrained(f"/mnt/h/sepalith/runs/sft_v1_{MODEL.split('/')[-1]}/final_lora")
print("DONE")
