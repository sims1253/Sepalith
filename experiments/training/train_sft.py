#!/usr/bin/env python3
"""Stage-2 Route-B SFT v1: LoRA fine-tune on zeta2-format finish-block data.

Run inside .venv-sft: python train_sft.py <model_id> [steps]
Defaults: openbmb/MiniCPM5-1B, 1 epoch-equivalent capped at 3000 steps.
Abort rule (shared-machine policy): GPU must be free at launch; re-check via
nvidia-smi before starting.

B8 midtrain instrument (2026-09-05, zcode-b8-patch): opt-in via MIDTRAIN_MASK=1
or --midtrain (plus MIDTRAIN_PACK=bucket|seq, default bucket). ON = completion-
only loss masking (prompt prefix -> -100, computed from the dataset's `prompt`
field against the single full-text tokenization) + the packing variant; see
experiments/training/midtrain_data.py. OFF = the legacy pipeline VERBATIM
(byte-compatibility contract for banked recipes: same map, same
shuffle(seed=42)+48k cap, same SFTConfig literals — regression-pinned in
experiments/training/test_midtrain_data.py).
"""
import json, os, subprocess, sys
from pathlib import Path

# B8 flag: snapshot argv, then strip --midtrain so its POSITION never shifts
# the positional args below (env MIDTRAIN_MASK=1 works identically).
_ARGV = list(sys.argv)
if "--midtrain" in sys.argv:
    sys.argv.remove("--midtrain")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openbmb/MiniCPM5-1B"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
DATA = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/mnt/h/sepalith/datasets/sft_v1")
# one output dir per (model, dataset): a shared dir lets the next chain
# overwrite the previous run's checkpoints (lost sft_v2_minicpm5 that way)
OUT = (Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].strip() else
       Path(f"/mnt/h/sepalith/runs/sft_{DATA.name}_{MODEL.split('/')[-1]}"))
# 5th arg: "auto" resumes from the newest checkpoint in OUT (v5 died at
# step 3086 in a GPU-contention incident; checkpoint-3000 was intact)
RESUME = sys.argv[5] if len(sys.argv) > 5 else ""

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
# SFT_TARGETS env: hybrid archs name their projections differently
# (LFM2: out_proj/in_proj/w1-3; Qwen3.5 GDN: in_proj_qkv/a/b/z/out_proj).
# Default = the llama-arch recipe, unchanged for the MiniCPM rungs.
# "regex:<pattern>" prefix (B13, 2026-09-05) passes a RAW regex string
# straight through to PEFT's re.fullmatch. Needed because unsloth_zoo's
# get_peft_regex list->regex conversion only knows attn/mlp parent tags,
# so an explicit leaf list silently drops modules under unlisted parents
# (B3's conv.in_proj/conv.out_proj never attached — its saved adapter has
# 72 modules, 0 conv; B13's full-attachment expectation is 166/48,922,624).
_SFT_TARGETS = os.environ.get(
    "SFT_TARGETS", "q_proj,k_proj,v_proj,o_proj,"
    "gate_proj,up_proj,down_proj")
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=64, lora_dropout=0,
    target_modules=(_SFT_TARGETS[6:] if _SFT_TARGETS.startswith("regex:")
                    else _SFT_TARGETS.split(",")),
    bias="none", use_gradient_checkpointing="unsloth", random_state=3407)

from datasets import load_dataset
ds = load_dataset("json", data_files={"train": str(DATA / "train.jsonl"),
                                      "eval": str(DATA / "eval.jsonl")})

# --- B8 midtrain instrument flag (parse BEFORE data prep; OFF = legacy) ---
from midtrain_data import (parse_midtrain_flag, assert_midtrain_safe,
                           midtrain_map_row, seam_guard, pack_examples_ffd,
                           MidtrainPackedCollator)
MIDTRAIN, MIDTRAIN_PACK = parse_midtrain_flag(_ARGV)
midtrain_kwargs, midtrain_collator = {}, None

if MIDTRAIN:
    # B4-saga guards: knobs mandatory on this box; seq packing refused for
    # GDN/linear-attention model types (see midtrain_data.py docstring).
    assert_midtrain_safe(os.environ,
                         model_type=getattr(model.config, "model_type", ""),
                         pack_mode=MIDTRAIN_PACK)
    MAXSEQ = 2048

    def _midtrain_row(row):
        return midtrain_map_row(row, lambda t: tokenizer(t)["input_ids"])

    def _midtrain_split(split, cap, shuffle):
        raw = ds[split]
        if "text" not in raw.column_names or not (
                "prompt" in raw.column_names or "target" in raw.column_names):
            raise SystemExit(
                f"MIDTRAIN needs a 'text'+'prompt' (astfim_v1) or "
                f"'text'+'target' (astfim_v1/fixed) dataset; {split} has "
                f"{raw.column_names}")
        if shuffle:  # same selection discipline as the legacy path
            raw = raw.shuffle(seed=42)
        raw = raw.select(range(min(cap, len(raw))))
        tokd = raw.map(_midtrain_row,
                       remove_columns=[c for c in raw.column_names])
        guard = seam_guard(tokd, split,
                           allow_dirty=os.environ.get("MIDTRAIN_ALLOW_DIRTY_SEAM", "") == "1")
        tokd = tokd.filter(lambda x: x["length"] <= MAXSEQ and x["n_loss"] > 0)
        routes = tokd["route"]
        n_loss_tok = sum(tokd["n_loss"]); n_tok = sum(tokd["length"])
        print(f"[midtrain:{split}] {guard['n0']} rows in -> {len(tokd)} kept "
              f"(prefix-route {sum(1 for r in routes if r == 0)}, suffix-route "
              f"{sum(1 for r in routes if r == 1)}; token-seam exact "
              f"{guard['n_exact']}/{guard['n_routed']}); {n_tok} tokens, "
              f"{n_loss_tok} loss tokens = "
              f"{100.0 * n_loss_tok / max(n_tok, 1):.1f}% completion", flush=True)
        return tokd

    train_ds = _midtrain_split("train", 48000, shuffle=True)
    eval_ds = _midtrain_split("eval", 500, shuffle=False)
    if MIDTRAIN_PACK == "seq":
        # FFD sequence packing (full-attention bases only; guard above).
        blocks, pstats = pack_examples_ffd(train_ds.to_list(), block_size=MAXSEQ)
        print(f"[midtrain:seq] {pstats}", flush=True)
        from datasets import Dataset
        train_ds = Dataset.from_list(blocks)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        midtrain_collator = MidtrainPackedCollator(pad_token_id=pad_id)
    else:
        # conservative variant: one sample per row; length-grouped batches
        # cut padding waste with EXACT cross-sample isolation (the stack's
        # own 2D padding-mask semantics — nothing new exercised on the GDN
        # recurrent layers; unsloth padding-free stays OFF per B4 saga).
        midtrain_kwargs = {"train_sampling_strategy": "group_by_length",
                           "length_column_name": "length"}
else:
    ds = ds.map(lambda x: {"text": x["text"]}, remove_columns=[
        c for c in ds["train"].column_names if c != "text"])

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=(train_ds if MIDTRAIN else
                   ds["train"].shuffle(seed=42).select(range(min(48000, len(ds["train"]))))),
    eval_dataset=(eval_ds if MIDTRAIN else
                  ds["eval"].select(range(min(500, len(ds["eval"]))))),
    data_collator=midtrain_collator,
    args=SFTConfig(
        # bs 4 x ga 4 = effective 16; expandable_segments (set by the chain
        # env) keeps fragmentation from spilling into shared memory.
        # SFT_PD_BATCH/SFT_GRAD_ACCUM knobs (B13, 2026-09-05; same names/
        # semantics as train_sft_trl.py): OOM fallback ONLY — keep the
        # product at 16 so optimizer math stays identical to the banked rows.
        per_device_train_batch_size=int(os.environ.get("SFT_PD_BATCH", "4")),
        gradient_accumulation_steps=int(os.environ.get("SFT_GRAD_ACCUM", "4")),
        num_train_epochs=1, max_steps=STEPS,
        learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=20, eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=1000, save_total_limit=2,
        output_dir=str(OUT),
        bf16=True, seed=3407, report_to="none", dataset_text_field="text",
        max_seq_length=2048, **midtrain_kwargs),
)
resume_from = None
if RESUME == "auto":
    cks = sorted(OUT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    resume_from = str(cks[-1]) if cks else None
    print(f"resume: {resume_from or '(no checkpoint found, fresh start)'}", flush=True)
elif RESUME:
    resume_from = RESUME
trainer.train(resume_from_checkpoint=resume_from)

# smoke generations: does it learn the format? (diagnostic only — never
# gates the save: hub bases of VL-capable classes — Qwen3.5-Base straight
# from HF — route tokenizer() through the VL processor, which misreads FIM
# markers as base64 images; the repo's locally text-stripped copies don't)
FastLanguageModel.for_inference(model)
try:
    sample = json.loads(open(DATA / "eval.jsonl").readline())
    gen = tokenizer(sample["prompt"], return_tensors="pt").to(model.device)
    out = model.generate(**gen, max_new_tokens=200, do_sample=False)
    print("=== SMOKE: prompt tail ===\n", sample["prompt"][-150:])
    print("=== SMOKE: generation ===\n", tokenizer.decode(out[0][gen.input_ids.shape[1]:], skip_special_tokens=True)[:400])
except Exception as e:  # noqa: BLE001 — diagnostic must not kill the run
    print(f"=== SMOKE: generation skipped ({type(e).__name__}: {e}) ===")
model.save_pretrained(str(OUT / "final_lora"))
print("DONE")
