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

PFT1 full fine-tuning mode (2026-09-06, zcode-pft1): opt-in via FULL_FT=1 or
--full-ft. OFF = the legacy LoRA path byte-for-byte (contract above). ON =
trains ALL weights on the SAME unsloth loader/trainer stack as the banked
runks (FastLanguageModel with full_finetuning=True, no get_peft_model —
B4-saga finding: the plain transformers/TRL path is 25s/it on this GDN base,
21h for 3000 steps; unsloth's patched kernels/fused CE are what make the
2-3h budget class possible), recipe per queue §3 PFT1: bf16, paged 8-bit
AdamW, gradient checkpointing (use_reentrant=False), lr 1.5e-5 cosine
(ONE pre-registered rescue at 5e-6/3e-5 via FULL_FT_LR — document if used),
seed 3407, seq 2048, the same sft_v7 selection discipline (shuffle(42)+48k
cap) and eval/save cadence as b4. Launch convention: SFT_PD_BATCH=2 x
SFT_GRAD_ACCUM=8 (B13 logits-block precedent; identical effective 16 to
b4's bs4xga4). Saves the FULL model to <OUT>/final_model (no adapter);
export via export_gguf.py NO_LORA=1. MIDTRAIN is refused on this path.

B9 SeleKT instrument (2026-09-06, zcode-b9-select): opt-in via SELEKT_MASK=1
or --selekt (plus SELEKT_KEEP=0.5, SELEKT_PROBE_BUDGET=4096). ON = token-space
gradient-importance masking — per-token importance I_t = ||softmax(z_{t-1}) -
onehot(y_t)||_2 from a forward-only probe at the b4 INIT state (base +
zero-init LoRA), a single global keep-threshold tau over all train target
positions, labels -100 below tau, >=1 kept token per row; FULL b4 attachment
(trainable must equal 21,823,488) so labels are the ONLY delta vs the banked
b4 rung (same rows/order/token streams/steps/seed). Token streams replicate
TRL's legacy text path exactly (+eos, truncate 2048). PRE-REGISTERED
adaptation + verdict rule: experiments/training/selekt_data.py header.
FULL_FT and MIDTRAIN are both refused on this path (one masking/training
mechanism per arm). OFF = the legacy pipeline VERBATIM (same contract as
B8; regression-pinned in experiments/training/test_selekt_data.py).
"""
import json, os, subprocess, sys
from pathlib import Path

# B8 flag: snapshot argv, then strip --midtrain so its POSITION never shifts
# the positional args below (env MIDTRAIN_MASK=1 works identically).
_ARGV = list(sys.argv)
if "--midtrain" in sys.argv:
    sys.argv.remove("--midtrain")
# PFT1 flag: same position-independence for --full-ft (env FULL_FT=1 works
# identically).
if "--full-ft" in sys.argv:
    sys.argv.remove("--full-ft")
# B9 flag: same snapshot/strip pattern for --selekt (env SELEKT_MASK=1 works
# identically; _ARGV above retains every flag for the parsers).
if "--selekt" in sys.argv:
    sys.argv.remove("--selekt")

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
# first GPU line only (multi-GPU hosts — Kaggle T4x2 — emit one line per
# device and the old whole-output split unpacked 3 values; zcode-kaggle-intel)
first_gpu = gpu.splitlines()[0] if gpu else "0 %, 0 MiB"
util, mem = [x.strip().split()[0] for x in first_gpu.split(",")]
print(f"GPU check: util={util}% mem={mem}MiB"
      + (f" ({len(gpu.splitlines())} GPUs, guarded on #0)" if len(gpu.splitlines()) > 1 else ""),
      flush=True)
if float(mem) > 8000:
    raise SystemExit("GPU busy (>8GB used) — aborting per shared-machine policy")

# --- PFT1 full fine-tuning branch (opt-in; exits before the legacy path) ---
if ("--full-ft" in _ARGV) or (os.environ.get("FULL_FT", "") == "1"):
    if ("--midtrain" in _ARGV) or (os.environ.get("MIDTRAIN_MASK", "") == "1"):
        raise SystemExit("FULL_FT and MIDTRAIN are mutually exclusive — PFT1 "
                         "uses the legacy data pipeline verbatim")
    from unsloth import FastLanguageModel
    import torch
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=2048, dtype=None,
        load_in_4bit=False, full_finetuning=True)
    torch.manual_seed(3407)  # same seed discipline as the LoRA path
    _n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _n_all = sum(p.numel() for p in model.parameters())
    print(f"PFT1 FULL-FT Trainable parameters = {_n_train:,} of {_n_all:,} "
          f"({100.0 * _n_train / max(_n_all, 1):.2f}%) — ALL weights, no adapter")

    from datasets import load_dataset
    ds = load_dataset("json", data_files={"train": str(DATA / "train.jsonl"),
                                          "eval": str(DATA / "eval.jsonl")})
    ds = ds.map(lambda x: {"text": x["text"]}, remove_columns=[
        c for c in ds["train"].column_names if c != "text"])

    # pre-registered lr 1.5e-5 (queue §3 PFT1); the env is the ONE rescue
    # channel (5e-6 or 3e-5, divergent-or-flat only — the chain log documents
    # any use); every other knob mirrors the b4 SFTConfig literals.
    _LR = float(os.environ.get("FULL_FT_LR", "1.5e-5"))
    print(f"PFT1 FULL-FT lr={_LR} "
          f"({'PRE-REGISTERED 1.5e-5' if _LR == 1.5e-5 else 'RESCUE VALUE — DOCUMENT'})",
          flush=True)
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=ds["train"].shuffle(seed=42).select(range(min(48000, len(ds["train"])))),
        eval_dataset=ds["eval"].select(range(500)),
        args=SFTConfig(
            # bs 2 x ga 8 = effective 16 (b4's optimizer math; B13 logits-block
            # precedent — full FT adds grads+moments on top of the activations)
            per_device_train_batch_size=int(os.environ.get("SFT_PD_BATCH", "2")),
            gradient_accumulation_steps=int(os.environ.get("SFT_GRAD_ACCUM", "8")),
            num_train_epochs=1, max_steps=STEPS,
            learning_rate=_LR, warmup_ratio=0.03, lr_scheduler_type="cosine",
            logging_steps=20, eval_strategy="steps", eval_steps=500,
            save_strategy="steps", save_steps=1000, save_total_limit=2,
            output_dir=str(OUT),
            bf16=True, seed=3407, report_to="none", dataset_text_field="text",
            max_seq_length=2048,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}),
    )
    resume_from = None
    if RESUME == "auto":
        cks = sorted(OUT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        resume_from = str(cks[-1]) if cks else None
        print(f"resume: {resume_from or '(no checkpoint found, fresh start)'}", flush=True)
    elif RESUME:
        resume_from = RESUME
    trainer.train(resume_from_checkpoint=resume_from)

    # smoke generation: same diagnostic-only convention as the legacy path
    FastLanguageModel.for_inference(model)
    try:
        sample = json.loads(open(DATA / "eval.jsonl").readline())
        gen = tokenizer(sample["prompt"], return_tensors="pt").to(model.device)
        out = model.generate(**gen, max_new_tokens=200, do_sample=False)
        print("=== SMOKE: prompt tail ===\n", sample["prompt"][-150:])
        print("=== SMOKE: generation ===\n", tokenizer.decode(out[0][gen.input_ids.shape[1]:], skip_special_tokens=True)[:400])
    except Exception as e:  # noqa: BLE001 — diagnostic must not kill the run
        print(f"=== SMOKE: generation skipped ({type(e).__name__}: {e}) ===")
    model.save_pretrained(str(OUT / "final_model"))
    tokenizer.save_pretrained(str(OUT / "final_model"))
    print("DONE")
    sys.exit(0)

from unsloth import FastLanguageModel
import torch

# SFT_FP16 (2026-09-06, zcode-kaggle-intel): T4-compat channel for cloud
# arms — Turing (sm75) has no bf16 and transformers hard-rejects bf16=True
# on pre-Ampere ("You need Ampere+ GPU"). ON = fp16 weights at load +
# fp16=True in the LoRA SFTConfig (HF Trainer auto GradScaler; T4 fp16
# tensor cores). OFF = banked bf16 recipe byte-identical. Sweep arms are
# internally consistent (all fp16 on Kaggle); the anchor arm doubles as
# the cross-platform normalization line.
_FP16 = os.environ.get("SFT_FP16", "0") == "1"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=2048,
    dtype=torch.float16 if _FP16 else None, load_in_4bit=False)
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

# --- B9 SeleKT instrument flag (parse BEFORE data prep; OFF = legacy) ---
from selekt_data import (parse_selekt_flag, assert_selekt_safe,
                         legacy_tokenize_row, probe_importances,
                         compute_keep_threshold, build_selekt_labels,
                         summarize_masking)
SELEKT, SELEKT_KEEP, SELEKT_BUDGET = parse_selekt_flag(_ARGV)
train_ds = eval_ds = None  # set by the MIDTRAIN or SELEKT branches only

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

if SELEKT:
    # B9: token-space gradient-importance masking (see selekt_data.py header
    # for the pre-registered adaptation). Guards first: one mechanism per arm.
    assert_selekt_safe(os.environ,
                       model_type=getattr(model.config, "model_type", ""),
                       midtrain_enabled=MIDTRAIN)
    if ("--full-ft" in _ARGV) or (os.environ.get("FULL_FT", "") == "1"):
        raise SystemExit("SELEKT and FULL_FT are mutually exclusive — B9 is "
                         "the LoRA b4 recipe with masked labels only")
    # SAME selection discipline as the legacy path (shuffle(42)+48k train cap,
    # first 500 eval rows) — selection made on the raw dataset is exactly what
    # the legacy SFTTrainer call below trains on; token streams via
    # legacy_tokenize_row == TRL's text path (eos append + truncate 2048).
    _sel_train_raw = ds["train"].shuffle(seed=42).select(range(min(48000, len(ds["train"]))))
    _sel_eval_raw = ds["eval"].select(range(min(500, len(ds["eval"]))))
    _eos = tokenizer.eos_token

    def _sel_map(row):
        ids = legacy_tokenize_row(row["text"],
                                  lambda t: tokenizer(t)["input_ids"], _eos, 2048)
        return {"input_ids": ids, "length": len(ids)}

    _sel_train = _sel_train_raw.map(_sel_map, remove_columns=_sel_train_raw.column_names)
    _sel_eval = _sel_eval_raw.map(_sel_map, remove_columns=_sel_eval_raw.column_names)
    _n_trunc = sum(1 for n in _sel_train["length"] if n >= 2048)
    print(f"[selekt:cfg] token-grad-imp masking keep_frac={SELEKT_KEEP} "
          f"probe_budget={SELEKT_BUDGET} state=init(base+zero-LoRA)", flush=True)
    print(f"[selekt:tokenize] train {len(_sel_train)} rows -> "
          f"{sum(_sel_train['length'])} tokens (len>=2048 rows: {_n_trunc}); "
          f"eval {len(_sel_eval)} rows -> {sum(_sel_eval['length'])} tokens",
          flush=True)

    # forward-only probe through the b4 init state (LoRA zero-init => exactly
    # the base's forward); eval() for dropout-free determinism, then back to
    # train() before the trainer takes over.
    model.eval()
    _pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    _imp_train = probe_importances(model, _sel_train["input_ids"],
                                   token_budget=SELEKT_BUDGET, pad_token_id=_pad_id)
    _imp_eval = probe_importances(model, _sel_eval["input_ids"],
                                  token_budget=SELEKT_BUDGET, pad_token_id=_pad_id)
    model.train()
    _thr = compute_keep_threshold(_imp_train, SELEKT_KEEP)
    print(f"[selekt:importance] positions={_thr['n_positions']} "
          f"mean={_thr['imp_mean']:.4f} median={_thr['imp_median']:.4f} "
          f"p05={_thr['imp_p05']:.4f} p95={_thr['imp_p95']:.4f} "
          f"min={_thr['imp_min']:.4f} max={_thr['imp_max']:.4f} "
          f"-> tau={_thr['tau']:.4f}", flush=True)

    def _sel_build(tokd, imps):
        from datasets import Dataset
        recs, fb = [], 0
        for ids, imp in zip(tokd["input_ids"], imps):
            labels, _n_kept, used_fb = build_selekt_labels(ids, imp, _thr["tau"])
            fb += used_fb
            recs.append({"input_ids": ids, "labels": labels})
        stats = summarize_masking([r["input_ids"] for r in recs],
                                  [r["labels"] for r in recs], fb)
        return Dataset.from_list(recs), stats

    train_ds, _s_tr = _sel_build(_sel_train, _imp_train)
    eval_ds, _s_ev = _sel_build(_sel_eval, _imp_eval)
    print(f"[selekt:mask] tau={_thr['tau']:.4f} train kept {_s_tr['kept']}/"
          f"{_s_tr['target_positions']} = {100.0 * _s_tr['kept_frac']:.1f}% "
          f"(fallback rows {_s_tr['fallback_rows']}); eval kept "
          f"{100.0 * _s_ev['kept_frac']:.1f}% (fallback "
          f"{_s_ev['fallback_rows']})", flush=True)
    # audit/crash-resume artifact BEFORE training starts
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"keep_frac": SELEKT_KEEP, "probe_budget": SELEKT_BUDGET,
               "threshold": _thr, "train": _s_tr, "eval": _s_ev,
               "state": "init(base+zero-LoRA)",
               "formula": "I_t=||softmax(z_{t-1})-onehot(y_t)||_2",
               "token_stream": "TRL-legacy-equivalent: +eos then truncate 2048"},
              open(OUT / "selekt_probe.json", "w"), indent=1)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=(train_ds if (MIDTRAIN or SELEKT) else
                   ds["train"].shuffle(seed=42).select(range(min(48000, len(ds["train"]))))),
    eval_dataset=(eval_ds if (MIDTRAIN or SELEKT) else
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
        # SFT_LR knob (2026-09-06, zcode-kaggle-intel): LoRA-LR sweep channel
        # for cloud arms; default = the banked b4 literal 2e-4 (OFF = byte-
        # identical contract). Same names/convention as SFT_PD_BATCH.
        learning_rate=float(os.environ.get("SFT_LR", "2e-4")),
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=20, eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=1000, save_total_limit=2,
        output_dir=str(OUT),
        bf16=not _FP16, fp16=_FP16, seed=3407, report_to="none", dataset_text_field="text",
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
