#!/usr/bin/env python3
"""PFT1 forgetting probe — causal-floor BPB on held-out general R text + a
general-text control (queue §3 PFT1 readout; the pre-named failure mode of
full fine-tuning).

Discipline ported VERBATIM from the banked bpb_eval machinery
(experiments/training/poc_twin/ladder/bpb_eval.py, FIM-Replica/P10
precedent): BPB = total_nats / (corpus_bytes * ln 2), teacher-forced full
forward, cross-entropy summed in fp32 chunks. The R corpus is the SAME
construction the ladder used for its causal floor: astfim_v1/fixed EVAL
rows re-rendered as plain documents in natural order (causal_from_row,
data_prep_ladder.py — PSM markers stripped; rows still carrying a '<|'
marker after stripping are dropped). The control corpus is
license_texts.jsonl (authored legal text — no training mixture contains
it; general-language forgetting would show here, R-specific forgetting
would not).

Models are HF dirs (bf16, CUDA): the full-FT arm's final_model, the b4
anchor MERGED (build once via peft merge_and_unload — see
scripts/pft1_merge_b4.py), and the raw base for context. Both arms are
scored on IDENTICAL bytes/tokens per model tokenizer, so the verdict gate
(general-R BPB regression of the FT arm vs b4 <= 1%) is a paired delta.

Usage (.venv-sft; GPU, ~10 min for all three models at the default caps):
  python3 experiments/eval/pft1_bpb_probe.py \
    --model base=experiments/models/qwen3.5-2b-base-text-hf \
    --model b4=/mnt/h/sepalith/runs/pft1_b4_merged \
    --model pft1=/mnt/h/sepalith/runs/pft1_fullft_qwen35_2b/final_model \
    --out experiments/eval/pft1_bpb_probe.json
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

ASTFIM_FIXED_EVAL = "/mnt/h/sepalith/datasets/astfim_v1/fixed/eval.jsonl"
LICENSE_TEXTS = "/mnt/h/sepalith/datasets/license_texts.jsonl"
CTX, HIST, SUF, END = "<|context|>", "<|history|>", "<|suffix|>\n", "\n<|end|>"


def causal_from_row(r):
    """Plain-document rendering (ported from data_prep_ladder.py)."""
    p, t = r["prompt"], r["target"]
    i_ctx = p.find(CTX)
    i_hist = p.find(HIST)
    i_sf = p.find(SUF)
    if i_ctx != 0 or i_hist == -1 or i_sf == -1:
        return None
    ctx = p[len(CTX):i_hist]                      # path + "\n" + prefix
    i_end = p.rfind(END)
    s0 = i_sf + len(SUF)
    suffix = p[s0:i_end] if i_end != -1 else p[s0:]
    span = t
    if span.endswith(END):
        span = span[:-len(END)]
    elif span.endswith("<|end|>"):
        span = span[:-len("<|end|>")]
    return ctx + span + "\n" + suffix


def load_corpus_r(cap_rows):
    docs, dropped = [], 0
    with open(ASTFIM_FIXED_EVAL, encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(docs) >= cap_rows:
                break
            c = causal_from_row(json.loads(line))
            if c is None or "<|" in c:
                dropped += 1
                continue
            docs.append(c)
    return docs, dropped


def load_corpus_control(cap_rows):
    docs = []
    with open(LICENSE_TEXTS, encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(docs) >= cap_rows:
                break
            r = json.loads(line)
            t = r.get("license_text", "")
            if t.strip():
                docs.append(t)
    return docs


@torch.no_grad()
def corpus_bpb(model, tokenizer, docs, block=2048, bs=4, chunk=4096):
    """nats over next-token CE of contiguous 2048-token blocks (no overlap;
    same block discipline as the ladder eval; doc boundaries lose at most
    block-1 tokens of context each — identical across arms). Doc-final
    blocks are shorter: batches are right-padded and pad positions are
    MASKED OUT of the CE sum (no pad token ever enters the score)."""
    nats, toks, bytes_total = 0.0, 0, 0
    blocks = []
    for d in docs:
        ids = tokenizer(d, add_special_tokens=False)["input_ids"]
        bytes_total += len(d.encode("utf-8"))
        for i in range(0, len(ids), block):
            b = ids[i:i + block]
            if len(b) >= 2:
                blocks.append(b)
    for i in range(0, len(blocks), bs):
        grp = blocks[i:i + bs]
        L = max(len(b) for b in grp)
        x = torch.zeros(len(grp), L, dtype=torch.long, device="cuda")
        keep = torch.zeros(len(grp), L, dtype=torch.bool, device="cuda")
        for j, b in enumerate(grp):
            x[j, :len(b)] = torch.tensor(b, dtype=torch.long, device="cuda")
            keep[j, :len(b)] = True
        logits = model(x[:, :-1]).logits
        lg = logits.reshape(-1, logits.size(-1))
        tg = x[:, 1:].reshape(-1)
        m = keep[:, 1:].reshape(-1)
        ce = torch.empty(lg.size(0), dtype=torch.float32, device="cuda")
        for c in range(0, lg.size(0), chunk):
            ce[c:c + chunk] = F.cross_entropy(lg[c:c + chunk].float(),
                                              tg[c:c + chunk], reduction="none")
        nats += ce[m].sum().item()
        toks += int(m.sum().item())
    return dict(bpb=nats / (bytes_total * math.log(2)),
                loss_per_tok=nats / max(toks, 1), tokens=toks,
                bytes=bytes_total, docs=len(docs), blocks=len(blocks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name=hf_dir (bf16, CUDA), repeatable")
    ap.add_argument("--r-rows", type=int, default=500)
    ap.add_argument("--control-rows", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    r_docs, r_drop = load_corpus_r(args.r_rows)
    c_docs = load_corpus_control(args.control_rows)
    print(f"corpus: general-R {len(r_docs)} docs ({r_drop} dropped dirty), "
          f"control {len(c_docs)} docs", flush=True)

    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        key = f"{name}"
        if key in out:
            print(f"{name}: cached, skipping"); continue
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
        res = dict(model_dir=path,
                   general_r=corpus_bpb(model, tok, r_docs),
                   general_text=corpus_bpb(model, tok, c_docs))
        out[key] = res
        print(f"{name}: general_r bpb={res['general_r']['bpb']:.4f} "
              f"({res['general_r']['tokens']} tok) | general_text "
              f"bpb={res['general_text']['bpb']:.4f} "
              f"({res['general_text']['tokens']} tok)", flush=True)
        del model
        torch.cuda.empty_cache()
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)

    if "b4" in out and "pft1" in out:
        for corpus in ("general_r", "general_text"):
            reg = 100.0 * (out["pft1"][corpus]["bpb"] / out["b4"][corpus]["bpb"] - 1.0)
            out.setdefault("verdict", {})[f"{corpus}_regression_pct"] = round(reg, 3)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out.get("verdict"), indent=1), flush=True)


if __name__ == "__main__":
    main()
