"""Arm-matrix readout evaluator for the decay/CMA POC (Tasks 4/8).

Per arm checkpoint:
  - per-stratum eval loss (nats/token) on the draw's position-disjoint
    holdout blocks (from eval_blocks.npy + its .src.json round-robin map)
  - R-BPB on the packed package-disjoint R eval slices
    (eval_causal / eval_rc / so_r_qa_eval; byte counts from
    a2/r/stats.json and so_r_qa_eval byte accounting)
  - dead-neuron census: SwiGLU row-norm distribution of every block's
    gate matrix (fraction of rows with norm < 1e-2 x median, sweep §5)
  - ELR trajectory + per-epoch slope come from the train JSONL logs
    (mpl_fit reads them; no GPU needed)

GPU discipline: LaunchGate + POC_MEM_FRACTION (default 0.42), memmap.
Usage:
  python eval_arms.py --ckpt /mnt/h/sepalith/runs/poc_cma/cma_C/final.pt \
      --tag C [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
POC_TWIN = os.path.abspath(os.path.join(HERE, "..", "poc_twin"))
sys.path.insert(0, POC_TWIN)

import train as T  # noqa: E402
from model import TinyGQA  # noqa: E402

R_SLICES = [
    # (label, path, byte count source)
    ("r_eval_causal", "/mnt/h/sepalith/a2/r/eval_causal.npy", 13404063),
    ("r_eval_rc", "/mnt/h/sepalith/a2/r/eval_rc.npy", 909391),
    ("so_r_qa_eval", "/mnt/h/sepalith/a2/r/so_r_qa_eval.npy", 1981802),
]


@torch.no_grad()
def eval_blocks_per_stratum(model, eval_npy, src_json, bs=8):
    """Round-robin eval blocks -> per-stratum mean CE (nats/token)."""
    src = json.load(open(eval_npy + ".src.json"))["src"]
    blocks = np.load(eval_npy, mmap_mode="r")
    strata = sorted({s[0] for s in src})
    nats = {si: 0.0 for si in strata}
    toks = {si: 0 for si in strata}
    row_si = np.array([s[0] for s in src])[:len(blocks)]
    for i in range(0, len(blocks), bs):
        x = torch.from_numpy(
            np.asarray(blocks[i:i + bs]).astype(np.int64)).cuda()
        h = model.trunk(x[:, :-1], probe=False)
        logits = F.linear(h, model.embed.weight)
        lg = logits.view(-1, logits.size(-1)).float()
        tg = x[:, 1:].reshape(-1)
        per = F.cross_entropy(lg, tg, reduction="none")
        per = per.view(x.size(0), -1).sum(dim=1).cpu().numpy()
        nt = x.size(1) - 1
        for j, si in enumerate(row_si[i:i + bs]):
            nats[si] += float(per[j])
            toks[si] += nt
    return {int(si): dict(loss=nats[si] / max(1, toks[si]),
                          tokens=toks[si]) for si in strata}


@torch.no_grad()
def bpb_slice(model, path, total_bytes, bs=8, chunk=4096):
    eb = np.load(path, mmap_mode="r")
    nats, toks = 0.0, 0
    for i in range(0, len(eb), bs):
        x = torch.from_numpy(
            np.asarray(eb[i:i + bs]).astype(np.int64)).cuda()
        h = model.trunk(x[:, :-1], probe=False)
        logits = F.linear(h, model.embed.weight)
        lg = logits.view(-1, logits.size(-1)).float()
        tg = x[:, 1:].reshape(-1)
        for c in range(0, lg.size(0), chunk):
            nats += F.cross_entropy(lg[c:c + chunk], tg[c:c + chunk],
                                    reduction="sum").item()
        toks += tg.numel()
    return dict(bpb=nats / (total_bytes * math.log(2)),
                loss_per_tok=nats / toks, tokens=toks)


@torch.no_grad()
def dead_neuron_census(model, rel_thresh=1e-2):
    """SwiGLU gate row-norm census across blocks (fraction of near-dead
    rows: norm < rel_thresh * median row norm of that matrix)."""
    fracs, rows_total = [], 0
    for name, p in model.named_parameters():
        # SwiGLU gate matrices only: blocks.N.Wg.weight, shape (h, d)
        if name.endswith("Wg.weight") and p.size(0) == model.cfg["ffn_hidden"]:
            rn = p.float().norm(dim=1)
            med = float(rn.median())
            dead = int((rn < rel_thresh * med).sum())
            fracs.append(dead / rn.numel())
            rows_total += rn.numel()
    return dict(dead_fraction=float(np.mean(fracs)) if fracs else None,
                matrices=len(fracs), rows=rows_total)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-blocks", default="/tmp/poc_cma/eval_blocks.npy")
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-r", action="store_true")
    ap.add_argument("--gate-min-free-mib", type=int, default=16 * 1024)
    args = ap.parse_args(argv)

    T.LaunchGate(min_free_mib=args.gate_min_free_mib).wait()
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("POC_MEM_FRACTION", "0.42")))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyGQA(ck["cfg"]).cuda().eval()
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})

    out = dict(tag=args.tag, ckpt=args.ckpt, step=ck.get("step"))
    out["per_stratum"] = eval_blocks_per_stratum(
        model, args.eval_blocks, args.eval_blocks)
    out["dead_neuron"] = dead_neuron_census(model)
    if not args.skip_r:
        out["r_bpb"] = {lbl: bpb_slice(model, p, b)
                        for lbl, p, b in R_SLICES}
    js = args.json or os.path.join(
        "/tmp/poc_cma", f"eval_{args.tag}.json")
    with open(js, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
