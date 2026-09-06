"""Per-exit held-out BPB for A2-structure checkpoints (the instrument's
eval side — the '16L ~ 24L -> ship M' gate reads THIS).

Loads an A2Model checkpoint (model_a2), scores held-out blocks at every
exit (trunk_taps -> tied head, chunked CE) and reports BPB per exit +
MTP NLL if present. Byte-parallel to bpb_eval.py (same blocks+bytes
convention): pass --bytes for the held-out set's utf-8 byte count.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC)
from model import model_config, chunked_eval_ce  # noqa: E402
from model_a2 import A2Model  # noqa: E402


@torch.no_grad()
def exit_bpbs(model, blocks, bs=8, chunk=4096):
    """{exit_label: nats} over the held-out blocks + mtp nats."""
    nats = {}
    n_tok = 0
    for i in range(0, len(blocks), bs):
        x = torch.from_numpy(blocks[i:i + bs].astype(np.int64)).cuda()
        inp = x[:, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h_pre, taps = model.trunk_taps(inp, probe=False)
            targets = x[:, 1:]
            all_h = {"top": model.ln_f(h_pre)}
            for e, tap in zip(model.exit_layers, taps.values()):
                all_h[f"exit_{e}"] = tap
            for label, h in all_h.items():
                tot, _ = chunked_eval_ce(h, model.embed.weight, targets, chunk=chunk)
                nats[label] = nats.get(label, 0.0) + tot
            if model.use_mtp:
                Tm = inp.size(1) - 1
                e_next = model.embed(inp[:, 1:Tm + 1])
                mtp_in = model.mtp_proj(torch.cat(
                    [model.mtp_norm(h_pre)[:, :Tm], e_next], dim=-1))
                cos = model.rope_cos[:Tm].cuda()
                sin = model.rope_sin[:Tm].cuda()
                mh = model.mtp_block(mtp_in, cos, sin, probe=False)
                tot, _ = chunked_eval_ce(
                    mh, model.embed.weight, x[:, 2:Tm + 2], chunk=chunk)
                nats["mtp"] = nats.get("mtp", 0.0) + tot
        n_tok += targets.numel()
    return nats, n_tok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--bytes", type=int, required=True,
                    help="utf-8 bytes of the held-out doc set")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    cfg = ck["cfg"]
    model = A2Model(cfg).cuda().eval()
    model.load_state_dict(ck["model"])
    blocks = np.load(args.blocks)
    nats, n_tok = exit_bpbs(model, blocks, bs=args.bs)
    out = {k: round(v / math.log(2) / args.bytes, 4) for k, v in nats.items()}
    print(json.dumps(dict(ckpt=args.ckpt, bpb_per_exit=out,
                          tokens=n_tok, bytes=args.bytes)), flush=True)


if __name__ == "__main__":
    main()
