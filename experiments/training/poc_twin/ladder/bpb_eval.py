"""
Final paired BPB eval for the FIM dose ladder (torch, same discipline as
parent evaluate.py): per arm, held-out BPB on BOTH objectives' slices —
  causal floor : eval_blocks_causal.npy / eval_bytes_causal (plain docs)
  PSM/FIM slice: eval_blocks.npy / eval_bytes (the twin-POC PSM eval stream)
The dose decision's causal-floor rule uses the CAUSAL column vs the 0% arm.
"""
import argparse, json, math, os, sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC)
from model import TinyGQA, model_config  # noqa: E402

TMP = "/tmp/poc_twin"
LAD = os.path.join(TMP, "ladder")


@torch.no_grad()
def bpb(model, blocks, total_bytes, bs=8, chunk=4096):
    nats, toks = 0.0, 0
    for i in range(0, len(blocks), bs):
        x = torch.from_numpy(blocks[i:i + bs].astype(np.int64)).cuda()
        h = model.trunk(x[:, :-1], probe=False)
        logits = F.linear(h, model.embed.weight)
        lg = logits.view(-1, logits.size(-1))
        tg = x[:, 1:].reshape(-1)
        for c in range(0, lg.size(0), chunk):
            nats += F.cross_entropy(lg[c:c + chunk].float(), tg[c:c + chunk],
                                    reduction="sum").item()
        toks += tg.numel()
    return dict(bpb=nats / (total_bytes * math.log(2)),
                loss_per_tok=nats / toks, tokens=toks)


@torch.no_grad()
def stop_accuracy(model, tok, rows, bs=4):
    """Teacher-forced stop readout (A2's stop-density gate, POC scale):
    for each held-out PSM row, feed prompt(without terminator)+target span
    and check the greedy next token == first token of the training
    terminator '\\n<|end|>'. Returns (argmax accuracy, mean prob)."""
    term_ids = tok("\n<|end|>", add_special_tokens=False)["input_ids"]
    first_term = term_ids[0]
    n_ok, n_tot, p_sum = 0, 0, 0.0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        seqs, cuts = [], []
        for r in chunk:
            p = r["prompt"]
            if p.endswith("<|end|>\n"):
                p = p[:-len("<|end|>\n")]
            pi = tok(p, add_special_tokens=False)["input_ids"]
            ti = tok(r["target"], add_special_tokens=False)["input_ids"]
            seqs.append(pi + ti)
            cuts.append(len(pi) + len(ti) - 1)   # position whose next token is the terminator
        L = max(len(s) for s in seqs)
        assert L <= 1024
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
        x = x.cuda()
        h = model.trunk(x, probe=False)
        logits = F.linear(h, model.embed.weight).float()
        for j, c in enumerate(cuts):
            lg = logits[j, c]
            ntok = int(lg.argmax().item())
            n_ok += int(ntok == first_term)
            p_sum += float(torch.softmax(lg, -1)[first_term].item())
            n_tot += 1
    return n_ok / max(1, n_tot), p_sum / max(1, n_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="tag:ckpt pairs")
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "bpb_eval.json"))
    args = ap.parse_args()

    meta_p = json.load(open(os.path.join(TMP, "meta.json")))
    meta_l = json.load(open(os.path.join(LAD, "meta_ladder.json")))
    eb_fim = np.load(os.path.join(TMP, "eval_blocks.npy"), mmap_mode="r")
    eb_cau = np.load(os.path.join(LAD, "eval_blocks_causal.npy"), mmap_mode="r")
    fim_rows = json.load(open(os.path.join(LAD, "fim_eval_rows.json")))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")

    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for spec in args.arms:
        tag, ckpt = spec.split(":", 1)
        if tag in out:
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = model_config(**{k: v for k, v in ck["cfg"].items()})
        model = TinyGQA(cfg).cuda().eval()
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        sa, sp = stop_accuracy(model, tok, fim_rows)
        r = dict(ckpt=ckpt, step=ck["step"],
                 tokens_trained=ck["step"] * ck["args"]["tokens_per_step"],
                 causal=bpb(model, eb_cau, meta_l["eval_bytes_causal"]),
                 fim=bpb(model, eb_fim, meta_p["eval_bytes"]),
                 stop_acc=round(sa, 4), stop_prob=round(sp, 4))
        out[tag] = r
        print(tag, json.dumps({k: v for k, v in r.items() if k != "ckpt"}),
              flush=True)
        del model
        torch.cuda.empty_cache()
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)

    # dose table
    print("\n== dose response (causal floor | FIM slice) ==")
    base = out.get([t for t in out if t.endswith("fim0")] [0], None) if any(
        t.endswith("fim0") for t in out) else None
    for tag in sorted(out, key=lambda t: float(t.split("fim")[-1]) if "fim" in t else 99):
        r = out[tag]
        line = (f"{tag:16s} step={r['step']:5d} causal_bpb={r['causal']['bpb']:.4f} "
                f"fim_bpb={r['fim']['bpb']:.4f}")
        if base and tag != [k for k in out if k.endswith('fim0')][0]:
            line += (f"  causal_penalty={100*(r['causal']['bpb']/base['causal']['bpb']-1):+.2f}%")
        print(line, flush=True)


if __name__ == "__main__":
    main()
