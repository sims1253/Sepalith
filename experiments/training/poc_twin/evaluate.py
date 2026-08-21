"""
Final paired evaluation for the twin POC:
- held-out BPB (bits per UTF-8 byte) over the packed 500-row eval stream
- 20 greedy generation samples (10 per arm) for PSM-format coherence
- aligned-by-token-count loss curve table (printed + CSV)
Usage: evaluate.py --ckptA ... --ckptB ... --armA muon --armB adamw
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyGQA, model_config  # noqa: E402

TMP = "/tmp/poc_twin"
POC_DIR = "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin"


@torch.no_grad()
def bpb(model, blocks, total_bytes, bs=8, chunk=4096):
    nats, toks = 0.0, 0
    for i in range(0, len(blocks), bs):
        x = torch.from_numpy(blocks[i:i + bs].astype(np.int64)).cuda()
        logits, _ = model(x[:, :-1], probe=False)
        lg = logits.view(-1, logits.size(-1))
        tg = x[:, 1:].reshape(-1)
        for c in range(0, lg.size(0), chunk):
            nats += F.cross_entropy(lg[c:c + chunk].float(), tg[c:c + chunk],
                                    reduction="sum").item()
        toks += tg.numel()
    ppl_tok = math.exp(min(20, nats / toks))
    return dict(bpb=nats / (total_bytes * math.log(2)),
                nats=nats, tokens=toks, loss_per_tok=nats / toks,
                ppl_tok=ppl_tok)


@torch.no_grad()
def generate_samples(model, tok, prompts, n_new=200):
    outs = []
    for p in prompts:
        x = torch.tensor([p], dtype=torch.long).cuda()
        y = model.generate(x, n_new)[0].tolist()
        outs.append(tok.decode(y))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckptA", required=True)
    ap.add_argument("--ckptB", required=True)
    ap.add_argument("--armA", default="muon")
    ap.add_argument("--armB", default="adamw")
    ap.add_argument("--out", default=os.path.join(POC_DIR, "logs", "final_eval.json"))
    args = ap.parse_args()

    meta = json.load(open(os.path.join(TMP, "meta.json")))
    eb = np.load(os.path.join(TMP, "eval_blocks.npy"), mmap_mode="r")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")

    results = {}
    gens = {}
    for arm, ckpt in [(args.armA, args.ckptA), (args.armB, args.ckptB)]:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = model_config(**{k: v for k, v in ck["cfg"].items()})
        model = TinyGQA(cfg).cuda().eval()
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        results[arm] = dict(bpb(model, eb, meta["eval_bytes"]),
                            ckpt=ckpt, step=ck["step"])
        print(arm, json.dumps(results[arm]), flush=True)
        prompts = json.load(open(os.path.join(TMP, "gen_prompts.json")))[:10]
        gens[arm] = generate_samples(model, tok, prompts)
        del model
        torch.cuda.empty_cache()

    a, b = results[args.armA], results[args.armB]
    rel = (a["bpb"] - b["bpb"]) / b["bpb"]  # >0: armA (muon) worse
    verdict = ("FLAG — sweep falsifier tripped: Muon >=1% worse than AdamW on paired BPB"
               if rel >= 0.01 else
               "PASS: Muon NOT >=1% worse than AdamW on paired BPB (falsifier clear)")
    results["comparison"] = dict(
        rel_bpb_A_vs_B=rel, falsifier_threshold=0.01, verdict=verdict)

    with open(args.out, "w") as f:
        json.dump(dict(results=results,
                       generations={k: v for k, v in gens.items()}), f, indent=1)
    print(json.dumps(results["comparison"], indent=1), flush=True)

    # aligned loss curves by token count
    import glob
    curves = {}
    for arm, tag in [(args.armA, args.armA), (args.armB, args.armB)]:
        recs = []
        for line in open(os.path.join(POC_DIR, "logs", f"{arm}.jsonl")):
            r = json.loads(line)
            if r.get("step") and "loss" in r:
                recs.append((r["tokens"], r["loss"]))
        curves[arm] = recs
    grid = np.arange(5, 101, 5) * 1e6 * 10  # every 50M tokens
    print("\n== loss curves aligned by tokens ==")
    print(f"{'tokens':>12} {'muon':>8} {'adamw':>8} {'diff':>8}")
    ca = dict(curves[args.armA]); cb = dict(curves[args.armB])
    for tk in grid:
        ka = min(ca, key=lambda k: abs(k - tk))
        kb = min(cb, key=lambda k: abs(k - tk))
        if abs(ka - tk) < 50e6 and abs(kb - tk) < 50e6:
            print(f"{int((ka+kb)//2):>12} {ca[ka]:>8.4f} {cb[kb]:>8.4f} {ca[ka]-cb[kb]:>8.4f}")


if __name__ == "__main__":
    main()
