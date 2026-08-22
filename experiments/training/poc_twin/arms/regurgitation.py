"""
Regurgitation canary (arm D, the sweep OPT-5 memorization readout).

20 greedy (temperature-0) generations of 200 new tokens per model:
  - 10 IN-CORPUS prompts: first 120 tokens of the first 10 docs of the
    seeded half split (in-corpus for BOTH the full-corpus and half-corpus
    models — paired).
  - 10 HELD-OUT prompts: gen_prompts.json rows 10..19 (eval corpus).
Metric (pre-registered): longest verbatim training-span reproduced — the
longest contiguous run of GENERATED tokens that appears anywhere in the
model's OWN training token stream. Implementation: uint64 polynomial-hash
index of all 12-grams of the training stream (sorted hashes + binary
search, collisions verified by direct token comparison), then forward
extension from each confirmed anchor. Sub-12-token spans are the noise
floor and are not counted.

Also reports the prompt-continuation match (for in-corpus prompts: how
far the greedy continuation agrees with the training document's actual
continuation), which is memorization in the plainest sense.
"""
import argparse, json, os, sys
import numpy as np
import torch

np.seterr(over="ignore")  # uint64 hash arithmetic wraps mod 2^64 by design

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from model import TinyGQA, model_config  # noqa: E402

TMP = "/tmp/poc_twin"
POC_DIR = os.path.dirname(HERE)
NGRAM = 12
B = np.uint64(0x100000001B3)  # FNV-style odd base for the poly hash


def stream_from_blocks(path):
    """Recover the packed token stream (overlap-1 blocks tile it exactly)."""
    blocks = np.load(path, mmap_mode="r")
    s = np.empty(blocks.shape[0] * 1024 + 1, dtype=np.int32)
    s[0] = blocks[0, 0]
    s[1:] = np.asarray(blocks[:, 1:]).reshape(-1)
    return s


def build_index(stream):
    """Sorted (hash, position) index with exact verification on query."""
    t = stream.astype(np.uint64)
    n_idx = len(t) - NGRAM + 1
    h = t[0:n_idx].copy()
    for j in range(1, NGRAM):
        h = h * B + t[j:n_idx + j]
    order = np.argsort(h, kind="stable")
    return dict(stream=stream, t=t, hashes=h[order], pos=order)


def find_gram(idx, gram):
    g = np.asarray(gram, dtype=np.uint64)
    q = g[0]
    for j in range(1, NGRAM):
        q = q * B + g[j]
    lo = int(np.searchsorted(idx["hashes"], q, side="left"))
    hi = int(np.searchsorted(idx["hashes"], q, side="right"))
    out = []
    stream = idx["stream"]
    for p in idx["pos"][lo:hi]:
        p = int(p)
        if int(stream[p + NGRAM - 1]) == int(gram[NGRAM - 1]) and \
                np.array_equal(stream[p:p + NGRAM], np.asarray(gram)):
            out.append(p)
    return out


def longest_verbatim_span(idx, gen):
    """Longest contiguous span of `gen` (token list) present in the stream.
    Anchored on exact 12-gram matches at every start position, extended
    forward. Exhaustive over start positions (no skipping) — 200 queries
    per generation is trivially cheap and never undercounts."""
    best, best_at = 0, -1
    s = 0
    while s + NGRAM <= len(gen):
        hits = find_gram(idx, gen[s:s + NGRAM])
        if hits:
            p = hits[0]
            L = NGRAM
            while (p + L < len(idx["stream"]) and s + L < len(gen)
                   and int(idx["stream"][p + L]) == int(gen[s + L])):
                L += 1
            if L > best:
                best, best_at = L, s
        s += 1
    return best, best_at


@torch.no_grad()
def greedy_new_tokens(model, prompt, n_new=200):
    x = torch.tensor([prompt], dtype=torch.long).cuda()
    out = model.generate(x, n_new, greedy=True)[0].tolist()
    return out[len(prompt):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--blocks", action="append", required=True,
                    help="training blocks .npy for THIS model's corpus")
    ap.add_argument("--n-new", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(POC_DIR, "logs", "arms_regurgitation.json"))
    args = ap.parse_args()
    assert len(args.name) == len(args.ckpt) == len(args.blocks)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    train_prompts = json.load(open(os.path.join(TMP, "gen_prompts_train.json")))[:10]
    heldout_prompts = json.load(open(os.path.join(TMP, "gen_prompts.json")))[10:20]

    results = {}
    for name, ckpt_path, blocks_path in zip(args.name, args.ckpt, args.blocks):
        print(f"[{name}] building 12-gram index over {blocks_path}", flush=True)
        idx = build_index(stream_from_blocks(blocks_path))
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = model_config(**{k: v for k, v in ck["cfg"].items()})
        model = TinyGQA(cfg).cuda().eval()
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        gens = {}
        for pkind, prompts in [("train", train_prompts), ("heldout", heldout_prompts)]:
            per = []
            for pi, p in enumerate(prompts):
                gen = greedy_new_tokens(model, p, args.n_new)
                span, at = longest_verbatim_span(idx, gen)
                # prompt-continuation agreement (memorization proper)
                cont = -1
                if pkind == "train":
                    locs = find_gram(idx, p[:NGRAM])
                    for loc in locs:
                        if list(idx["stream"][loc:loc + len(p)]) == list(p):
                            cont = 0
                            while (loc + len(p) + cont < len(idx["stream"])
                                   and cont < len(gen)
                                   and int(idx["stream"][loc + len(p) + cont]) == int(gen[cont])):
                                cont += 1
                            break
                per.append(dict(prompt=pi, longest_span=int(span),
                                span_at=int(at), prompt_cont_match=int(cont),
                                text=tok.decode(gen)))
                print(f"  [{name}/{pkind}#{pi}] span={span} cont={cont}", flush=True)
            gens[pkind] = dict(
                per_prompt=per,
                max_span=int(max(r["longest_span"] for r in per)),
                mean_span=float(np.mean([r["longest_span"] for r in per])),
                n_ge30=int(sum(r["longest_span"] >= 30 for r in per)),
                n_ge60=int(sum(r["longest_span"] >= 60 for r in per)),
                mean_cont_match=float(np.mean([r["prompt_cont_match"] for r in per])),
                max_cont_match=int(max(r["prompt_cont_match"] for r in per)))
        results[name] = dict(ckpt=ckpt_path, corpus_blocks=blocks_path, **gens)
        del model
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}", flush=True)
    for name, r in results.items():
        for k in ("train", "heldout"):
            print(f"{name:12s} {k:8s} max_span={r[k]['max_span']:3d} "
                  f"mean={r[k]['mean_span']:6.1f} ge30={r[k]['n_ge30']} "
                  f"ge60={r[k]['n_ge60']} mean_cont={r[k].get('mean_cont_match', -1):.1f}")


if __name__ == "__main__":
    main()
