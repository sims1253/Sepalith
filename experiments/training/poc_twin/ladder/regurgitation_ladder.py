"""
Memorization canary for the FIM dose ladder (per-dose regurgitation readout).

Same instrument as arms/regurgitation.py (12-gram poly-hash index over the
training token stream, exact verification, forward extension; sub-12-token
spans are the noise floor): does higher PSM-FIM dose amplify verbatim
memorization (arXiv 2605.22981 prediction)?

Both packed streams are IDENTICAL across dose arms (only the mix differs),
so each index is built once and reused for all four arms:
  causal index : /tmp/poc_twin/ladder/train_blocks_causal.npy
  fim index    : /tmp/poc_twin/train_blocks.npy  (the PSM text stream)
Prompts (120 tok, greedy 200 new):
  train_causal : first 10 causal-rendered train docs   (in-corpus, plain)
  heldout      : eval rows 10..19 causal-rendered      (noise floor)
  train_fim    : first 10 PSM-text train docs          (in-corpus, PSM)
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
ARMS = os.path.join(POC, "arms")
sys.path.insert(0, POC)
sys.path.insert(0, ARMS)
from model import TinyGQA, model_config                    # noqa: E402
from regurgitation import (build_index, stream_from_blocks,  # noqa: E402
                           greedy_new_tokens, longest_verbatim_span,
                           find_gram, NGRAM)

LAD = "/tmp/poc_twin/ladder"
FIM_BLOCKS = "/tmp/poc_twin/train_blocks.npy"
CAUSAL_BLOCKS = os.path.join(LAD, "train_blocks_causal.npy")


def eval_prompts(model, prompts, idx, kind):
    per = []
    for pi, p in enumerate(prompts):
        gen = greedy_new_tokens(model, p, 200)
        span, at = longest_verbatim_span(idx, gen)
        cont = -1
        if kind == "train":
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
                        prompt_cont_match=int(cont)))
        print(f"  [{kind}#{pi}] span={span} cont={cont}", flush=True)
    return dict(per_prompt=per,
                max_span=int(max(r["longest_span"] for r in per)),
                mean_span=float(np.mean([r["longest_span"] for r in per])),
                n_ge30=int(sum(r["longest_span"] >= 30 for r in per)),
                n_ge60=int(sum(r["longest_span"] >= 60 for r in per)),
                mean_cont_match=float(np.mean([r["prompt_cont_match"] for r in per])),
                max_cont_match=int(max(r["prompt_cont_match"] for r in per)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="tag:ckptpath pairs like ladder_fim0:/tmp/.../final.pt")
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "canary.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    p_train_causal = json.load(open(os.path.join(LAD, "canary_prompts_causal_train.json")))
    p_heldout = json.load(open(os.path.join(LAD, "canary_prompts_causal_heldout.json")))
    p_train_fim = json.load(open(os.path.join(LAD, "canary_prompts_fim_train.json")))

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}

    print("[index] causal stream...", flush=True)
    idx_c = build_index(stream_from_blocks(CAUSAL_BLOCKS))
    print("[index] fim stream...", flush=True)
    idx_f = build_index(stream_from_blocks(FIM_BLOCKS))

    for spec in args.arms:
        tag, ckpt = spec.split(":", 1)
        if tag in results:
            print(f"[{tag}] already done, skip", flush=True)
            continue
        print(f"[{tag}] {ckpt}", flush=True)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = model_config(**{k: v for k, v in ck["cfg"].items()})
        model = TinyGQA(cfg).cuda().eval()
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        results[tag] = dict(ckpt=ckpt,
                            train_causal=eval_prompts(model, p_train_causal, idx_c, "train"),
                            heldout=eval_prompts(model, p_heldout, idx_c, "heldout"),
                            train_fim=eval_prompts(model, p_train_fim, idx_f, "train"))
        del model
        torch.cuda.empty_cache()
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        r = results[tag]
        for k in ("train_causal", "heldout", "train_fim"):
            print(f"{tag:14s} {k:13s} max_span={r[k]['max_span']:3d} "
                  f"mean={r[k]['mean_span']:6.1f} ge30={r[k]['n_ge30']} "
                  f"ge60={r[k]['n_ge60']} mean_cont={r[k]['mean_cont_match']:.1f}",
                  flush=True)

    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
