"""
MLP neuron-utilization census (Aurora's mechanism-of-action check).

Two independent dead-neuron criteria:

1. MOMENTUM LEVERAGE (the Aurora paper's own in-training criterion,
   arXiv:2606.27715 Sec. 4.1): for the tall up/gate momentum buffers
   M (m x n), thin-SVD M = U S V^T; leverage_i = ||U_i||^2 (row norm sq).
   A neuron is DEAD iff leverage < 1% of the uniform leverage (n/m) in
   BOTH the gate (Wg) and up (Wu) momentum buffer of its layer.
   Read from the checkpoint's saved optimizer state (both last night's
   Muon ckpt and today's Aurora ckpt carry momentum buffers).

2. ACTIVATION RMS (functional): forward K held-out eval blocks; hook the
   input of each block's Wd (= h = silu(Wg x) * (Wu x), the SwiGLU hidden
   activation); per-neuron RMS over all token positions; near-dead iff
   RMS < 5% of the layer median (secondary: < 1%).

Usage: neuron_census.py --name muon --ckpt ../checkpoints/muon_final.pt \
                          --name aurora --ckpt /tmp/poc_twin/ckpt_aurora/final.pt
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from model import TinyGQA, model_config  # noqa: E402

TMP = "/tmp/poc_twin"
POC_DIR = os.path.dirname(HERE)


def momentum_leverage_census(ckpt):
    """Paper criterion: leverage < 1% of uniform (n/m) in BOTH Wg and Wu.
    Optimizer params were added in filtered named_parameters order, which
    for TinyGQA is per block: attn.Wq, Wk, Wv, Wo, Wg, Wu, Wd (7 x 12)."""
    names = []
    for l in range(12):
        names += [f"blocks.{l}.attn.Wq.weight", f"blocks.{l}.attn.Wk.weight",
                  f"blocks.{l}.attn.Wv.weight", f"blocks.{l}.attn.Wo.weight",
                  f"blocks.{l}.Wg.weight", f"blocks.{l}.Wu.weight",
                  f"blocks.{l}.Wd.weight"]
    opt_sd = ckpt["opt"][0]
    state = opt_sd["state"]
    n_params = len(opt_sd["param_groups"][0]["params"])
    assert n_params == len(names), (n_params, len(names))
    res = {}
    for idx, name in enumerate(names):
        if not (name.endswith("Wg.weight") or name.endswith("Wu.weight")):
            continue
        M = state[idx]["momentum_buffer"].float()      # (m, n) tall
        assert tuple(M.shape) == (3072, 768), (name, M.shape)
        m, n = M.shape
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        lev = (U ** 2).sum(dim=1)                       # leverage scores, sum=n
        res[name] = dict(
            leverage=lev,
            uniform=float(n / m),
            dead_frac_1pct=float((lev < 0.01 * n / m).float().mean()),
            dead_frac_10pct=float((lev < 0.10 * n / m).float().mean()),
            cv=float(lev.std() / lev.mean()),
        )
    per_layer = {}
    for l in range(12):
        g = res[f"blocks.{l}.Wg.weight"]; u = res[f"blocks.{l}.Wu.weight"]
        both = (res[f"blocks.{l}.Wg.weight"]["leverage"] < 0.01 * g["uniform"]) & \
               (res[f"blocks.{l}.Wu.weight"]["leverage"] < 0.01 * u["uniform"])
        per_layer[l] = dict(
            dead_frac_Wg_1pct=g["dead_frac_1pct"],
            dead_frac_Wu_1pct=u["dead_frac_1pct"],
            dead_frac_BOTH_1pct=float(both.float().mean()),
            lev_cv_Wg=g["cv"], lev_cv_Wu=u["cv"])
    tot = float(np.mean([v["dead_frac_BOTH_1pct"] for v in per_layer.values()]))
    return per_layer, tot


@torch.no_grad()
def activation_census(model, n_blocks=64, bs=8):
    """Functional criterion: per-neuron activation RMS over eval blocks."""
    sums = {l: torch.zeros(model.cfg["ffn_hidden"], dtype=torch.float64)
            for l in range(model.cfg["n_layers"])}
    sqs = {l: torch.zeros_like(next(iter(sums.values()))) for l in sums}
    counts = {l: 0 for l in sums}
    hooks = []
    # hook the INPUT of Wd — exactly the SwiGLU hidden activation h
    def make_hook(l):
        def hook(mod, inp, out):
            h = inp[0].detach()                     # (B, T, ffn)
            h2 = h.reshape(-1, h.size(-1)).float()
            sums[l] += h2.sum(dim=0).cpu().double()
            sqs[l] += (h2 ** 2).sum(dim=0).cpu().double()
            counts[l] += h2.size(0)
        return hook
    for l, b in enumerate(model.blocks):
        hooks.append(b.Wd.register_forward_hook(make_hook(l)))
    eb = np.load(os.path.join(TMP, "eval_blocks.npy"), mmap_mode="r")
    for i in range(0, min(n_blocks, len(eb)), bs):
        x = torch.from_numpy(eb[i:i + bs].astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x[:, :-1], targets=None, probe=False)
    for h_ in hooks:
        h_.remove()
    per_layer, dead5, dead1 = {}, [], []
    for l in sums:
        rms = (sqs[l] / counts[l]).sqrt()            # (ffn,)
        med = rms.median()
        per_layer[l] = dict(
            act_rms_median=float(med),
            near_dead_frac_5pct_of_median=float((rms < 0.05 * med).float().mean()),
            near_dead_frac_1pct_of_median=float((rms < 0.01 * med).float().mean()),
            act_rms_cv=float(rms.std() / rms.mean()))
        dead5.append(per_layer[l]["near_dead_frac_5pct_of_median"])
        dead1.append(per_layer[l]["near_dead_frac_1pct_of_median"])
    return per_layer, float(np.mean(dead5)), float(np.mean(dead1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--act-blocks", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(POC_DIR, "logs", "arms_census.json"))
    args = ap.parse_args()
    assert len(args.name) == len(args.ckpt)

    out = {}
    for name, ckpt_path in zip(args.name, args.ckpt):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = model_config(**{k: v for k, v in ck["cfg"].items()})
        model = TinyGQA(cfg).cuda().eval()
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        try:
            lev_layer, lev_tot = momentum_leverage_census(ck)
            lev = dict(per_layer=lev_layer, mean_dead_both_1pct=lev_tot)
        except (KeyError, AssertionError) as e:
            lev = dict(per_layer=None, mean_dead_both_1pct=None,
                       note=f"leverage census skipped ({type(e).__name__}: no Muon-class momentum state)")
        act_layer, dead5, dead1 = activation_census(model, n_blocks=args.act_blocks)
        out[name] = dict(
            ckpt=ckpt_path, step=ck["step"],
            leverage_census=lev,
            activation_census=dict(per_layer=act_layer,
                                   mean_near_dead_5pct_of_median=dead5,
                                   mean_near_dead_1pct_of_median=dead1))
        print(f"[{name}] leverage-dead(BOTH,1pct)={lev['mean_dead_both_1pct']} "
              f"act-near-dead(5pct)={dead5:.5f} (1pct)={dead1:.5f}", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
