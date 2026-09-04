#!/usr/bin/env python3
"""O3-S0: suffix-entropy + length telemetry on banked RL runs.

Queue docs/EXPERIMENT-QUEUE.md §3 O3, pre-registered question: does a
suffix-first entropy rise precede stall/collapse (flattening exact /
reward) in the banked GRPO runs? Mapped from the OPD papers
(2604.13016 / 2609.04172) to our teacher-free GRPO (rl_smoke.py,
reward = exact + 0.2*line_f1, num_generations=4).

Two legs:

  curves  (CPU, anytime)  — parses what the runs ALREADY logged:
       * rl_metrics.jsonl (per optimizer step: reward, exact, per-family
         exact/n — the MetricsCb flush of the reward-fn stash)
       * trainer_state.json in the LAST checkpoint (TRL log_history at
         logging_steps=10: grad_norm, frac_reward_zero_std [= share of
         groups with zero reward std = unanimous groups = zero
         advantage: THE advantage-concentration signal], reward_std,
         kl, completion lengths, clip ratios)
     and aligns signal onsets against exact/reward stall points.

  replay  (GPU, claim first) — the one thing never logged: per-position
     output entropy and top-k token stability. Re-merges the SFT base
     (base HF model + banked SFT LoRA, in memory), loads each banked
     checkpoint LoRA, replays a fixed rebuilt prompt set (same
     build_dataset seed/quotas/data the run used — NAS files verified
     present) and records, per weight point (step 0 base + each
     surviving checkpoint):
       * own-rollout greedy generation with per-position next-token
         entropy (nats) — suffix (positions 96-191) vs forward (0-95),
       * teacher-forced pass on the BASE's reference completion ->
         matched-token-position entropy + top-1/top-5 churn between
         consecutive weight points.

Pre-registered stall/signal definitions (fixed before reading curves):
  smoothed exact   = centered rolling mean, W=21 steps
  slope30(t)       = OLS slope of smoothed exact over [t-29, t]
  stall_step       = first t* >= 40 with slope30 < 5e-4 for ALL windows
                     in [t*, T-10], given a real rise earlier
                     (max slope30 > 1.5e-3) and >= 30 steps after t*
  collapse_step    = argmax of smoothed exact if final smoothed is
                     >= 0.10 below that max (else no collapse)
  onset(signal,th) = first log step where signal crosses threshold and
                     stays crossed for >= 3 consecutive logs

Usage:
  python entropy_telemetry.py curves [--runs rl_grpo_v2c,...] [--out DIR]
  python entropy_telemetry.py inventory
  python entropy_telemetry.py replay --run rl_grpo_v2c,...  # GPU: claim!
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL_DIR = HERE.parent                       # experiments/training
RUNS = Path("/mnt/h/sepalith/runs")
DEFAULT_OUT = HERE / "results" / "o3_s0"

# ---------------------------------------------------------------- run configs
# Verified against metrics family shares + docs/research night notes
# (2026-08-23-night-session.md) + adapter_config base paths. Replay uses
# these to rebuild each run's prompt set via rl_smoke.build_dataset.
V6 = dict(base_lora="/mnt/h/sepalith/runs/sft_v6_minicpm5/final_lora",
          data="/mnt/h/sepalith/datasets/sft_v6/train.jsonl",
          run2=False, no_op_n=None)
V7 = dict(base_lora="/mnt/h/sepalith/runs/sft_v7_minicpm5/final_lora",
          data="/mnt/h/sepalith/datasets/sft_v7/train.jsonl",
          run2=True, no_op_n=None)
RUN_CONFIGS = {
    "rl_grpo_v1": V6,
    "rl_grpo_v2": V7,
    "rl_grpo_v2b": dict(V7, data="/mnt/h/sepalith/datasets/sft_v8/train.jsonl"),
    "rl_grpo_v2c": dict(V7, data="/mnt/h/sepalith/datasets/sft_v8_1/train.jsonl"),
    "rl_grpo_v2d": dict(V7, data="/mnt/h/sepalith/datasets/sft_v8_1/train.jsonl",
                        no_op_n=800),                    # knee test ~15% draw
    "rl_grpo_v4_tether": V6,
    "rl_grpo_v5_loo_unnorm": V6,
    "rl_grpo_t1_dapo": V6,
}
REFINE = "/mnt/h/sepalith/datasets/rl_refinement_v1.jsonl"

ALL_RUNS = ["rl_grpo_v1", "rl_grpo_v2", "rl_grpo_v2b", "rl_grpo_v2c",
            "rl_grpo_v2d", "rl_grpo_v3", "rl_grpo_v3b", "rl_grpo_v4_tether",
            "rl_grpo_v5_loo_unnorm", "rl_grpo_t1_dapo"]


# ---------------------------------------------------------------- loading
def load_run(run):
    d = RUNS / run
    metrics = []
    with open(d / "rl_metrics.jsonl") as f:
        for line in f:
            metrics.append(json.loads(line))
    ckpts = sorted((p for p in d.glob("checkpoint-*") if p.is_dir()),
                   key=lambda p: int(p.name.split("-")[1]))
    loghist, ckpt_steps = [], []
    if ckpts:
        st = json.load(open(ckpts[-1] / "trainer_state.json"))
        loghist = [e for e in st["log_history"] if "loss" in e]
        ckpt_steps = [int(p.name.split("-")[1]) for p in ckpts]
    return dict(run=run, metrics=metrics, loghist=loghist,
                ckpt_steps=ckpt_steps,
                final_lora=(d / "final_lora").exists())


# ---------------------------------------------------------------- smoothing
def fill_nan(y):
    y = np.asarray(y, float)
    bad = ~np.isfinite(y)
    if bad.all() or not bad.any():
        return y
    idx = np.where(~bad)[0]
    y[bad] = np.interp(np.where(bad)[0], idx, y[idx])
    return y


def roll_mean(y, w):
    y = fill_nan(y)
    if len(y) < w:
        return y.copy()
    sm = np.convolve(y, np.ones(w) / w, mode="valid")
    pad = w // 2
    out = np.empty_like(y)
    out[:pad] = sm[0]
    out[pad:pad + len(sm)] = sm
    out[pad + len(sm):] = sm[-1]
    return out


def roll_slope(y, w=30):
    y = fill_nan(y)
    out = np.zeros(len(y))
    for i in range(len(y)):
        lo = max(0, i - w + 1)
        seg = y[lo:i + 1]
        if len(seg) < max(8, w // 3):
            continue
        xs = np.arange(len(seg), dtype=float)
        xs = xs - xs.mean()
        out[i] = float((xs * (seg - seg.mean())).sum()
                       / max((xs ** 2).sum(), 1e-9))
    return out


# ------------------------------------------------- stall / onset machinery
def stall_step(smoothed, min_start=40, slope_flat=5e-4, slope_rise=1.5e-3,
               tail=10, need_after=30):
    """First t where the run has stopped improving for good."""
    slope = roll_slope(smoothed)
    T = len(smoothed)
    if not len(slope) or slope.max() <= slope_rise:
        return None, slope
    for t in range(min_start, T - tail):
        if T - t < need_after:
            break
        if slope[t] < slope_flat and all(s < slope_flat for s in slope[t:T - tail]):
            return t, slope
    return None, slope


def collapse_step(smoothed, drop=0.10):
    imax = int(np.argmax(smoothed))
    if smoothed[-1] <= smoothed[imax] - drop and imax < len(smoothed) - 15:
        return imax
    return None


def onset(steps, vals, threshold, direction="above", sustain=3):
    """First step where the series crosses threshold and stays crossed."""
    ok = [v >= threshold if direction == "above" else v <= threshold
          for v in vals]
    run = 0
    for s, o in zip(steps, ok):
        run = run + 1 if o else 0
        if run >= sustain:
            return int(s) - (sustain - 1) * 10        # first-crossing log step
    return None


# ---------------------------------------------------------------- curves leg
SIGNALS = [
    ("frac_reward_zero_std>=0.70", "frac_reward_zero_std", 0.70, "above"),
    ("frac_reward_zero_std>=0.80", "frac_reward_zero_std", 0.80, "above"),
    ("grad_norm<=0.010", "grad_norm", 0.010, "below"),
    ("grad_norm<=0.001", "grad_norm", 0.001, "below"),
    ("reward_std<=0.150", "reward_std", 0.150, "below"),
]
FAMS = ("rename_propagation", "format_propagation", "no_op")


def analyze_run(run):
    R = load_run(run)
    m = R["metrics"]
    steps = [r["step"] for r in m]
    exact = [r["exact"] for r in m]
    reward = [r["reward"] for r in m]
    sm = roll_mean(exact, 21)
    st_all, slope = stall_step(sm)
    col = collapse_step(sm)

    fam_out = {}
    for fam in FAMS:
        key = f"exact_{fam}"
        if key not in m[0]:
            continue
        fe = roll_mean([r.get(key) for r in m], 31)
        fs = fcol = None
        if np.isfinite(fe).sum() > 40:
            fs, _ = stall_step(fe, min_start=40, need_after=25)
            fcol = collapse_step(fe)
        fam_out[fam] = dict(stall=fs, collapse=fcol,
                            first=round(float(fe[0]), 3),
                            last=round(float(fe[-1]), 3))

    lh = R["loghist"]
    lh_steps = [e["step"] for e in lh]
    sig_out = {}
    for name, key, th, dr in SIGNALS:
        pairs = [(e["step"], e[key]) for e in lh if key in e
                 and np.isfinite(e[key])]
        if len(pairs) < 5:
            continue
        sig_out[name] = dict(
            onset=onset([s for s, _ in pairs], [v for _, v in pairs], th, dr),
            first=round(pairs[0][1], 4), last=round(pairs[-1][1], 4),
            median=round(float(np.median([v for _, v in pairs])), 4))

    leads = {}
    if st_all is not None:
        for name, so in sig_out.items():
            if so["onset"] is not None:
                leads[name] = int(st_all - so["onset"])

    comp = [e["completions/mean_length"] for e in lh
            if "completions/mean_length" in e]
    return dict(
        run=run, n_steps=steps[-1], ckpt_steps=R["ckpt_steps"],
        exact_first=round(float(exact[0]), 3),
        exact_last=round(float(exact[-1]), 3),
        exact_smooth_last=round(float(sm[-1]), 3),
        exact_smooth_max=round(float(sm.max()), 3),
        stall=st_all, collapse=col, family=fam_out,
        signals=sig_out, leads=leads,
        completion_length=dict(min=float(np.min(comp)),
                               max=float(np.max(comp))) if comp else None,
        _series=dict(steps=list(map(int, steps)), exact=list(map(float, exact)),
                     reward=list(map(float, reward)), smoothed=list(map(float, sm)),
                     slope=list(map(float, slope)),
                     lh_steps=list(map(int, lh_steps)),
                     lh={k: [e.get(k) for e in lh] for k in
                         ("grad_norm", "frac_reward_zero_std", "reward_std",
                          "kl", "completions/mean_length", "loss")}),
    )


def plot_run(res, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    s = res["_series"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax[0].plot(s["steps"], s["exact"], alpha=.25, lw=.7, label="exact (raw)")
    ax[0].plot(s["steps"], s["smoothed"], lw=1.6, label="exact (smoothed)")
    ax[0].plot(s["steps"], s["reward"], alpha=.25, lw=.7, label="reward (raw)")
    for x, lab, c in ((res["stall"], "stall", "red"),
                      (res["collapse"], "collapse", "darkred")):
        if x is not None:
            ax[0].axvline(x, color=c, ls="--", lw=1, label=f"{lab} @{x}")
    ax[0].legend(fontsize=7); ax[0].set_ylabel("exact / reward")
    ax[0].set_title(f'{res["run"]} — O3-S0 telemetry')
    ax[1].plot(s["lh_steps"], s["lh"]["frac_reward_zero_std"], ".-",
               label="frac_reward_zero_std (unanimous groups)")
    ax[1].plot(s["lh_steps"], s["lh"]["reward_std"], ".-", label="reward_std")
    ax[1].axhline(0.7, color="gray", ls=":", lw=.8)
    for x in res["ckpt_steps"]:
        for a in ax:
            a.axvline(x, color="green", ls=":", lw=.8)
    ax[1].legend(fontsize=7); ax[1].set_ylabel("concentration")
    ax[2].plot(s["lh_steps"], s["lh"]["grad_norm"], ".-", label="grad_norm")
    ax[2].set_yscale("log"); ax[2].legend(fontsize=7)
    ax[2].set_ylabel("grad_norm"); ax[2].set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(outdir / f'{res["run"]}_curves.png', dpi=130)
    plt.close(fig)
    return True


def cmd_curves(args):
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    runs = args.runs.split(",") if args.runs else ALL_RUNS
    summary = []
    for r in runs:
        res = analyze_run(r)
        ser = res.pop("_series")
        json.dump(dict(res, series=ser), open(outdir / f"{r}_curves.json", "w"),
                  indent=1)
        png = plot_run(dict(res, _series=ser), outdir)
        summary.append(res)
        fam_s = ", ".join(f"{k}: stall={v['stall']} col={v['collapse']}"
                          for k, v in res["family"].items())
        print(f"{r}: stall={res['stall']} collapse={res['collapse']} | {fam_s}")
    json.dump(summary, open(outdir / "curves_summary.json", "w"), indent=1)
    print("\n-- lead/lag (stall_step - onset; + = signal EARLIER) --")
    for r in summary:
        if r["stall"] is None:
            print(f"{r['run']}: no mid-run stall detected")
            continue
        for k, v in sorted(r["leads"].items(), key=lambda kv: -kv[1]):
            print(f"  {r['run']}: {k} onset lead {v:+d} steps")


def cmd_inventory(args):
    out = {}
    for r in ALL_RUNS:
        R = load_run(r)
        out[r] = dict(
            n_metric_steps=len(R["metrics"]),
            last_step=R["metrics"][-1]["step"],
            metric_fields=sorted(R["metrics"][0].keys()),
            ckpt_steps=R["ckpt_steps"], final_lora=R["final_lora"],
            loghist_n=len(R["loghist"]),
            loghist_fields=sorted({k for e in R["loghist"] for k in e}),
            mtimes={p.name: int(p.stat().st_mtime)
                    for p in (RUNS / r).iterdir()},
        )
    print(json.dumps(out, indent=1))


# ------------------------------------------------------ prediction test
def cmd_predict(args):
    """Does banked telemetry at step t predict REMAINING exact gain
    beyond exact(t) itself? (the honest early-warning test — raw onsets
    are confounded because concentration rises as exact rises)"""
    probes = [50, 100, 150, 200, 250]
    rows = []
    for r in (args.runs.split(",") if args.runs else ALL_RUNS):
        res = analyze_run(r)
        s = res["_series"]
        sm = fill_nan(s["smoothed"])
        steps = np.array(s["steps"])
        lh = {k: fill_nan(s["lh"][k]) for k in
              ("frac_reward_zero_std", "reward_std", "grad_norm")}
        lh_steps = np.array(s["lh_steps"])
        for t in probes:
            if t >= len(sm) - 20:
                continue
            ex_t = float(sm[t])
            rem = float(sm[-1] - sm[t])
            nxt = float(sm[min(t + 50, len(sm) - 1)] - sm[t])
            zi = int(np.argmin(np.abs(lh_steps - t)))
            if abs(lh_steps[zi] - t) > 12:
                continue
            rows.append(dict(run=r, t=t, exact=ex_t, rem=rem, nxt50=nxt,
                             Z=float(lh["frac_reward_zero_std"][zi]),
                             R=float(lh["reward_std"][zi]),
                             G=float(lh["grad_norm"][zi])))
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "run"}

    def pear(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        good = np.isfinite(a) & np.isfinite(b)
        if good.sum() < 5:
            return None
        a, b = a[good] - a[good].mean(), b[good] - b[good].mean()
        den = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()))
        return float((a * b).sum() / den) if den else None

    def partial(x, y, c):
        """corr(x,y) controlling for c (single control, simple residuals)."""
        rxy, rxc, ryc = pear(x, y), pear(x, c), pear(y, c)
        if None in (rxy, rxc, ryc):
            return None
        den = np.sqrt((1 - rxc ** 2) * (1 - ryc ** 2))
        return float((rxy - rxc * ryc) / den) if den > 1e-12 else None

    out = dict(n_points=len(rows), rows=rows)
    for target in ("rem", "nxt50"):
        out[target] = {}
        for feat in ("exact", "Z", "R", "G"):
            out[target][f"r_{feat}"] = pear(arr[feat], arr[target])
        for feat in ("Z", "R", "G"):
            out[target][f"r_{feat}|exact"] = partial(
                arr[feat], arr[target], arr["exact"])
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    json.dump(out, open(Path(args.out) / "predict_test.json", "w"), indent=1)


# ---------------------------------------------------------------- replay leg
def entropy_rows(logits_row):
    """Per-position entropy (nats) of one logits row [vocab]."""
    lp = torch.log_softmax(logits_row.float(), dim=-1)
    return float((-(lp.exp() * lp)).sum().item())


def cmd_replay(args):
    """GPU leg — claim in comms/gpu.md BEFORE running this."""
    global torch
    sys.path.insert(0, str(RL_DIR))            # rl_smoke import
    import random

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import rl_smoke

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    n_prompts, batch, max_new = args.n_prompts, 8, 192
    runs = args.run.split(",")
    HF_BASE = "openbmb/MiniCPM5-1B"

    # group runs by merged base (one in-memory merge per group)
    groups = {}
    for r in runs:
        groups.setdefault(RUN_CONFIGS[r]["base_lora"], []).append(r)

    results = {}
    for base_lora, grp in groups.items():
        print(f"[replay] merging base: {base_lora}", flush=True)
        raw = AutoModelForCausalLM.from_pretrained(
            HF_BASE, dtype=torch.bfloat16, device_map="cuda")
        base = PeftModel.from_pretrained(raw, base_lora).merge_and_unload()
        base.eval()
        tok = AutoTokenizer.from_pretrained(HF_BASE)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        for run in grp:
            cfg = RUN_CONFIGS[run]
            quotas = dict(rl_smoke.FAMILY_QUOTA_RUN2 if cfg["run2"]
                          else rl_smoke.FAMILY_QUOTA)
            if cfg["no_op_n"]:
                quotas["no_op"] = cfg["no_op_n"]
            rows, dstat = rl_smoke.build_dataset(
                tok, quotas, data_path=Path(cfg["data"]), refine_path=REFINE)
            rng = random.Random(777)
            idx = sorted(rng.sample(range(len(rows)),
                                    min(n_prompts, len(rows))))
            prompts = [rows[i] for i in idx]
            print(f"[replay {run}] {len(prompts)} prompts rebuilt "
                  f"(dataset stats: {dstat['counts']})", flush=True)

            points = [(0, None)] + [(cs, RUNS / run / f"checkpoint-{cs}")
                                    for cs in load_run(run)["ckpt_steps"]]

            tok.padding_side = "left"       # for generation
            ref_store = [None] * len(prompts)   # base's greedy completions
            run_res = {"prompts": len(prompts), "points": [],
                       "topk_churn": [], "prompt_families":
                       [p["family"] for p in prompts]}
            topk_all = []

            for step, lora in points:
                model = base if lora is None else \
                    PeftModel.from_pretrained(base, str(lora))
                model.eval()
                own_H = np.zeros((len(prompts), max_new))
                tf_H = np.zeros((len(prompts), max_new))
                topk_own = np.zeros((len(prompts), max_new, 5), dtype=np.int64)
                topk_tf = np.zeros((len(prompts), max_new, 5), dtype=np.int64)

                for i in range(0, len(prompts), batch):
                    chunk = prompts[i:i + batch]
                    enc = tok([p["prompt"] for p in chunk],
                              return_tensors="pt", padding=True,
                              add_special_tokens=False).to("cuda")
                    with torch.no_grad():
                        o = model.generate(
                            **enc, max_new_tokens=max_new, do_sample=False,
                            output_scores=True, return_dict_in_generate=True,
                            pad_token_id=tok.pad_token_id,
                            use_cache=True)
                    plen = enc.input_ids.shape[1]
                    gen = o.sequences[:, plen:plen + max_new]   # [b, max_new]
                    for t, sc in enumerate(o.scores[:max_new]):
                        lp = torch.log_softmax(sc.float(), dim=-1)
                        own_H[i:i + len(chunk), t] = \
                            (-(lp.exp() * lp)).sum(-1).cpu().numpy()
                        topk_own[i:i + len(chunk), t] = \
                            torch.topk(sc, 5, dim=-1).indices.cpu().numpy()

                    # reference tokens: the BASE's greedy rollout (matched
                    # token positions across all weight points)
                    ref = gen if step == 0 else ref_store[i:i + len(chunk)]

                    # teacher-forced pass (right padding, own lengths)
                    ids_list = []
                    for j, p in enumerate(chunk):
                        pid = enc.input_ids[j][enc.attention_mask[j] == 1]
                        ids_list.append(torch.cat([pid.cpu(), ref[j].cpu()]))
                    maxlen = max(x.numel() for x in ids_list)
                    ids = torch.full((len(chunk), maxlen), tok.pad_token_id,
                                     dtype=torch.long)
                    att = torch.zeros((len(chunk), maxlen), dtype=torch.long)
                    for j, x in enumerate(ids_list):
                        ids[j, :x.numel()] = x
                        att[j, :x.numel()] = 1
                    with torch.no_grad():
                        logits = model(input_ids=ids.cuda(),
                                       attention_mask=att.cuda()).logits
                    for j in range(len(chunk)):
                        pj = int(att[j].sum()) - max_new   # prompt length
                        for t in range(max_new):
                            pos = pj + t - 1
                            if 0 <= pos < logits.shape[1]:
                                row = logits[j, pos]
                                tf_H[i + j, t] = entropy_rows(row)
                                topk_tf[i + j, t] = torch.topk(
                                    row, 5).indices.cpu().numpy()
                    if step == 0:
                        for j in range(len(chunk)):
                            ref_store[i + j] = gen[j].cpu()

                run_res["points"].append(dict(
                    step=step,
                    own_entropy_by_pos=[round(float(v), 4)
                                        for v in own_H.mean(0)],
                    own_entropy_fwd=round(float(own_H[:, :96].mean()), 4),
                    own_entropy_sfx=round(float(own_H[:, 96:].mean()), 4),
                    tf_entropy_by_pos=[round(float(v), 4)
                                       for v in tf_H.mean(0)],
                    tf_entropy_fwd=round(float(tf_H[:, :96].mean()), 4),
                    tf_entropy_sfx=round(float(tf_H[:, 96:].mean()), 4),
                ))
                topk_all.append((topk_own, topk_tf))
                if lora is not None:
                    model = model.unload()   # drop adapter, keep base
                print(f"[replay {run}] step {step}: own H fwd/sfx = "
                      f"{own_H[:, :96].mean():.3f}/{own_H[:, 96:].mean():.3f} "
                      f"tf H fwd/sfx = "
                      f"{tf_H[:, :96].mean():.3f}/{tf_H[:, 96:].mean():.3f}",
                      flush=True)

            # top-k churn between consecutive weight points, own rollout
            # (same prompt, own token contexts) AND teacher-forced on the
            # base's reference completion (matched tokens AND positions)
            def _churn(tk0, tk1):
                top1 = float((tk0[:, :, 0] == tk1[:, :, 0]).mean())
                top5 = float(np.mean([
                    len(set(tk0[i, t].tolist()) & set(tk1[i, t].tolist())) / 5
                    for i in range(tk0.shape[0])
                    for t in range(tk0.shape[1])]))
                return round(top1, 4), round(top5, 4)

            for a in range(len(run_res["points"]) - 1):
                o0, o1 = _churn(topk_all[a][0], topk_all[a + 1][0])
                f0, f1 = _churn(topk_all[a][1], topk_all[a + 1][1])
                run_res["topk_churn"].append(dict(
                    from_step=run_res["points"][a]["step"],
                    to_step=run_res["points"][a + 1]["step"],
                    own_top1_match=o0, own_top5_overlap=o1,
                    tf_top1_match=f0, tf_top5_overlap=f1))
            results[run] = run_res
            json.dump(results, open(outdir / f"{run}_entropy.json", "w"),
                      indent=1)
        del base, raw
        torch.cuda.empty_cache()
    json.dump(results, open(outdir / "entropy_replay.json", "w"), indent=1)
    print("[replay] done")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves")
    c.add_argument("--runs", default=None)
    c.add_argument("--out", default=str(DEFAULT_OUT))
    i = sub.add_parser("inventory")
    p = sub.add_parser("predict")
    p.add_argument("--runs", default=None)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    r = sub.add_parser("replay")
    r.add_argument("--run", required=True, help="comma-separated run ids")
    r.add_argument("--n-prompts", type=int, default=48)
    r.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    if args.cmd == "curves":
        cmd_curves(args)
    elif args.cmd == "inventory":
        cmd_inventory(args)
    elif args.cmd == "predict":
        Path(args.out).mkdir(parents=True, exist_ok=True)
        cmd_predict(args)
    else:
        cmd_replay(args)


if __name__ == "__main__":
    main()
