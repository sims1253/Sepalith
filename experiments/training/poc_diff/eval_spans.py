"""Task 5/6: span-level eval harness — scores BOTH arms identically.

AR arm  : poc_twin muon_final.pt, greedy decode of the span after the PSM
          prompt tokens (right-padded batch; causal attention makes pads
          invisible), stop at the "\\n<|end|>" token suffix / eos / cap
          min(384, 1024-prompt_len) — the ladder GEN_TOK_CAP convention.
MD arm  : poc_diff md_final.pt, low-confidence-remasking sampler at
          steps {8,16,32,64}; k=1 greedy for the pre-registered point
          metrics, k=8 temperature-1.0 for the multimodality metrics
          (best-of-8, distinct spans per cursor).

Pre-registered metrics (plan, frozen): span exact-match (line-normalized
text — BPE merges across the prefix|span boundary on 93.5% of rows make
token-level comparison unfair), prefix-match >=8 tok, position-match,
edit similarity, best-of-8 exact match, distinct spans per cursor (k=8),
p50/p95 latency per span at each step count. Buckets: 0, 1-10, 11-50,
51-256 tokens (GT length; astfim has no empty spans — bucket kept for the
calibration readout). Latency passes run batch=1 per span (true per-span
wall, CUDA events on the MD side, perf_counter on the AR side); quality
passes run batched.

Usage:
  uv run python -m experiments.training.poc_diff.eval_spans \
      --ar-ckpt experiments/training/poc_twin/checkpoints/muon_final.pt \
      --md-ckpt /mnt/h/sepalith/runs/poc_diff/md_final.pt
  (--limit N --tiny: plumbing smoke on tiny random models, CPU-safe)
"""
import argparse
import difflib
import json
import time

import numpy as np
import torch

from experiments.training.poc_diff import TMP
from experiments.training.poc_diff.sample import decode_span, sample_spans

END_IDS = [220, 49, 113, 508, 113, 51]  # tok("\n<|end|>") MiniCPM5
GEN_TOK_CAP = 384                        # ladder convention
STEP_GRID = (8, 16, 32, 64)
K_BESTOF = 8
BUCKETS = [("0", 0, 0), ("1-10", 1, 10), ("11-50", 11, 50),
           ("51-256", 51, 256)]


def norm_lines(text):
    """House norm (eval_fim_served.py): rstrip lines, drop trailing blanks."""
    lines = [l.rstrip() for l in text.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# ---------------- AR arm ----------------

@torch.no_grad()
def ar_greedy_span(model, prompt_ids, tok, max_new, timings=None):
    """Batch-1 greedy decode (latency-honest). Returns (pred_span_ids,
    latency_ms). Stop: end-marker suffix, eos, or cap."""
    device = next(model.parameters()).device
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen = []
    t0 = time.perf_counter()
    for _ in range(max_new):
        h = model.trunk(x, probe=False)
        nxt = int(torch.argmax(h[0, -1] @ model.embed.weight.T))
        gen.append(nxt)
        if nxt == 1 or (len(gen) >= len(END_IDS) and
                        gen[-len(END_IDS):] == END_IDS):
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if x.shape[1] >= model.cfg["max_seq"]:
            break
    if timings is not None:
        timings.append((time.perf_counter() - t0) * 1000.0)
    if gen[-len(END_IDS):] == END_IDS:
        gen = gen[:-len(END_IDS)]
    return [t for t in gen if t != 1]  # strip any eos bleed


# ---------------- metrics ----------------

def token_metrics(pred_text, gt_text, tok):
    p = tok(pred_text, add_special_tokens=False)["input_ids"]
    g = tok(gt_text, add_special_tokens=False)["input_ids"]
    prefix8 = int(len(p) >= 8 and len(g) >= 8 and p[:8] == g[:8])
    n = min(len(p), len(g))
    pos = int(np.mean([p[i] == g[i] for i in range(n)])) if n else 0.0
    return prefix8, pos


def point_metrics(pred_text, gt_text, tok):
    """The per-row pre-registered point metrics (text-level exact)."""
    a, b = norm_lines(pred_text), norm_lines(gt_text)
    exact = int(a == b)
    edit = difflib.SequenceMatcher(None, a, b).ratio()
    prefix8, pos = token_metrics(a, b, tok)
    return dict(exact=exact, edit_sim=edit, prefix8=prefix8, position=pos)


# ---------------- harness ----------------

def load_md(path, device, tiny=False):
    from experiments.training.poc_diff.model_md import MDGQA, model_config_md
    if tiny:
        model = MDGQA(model_config_md(d_model=32, n_layers=1, n_q=2, n_kv=1,
                                      head_dim=16, ffn_hidden=64, max_seq=1024,
                                      vocab=130560)).to(device).eval()
        return model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MDGQA(ck["cfg"]).to(device).eval()
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return model


def load_ar(path, device, tiny=False):
    from experiments.training.poc_twin.model import TinyGQA, model_config
    if tiny:
        model = TinyGQA(model_config(d_model=32, n_layers=1, n_q=2, n_kv=1,
                                     head_dim=16, ffn_hidden=64, max_seq=1024,
                                     vocab=130560)).to(device).eval()
        return model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyGQA(ck["cfg"]).to(device).eval()
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return model


def pct(vals, q):
    return float(np.percentile(vals, q)) if vals else float("nan")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")

    rows = [json.loads(l) for l in
            open(f"{TMP}/eval_triples.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"eval rows: {len(rows)}", flush=True)

    ar = load_ar(args.ar_ckpt, device, tiny=args.tiny)
    md = load_md(args.md_ckpt, device, tiny=args.tiny)

    # ---- AR arm: greedy quality pass (batched semantics via batch-1) ----
    ar_rows = []
    for i, r in enumerate(rows):
        max_new = int(min(GEN_TOK_CAP, 1024 - len(r["prompt_ids"])))
        t = []
        ids = ar_greedy_span(ar, r["prompt_ids"], tok, max_new, timings=t)
        text = tok.decode(ids)
        ar_rows.append(dict(metrics=point_metrics(text, r["span_text"], tok),
                            lat_ms=t[0], text=text))
    ar_lat = [r["lat_ms"] for r in ar_rows]

    # ---- MD arm: per-row pass (k=1 greedy, batch-1: serves quality AND
    #      latency) + k=8 temperature-1.0 multimodality, at each steps ----
    md_rows_by_steps = {}
    md_out = {}
    for steps in args.steps:
        q_rows, lat = [], []
        for r in rows:
            L = max(1, len(r["span_ids"]))
            out = sample_spans(md, [r["prompt_ids"]], [L], steps,
                               temperature=0.0)
            text = decode_span(tok, out["pred_ids"][0, :L], md.empty_id)
            q_rows.append(dict(metrics=point_metrics(text, r["span_text"], tok),
                               lat_ms=out["latency_ms"]))
        # k=8 multimodality: one batched call, seeded generator (device-
        # matched) per row; reproducible by construction
        bo8, distinct = [], []
        for i, r in enumerate(rows):
            L = max(1, len(r["span_ids"]))
            g = torch.Generator(device=device).manual_seed(1273 + steps)
            out = sample_spans(md, [r["prompt_ids"]] * K_BESTOF, [L] * K_BESTOF,
                               steps, temperature=1.0, generator=g)
            texts = {norm_lines(decode_span(tok, out["pred_ids"][k, :L],
                                            md.empty_id))
                     for k in range(K_BESTOF)}
            bo8.append(int(norm_lines(r["span_text"]) in texts))
            distinct.append(len(texts) / K_BESTOF)
        md_rows_by_steps[steps] = q_rows
        md_out[steps] = dict(
            exact=np.mean([m["metrics"]["exact"] for m in q_rows]),
            edit_sim=np.mean([m["metrics"]["edit_sim"] for m in q_rows]),
            prefix8=np.mean([m["metrics"]["prefix8"] for m in q_rows]),
            position=np.mean([m["metrics"]["position"] for m in q_rows]),
            best_of_8=np.mean(bo8), distinct=np.mean(distinct),
            lat_p50=pct([m["lat_ms"] for m in q_rows], 50),
            lat_p95=pct([m["lat_ms"] for m in q_rows], 95))
        print(f"MD steps={steps}: {json.dumps(md_out[steps])}", flush=True)

    ar_summary = dict(
        exact=np.mean([r["metrics"]["exact"] for r in ar_rows]),
        edit_sim=np.mean([r["metrics"]["edit_sim"] for r in ar_rows]),
        prefix8=np.mean([r["metrics"]["prefix8"] for r in ar_rows]),
        position=np.mean([r["metrics"]["position"] for r in ar_rows]),
        lat_p50=pct(ar_lat, 50), lat_p95=pct(ar_lat, 95))
    print(f"AR: {json.dumps(ar_summary)}", flush=True)

    # ---- per-bucket breakdown (from the stored per-row records) ----
    def bucket_of(n):
        for name, lo, hi in BUCKETS:
            if lo <= n <= hi:
                return name
        return None

    buckets = {}
    for name, lo, hi in BUCKETS:
        sel = [i for i, r in enumerate(rows) if bucket_of(r["span_len"]) == name]
        if not sel:
            continue
        b = dict(n=len(sel),
                 ar_exact=float(np.mean([ar_rows[i]["metrics"]["exact"]
                                         for i in sel])))
        for steps in args.steps:
            b[f"md{steps}_exact"] = float(np.mean(
                [md_rows_by_steps[steps][i]["metrics"]["exact"] for i in sel]))
        buckets[name] = b

    results = dict(ar=ar_summary, md={str(k): v for k, v in md_out.items()},
                   buckets=buckets, n_rows=len(rows), step_grid=list(args.steps))
    with open(args.json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.json}", flush=True)
    return results


def write_results_md(results, path):
    md_s = results["md"]
    steps = sorted(int(k) for k in md_s)
    cols = ["AR greedy"] + [f"MD@{s}" for s in steps]
    lines = [
        "# POC-DIFF results — masked-diffusion twin vs AR-FIM twin", "",
        "Paired eval, both arms loaded fresh and scored the same day with "
        "this file's harness (`eval_spans.py`). Pre-registered metrics; "
        "kill test verbatim from the plan.", "",
        f"Eval triples: {results['n_rows']} held-out rows "
        f"(span<=256, prompt<=640). Step grid: {results['step_grid']}.", "",
        "## Metric table", "",
        "| metric | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for key, label in [("exact", "span exact-match"), ("prefix8", "prefix-match (>=8 tok)"),
                       ("position", "position-match"), ("edit_sim", "edit similarity"),
                       ("best_of_8", "best-of-8 exact"), ("distinct", "distinct spans (k=8)"),
                       ("lat_p50", "p50 latency ms"), ("lat_p95", "p95 latency ms")]:
        row = [label, f"{results['ar'].get(key, float('nan')):.4f}"]
        for s in steps:
            row.append(f"{md_s[str(s)].get(key, float('nan')):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Per-bucket exact-match (GT span length)", "",
              "| bucket | n | AR | " + " | ".join(f"MD@{s}" for s in steps) + " |",
              "|" + "---|" * (len(steps) + 3)]
    for name, b in results["buckets"].items():
        row = [name, str(b["n"]), f"{b['ar_exact']:.3f}"]
        row += [f"{b.get(f'md{s}_exact', float('nan')):.3f}" for s in steps]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "", "## VERDICT (Task 6 — apply the kill test verbatim)", "",
        "KILL TEST: diffusion @ <=32 steps must reach >=90% of AR exact-match "
        "OR beat AR on best-of-8 OR beat AR p95 latency; one rescue allowed "
        "(AR-init, Task 8) before the final verdict.", "",
        "- [ ] VALIDATED / KILLED / RESCUE: ______", "- [ ] numbers: ______", "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-ckpt",
                    default="experiments/training/poc_twin/checkpoints/muon_final.pt")
    ap.add_argument("--md-ckpt", default="/mnt/h/sepalith/runs/poc_diff/md_final.pt")
    ap.add_argument("--steps", type=int, nargs="+", default=list(STEP_GRID))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tiny", action="store_true",
                    help="plumbing smoke: tiny random models, no ckpts")
    ap.add_argument("--json", default=f"{TMP}/eval_results.json")
    ap.add_argument("--out", default="experiments/training/poc_diff/RESULTS.md")
    args = ap.parse_args()
    results = run(args)
    write_results_md(results, args.out)


if __name__ == "__main__":
    main()
