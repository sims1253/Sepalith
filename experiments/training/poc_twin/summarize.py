"""Render RESULTS.md tables from the arm logs (markdown-ready)."""
import json, sys

POC = "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin"


def load(arm):
    rows, evals, meta = [], [], None
    for line in open(f"{POC}/logs/{arm}.jsonl"):
        r = json.loads(line)
        if "loss" in r and "step" in r:
            rows.append(r)
        elif r.get("event") == "eval":
            evals.append(r)
        elif r.get("event") == "done":
            meta = r
    return rows, evals, meta


def main():
    out = []
    arms = [a for a in (sys.argv[1:] or ["muon", "adamw"])]
    data = {a: load(a) for a in arms}

    out.append("### Loss curves aligned by token count (train loss, 100-step windows)\n")
    out.append("| tokens | " + " | ".join(f"{a} loss" for a in arms) + " | " +
               " | ".join(f"{a} qk_max" for a in arms) + " |")
    out.append("|---" * (1 + 2 * len(arms)) + "|")
    by_tok = {a: {r["tokens"]: r for r in data[a][0]} for a in arms}
    toks = sorted(set.intersection(*[set(by_tok[a]) for a in arms])) if len(arms) > 1 else sorted(by_tok[arms[0]])
    for t in toks:
        cells = [f"{t/1e6:.0f}M"]
        for a in arms:
            cells.append(f"{by_tok[a][t]['loss']:.4f}")
        for a in arms:
            cells.append(f"{by_tok[a][t]['qk_max']:.1f}")
        out.append("| " + " | ".join(cells) + " |")

    out.append("\n### Held-out eval loss (64-block quick eval, nats/token)\n")
    out.append("| tokens | " + " | ".join(arms) + " |")
    out.append("|---" * (1 + len(arms)) + "|")
    ev = {a: {r["tokens"]: r["eval_loss"] for r in data[a][1]} for a in arms}
    for t in sorted(set.intersection(*[set(ev[a]) for a in arms])) if len(arms) > 1 else sorted(ev[arms[0]]):
        out.append(f"| {t/1e6:.0f}M | " + " | ".join(f"{ev[a][t]:.4f}" for a in arms) + " |")

    out.append("\n### Per-arm summary\n")
    for a in arms:
        rows, evals, meta = data[a]
        tps = [r["tok_per_s"] for r in rows if r.get("tok_per_s")]
        qmax = [r["qk_max"] for r in rows if r.get("qk_max") is not None]
        clips = [r["qk_clipped_heads"] for r in rows if "qk_clipped_heads" in r]
        out.append(f"- **{a}**: final train loss {rows[-1]['loss']:.4f} @ {rows[-1]['tokens']/1e6:.0f}M tokens "
                   f"({rows[-1]['epoch']:.2f} epochs); mean tok/s {sum(tps)/len(tps):.0f} "
                   f"(min {min(tps):.0f}, max {max(tps):.0f}); QK max-logit range "
                   f"{min(qmax):.1f}-{max(qmax):.1f} (tau=100); head-clips/100-step trajectory: " +
                   ", ".join(str(c) for c in clips) + "; evals: " +
                   ", ".join(f"{e['eval_loss']:.4f}@{e['tokens']/1e6:.0f}M" for e in evals))
    print("\n".join(out))


if __name__ == "__main__":
    main()
