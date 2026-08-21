"""Pick LR winners from probe logs and the step budget from measured
throughput + remaining window; writes logs/plan.json."""
import glob, json, time

POC = "/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin"
HARD_END = time.mktime(time.strptime("2026-08-21 13:30", "%Y-%m-%d %H:%M"))


def probe_results(arm):
    out = {}
    for f in glob.glob(f"{POC}/logs/probe_{arm}_*.jsonl"):
        lr = float(f.rsplit("_", 1)[1].replace(".jsonl", ""))
        best = None
        for line in open(f):
            r = json.loads(line)
            if r.get("event") == "probe_final":
                best = r
        if best:
            out[lr] = best["eval_loss"]
    return out


def main():
    res_a, res_m = probe_results("adamw"), probe_results("muon")
    lr_adamw = min(res_a, key=res_a.get)
    lr_muon = min(res_m, key=res_m.get)
    # throughput: last tok/s window from the muon probe log
    tps = 20000.0
    for line in open(f"{POC}/logs/probe_muon_{lr_muon:g}.jsonl"):
        r = json.loads(line)
        if "tok_per_s" in r and r["tok_per_s"]:
            tps = r["tok_per_s"]
    # per-arm wall budget: split the remaining window, keep 1.5h for eval +
    # writeup + slack; each arm gets half. Clamp 1h..4.5h.
    remaining_s = HARD_END - time.time()
    per_arm = max(3600, min(4.5 * 3600, (remaining_s - 5400) / 2))
    steps = int(per_arm * tps / 524288 / 1.18 / 50) * 50
    steps = max(400, min(1760, steps))
    plan = dict(lr_adamw=lr_adamw, lr_muon=lr_muon, steps=steps,
                measured_tok_per_s=tps, per_arm_budget_s=per_arm,
                remaining_s=remaining_s,
                rationale=dict(adamw_probes=res_a, muon_probes=res_m))
    with open(f"{POC}/logs/plan.json", "w") as f:
        json.dump(plan, f, indent=1)
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
