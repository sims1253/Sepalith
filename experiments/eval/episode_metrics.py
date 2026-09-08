#!/usr/bin/env python3
"""V1a episode-metrics leg — judge_loop wiring for the model battery
(docs/research/2026-09-02-eval-strategy-v2.md §3 V1a "session feel").

Three modes:

  1) REPLAY (zero serving) — recompute the episode metrics from STORED
     judged rows (old rows lack `extras`; the identical functions from
     judge_loop score them):

     python3 experiments/eval/episode_metrics.py --replay \
       v7=/mnt/h/sepalith/datasets/sim_trajectories_v1/judged_v7.jsonl \
       rl_v2c=/mnt/h/sepalith/datasets/sim_trajectories_v1/judged_rl_v2c.jsonl \
       --fp-gap v7 rl_v2c --out /mnt/h/sepalith/runs/episode_metrics_calibration1.json

  2) LIVE (CPU serving) — spawn a nice'd CPU-only llama-server (repo
     binary experiments/bin/llama/llama-b10453, -ngl 0, -t 6, ports
     18200-18219, ONE server at a time), point judge_loop at it as an
     EXTERNAL server (its CUDA spawn path never fires), aggregate:

     python3 experiments/eval/episode_metrics.py --live \
       --model experiments/models/sft_v8_2_minicpm5-Q8_0.gguf --tag sft_v8_2 \
       --n 60 --traj /mnt/h/sepalith/datasets/sim_trajectories_v1/trajectories_v2.jsonl

  3) COMPARE — side-by-side arms + calibration verdicts into the results
     JSON (+ markdown table):

     python3 experiments/eval/episode_metrics.py --compare \
       base=/mnt/h/sepalith/runs/episode_metrics_base.json \
       sft_v8_2=/mnt/h/sepalith/runs/episode_metrics_sft_v8_2.json \
       --accept-sanity sft_v8_2 base \
       --out experiments/eval/episode_metrics_v8_2_vs_base.json \
       --md experiments/eval/episode_metrics_v8_2_vs_base.md

BATTERY ONE-LINER (add leg 5 / V1a to any rung; CPU box, serialized):

  for M in sft_v8_2_minicpm5 minicpm5-1b; do
    python3 experiments/eval/episode_metrics.py --live \
      --model experiments/models/${M}-Q8_0.gguf --tag ${M%%_*} --n 60 \
      --traj /mnt/h/sepalith/datasets/sim_trajectories_v1/trajectories_v2.jsonl
  done

New (additive) metrics, per trajectory + aggregate — computed on the
SIMULATOR clock, so serving latency cannot distort them:
  time-to-edit  t_ms(first ACCEPTABLE-class accept) + 1500ms debounce
                (null if no accept; censored mean penalizes at episode end)
  interruption  proposals during Thinking/Navigating windows
                (ctx pre_task_think/navigation — noop points by
                construction, so any non-empty proposal there interrupts)

Calibration anchors (design doc §3 V1a / §4):
  - banked noopFP pair v7 vs RL-v2c: 0.706 -> 0.466 (204 pts, 49/0
    discordant, p~=1.5e-13; 2026-08-29-paired-significance-audit.md).
    REPLAY must reproduce the DIRECTION v7_fp > rl_v2c_fp on judged rows.
  - live v8_2 vs base: v8_2 accept_rate >= base (scenario direction:
    exact 0.722 vs 0.000 on results_scenarios_*_minicpm5.jsonl).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "training" / "coding_simulator"))
from judge_loop import episode_extras  # noqa: E402  pure function, no serving

RUNS = Path("/mnt/h/sepalith/runs")
LOG = RUNS / "episode_metrics.log"
SERVER_BIN = REPO / "experiments" / "bin" / "llama" / "llama-b10453" / "llama-server"
JUDGE_LOOP = REPO / "experiments" / "training" / "coding_simulator" / "judge_loop.py"
PORT_MIN, PORT_MAX = 18200, 18219          # task-mandated serving range
THREADS = 6                                # task-mandated thread cap
CTX, PARALLEL = 4096, 2                    # judge_loop's own server config


def log(msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# aggregation (works on any judged jsonl, with or without `extras`)
# ---------------------------------------------------------------------------

def summarize(label: str, rows: list[dict], source: str) -> dict:
    agg = dict(shown=0, accepted=0, dismissed=0, false_sug=0, correct_stop=0,
               chars_saved=0, chars_typed=0)
    extras = []
    per_traj = []
    mismatch = 0
    for r in rows:
        rec = r.get("points", [])
        dec = dict(accepted=0, dismissed=0, false_sug=0, correct_stop=0)
        for p in rec:
            if p.get("decision") == "false_suggestion":
                dec["false_sug"] += 1
            elif p.get("decision") in dec:
                dec[p["decision"]] += 1
        recomputed = dict(shown=len(rec), **dec)
        for k in ("shown", "accepted", "dismissed", "false_sug", "correct_stop"):
            if recomputed[k] != r.get("stats", {}).get(k, recomputed[k]):
                mismatch += 1
                break
        for k in agg:
            agg[k] += r.get("stats", {}).get(k, 0)
        e = r.get("extras") or episode_extras(rec)
        extras.append(e)
        per_traj.append(dict(key=r.get("key"), variant=r.get("variant"),
                             time_to_edit_ms=e["time_to_edit_ms"],
                             think_nav_windows=e["think_nav_windows"],
                             think_nav_interruptions=e["think_nav_interruptions"],
                             interruption_rate=e["interruption_rate"],
                             n_accepts=e["n_accepts"]))
    ttes = [e["time_to_edit_ms"] for e in extras
            if e["time_to_edit_ms"] is not None]
    cens = [e["time_to_edit_ms"] if e["time_to_edit_ms"] is not None
            else e["session_end_ms"] for e in extras]
    tn_w = sum(e["think_nav_windows"] for e in extras)
    tn_i = sum(e["think_nav_interruptions"] for e in extras)
    noop_by_ctx: dict[str, dict] = {}
    for e in extras:
        for ctx, d in e.get("noop_false_sug_by_ctx", {}).items():
            m = noop_by_ctx.setdefault(ctx, dict(points=0, false_sug=0))
            m["points"] += d["points"]
            m["false_sug"] += d["false_sug"]
    return dict(
        label=label, source=source, n_traj=len(rows),
        stats_mismatch_rows=mismatch,
        **agg,
        accept_rate=round(agg["accepted"] / max(1, agg["shown"]), 4),
        fp_rate=round(agg["false_sug"] /
                      max(1, agg["false_sug"] + agg["correct_stop"]), 4),
        saved_ratio=round(agg["chars_saved"] / max(1, agg["chars_typed"]), 4),
        frac_traj_with_accept=round(len(ttes) / max(1, len(extras)), 4),
        tte_mean_ms=round(sum(ttes) / len(ttes)) if ttes else None,
        tte_median_ms=round(statistics.median(ttes)) if ttes else None,
        tte_censored_mean_ms=round(sum(cens) / len(cens)) if cens else None,
        think_nav_windows=tn_w,
        think_nav_interruptions=tn_i,
        interruption_rate=round(tn_i / max(1, tn_w), 4),
        noop_false_sug_by_ctx={k: dict(points=v["points"],
                                       false_sug_rate=round(
                                           v["false_sug"] / max(1, v["points"]), 4))
                               for k, v in sorted(noop_by_ctx.items())},
        per_trajectory=per_traj,
    )


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def paired_noop_fp(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Exact McNemar on paired noop points (key, t_ms) — the audit's
    discordant-count reading of the FP gap, on episode rows."""
    def noop_map(rows):
        m = {}
        for r in rows:
            for p in r.get("points", []):
                if p.get("label") == "noop":
                    m[(r.get("key"), p.get("t_ms"))] = \
                        p.get("decision") == "false_suggestion"
        return m
    a, b = noop_map(rows_a), noop_map(rows_b)
    shared = set(a) & set(b)
    d_a = sum(1 for k in shared if a[k] and not b[k])
    d_b = sum(1 for k in shared if b[k] and not a[k])
    n = d_a + d_b
    # exact two-sided binomial p on discordants
    if n == 0:
        p = 1.0
    else:
        k = min(d_a, d_b)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return dict(paired_noop_points=len(shared), discordant_a=d_a,
                discordant_b=d_b, mcnemar_exact_p=round(p, 8))


def md_table(summs: list[dict]) -> str:
    cols = [("n_traj", "traj"), ("shown", "shown"),
            ("accept_rate", "accept_rate"), ("fp_rate", "fp_rate"),
            ("saved_ratio", "saved_ratio"),
            ("frac_traj_with_accept", "frac_w_accept"),
            ("tte_mean_ms", "tte_mean_ms"),
            ("tte_median_ms", "tte_median_ms"),
            ("tte_censored_mean_ms", "tte_cens_ms"),
            ("think_nav_windows", "think_nav_win"),
            ("think_nav_interruptions", "interrupts"),
            ("interruption_rate", "interruption_rate")]
    head = "| metric | " + " | ".join(s["label"] for s in summs) + " |"
    sep = "|---" * (len(summs) + 1) + "|"
    lines = [head, sep]
    for key, name in cols:
        lines.append(f"| {name} | " + " | ".join(
            str(s.get(key) if s.get(key) is not None else "—")
            for s in summs) + " |")
    return "\n".join(lines)


def write_out(out: str, payload: dict) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    log(f"wrote {out}")


# ---------------------------------------------------------------------------
# live mode: own CPU server (nice 15, -t 6, ports 18200-18219, serialized)
# ---------------------------------------------------------------------------

def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def complete(port: int, prompt: str, max_tokens: int = 160,
             timeout: int = 300) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(dict(prompt=prompt, max_tokens=max_tokens,
                             temperature=0.0)).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["text"]


def spawn_server(model: str, port: int) -> subprocess.Popen:
    cmd = ["nice", "-n", "15", str(SERVER_BIN), "-m", model,
           "--port", str(port), "--host", "127.0.0.1",
           "-c", str(CTX), "--parallel", str(PARALLEL),
           "-t", str(THREADS), "-ngl", "0"]
    RUNS.mkdir(parents=True, exist_ok=True)
    log(f"[serve] spawning (own pid only): {' '.join(cmd)}")
    with open(LOG, "ab") as lf:
        lf.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited rc={proc.returncode}; see {LOG}")
        try:
            complete(port, "x", 1, timeout=30)
            log(f"[serve] ready on :{port} after {time.time()-t0:.0f}s "
                f"(pid {proc.pid})")
            return proc
        except Exception:
            time.sleep(3)
    proc.terminate()
    sys.exit(f"server not ready in 600s; see {LOG}")


def stop_server(proc: subprocess.Popen) -> None:
    pid = proc.pid
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(75):
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        if proc.poll() is None:
            os.kill(pid, signal.SIGKILL)
        proc.wait(10)
    except (ProcessLookupError, PermissionError):
        pass
    log(f"[serve] torn down pid {pid} (ours only)")


def selected_trajs(traj_path: str, n: int) -> list[dict]:
    import random
    trajs = []
    for line in open(traj_path):
        try:
            t = json.loads(line)
        except ValueError:
            continue
        if t.get("n_points", 0) >= 5:
            trajs.append(t)
    random.Random(3).shuffle(trajs)          # judge_loop's own selection
    return trajs[:n]


def preflight(port: int, traj_path: str, n_want: int,
              max_serve_min: int) -> tuple[int, dict]:
    """Time a handful of REAL prompts to project the live run's cost,
    then apply the owner's rule: if n=60 projects to more than
    max_serve_min (90) minutes of CPU serving, drop to n=40 (the floor).
    Returns (chosen_n, info)."""
    trajs = selected_trajs(traj_path, n_want)
    pts = [p for t in trajs for p in t["points"][:30]]
    n_req = len(pts)
    sample = [pts[i] for i in range(0, n_req, max(1, n_req // 6))][:6]
    lat = []
    for p in sample:
        t0 = time.time()
        try:
            complete(port, p["prompt"])
        except Exception as e:
            log(f"[preflight] request failed: {e}")
            continue
        lat.append(time.time() - t0)
    n_req40 = sum(min(30, len(t["points"])) for t in trajs[:40])
    if not lat:
        return n_want, dict(note="preflight failed; keeping requested n",
                            n_requests=n_req)
    mean_lat = sum(lat) / len(lat)
    # 2 judge_loop workers share the 6 threads across 2 server slots;
    # measured effective speedup on this box ~1.8x -> report a band.
    def proj(reqs):
        return (round(reqs * mean_lat / 1.8 / 60),
                round(reqs * mean_lat / 60))
    lo60, hi60 = proj(n_req)
    lo40, hi40 = proj(n_req40)
    n = n_want
    drop_note = ""
    if lo60 > max_serve_min:
        n = min(40, n_want)
        drop_note = (f"n=60 projected at {lo60}-{hi60} min > "
                     f"{max_serve_min} min budget -> DROPPED to n={n} "
                     f"(projected {lo40}-{hi40} min)")
        if lo40 > max_serve_min:
            drop_note += ("; WARNING: even n=40 exceeds the budget — "
                          "running anyway (n=40 is the agreed floor)")
    info = dict(preflight_n=len(lat), mean_request_s=round(mean_lat, 2),
                n_requests=n_req, projected_min_60=[lo60, hi60],
                projected_min_40=[lo40, hi40], chosen_n=n, note=drop_note)
    log(f"[preflight] {json.dumps(info)}")
    return n, info


def live(args) -> dict:
    if not Path(args.model).exists():
        sys.exit(f"missing model: {args.model}")
    port = args.port or next((p for p in range(PORT_MIN, PORT_MAX + 1)
                              if port_free(p)), None)
    if port is None:
        sys.exit(f"no free port in {PORT_MIN}-{PORT_MAX}")
    out_rows = args.out or str(RUNS / f"episode_judged_{args.tag}.jsonl")
    summary_out = args.summary or str(RUNS / f"episode_metrics_{args.tag}.json")

    proc = spawn_server(args.model, port)
    try:
        n, pre = preflight(port, args.traj, args.n, args.max_serve_min)
        cmd = [sys.executable, str(JUDGE_LOOP), "--model", args.model,
               "--traj", args.traj, "--out", out_rows, "--port", str(port),
               "--n", str(n)]
        log(f"[judge_loop] {' '.join(cmd)}")
        t0 = time.time()
        with open(LOG, "ab") as lf:
            lf.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
            r = subprocess.run(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
            lf.write(r.stdout.encode(errors="replace"))
        for line in r.stdout.splitlines():
            if line.startswith("{"):
                log(f"[judge_loop] {line}")     # echo progress/summary lines
        if r.returncode != 0:
            sys.exit(f"judge_loop rc={r.returncode}; see {LOG}")
        rows = load_jsonl(out_rows)
        summ = summarize(args.tag, rows, out_rows)
        summ["server"] = dict(bin=str(SERVER_BIN), port=port, threads=THREADS,
                              ngl=0, nice=15, ctx=CTX, parallel=PARALLEL,
                              model=args.model, n_requested=args.n, n_run=n,
                              traj=args.traj,
                              elapsed_s=round(time.time() - t0),
                              preflight=pre)
        write_out(summary_out, summ)
        print(md_table([summ]))
        return summ
    finally:
        stop_server(proc)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", nargs="+", metavar="LABEL=PATH",
                    help="recompute metrics from stored judged jsonl rows")
    ap.add_argument("--live", action="store_true",
                    help="serve on CPU + run judge_loop (one arm)")
    ap.add_argument("--compare", nargs="+", metavar="LABEL=SUMMARY_JSON",
                    help="side-by-side arm summaries + verdicts")
    ap.add_argument("--model", help="(live) GGUF path")
    ap.add_argument("--tag", help="(live) arm label")
    ap.add_argument("--traj",
                    default="/mnt/h/sepalith/datasets/sim_trajectories_v1/"
                            "trajectories_v2.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-serve-min", type=int, default=90,
                    help="(live) if n=60 projects above this many minutes "
                         "of CPU serving, drop to n=40 (owner's rule)")
    ap.add_argument("--port", type=int, choices=range(PORT_MIN, PORT_MAX + 1))
    ap.add_argument("--out", help="(live) judged rows path")
    ap.add_argument("--summary", help="(live) arm summary json path")
    ap.add_argument("--fp-gap", nargs=2, metavar=("HI", "LO"),
                    help="verdict: fp[HI] > fp[LO] expected (banked anchor "
                         "0.706->0.466, noopFP v7 vs rl_v2c)")
    ap.add_argument("--accept-sanity", nargs=2, metavar=("GE", "LE"),
                    help="verdict: accept[GE] >= accept[LE] expected "
                         "(v8_2 >= base)")
    ap.add_argument("--anchors", help="extra anchors json merged into output")
    ap.add_argument("--md", help="(compare) also write the markdown table here")
    args = ap.parse_args()
    if not (args.replay or args.live or args.compare):
        ap.error("one of --replay / --live / --compare required")

    if args.live:
        if not (args.model and args.tag):
            ap.error("--live needs --model and --tag")
        live(args)
        return

    if args.replay:
        labels, rows_by = [], {}
        for spec in args.replay:
            label, path = spec.split("=", 1)
            rows = load_jsonl(path)
            rows_by[label] = rows
            labels.append(label)
            log(f"[replay] {label}: {len(rows)} rows from {path}")
        summs = [summarize(l, rows_by[l], s.split("=", 1)[1])
                 for l, s in zip(labels, args.replay)]
        payload = dict(kind="episode_metrics replay (calibration run 1)",
                       generated=time.strftime("%F %T"),
                       md_table=md_table(summs), arms=summs)
        payload["verdicts"] = verdicts(payload, summs, args)
        if len(summs) >= 2:
            payload["paired_noop_fp"] = paired_noop_fp(
                rows_by[labels[0]], rows_by[labels[1]])
        out = args.out or str(RUNS / "episode_metrics_replay.json")
        write_out(out, payload)
        print(payload["md_table"])
        for v in payload["verdicts"]:
            print(f"VERDICT {v['check']}: {v['verdict']} — {v['observed']} "
                  f"(expectation: {v['expectation']})")
        if "paired_noop_fp" in payload:
            print("paired noop FP:", payload["paired_noop_fp"])
        return

    if args.compare:
        summs = []
        for spec in args.compare:
            label, path = spec.split("=", 1)
            s = json.load(open(path))
            s["label"] = label
            summs.append(s)
        payload = dict(kind="episode_metrics compare (calibration run 2)",
                       generated=time.strftime("%F %T"),
                       md_table=md_table(summs), arms=summs)
        payload["verdicts"] = verdicts(payload, summs, args)
        if args.anchors and Path(args.anchors).exists():
            payload["anchors"] = json.load(open(args.anchors))
        out = args.out or str(HERE / "episode_metrics_compare.json")
        write_out(out, payload)
        if args.md:
            with open(args.md, "w") as fh:
                fh.write(payload["md_table"] + "\n")
            log(f"wrote {args.md}")
        print(payload["md_table"])
        for v in payload["verdicts"]:
            print(f"VERDICT {v['check']}: {v['verdict']} — {v['observed']} "
                  f"(expectation: {v['expectation']})")


def verdicts(payload: dict, summs: list[dict], args) -> list[dict]:
    by = {s["label"]: s for s in summs}
    out = []
    if args.fp_gap:
        hi, lo = args.fp_gap
        ok = by[hi]["fp_rate"] > by[lo]["fp_rate"]
        out.append(dict(
            check="fp_gap_direction",
            expectation=(f"banked noopFP anchor v7 0.706 -> rl_v2c 0.466 "
                         f"(2026-08-29 audit); judged-rows recompute must "
                         f"keep the direction fp[{hi}] > fp[{lo}]"),
            observed=f"fp[{hi}]={by[hi]['fp_rate']} fp[{lo}]={by[lo]['fp_rate']}",
            verdict="REPRODUCED" if ok else "NOT REPRODUCED"))
    if args.accept_sanity:
        ge, le = args.accept_sanity
        ok = by[ge]["accept_rate"] >= by[le]["accept_rate"]
        out.append(dict(
            check="accept_sanity",
            expectation=(f"design doc §3 V1a: {ge} >= {le} on accept_rate "
                         f"(scenario direction: exact 0.722 vs 0.000)"),
            observed=(f"accept[{ge}]={by[ge]['accept_rate']} "
                      f"accept[{le}]={by[le]['accept_rate']}"),
            verdict="REPRODUCED" if ok else "NOT REPRODUCED"))
    return out


if __name__ == "__main__":
    main()
