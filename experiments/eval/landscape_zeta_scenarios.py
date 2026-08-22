#!/usr/bin/env python3
"""Landscape leg: Zed's ZETA 2.1 (open big-player edit-prediction specialist)
served locally on the free 5090, on ITS NATIVE prompt format.

Prompt + output contract are the faithful V0318 port in landscape_zeta_render.py
(ported from Zed's production Rust, crates/zeta_prompt — see that module's
docstring for the format facts and the delta vs our legacy
run_eval.render_zeta2_1). Row selection is IDENTICAL to
eval_scenarios.load_heldout (same held-out rows, same file order, first
n<=60 per family), and row ids are the eval_scenarios convention (sha1 of
the zeta2 render prompt), so rows join 1:1 across the v7 / glm-5.3 / zeta
legs.

Serving (ops rules): one llama-server child on port 18106 with -ngl 99 (the
5090 is free — the v7 train finished; the v7 battery chain owns port 18103
on CPU and is NOT touched). Readiness = a POST /v1/completions returning
200 (never /health). Teardown signals ONLY the tracked child PID.

Scoring: the zeta output is resolved through ITS OWN contract
(apply_marker_span_v0318 — marker span replace over the block-partitioned
editable text; repeated marker = no-edit), the predicted region lines are
extracted from the rewritten editable by line-diff, normalized with the
run_eval zeta2_1 conventions (rstrip per line, strip cursor markers, pop
leading/trailing blanks), and then scored by the SAME eval_scenarios path
as the other legs: validator_verdict (the EXACT scenarios.validate_example
call), exact line match, exact_reward.

Usage:
  python3 landscape_zeta_scenarios.py [--n 60] [--limit N]

Writes results_scenarios_zeta21_native.jsonl next to this script (resume by
id) and prints the per-family aggregate LAST.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                        # eval_scenarios
sys.path.insert(0, str(HERE.parent / "post-processing"))             # assemble_sft_v2
sys.path.insert(0, str(HERE.parent / "synthetic-data"))              # scenarios

from eval_scenarios import (DEFAULT_SERVER, FAMILIES, load_heldout,      # noqa: E402
                            validator_verdict)
import scenarios                                                     # noqa: E402
from landscape_zeta_render import (V0318_END_MARKER, apply_marker_span_v0318,   # noqa: E402
                                   build_v0318_prompt, compute_marker_offsets_v0318,
                                   diff_body_from_event_diff,
                                   extract_region_lines)

MODEL = "/tmp/zeta21-Q6_K.gguf"        # mradermacher Q6_K of zeta-2.1 (8B)
PORT = 18106
OUT_PATH = HERE / "results_scenarios_zeta21_native.jsonl"
CURSOR2 = "<|user_cursor|>"


def norm_pred_lines(lines):
    """run_eval zeta2_1 conventions: strip echoed cursor markers, rstrip per
    line, pop leading and trailing blanks."""
    lines = [l.replace(CURSOR2, "").rstrip() for l in lines]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def port_open(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


class Server:
    """llama-server child on the GPU; readiness via a real completion POST;
    teardown signals only the tracked PID (never pkill by name)."""

    def __init__(self, binary, model, port, log_path):
        self.binary, self.model, self.port = str(binary), str(model), port
        self.log_path = Path(log_path)
        self.proc = None

    def start(self, ready_timeout=900):
        if port_open(self.port):
            raise RuntimeError(f"port {self.port} already in use; refusing "
                               f"to touch a server we did not start")
        cmd = [self.binary, "-m", self.model, "--port", str(self.port),
               "--host", "127.0.0.1", "-t", "8", "--parallel", "1",
               "-c", "8192", "-ngl", "99"]
        with open(self.log_path, "ab") as log:
            log.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n"
                      .encode())
            log.flush()
            self.proc = subprocess.Popen(cmd, stdout=log,
                                         stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL,
                                         start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited rc="
                                   f"{self.proc.returncode}; log tail:\n"
                                   f"{self.log_tail()}")
            try:
                body = json.dumps({"prompt": "readiness", "max_tokens": 1,
                                   "temperature": 0}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.port}/v1/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                    json.JSONDecodeError):
                pass
            time.sleep(2)
        raise RuntimeError(f"llama-server not ready in {ready_timeout}s; "
                           f"log tail:\n{self.log_tail()}")

    def log_tail(self, n=1500):
        try:
            return self.log_path.read_text(errors="replace")[-n:]
        except OSError:
            return "<no log>"

    def stop(self):
        if self.proc is None:
            return
        pid = self.proc.pid
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(75):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.2)
            if self.proc.poll() is None:
                os.kill(pid, signal.SIGKILL)
            self.proc.wait(10)
        except (ProcessLookupError, PermissionError):
            pass
        self.proc = None


def complete(port, prompt, max_tokens):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": [V0318_END_MARKER],
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"], time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="rows per family cap")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: only this many rows per family")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--server-bin", default=DEFAULT_SERVER,
                    help="llama-server binary (use a CUDA build for -ngl 99; "
                         "the stock experiments/bin build is CPU-only)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"missing: {args.model}")
    examples, report = load_heldout(150)   # EXACT eval_scenarios selection

    done = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            try:
                done.add(json.loads(line).get("id"))
            except ValueError:
                pass
        print(f"resume: {len(done)} row(s) already scored", flush=True)

    server = Server(args.server_bin, args.model, args.port,
                    HERE / "llama-server-zeta21.log")
    server.start()
    print(f"server ready on {args.port}", flush=True)
    try:
        with open(OUT_PATH, "a") as out:
            for fam in FAMILIES:
                rows = examples[fam][: args.n]
                if args.limit:
                    rows = rows[: args.limit]
                for i, row in enumerate(rows):
                    rid = hashlib.sha1(row["_prompt"].encode()).hexdigest()[:12]
                    if rid in done:
                        continue
                    prompt, ctx, editable, cursor = build_v0318_prompt(
                        row["prefix"], row["region_old"], row["cursor_idx"],
                        row["path"], diff_body_from_event_diff(row["event_diff"]))
                    rec = dict(id=rid, family=fam, i=i, package=row["package"],
                               path=row["path"], note=row.get("note", ""),
                               model="zeta-2.1-native",
                               n_blocks=len(compute_marker_offsets_v0318(
                                   editable)) - 1)
                    try:
                        raw, dt = complete(args.port, prompt, 640)
                        rec.update(latency_s=round(dt, 2), raw=raw[:400])
                        ok, new_editable = apply_marker_span_v0318(editable, raw)
                        if not ok:
                            pred, kind, reason = [], "parse", str(new_editable)[:160]
                        else:
                            pred = extract_region_lines(
                                editable, new_editable, row["region_old"],
                                len(row["prefix"]))
                            if pred is None:
                                pred, kind, reason = [], "parse", \
                                    "region not mappable in rewritten text"
                            else:
                                pred = norm_pred_lines(pred)
                                kind, reason = None, None
                        if kind is None:
                            v_ok, kind, reason = validator_verdict(row, pred)
                            rec.update(valid_pass=int(v_ok))
                        else:
                            rec.update(valid_pass=0)
                        gt = [l.rstrip() for l in row["region_new"]]
                        while gt and not gt[-1]:
                            gt.pop()
                        rec.update(n_pred_lines=len(pred),
                                   pred="\n".join(pred)[:400],
                                   fail_kind=kind, valid_reason=reason,
                                   exact=int(pred == gt),
                                   reward=round(scenarios.exact_reward(
                                       pred, row["region_new"]), 4))
                    except Exception as e:
                        rec.update(latency_s=0.0, n_pred_lines=0, pred=None,
                                   raw=None, valid_pass=0, fail_kind="error",
                                   valid_reason=str(e)[:160], exact=0,
                                   reward=0.0, error=str(e)[:120])
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(json.dumps({k: rec[k] for k in
                                      ("id", "family", "i", "valid_pass",
                                       "fail_kind", "exact", "reward",
                                       "latency_s")}), flush=True)
    finally:
        server.stop()
        print("server stopped", flush=True)

    # ---- per-family aggregate over ALL rows in the file (incl. resumed) ----
    rows = [json.loads(l) for l in open(OUT_PATH)]
    agg = {}
    for fam in FAMILIES:
        rs = [r for r in rows if r["family"] == fam]
        if not rs:
            agg[fam] = dict(n_scored=0)
            continue
        lat = sorted(r["latency_s"] for r in rs)

        def frac(k):
            return round(sum(1 for r in rs if r.get(k)) / len(rs), 4)
        agg[fam] = dict(
            n_scored=len(rs),
            validator_pass=frac("valid_pass"),
            exact=frac("exact"),
            mean_reward=round(sum(r["reward"] for r in rs) / len(rs), 4),
            fail_parse=round(sum(1 for r in rs if r.get("fail_kind") == "parse")
                             / len(rs), 4),
            fail_shape=round(sum(1 for r in rs if r.get("fail_kind") == "shape")
                             / len(rs), 4),
            fail_transform=round(sum(1 for r in rs if r.get("fail_kind")
                                     == "transform") / len(rs), 4),
            fail_error=round(sum(1 for r in rs if r.get("fail_kind") == "error")
                             / len(rs), 4),
            p50_latency_s=lat[len(lat) // 2],
            p95_latency_s=lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))],
        )
    print(json.dumps(dict(aggregate=dict(
        model="zeta-2.1-native", gguf=str(Path(args.model).resolve()),
        port=args.port, temperature=0, max_tokens=640,
        stop=[V0318_END_MARKER],
        render="landscape_zeta_render.build_v0318_prompt (faithful port of "
        "Zed crates/zeta_prompt V0318SeedMultiRegions, alias Zeta2.1)",
        output_contract="apply_marker_span_v0318 + line-diff region extraction",
        validator="scenarios.validate_example",
        cap_per_family=args.n, smoke_limit=args.limit or None,
        selection={f: report[f] for f in FAMILIES},
        families=agg, total_rows_scored=len(rows),
    )), indent=1))


if __name__ == "__main__":
    main()
