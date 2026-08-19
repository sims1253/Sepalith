#!/usr/bin/env python3
"""Programmatic scenario-family eval with the exact per-family validators.

Scores the five programmatic edit-scenario families (rename_propagation,
pipe_rewrite, format_propagation, doc_sync, na_rm_propagation) on their
HELD-OUT rows, one model at a time, on a llama-server this script owns.

Held-out split. The mixture assembly (post-processing/assemble_sft_v2.py)
holds out 3% of each family's packages (min 1, seed 42) for eval. That
split is materialized in /mnt/h/sepalith/datasets/sft_v3/eval.jsonl (the
v2-mixture assembly the sft_v3 model was trained against; the rng draw
depends on which scenario files existed at assembly time, so the jsonl is
the authoritative split, not a fresh re-draw). This script selects the
scenarios_v1 rows whose zeta2 render — re-rendered through the exact
edit_row()/render_zeta2() functions the models were trained on — matches a
prompt in that eval file. The abl_dropout arm trained on finish-block
records only and saw no scenario rows, so the same split is clean for both
models. Cap: first 150 held-out rows per family, in file order.

Serving. One llama-server per model (CPU: -t 8 -ngl 0, --parallel 1,
-c 8192, --host 127.0.0.1). Readiness is a POST /v1/completions that
returns HTTP 200 (never /health, which can report ready before the first
real completion works). Teardown signals ONLY the tracked child PID —
never pkill by name; other eval servers on this machine are not ours.

Scoring. Each prediction is checked with the EXACT validator imported from
experiments/synthetic-data/scenarios.py: validate_example() is rerun on the
scenario row with region_new replaced by the predicted lines, i.e. the same
code path that gated every constructed training row. Call adaptations (not
reimplementations), verified against all held-out rows up front:
  - predicted lines are rstripped per line with trailing blanks popped
    (run_eval.parse_pred conventions; the training target itself is the
    rstripped join of region_new);
  - the three original families are single-line-region constructions, so a
    prediction with any line count != 1 cannot be a valid region and is
    failed before the validator runs (validate_example would otherwise only
    inspect pred[0]);
  - rows lacking construction metadata cannot be validated — none exist:
    every scenarios_v1 row carries family/package/path/prefix/region_old/
    region_new/cursor_idx/event_diff/note.
Also reported per row: exact (normalized line match vs ground truth) and
reward (scenarios.exact_reward line-F1). doc_sync's line-F1 no-op is high
by construction (pure insertion); its gate is exact/validator, see
calibrate_new() in scenarios.py.

Usage (absolute paths; shell cwd drifts):
  python3 experiments/eval/eval_scenarios.py \
      --model /home/m0hawk/Documents/Sepalith/experiments/models/sft_v3_minicpm5-Q8_0.gguf

Writes results_scenarios_<model>.jsonl next to this script (one row per
example; reruns resume) and prints one progress JSON per row, with the
per-family aggregate block printed LAST.
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
sys.path.insert(0, str(HERE))                                  # run_eval
sys.path.insert(0, str(HERE.parent / "post-processing"))       # assemble_sft_v2
sys.path.insert(0, str(HERE.parent / "synthetic-data"))        # scenarios

import scenarios                                                    # noqa: E402
from assemble_sft_v2 import edit_row                                # noqa: E402
from run_eval import parse_pred                                     # noqa: E402

FAMILIES = ("rename_propagation", "pipe_rewrite", "format_propagation",
            "doc_sync", "na_rm_propagation")
SINGLE_LINE_FAMILIES = set(scenarios.FAMILIES)  # rename/pipe/na_rm by construction
SCEN_DIR = Path("/mnt/h/sepalith/datasets/scenarios_v1")
HOLDOUT_REF = Path("/mnt/h/sepalith/datasets/sft_v3/eval.jsonl")
DEFAULT_SERVER = HERE.parent / "bin" / "llama" / "llama-b10453" / "llama-server"
STOP = ">>>>>>> UPDATED"

# validate_example assertion messages that mean "wrong shape", not "wrong
# transformation" (used only to bucket failures in the aggregate)
_SHAPE_MARKS = ("must be non-empty list", "GT must change the region",
                "single-line strings", "cursor_idx")


def load_heldout(cap):
    """Held-out scenario rows: render every row of the 5 family files with
    the training-time edit_row() and keep those whose prompt is in the
    materialized eval split. Returns ({family: [rows]}, unmatched report)."""
    ref = {f: set() for f in FAMILIES}
    for line in open(HOLDOUT_REF):
        r = json.loads(line)
        if r.get("family") in ref:
            ref[r["family"]].add(r["prompt"])
    out, report = {}, {}
    for fam in FAMILIES:
        rows, matched = [], 0
        for line in open(SCEN_DIR / f"{fam}.jsonl"):
            row = json.loads(line)
            rr = edit_row(dict(row), fam, row["package"])
            if rr is None or rr["prompt"] not in ref[fam]:
                continue
            matched += 1
            if len(rows) < cap:
                # prompt+target from the exact training render; keep the raw
                # scenario fields alongside for the validator
                row = dict(row, _prompt=rr["prompt"], _target=rr["target"])
                rows.append(row)
        report[fam] = dict(held_out=matched, scored=len(rows),
                           capped_away=max(0, matched - cap))
        out[fam] = rows
    return out, report


def validator_verdict(row, pred_lines):
    """(passed, fail_kind, reason) from the EXACT scenarios validator.

    fail_kind: None on pass, "shape" for impossible-region shapes,
    "transform" when the validator's transformation asserts fire."""
    if row["family"] in SINGLE_LINE_FAMILIES and len(pred_lines) != 1:
        return False, "shape", f"single-line region family, got {len(pred_lines)} line(s)"
    ex = dict(row)
    ex["region_new"] = pred_lines
    try:
        scenarios.validate_example(ex)
        return True, None, None
    except AssertionError as e:
        msg = str(e).replace("\n", " ")[:160]
        kind = "shape" if any(m in msg for m in _SHAPE_MARKS) else "transform"
        return False, kind, msg


def complete(port, prompt, max_tokens):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": [STOP],
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"], time.time() - t0


def port_open(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


class Server:
    """Own llama-server child; readiness via a real completion POST;
    teardown signals only the tracked PID (never pkill by name)."""

    def __init__(self, binary, model, port, threads, ctx, log_path):
        self.binary, self.model, self.port = str(binary), str(model), port
        self.threads, self.ctx = threads, ctx
        self.log_path = log_path
        self.proc = None

    def start(self, ready_timeout=1800):
        if port_open(self.port):
            raise RuntimeError(
                f"port {self.port} already in use; refusing to touch a server "
                f"we did not start (tracked-PID-only policy)")
        cmd = [self.binary, "-m", self.model, "--port", str(self.port),
               "--host", "127.0.0.1", "-t", str(self.threads),
               "--parallel", "1", "-c", str(self.ctx), "-ngl", "0"]
        log = open(self.log_path, "ab")
        log.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited rc={self.proc.returncode} during "
                    f"startup; log tail:\n{self.log_tail()}")
            try:  # readiness = a real completion returning HTTP 200
                body = json.dumps({"prompt": "readiness", "max_tokens": 1,
                                   "temperature": 0}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.port}/v1/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, urllib.error.HTTPError,
                    OSError, json.JSONDecodeError):
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
        try:  # SIGTERM the tracked child only; escalate after a grace period
            os.kill(pid, signal.SIGTERM)
            for _ in range(75):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.2)
            if self.proc.poll() is None:
                os.kill(pid, signal.SIGKILL)
            self.proc.wait(10)
        except (ProcessLookupError, PermissionError):
            pass  # already gone; never signal anything but this pid
        self.proc = None
        if port_open(self.port):
            print(f"WARNING: port {self.port} still open after teardown "
                  f"(not our process anymore; leaving it alone)", file=sys.stderr)


def run_model(args, examples, report):
    model_name = Path(args.model).stem
    if model_name.endswith("-Q8_0"):
        model_name = model_name[: -len("-Q8_0")]
    out_path = HERE / f"results_scenarios_{model_name}.jsonl"
    done = set()
    if args.resume and out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line).get("id"))
            except (ValueError, AttributeError):
                pass
        print(f"resume: {len(done)} row(s) already scored in {out_path.name}",
              flush=True)

    server = Server(args.server_bin, args.model, args.port, args.threads,
                    args.ctx, HERE / f"llama-server-scenarios-{model_name}.log")
    server.start()
    try:
        with open(out_path, "a") as out:
            for fam in FAMILIES:
                for i, row in enumerate(examples[fam]):
                    rid = hashlib.sha1(row["_prompt"].encode()).hexdigest()[:12]
                    if rid in done:
                        continue
                    rec = dict(id=rid, family=fam, i=i, package=row["package"],
                               path=row["path"], note=row.get("note", ""),
                               model=model_name)
                    try:
                        text, dt = complete(args.port, row["_prompt"],
                                            args.max_tokens)
                        pred = parse_pred("zeta2", text)  # rstrip+trailing-blank norm
                        ok, kind, reason = validator_verdict(row, pred)
                        gt = [l.rstrip() for l in row["region_new"]]
                        while gt and not gt[-1]:
                            gt.pop()
                        rec.update(latency_s=round(dt, 2),
                                   n_pred_lines=len(pred),
                                   pred="\n".join(pred)[:400],
                                   valid_pass=int(ok), fail_kind=kind,
                                   valid_reason=reason,
                                   exact=int(pred == gt),
                                   reward=round(scenarios.exact_reward(
                                       pred, row["region_new"]), 4))
                    except Exception as e:  # transport/timeout: scored 0
                        rec.update(latency_s=0.0, n_pred_lines=0, pred=None,
                                   valid_pass=0, fail_kind="error",
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

    # ---- per-family aggregate over ALL rows in the file (incl. resumed) ----
    rows = [json.loads(l) for l in open(out_path)]
    agg = {}
    for fam in FAMILIES:
        rs = [r for r in rows if r["family"] == fam]
        raw = examples.get(fam, [])
        if not rs:
            agg[fam] = dict(n_scored=0)
            continue
        lat = sorted(r["latency_s"] for r in rs)

        def frac(k):
            return round(sum(1 for r in rs if r.get(k)) / len(rs), 4)
        agg[fam] = dict(
            n_scored=len(rs),
            held_out_available=report[fam]["held_out"],
            validator_pass=frac("valid_pass"),
            exact=frac("exact"),
            mean_reward=round(sum(r["reward"] for r in rs) / len(rs), 4),
            fail_shape=round(sum(1 for r in rs if r.get("fail_kind") == "shape")
                             / len(rs), 4),
            fail_transform=round(sum(1 for r in rs if r.get("fail_kind")
                                     == "transform") / len(rs), 4),
            fail_error=round(sum(1 for r in rs if r.get("fail_kind") == "error")
                             / len(rs), 4),
            noop_line_f1=round(sum(scenarios.noop_baseline_score(e) for e in raw)
                               / len(raw), 4) if raw else None,
            noop_exact=round(sum(scenarios.noop_exact_score(e) for e in raw)
                             / len(raw), 4) if raw else None,
            p50_latency_s=lat[len(lat) // 2],
            p95_latency_s=lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))],
        )
    block = dict(aggregate=dict(
        model=model_name, gguf=str(Path(args.model).resolve()),
        port=args.port, threads=args.threads, ctx=args.ctx,
        temperature=0, max_tokens=args.max_tokens, stop=[STOP],
        render="assemble_sft_v2.edit_row (zeta2)", validator="scenarios.validate_example",
        holdout_ref=str(HOLDOUT_REF), cap_per_family=args.cap,
        selection={f: report[f] for f in FAMILIES},
        families=agg,
        total_rows_scored=len(rows),
    ))
    print(json.dumps(block, indent=1))  # the aggregate is printed LAST


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="GGUF path")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--cap", type=int, default=150, help="rows per family")
    ap.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    for p in (args.model, args.server_bin, HOLDOUT_REF):
        if not Path(p).exists():
            sys.exit(f"missing: {p}")
    examples, report = load_heldout(args.cap)
    for fam in FAMILIES:
        if report[fam]["held_out"] == 0:
            sys.exit(f"no held-out rows matched for {fam}; split drift?")
        # up-front: every evaluable row must carry valid construction metadata
        for row in examples[fam]:
            scenarios.validate_example(row)
    print(json.dumps(dict(holdout=report, model=str(Path(args.model).resolve()),
                          port=args.port)), flush=True)
    run_model(args, examples, report)


if __name__ == "__main__":
    main()
