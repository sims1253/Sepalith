#!/usr/bin/env python3
"""Landscape leg: gemma-4 e2b (small-model PEER, ~2B class generalist)
zero-shot on our scenario task — the fourth landscape corner.

Same held-out rows / render / validators as the other legs (see
landscape_glm_scenarios.py). The gemma-4-e2b QAT Q4_0 GGUF is served on
port 18106 by a server this script does NOT own or manage (user-authorized,
coordinator-launched CUDA server) — this script is a CLIENT only and never
starts/stops servers. Treatment mirrors the glm-5.3 leg exactly (a zero-shot
generalist with one short instruction): chat-completions endpoint, the
zeta2 render + FMT_INSTRUCTION as a single user message, temperature 0,
max_tokens 2048, NO stop string (parse_pred cuts at >>>>>>>), then the
identical parse_pred("zeta2") + validator_verdict + exact/reward path.

Usage: python3 landscape_gemma_scenarios.py [--n 60] [--limit N]

Writes results_scenarios_gemma4e2b_zeroshot.jsonl next to this script
(resume by id); prints the per-family aggregate LAST.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "post-processing"))
sys.path.insert(0, str(HERE.parent / "synthetic-data"))

from eval_scenarios import FAMILIES, load_heldout, validator_verdict  # noqa: E402
from landscape_glm_scenarios import FMT_INSTRUCTION                  # noqa: E402
from run_eval import parse_pred                                      # noqa: E402
import scenarios                                                     # noqa: E402

OUT_PATH = HERE / "results_scenarios_gemma4e2b_zeroshot.jsonl"
# The externally-owned 18106 server (user-authorized peer server) was reaped
# by its owner mid-run; we serve the SAME gguf ourselves (tracked PID, CUDA
# build) on 18110 when 18106 is not answering.
FALLBACK_PORT = 18110
GEMMA_GGUF = str(HERE.parent / "models" / "gemma4-e2b-qat-q4_0.gguf")
CUDA_BIN = "/tmp/llama.cpp-b10453/build-cuda/bin/llama-server"
PORT = 18106           # externally-owned peer server, when alive
PORT_GLOBAL = [PORT]   # resolved port (mutable so chat() sees the fallback)


def chat(port, prompt, max_tokens=2048, timeout=600):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"], time.time() - t0


def _alive(port):
    try:
        chat(port, "readiness", max_tokens=1, timeout=15)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--port", type=int, default=0,
                    help="force a port; default: 18106 if alive, else our "
                         "own tracked-PID server on 18110")
    args = ap.parse_args()

    port, owned_server = args.port or PORT, None
    if not args.port and not _alive(port):
        from landscape_zeta_scenarios import Server
        port = FALLBACK_PORT
        print(f"18106 not answering; serving {GEMMA_GGUF} ourselves on "
              f"{port} (tracked PID)", flush=True)
        owned_server = Server(CUDA_BIN, GEMMA_GGUF, port,
                              HERE / "llama-server-gemma4e2b.log")
        owned_server.start()
    PORT_GLOBAL[0] = port

    examples, report = load_heldout(150)   # EXACT eval_scenarios selection
    done = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            try:
                done.add(json.loads(line).get("id"))
            except ValueError:
                pass
        print(f"resume: {len(done)} row(s) already scored", flush=True)

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
                  prompt = row["_prompt"] + "\n\n" + FMT_INSTRUCTION
                  rec = dict(id=rid, family=fam, i=i, package=row["package"],
                             path=row["path"], note=row.get("note", ""),
                             model="gemma-4-e2b-zeroshot")
                  t0 = time.time()
                  try:
                      text, _ = chat(PORT_GLOBAL[0], prompt)
                      pred = parse_pred("zeta2", text)  # same parse/norm
                      ok, kind, reason = validator_verdict(row, pred)
                      gt = [l.rstrip() for l in row["region_new"]]
                      while gt and not gt[-1]:
                          gt.pop()
                      rec.update(latency_s=round(time.time() - t0, 2),
                                 n_pred_lines=len(pred),
                                 pred="\n".join(pred)[:400],
                                 raw=text[:600],
                                 valid_pass=int(ok), fail_kind=kind,
                                 valid_reason=reason,
                                 exact=int(pred == gt),
                                 reward=round(scenarios.exact_reward(
                                     pred, row["region_new"]), 4))
                  except Exception as e:
                      rec.update(latency_s=round(time.time() - t0, 2),
                                 n_pred_lines=0, pred=None, raw=None,
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
      if owned_server is not None:
          owned_server.stop()
          print("owned gemma server stopped", flush=True)

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
            n_scored=len(rs), validator_pass=frac("valid_pass"),
            exact=frac("exact"),
            mean_reward=round(sum(r["reward"] for r in rs) / len(rs), 4),
            fail_shape=round(sum(1 for r in rs if r.get("fail_kind") == "shape")
                             / len(rs), 4),
            fail_transform=round(sum(1 for r in rs if r.get("fail_kind")
                                     == "transform") / len(rs), 4),
            fail_error=round(sum(1 for r in rs if r.get("fail_kind") == "error")
                             / len(rs), 4),
            p50_latency_s=lat[len(lat) // 2],
            p95_latency_s=lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))])
    print(json.dumps(dict(aggregate=dict(
        model="gemma-4-e2b-zeroshot",
        serving=f"port {PORT_GLOBAL[0]} (18106 = externally-owned "
        "peer server when alive; else our tracked-PID CUDA server on 18110)",
        treatment="glm-5.3 parity: chat completions, temp 0, max_tokens "
        "2048, no stop, zeta2 render + one instruction",
        instruction=FMT_INSTRUCTION[:250],
        cap_per_family=args.n, smoke_limit=args.limit or None,
        selection={f: report[f] for f in FAMILIES},
        families=agg, total_rows_scored=len(rows),
    )), indent=1))


if __name__ == "__main__":
    main()
