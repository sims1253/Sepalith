#!/usr/bin/env python3
"""C09: DEV75/DEV250 generation for one GGUF in llama.cpp, scored like the PC evaluator.

    uv run python docs/training/plan-20260927/c09/gguf_dev_eval.py \\
        --gguf .../arm-a-Q4_K_M.gguf --tokenizer-dir <HF checkpoint dir> \\
        --panel dev250 --output .../dev250-Q4_K_M.json [--threads 8] [--build cuda --gpu-window]

Prompts are the PC evaluator's pre-tokenized ids (``build_training_row`` with
the checkpoint's HF tokenizer), sent to llama-server as an id list.  Greedy,
cap 1,024, stop ids 1 and 130073, one fresh request per case without prompt
caching.  Scoring calls ``campaign_eval.classify`` and
``generation_termination``; the per-case records and summary keep the PC keys
(see ``sepalith.quant.gguf_dev``), so results pair case by case with PC result
JSONs and with each other.

The CPU build runs under ``nice -n 10`` with every CUDA device hidden and at
most 8 threads.  The CUDA build needs ``--gpu-window`` (the night window owns
the GPU).  The result records the llama.cpp build, server flags, request
template, GGUF sha256 and tokenizer sha256.  ``--limit``/``--cap`` make a
smoke result, which the gate marks as not gate-eligible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages" / "sepalith" / "src"))
from sepalith.quant import gguf_dev as gd
from sepalith.quant.llama_tools import NICE, file_sha256, gpu_layers, llama_bin, llama_version, refuse_sealed, tool_env

TOKENIZER_JSON_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp, path)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--tokenizer-dir", type=Path, required=True, help="HF checkpoint dir (tokenizer.json is pinned)")
    ap.add_argument("--panel", choices=tuple(gd.PANELS), required=True)
    ap.add_argument("--panel-file", type=Path, help="override the panel path; its sha256 must still match")
    ap.add_argument("--output", type=Path, required=True, help="new result JSON; cases go to <output>.cases/cases.json")
    ap.add_argument("--cap", type=int, default=gd.GATE_CAP)
    ap.add_argument("--limit", type=int, help="evaluate only N cases (smoke)")
    ap.add_argument("--order", choices=("panel", "shortest"), default="panel", help="case order before --limit")
    ap.add_argument("--build", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--gpu-window", action="store_true", help="required for --build cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, help="server context (default: longest prompt + cap, rounded up to 256)")
    ap.add_argument("--case-timeout", type=float, default=3600.0)
    ap.add_argument("--label", help="free-form variant label recorded in the result")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    for path in (a.gguf, a.tokenizer_dir, a.output, a.panel_file or ""):
        refuse_sealed(path)
    if a.build == "cuda" and not a.gpu_window:
        raise SystemExit("the CUDA build runs only inside the GPU window (--gpu-window)")
    if a.threads < 1 or (a.threads > 8 and not a.gpu_window):
        raise SystemExit("daytime CPU rule: 1 to 8 threads")
    output = a.output.resolve()
    cases_path = output.with_suffix(".cases") / "cases.json"
    if output.exists() or cases_path.parent.exists():
        raise SystemExit(f"result already exists: {output}")
    started = time.monotonic()
    panel_path, panel_sha, denominator = gd.PANELS[a.panel]
    panel_path = a.panel_file or Path(panel_path)
    if file_sha256(panel_path) != panel_sha:
        raise SystemExit(f"{a.panel} panel hash differs: {panel_path}")
    if file_sha256(a.tokenizer_dir / "tokenizer.json") != TOKENIZER_JSON_SHA256:
        raise SystemExit("tokenizer.json differs from the pinned tokenizer")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(a.tokenizer_dir), local_files_only=True, trust_remote_code=False)
    cases = gd.load_panel(panel_path)
    if len(cases) != denominator:
        raise SystemExit(f"{a.panel} has {len(cases)} cases, expected {denominator}")
    prepared = gd.select(gd.prepare(cases, tokenizer), a.limit, a.order)
    prompt_max = max(row["target_start"] for _, _, row in prepared)
    ctx = a.ctx or -(-(prompt_max + a.cap) // 256) * 256
    if ctx < prompt_max + a.cap:
        raise SystemExit("server context is smaller than the longest prompt plus the cap")

    gguf = a.gguf.resolve()
    gguf_info = {"path": str(gguf), "bytes": gguf.stat().st_size, "sha256": file_sha256(gguf)}
    port = gd.free_port()
    server = llama_bin("llama-server", a.build)
    server_flags = ["-m", str(gguf), "-c", str(ctx), "-np", "1", "-t", str(a.threads), "-tb", str(a.threads),
                    *gpu_layers(a.build), "--host", "127.0.0.1", "--port", str(port), "--no-webui",
                    "--seed", "3407", "--no-context-shift", "--no-cache-prompt"]
    argv = [*NICE, str(server), *server_flags]
    runtime = {"engine": "llama-server", "llama_cpp": llama_version("llama-server", a.build), "build": a.build,
               "nice": 10, "threads": a.threads, "ctx": ctx, "server_argv": argv,
               "cuda_visible_devices": tool_env(a.build).get("CUDA_VISIBLE_DEVICES"),
               "request_template": {**gd.completion_request([], a.cap), "prompt": "<pre-tokenized ids>"},
               "stop_ids": list(gd.NATIVE_EOG_IDS), "per_case_fresh_request": True, "batch_size": 1}
    gate_eligible = a.cap == gd.GATE_CAP and not a.limit and a.order == "panel"
    results = []
    log_path = output.with_suffix(".server.log")
    output.parent.mkdir(parents=True, exist_ok=True)
    with gd.llama_server(argv, port, tool_env(a.build), log_path) as url:
        for case, context, row in prepared:
            prompt = row["input_ids"][:row["target_start"]]
            began = time.monotonic()
            response = gd.post_json(url + "/completion", gd.completion_request(prompt, a.cap), a.case_timeout)
            seconds = time.monotonic() - began
            generated = gd.generated_ids(response, len(prompt), a.cap)
            record = gd.case_record(case, context, row, generated, seconds, tokenizer, a.cap)
            record["llama_timings"] = response.get("timings")
            results.append(record)
            write_json(cases_path, {"status": "partial", "results": results,
                                    "expected_case_ids": [c["id"] for c, _, _ in prepared]})
            print(json.dumps({"case": case["id"], "generated": len(generated), "exact": record["exact_region"],
                              "valid": record["protocol_valid"], "seconds": round(seconds, 1)}), flush=True)
    summary = gd.summarize(results, a.cap, panel_sha, str(cases_path))
    summary["generation_runtime"] = runtime
    write_json(cases_path, {"status": "complete", "summary": summary, "results": results})
    result = {
        "schema": gd.SCHEMA, "status": "complete", "label": a.label, "panel_name": a.panel,
        "panel": {"path": str(panel_path), "sha256": panel_sha, "cases_in_panel": denominator},
        "gate_eligible": gate_eligible,
        "smoke": None if gate_eligible else {"limit": a.limit, "order": a.order, "cap": a.cap},
        "gguf": gguf_info, "tokenizer": {"dir": str(a.tokenizer_dir), "tokenizer_json_sha256": TOKENIZER_JSON_SHA256},
        "summary": summary, "cases": {"path": str(cases_path), "sha256": file_sha256(cases_path)},
        "server_log": str(log_path), "elapsed_seconds": round(time.monotonic() - started, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "final_set_access": False,
    }
    write_json(output, result)
    counts = summary["counts"]
    print(json.dumps({"output": str(output), "cases": len(results), "valid": counts["protocol_valid"],
                      "exact": counts["exact_region"], "edit": counts["edit_exact"],
                      "nofp": counts["strict_noop_false_suggestions"], "cap": counts["cap_hit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
