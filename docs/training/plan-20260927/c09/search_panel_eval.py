#!/usr/bin/env python3
"""C09 search: TRAIN-panel greedy generation for one GGUF, for quant recipe search.

    uv run python docs/training/plan-20260927/c09/search_panel_eval.py \\
        --gguf .../candidate.gguf --tokenizer-dir <HF checkpoint dir> \\
        --panel /mnt/e/.../autoresearch-20261017/search-panel.json \\
        --output .../search-<label>.json [--threads 8] [--build cpu]

Unlike ``gguf_dev_eval.py`` (DEV75/DEV250, gate confirmations only), this tool
evaluates the TRAIN-derived search panel from
``sepalith.quant.search_panel``: family-balanced rows excluded from the
imatrix calibration set, the held-out KLD slice and the DEV-excluded rows.  Prompts are the rows'
pre-tokenized ``input_ids[:target_start]``, sent as token ids — the same
request semantics as the DEV evaluator (greedy, cap, stop ids 1 and 130073,
fresh request per case, no prompt caching).

The result JSON holds one record per case (``generated_ids`` included) plus a
summary; compare two result files with ``--compare REF CAND`` to score greedy
agreement with the bf16 reference (identical generations, cap transitions,
length drift) without touching DEV data.
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
from sepalith.quant import search_panel as sp
from sepalith.quant import gguf_dev as gd
from sepalith.quant.llama_tools import NICE, file_sha256, gpu_layers, llama_bin, llama_version, refuse_sealed, tool_env

TOKENIZER_JSON_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"


def build(args) -> dict:
    """Build the panel file from TRAIN rows minus calibration/KLD ids."""
    rows = []
    with open(args.train_rows) as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    reserved = sp.manifest_row_ids(Path(args.calibration_manifest))
    cases = sp.build_panel(rows, per_family=args.per_family, seed=args.seed, calibration_ids=reserved["calibration"],
                           kld_ids=reserved["kld"], dev_excluded_ids=reserved["dev_excluded"])
    payload = {"schema": sp.SCHEMA, "seed": args.seed, "per_family": args.per_family,
               "train_rows": str(args.train_rows), "excluded": {k: len(v) for k, v in reserved.items()},
               "cases": cases}
    Path(args.panel).write_text(json.dumps(payload, indent=2) + "\n")
    return {"panel": str(args.panel), "cases": len(cases),
            "sha256_of_ids": sp.panel_sha256(cases)}


def generate(args) -> int:
    for path in (args.gguf, args.tokenizer_dir, args.output, args.panel):
        refuse_sealed(path)
    if args.build == "cuda" and not args.gpu_window:
        raise SystemExit("the CUDA build runs only inside the GPU window (--gpu-window)")
    if args.threads < 1 or (args.threads > 8 and not args.gpu_window):
        raise SystemExit("daytime CPU rule: 1 to 8 threads")
    if file_sha256(args.tokenizer_dir / "tokenizer.json") != TOKENIZER_JSON_SHA256:
        raise SystemExit("tokenizer.json differs from the pinned tokenizer")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"result already exists: {output}")
    cases = sp.load_panel(Path(args.panel))
    if args.limit:
        cases = cases[: args.limit]
    prompt_max = max(len(c["prompt_ids"]) for c in cases)
    ctx = args.ctx or -(-(prompt_max + args.cap) // 256) * 256
    gguf = args.gguf.resolve()
    port = gd.free_port()
    server = llama_bin("llama-server", args.build)
    server_flags = ["-m", str(gguf), "-c", str(ctx), "-np", "1", "-t", str(args.threads), "-tb", str(args.threads),
                    *gpu_layers(args.build), "--host", "127.0.0.1", "--port", str(port), "--no-webui",
                    "--seed", "3407", "--no-context-shift", "--no-cache-prompt"]
    argv = [*NICE, str(server), *server_flags]
    started = time.monotonic()
    records = []
    with tempfile.TemporaryDirectory(prefix="search-panel-") as tmp:
        with gd.llama_server(argv, port, tool_env(args.build), Path(tmp) / "server.log") as url:
            for index, case in enumerate(cases):
                t0 = time.monotonic()
                body = sp.completion_for(case["prompt_ids"], args.cap)
                response = gd.post_json(f"{url}/completion", body, args.case_timeout)
                tokens = sp.generated(response, len(case["prompt_ids"]), args.cap)
                records.append(sp.case_record(case, tokens, time.monotonic() - t0, args.cap))
                if args.progress and (index + 1) % args.progress == 0:
                    print(f"[{index + 1}/{len(cases)}] {records[-1]['generated_tokens']} tok "
                          f"{records[-1]['generation_seconds']:.1f}s", flush=True)
    summary = {"cases": len(records), "cap_hits": sum(r["cap_hit"] for r in records),
               "identical_pairs": None,
               "mean_generated_tokens": sum(r["generated_tokens"] for r in records) / len(records),
               "wall_seconds": time.monotonic() - started}
    result = {"schema": "sepalith.c09.quant-search-generation.v1", "label": args.label,
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "gguf": {"path": str(gguf), "bytes": gguf.stat().st_size, "sha256": file_sha256(gguf)},
              "panel": {"path": str(args.panel), "ids_sha256": sp.panel_sha256(cases)},
              "cap": args.cap, "threads": args.threads, "build": args.build,
              "llama_cpp": llama_version("llama-server", args.build), "server_argv": argv,
              "results": records, "summary": summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="." + output.name, dir=output.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    os.replace(tmp_path, output)
    print(json.dumps(summary))
    return 0


def compare(args) -> int:
    ref = json.loads(Path(args.reference).read_text())
    cand = json.loads(Path(args.candidate).read_text())
    stats = sp.agreement(ref["results"], cand["results"], cand["cap"])
    payload = {"schema": "sepalith.c09.quant-search-agreement.v1", "reference": str(args.reference),
               "candidate": str(args.candidate), "agreement": stats}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the TRAIN search panel file")
    b.add_argument("--train-rows",
                   default="/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl")
    b.add_argument("--calibration-manifest",
                   default="/mnt/e/sepalith/campaign-20260915/quant-work/calibration-v1/manifest.json")
    b.add_argument("--panel", required=True)
    b.add_argument("--per-family", type=int, default=28)
    b.add_argument("--seed", type=int, default=20261017)

    g = sub.add_parser("generate", help="greedy generation for one GGUF")
    g.add_argument("--gguf", type=Path, required=True)
    g.add_argument("--tokenizer-dir", type=Path, required=True)
    g.add_argument("--panel", required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--label", default="")
    g.add_argument("--cap", type=int, default=gd.GATE_CAP)
    g.add_argument("--limit", type=int)
    g.add_argument("--threads", type=int, default=8)
    g.add_argument("--ctx", type=int)
    g.add_argument("--case-timeout", type=float, default=3600.0)
    g.add_argument("--build", choices=("cpu", "cuda"), default="cpu")
    g.add_argument("--gpu-window", action="store_true")
    g.add_argument("--progress", type=int, default=25)

    c = sub.add_parser("compare", help="agreement of a candidate result vs the bf16 reference result")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--output", type=Path)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "build":
        print(json.dumps(build(args)))
        return 0
    if args.cmd == "generate":
        return generate(args)
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
