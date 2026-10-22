"""Difficulty probe: arm A (Q8_0 GGUF, llama.cpp CPU) greedy on a stratified
SYNTH-v1 sample, scored with reward v2.1 exactly as the gold self-check.

CPU only (CUDA hidden), 8 threads, nice 10, one request at a time; prompts are
sent as the rows' token ids (no re-tokenisation). Q8_0 is close to but not
identical with the bf16 checkpoint, so read rates as approximate.

    PYTHONPATH=packages/sepalith/src .venv-sft/bin/python docs/training/plan-20260927/synth/probe_arm_a.py \
        --data /mnt/e/.../SYNTH-v1 --out docs/training/plan-20260927/receipts/SYNTH-v1-arm-a-probe.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import tempfile
import time

from sepalith.campaign_protocol import PromptContext
from sepalith.data.synth.build import make_scorer, sha256_file
from sepalith.quant import gguf_dev as gd
from sepalith.quant.llama_tools import gpu_layers, llama_bin, tool_env
from sepalith.training.rl.reward_v21 import Evidence, ReferenceSpec, Sample

GGUF = Path("/mnt/e/sepalith/campaign-20260915/quant-work/arm-a/arm-a-Q8_0.gguf")
STRATA = {("finish_block", "short"): 60, ("finish_block", "medium"): 50, ("finish_block", "long"): 10,
          ("roxygen_drafting", None): 30}
CAP = 1024
SEED = 3407  # sampling seed; the audit pack uses 20261022 and its ids are excluded here


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--exclude-ids", type=Path, default=None, help="JSON list of row ids to leave out")
    args = parser.parse_args()
    if args.threads > 8:
        raise SystemExit("at most 8 threads")
    records = {r["row_id"]: r for r in map(json.loads, (args.data / "cases.jsonl").open())}
    rows = {r["id"]: r for r in map(json.loads, (args.data / "train-token-rows.jsonl").open())}
    excluded = set(json.loads(args.exclude_ids.read_text())) if args.exclude_ids else set()
    pools = defaultdict(list)
    for rid, record in sorted(records.items()):
        if rid in excluded:
            continue
        bucket = record["meta"].get("bucket") if record["family"] == "finish_block" else None
        pools[(record["family"], bucket)].append(rid)
    rng = random.Random(SEED)
    chosen = [(key, rid) for key, n in STRATA.items() for rid in rng.sample(pools[key], min(n, len(pools[key])))]
    prompt_max = max(rows[rid]["target_start"] for _, rid in chosen)
    ctx = -(-(prompt_max + CAP) // 256) * 256
    port = gd.free_port()
    argv = ["nice", "-n", "10", str(llama_bin("llama-server", "cpu")), "-m", str(GGUF), "-c", str(ctx), "-np", "1",
            "-t", str(args.threads), "-tb", str(args.threads), *gpu_layers("cpu"), "--host", "127.0.0.1",
            "--port", str(port), "--no-webui", "--seed", str(SEED), "--no-context-shift", "--no-cache-prompt"]
    generated: dict[str, list[int]] = {}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="synth-probe-") as tmp:
        with gd.llama_server(argv, port, tool_env("cpu"), Path(tmp) / "server.log") as url:
            for index, (_key, rid) in enumerate(chosen):
                prompt = rows[rid]["input_ids"][:rows[rid]["target_start"]]
                response = gd.post_json(f"{url}/completion", gd.completion_request(prompt, CAP, SEED), 900)
                generated[rid] = gd.generated_ids(response, len(prompt), CAP)
                if (index + 1) % 25 == 0:
                    print(f"[{index + 1}/{len(chosen)}] {time.monotonic() - started:.0f}s", flush=True)
    scorer, _formatter, decoder = make_scorer(8)
    specs = [ReferenceSpec(rid, records[rid]["family"], PromptContext.from_dict(records[rid]["context"]), "replace",
                           rows[rid]["target_body_text"], Evidence("unverified")) for _, rid in chosen]
    samples = [Sample(rid, completion_ids=tuple(generated[rid]), sample_id="arm-a-q8-greedy") for _, rid in chosen]
    scored = {r["row_id"]: r for r in scorer.score(scorer.build_references(specs), samples, decoder)}
    table: dict[str, dict] = {}
    for key, _n in STRATA.items():
        ids = [rid for k, rid in chosen if k == key]
        outcomes: dict[str, int] = defaultdict(int)
        for rid in ids:
            outcomes[str(scored[rid]["outcome"])] += 1
        table[f"{key[0]}:{key[1] or 'all'}"] = {
            "cases": len(ids), "exact": sum(bool(scored[rid]["exact"]) for rid in ids),
            "mean_reward": round(sum(float(str(scored[rid]["reward"])) for rid in ids) / max(1, len(ids)), 4),
            "outcomes": dict(outcomes)}
    receipt = {
        "schema": "sepalith.synth.arm-a-probe.v1",
        "model": {"gguf": str(GGUF), "sha256": sha256_file(GGUF), "note": "arm A checkpoint 536, Q8_0"},
        "decoding": {"greedy": True, "cap": CAP, "engine": "llama.cpp CPU llama-server", "threads": args.threads},
        "sample": {"seed": SEED, "strata": {f"{k[0]}:{k[1] or 'all'}": n for k, n in STRATA.items()},
                   "excluded_ids": len(excluded), "overlap_with_excluded": len(excluded & {rid for _, rid in chosen}),
                   "row_ids": [rid for _, rid in chosen]},
        "by_stratum": table,
        "total": {"cases": len(chosen), "exact": sum(bool(r["exact"]) for r in scored.values())},
        "wall_seconds": round(time.monotonic() - started, 1),
        "data": {"train-token-rows.jsonl": sha256_file(args.data / "train-token-rows.jsonl"),
                 "manifest.json": sha256_file(args.data / "manifest.json")},
    }
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"by_stratum": {k: (v["exact"], v["cases"]) for k, v in table.items()},
                      "total": receipt["total"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
