#!/usr/bin/env python3
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in values))


def terminal(path: Path, source: Path, output: Path) -> None:
    path.write_text(json.dumps({
        "status": "complete",
        "input": {"sha256": sha(source)},
        "output": {"sha256": sha(output)},
    }) + "\n")


with tempfile.TemporaryDirectory(prefix="semantic10948-policy-") as td:
    root = Path(td)
    inputs, render16, fallback, render32, selected = [root / x for x in ("inputs", "render16", "fallback", "render32", "selected")]
    entries = []
    cursor = 0
    for shard in range(12, 27):
        count = 730 if shard < 25 else 729
        if shard == 26:
            count = 729
        values = [{"row_id": f"r{i:05d}"} for i in range(cursor, cursor + count)]
        cursor += count
        source = inputs / f"shard-{shard:04d}.jsonl"
        write_rows(source, values)
        entries.append({"shard": shard, "rows": count, "path": source.name, "sha256": sha(source), "bytes": source.stat().st_size})
        r16 = render16 / source.name
        rows16 = [{"row_id": x["row_id"], "status": "supported" if int(x["row_id"][1:]) % 2 == 0 else "hold", "reasons": ["complete_span_context_budget"]} for x in values]
        write_rows(r16, rows16)
        terminal(render16 / f"shard-{shard:04d}.terminal.json", source, r16)
    assert cursor == 10948
    inputs.mkdir(exist_ok=True)
    (inputs / "manifest.json").write_text(json.dumps({"status": "complete_target_free", "rows": 10948, "shards": entries}) + "\n")
    subprocess.run(["python3", str(HERE / "prepare_fallback.py"), "--inputs", str(inputs), "--render16", str(render16), "--output", str(fallback)], check=True, capture_output=True, text=True)
    fallback_manifest = json.loads((fallback / "manifest.json").read_text())
    assert fallback_manifest["rerun32_rows"] == 5474
    for entry in fallback_manifest["shards"]:
        shard = entry["shard"]
        source = fallback / entry["path"]
        vals = [json.loads(x) for x in source.read_text().splitlines()]
        out = render32 / source.name
        write_rows(out, [{"row_id": x["row_id"], "status": "hold" if x["row_id"] == "r00001" else "supported", "reasons": ["complete_span_context_budget"] if x["row_id"] == "r00001" else []} for x in vals])
        terminal(render32 / f"shard-{shard:04d}.terminal.json", source, out)
    subprocess.run(["python3", str(HERE / "finalize_policy.py"), "--inputs", str(inputs), "--render16", str(render16), "--fallback_inputs", str(fallback), "--render32", str(render32), "--output", str(selected)], check=True, capture_output=True, text=True)
    manifest = json.loads((selected / "manifest.json").read_text())
    assert manifest["denominator"] == 10948
    assert manifest["supported"] == 10947
    assert manifest["holds"] == 1
    assert manifest["counts"]["context_only_followup_64k_128k"] == 1
    assert sum(x["rows"] for x in manifest["followup_context_inputs"]["shards"]) == 1
    damaged = render32 / "shard-0012.jsonl"
    damaged.write_text(damaged.read_text() + "{}\n")
    failed = subprocess.run(["python3", str(HERE / "finalize_policy.py"), "--inputs", str(inputs), "--render16", str(render16), "--fallback_inputs", str(fallback), "--render32", str(render32), "--output", str(root / "must-fail")], capture_output=True, text=True)
    assert failed.returncode != 0

print("PASS test_policy_pipeline: exact 10948 closure, 16K/32K selection, 64K/128K queue, corrupt binding rejection")
