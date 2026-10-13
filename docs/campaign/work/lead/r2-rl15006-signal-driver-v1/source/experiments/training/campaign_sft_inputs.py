"""Project an approved registry into immutable SFT inputs and finite schedules."""
import argparse
from collections import Counter
import json
from pathlib import Path

from campaign_checkpoint import digest, write_json
from campaign_sampling import build_draw_manifest, validate_draw_manifest


def build(registry, provenance, report, output, split_id):
    approval = json.loads(report.read_text())
    if approval.get("status") != "admitted" or not approval.get("admission_ready"):
        raise ValueError("An admitted registry report is required")
    for name, path in (("registry", registry), ("provenance", provenance)):
        record = approval["output"]
        if str(path) != record[name + "_path"] or digest(path) != record[name + "_sha256"]:
            raise ValueError(f"Approved {name} bytes differ")
    if output.exists():
        raise ValueError("Use a new output directory")
    rows = [json.loads(line) for line in registry.read_text().splitlines()]
    refs = {item["id"]: item for item in
            (json.loads(line) for line in provenance.read_text().splitlines())
            if item["decision"] == "admitted"}
    if (len(rows) != approval["admitted_rows"] or len({r["id"] for r in rows}) != len(rows)
            or set(refs) != {r["id"] for r in rows}):
        raise ValueError("Registry, provenance and report identities differ")
    train = [row for row in rows if row["split"] == "train"]
    output.mkdir(parents=True)
    target = output / "train-token-rows.jsonl"
    with target.open("x") as stream:
        for row in train:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = []
    for row in train:
        ref = refs[row["id"]]
        source = ref["source_ref"]["source"]
        if not source:
            raise ValueError("A training row lacks a source identity")
        metadata.append({
            "row_id": row["id"], "family": row["family"],
            "source_id": source, "package_id": row["package_id"], "split": "train",
            "semantic_noop": row["target_operation"] == "no_op",
            "operation": row["target_operation"],
            "prompt_tokens": row["target_start"],
            "target_tokens": len(row["input_ids"]) - row["target_start"],
            "total_tokens": len(row["input_ids"]),
            "length_bucket": "long" if len(row["input_ids"]) > 2048 else "short",
            "naturally_long": len(row["input_ids"]) > 2048,
            "source_kind": "ordinary", "admitted": True,
            "provenance": ref["approval_id"],
        })
    write_json(output / "sampler-metadata.json", metadata)
    schedules = {}
    for steps in (50, 3000):
        manifest = build_draw_manifest(
            metadata, max_steps=steps, effective_batch=16, split_id=split_id,
            seed=3407, token_rows_sha256=digest(target),
        )
        validate_draw_manifest(manifest)
        if manifest["status"] != "complete":
            write_json(output / f"infeasible-{steps}.json", manifest)
            raise ValueError(f"Finite schedule for {steps} updates is infeasible")
        path = output / f"draws-{steps}.json"
        write_json(path, manifest)
        schedules[str(steps)] = {"path": str(path), "sha256": digest(path)}
    result = {
        "status": "prepared_from_admitted_registry", "split_id": split_id,
        "registry": {"path": str(registry), "sha256": digest(registry)},
        "provenance": {"path": str(provenance), "sha256": digest(provenance)},
        "admission_report": {"path": str(report), "sha256": digest(report)},
        "train_token_rows": {"path": str(target), "sha256": digest(target)},
        "train_rows": len(train), "family_counts": dict(Counter(r["family"] for r in train)),
        "schedules": schedules,
        "source_kind_decision": "Existing source corpora only; no newly authored small-pack rows. Ordinary replay cap8 applies, small-pack cap3 remains configured.",
        "token_count_policy": "Prompt count includes the one manual BOS; target count includes complete output protocol and terminal EOS. Positional loss masks padding only.",
    }
    write_json(output / "receipt.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "provenance", "report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split-id", required=True)
    args = parser.parse_args()
    result = build(args.registry, args.provenance, args.report, args.output, args.split_id)
    print(json.dumps({"status": result["status"], "train_rows": result["train_rows"]}))
