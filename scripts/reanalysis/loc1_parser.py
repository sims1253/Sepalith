#!/usr/bin/env python3
"""Check saved LOC1 gold chunks and parent names with R parse(), never eval()."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from common import Evidence, dump, keyed


def parse_sources(sources):
    if not sources:
        return []
    with tempfile.TemporaryDirectory(prefix="loc1-parse-") as directory:
        paths = []
        for i, source in enumerate(sources):
            path = Path(directory) / f"{i}.R"
            path.write_text(source)
            paths.append(str(path))
        result = subprocess.run(
            [
                "Rscript",
                "--vanilla",
                str(Path(__file__).with_name("r_functions.R")),
                *paths,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    lines = result.stdout.splitlines()
    if len(lines) != len(sources):
        raise ValueError("R parser returned an unexpected number of records")
    records = []
    for line in lines:
        fields = line.split("\t")
        if fields == ["ERROR"]:
            records.append({"parse_ok": False, "expressions": None, "names": []})
        elif len(fields) >= 2 and fields[0] == "OK":
            records.append(
                {
                    "parse_ok": True,
                    "expressions": int(fields[1]),
                    "names": [bytes.fromhex(n).decode() for n in fields[2:]],
                }
            )
        else:
            raise ValueError("Unexpected R parser record")
    return records


def run(e, parents, receipts):
    parents, receipts = keyed(parents, "qid"), keyed(receipts["rows"], "qid")
    saved = keyed(e.json("/loc1_s0_results/results.json")["rows"], "qid")
    source = keyed(e.rows("/loc1_s0_r/set.jsonl"), "qid")
    corpora = keyed(e.rows("/loc1_s0_r/corpus.jsonl"), "qid")
    if parents.keys() != saved.keys() or receipts.keys() != saved.keys():
        raise ValueError("Parent and saved query sets differ")
    inputs, metadata = [], []
    for qid in sorted(saved):
        p, receipt = parents[qid], receipts[qid]
        blob = p["parent_source"]
        if blob is not None:
            if (
                hashlib.sha256(blob.encode()).hexdigest()
                != receipt["parent_source_sha256"]
            ):
                raise ValueError("Parent source hash mismatch")
            inputs.append(blob)
            metadata.append(
                {
                    "qid": qid,
                    "kind": "parent",
                    "source_sha256": receipt["parent_source_sha256"],
                    "gold_names": [g["name"] for g in source[qid]["gold"]],
                }
            )
        elif receipt["parent_source_available"]:
            raise ValueError("Missing available parent source")
        gold = {(g["path"], g["name"]) for g in source[qid]["gold"]}
        found = set()
        for i, chunk in enumerate(corpora[qid]["chunks"]):
            if (chunk["path"], chunk["name"]) in gold:
                found.add((chunk["path"], chunk["name"]))
                inputs.append(chunk["text"])
                metadata.append(
                    {
                        "qid": qid,
                        "kind": "gold_chunk",
                        "candidate_index": i,
                        "name": chunk["name"],
                        "source_sha256": hashlib.sha256(
                            chunk["text"].encode()
                        ).hexdigest(),
                    }
                )
        if found != gold:
            raise ValueError("Gold missing from corpus")
    rows = []
    for meta, parsed in zip(metadata, parse_sources(inputs), strict=True):
        row = meta | parsed
        if meta["kind"] == "gold_chunk":
            row["single_named_function"] = (
                parsed["parse_ok"]
                and parsed["expressions"] == 1
                and parsed["names"] == [meta["name"]]
            )
        else:
            row["gold_names_present"] = (
                sum(n in parsed["names"] for n in meta["gold_names"])
                if parsed["parse_ok"]
                else None
            )
        row["top_level_function_count"] = len(row.pop("names"))
        rows.append(row)
    return {
        "retrospective": True,
        "method": "R parse(), direct top-level named function assignments only; no code evaluation or corrected retrieval scores.",
        "r_version": subprocess.run(
            ["Rscript", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "unavailable_parent_qids": sorted(
            q for q in saved if parents[q]["parent_source"] is None
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "snapshot", "parent-sources", "receipts", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(
        Evidence(args.inventory, args.snapshot),
        json.loads(Path(args.parent_sources).read_text()),
        json.loads(Path(args.receipts).read_text()),
    )
    dump(args.out, result)


if __name__ == "__main__":
    main()
