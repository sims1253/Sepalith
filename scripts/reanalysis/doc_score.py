#!/usr/bin/env python3
"""Aggregate retrospective human/agent labels; never overwrite historical scores."""

import argparse
import json
import re
from pathlib import Path

from common import dump, keyed

TAG = re.compile(r"^\s*#'\s+@param\s+(\S+)\s+(.+)$")


def structural(packet):
    old, pred = packet["old"], packet["prediction"]
    arg = "call" if packet["case"] == "aacd773af914" else "verbose"
    old_names = {m[1] for line in old if (m := TAG.match(line))}
    tags = [(i, m[1], m[2]) for i, line in enumerate(pred) if (m := TAG.match(line))]
    added = [i for i, name, desc in tags if name == arg]
    allowed = old_names | {arg}
    coverage = len(added) == 1 and all(n in allowed for _, n, _ in tags)
    anchors = [
        i
        for i, line in enumerate(pred)
        if re.match(r"^\s*#'\s+@(return|export)\b", line)
    ]
    placement = len(added) == 1 and bool(anchors) and added[0] < min(anchors)
    formatting = bool(pred) and all(
        not line.strip() or re.match(r"^\s*#'(\s|$)", line) for line in pred
    )
    # Exact preservation is only an initial screen. Reviewed semantic equivalents
    # and harmful additions are resolved by the packet annotations.
    remaining = iter(line.strip() for line in pred)
    preservation = all(
        any(candidate == line.strip() for candidate in remaining) for line in old
    )
    return {
        "coverage": bool(coverage),
        "placement": placement,
        "formatting": bool(formatting),
        "preservation": preservation,
    }


def aggregate(packets, mapping, annotations):
    labels = keyed(annotations, "packet")
    if set(labels) != {p["packet"] for p in packets}:
        raise ValueError("Annotation coverage differs from packets")
    scored = {}
    for p in packets:
        label = labels[p["packet"]]
        if label["target_semantic"] not in ("pass", "fail", "unknown") or label[
            "factual"
        ] not in ("pass", "fail", "unknown"):
            raise ValueError("Invalid semantic label")
        s = structural(p)
        overrides = label.get("structural_overrides", {})
        if not overrides.keys() <= s.keys():
            raise ValueError("Unknown structural override")
        s.update(overrides)
        if any(
            type(s[k]) is not bool
            for k in ("coverage", "placement", "formatting", "preservation")
        ):
            raise ValueError("Structural labels must be boolean")
        s.update(target_semantic=label["target_semantic"], factual=label["factual"])
        s["target_usable"] = (
            all(s[k] for k in ("coverage", "placement", "formatting", "preservation"))
            and s["target_semantic"] == "pass"
        )
        s["prompt_supported_usable"] = s["target_usable"] and s["factual"] == "pass"
        if p["raw"] is None:
            s = {k: None for k in s}
        s["raw_available"] = p["raw"] is not None
        scored[p["packet"]] = s
    rows = []
    for m in mapping:
        # Do not publish raw validator reasons: they can contain source contexts.
        result = {
            k: m[k]
            for k in (
                "arm",
                "case",
                "packet",
                "exact",
                "valid_pass",
                "fail_kind",
                "source_sha256",
                "evidence_status",
            )
        } | scored[m["packet"]]
        if m["evidence_status"] != "complete":
            for k in (
                "coverage",
                "placement",
                "formatting",
                "preservation",
                "target_semantic",
                "factual",
                "target_usable",
                "prompt_supported_usable",
            ):
                result[k] = None
        result["request_prompt_sha256"] = m.get("request_prompt_sha256")
        result["adjudicable"] = m["evidence_status"] == "complete"
        rows.append(result)
    totals = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        totals[arm] = {
            "n": len(rs),
            "raw_available": sum(r["raw_available"] for r in rs),
            "adjudicable": sum(r["adjudicable"] for r in rs),
        }
        for k in (
            "exact",
            "valid_pass",
            "coverage",
            "placement",
            "formatting",
            "preservation",
            "target_usable",
            "prompt_supported_usable",
        ):
            values = [r[k] for r in rs if r[k] is not None]
            totals[arm][k] = sum(values) if values else None
    return {"retrospective": True, "arms": totals, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packets", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    def read(f):
        return json.loads(Path(f).read_text())

    dump(a.out, aggregate(read(a.packets), read(a.mapping), read(a.annotations)))


if __name__ == "__main__":
    main()
