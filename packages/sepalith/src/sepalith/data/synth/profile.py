"""Distribution profile of editing cases, computed identically for TRAIN and SYNTH.

Inputs are (family, context, target_lines) triples, where ``context`` is the
PRM03 context object (sidecar schema). Metrics, all as shares of the family:

- target length buckets (lines), median target lines;
- targets containing a blank line / a full-line ``#`` comment (roxygen lines
  excluded); targets that start with a blank line; single-line ``}`` targets;
- prefixes without any blank line;
- finish_block: full-body cuts (the prompt's last code line opens the body,
  or the region is the signature line) and test-file cuts (path under tests/);
- roxygen_drafting: per-tag frequency (share of targets with the tag) and the
  most common tag orders.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
import re
import statistics

BUCKETS = ((1, "1"), (2, "2"), (4, "3-4"), (8, "5-8"), (12, "9-12"), (20, "13-20"), (30, "21-30"), (45, "31-45"))
TAG_RE = re.compile(r"^#'\s*@(\w+)")
TRAIN_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl")
TRAIN_SIDECAR = TRAIN_ROWS.with_name("context-sidecar.jsonl")


def bucket(n: int) -> str:
    for bound, name in BUCKETS:
        if n <= bound:
            return name
    return ">45"


def is_comment(line: str) -> bool:
    s = line.strip()
    return s.startswith("#") and not s.startswith("#'")


SIG_TAIL_RE = re.compile(r"^\s*[\w.]+\s*(=.*)?\)\s*\{$")


def full_body(context: Mapping, target: Sequence[str]) -> bool:
    """The cut sits directly after a function signature (the body is entirely the target)."""
    if context["region_old"]:
        last = context["region_old"][-1]
        return last.rstrip().endswith("{") and bool(target) and target[0] == context["region_old"][0]
    code = [line for line in context["prefix"] if line.strip() and not line.lstrip().startswith("#")]
    if not code or not code[-1].rstrip().endswith("{"):
        return False
    if "function(" in code[-1] or "function (" in code[-1]:
        return not code[-1][:1].isspace()  # a top-level signature, not an anonymous function
    # multi-line signature: ...function(\n  a,\n  b\n) {
    if not SIG_TAIL_RE.match(code[-1]) and code[-1].strip() != ") {":
        return False
    for line in reversed(code[:-1]):
        if "function(" in line or "function (" in line:
            return not line[:1].isspace() and line.rstrip().endswith(("(", ","))
        if not line.rstrip().endswith((",", "(")):
            return False
    return False


def test_file(context: Mapping) -> bool:
    """The cut unit is a testthat block (or the file is under tests/)."""
    if "test" in context["path"].lower():
        return True
    tops = [line for line in context["prefix"] if line and not line[0].isspace() and not line.startswith(("#", "}", ")"))]
    return bool(tops) and tops[-1].startswith(("test_that(", "describe(", "it("))


def profile(items: Iterable[tuple[str, Mapping, Sequence[str]]]) -> dict:
    acc: dict[str, dict] = defaultdict(lambda: {"n": 0, "lengths": [], "buckets": Counter(), "blank": 0,
                                                 "comment": 0, "leading_blank": 0, "lone_close": 0,
                                                 "prefix_no_blank": 0, "full_body": 0, "full_body_signature_region": 0, "test_file": 0,
                                                 "tags": Counter(), "orders": Counter(),
                                                 "cells": defaultdict(lambda: [0, 0, 0])})
    for family, context, target in items:
        a = acc[family]
        a["n"] += 1
        a["lengths"].append(len(target))
        a["buckets"][bucket(len(target))] += 1
        body = target[1:] if target and not target[0].strip() else target
        cell = a["cells"][bucket(len(body))]
        cell[0] += 1
        cell[1] += any(not line.strip() for line in body)
        cell[2] += any(is_comment(line) for line in body)
        a["blank"] += any(not line.strip() for line in target)
        a["comment"] += any(is_comment(line) for line in target)
        a["leading_blank"] += bool(target) and not target[0].strip()
        a["lone_close"] += len(target) == 1 and target[0].strip() == "}"
        a["prefix_no_blank"] += not any(not line.strip() for line in context["prefix"])
        if family == "finish_block":
            body = full_body(context, target)
            a["full_body"] += body
            a["full_body_signature_region"] += body and bool(context["region_old"])
            a["test_file"] += test_file(context)
        if family == "roxygen_drafting":
            tags = [m.group(1) for m in map(TAG_RE.match, (line.strip() for line in target)) if m]
            for tag in set(tags):
                a["tags"][tag] += 1
            order = []
            for tag in tags:
                if not order or order[-1] != tag:
                    order.append(tag)
            a["orders"][" ".join(order)] += 1
    out = {}
    for family, a in sorted(acc.items()):
        n = a["n"]
        share = lambda k, a=a, n=n: round(a[k] / n, 4)  # noqa: E731
        entry = {"cases": n, "median_target_lines": statistics.median(a["lengths"]),
                 "target_length_buckets": {name: round(a["buckets"][name] / n, 4)
                                           for name in [b for _, b in BUCKETS] + [">45"]},
                 "any_blank_line": share("blank"), "any_full_line_comment": share("comment"),
                 "leading_blank_line": share("leading_blank"), "single_close_brace": a["lone_close"],
                 "prefix_without_blank_line": share("prefix_no_blank"),
                 "by_bucket_blank_comment": {b: {"cases": c[0], "blank": round(c[1] / c[0], 4),
                                                 "comment": round(c[2] / c[0], 4)}
                                             for b, c in sorted(a["cells"].items())}}
        if family == "finish_block":
            entry.update(full_body=share("full_body"), full_body_signature_region=share("full_body_signature_region"),
                         full_body_empty_region=round((a["full_body"] - a["full_body_signature_region"]) / n, 4),
                         test_file=share("test_file"))
        if family == "roxygen_drafting":
            entry["tag_frequency"] = {t: round(c / n, 4) for t, c in a["tags"].most_common()}
            entry["top_tag_orders"] = [[o, c] for o, c in a["orders"].most_common(8)]
            entry["distinct_tag_orders"] = len(a["orders"])
        out[family] = entry
    return out


def train_items(families: set[str] | None = None) -> Iterable[tuple[str, Mapping, list[str]]]:
    targets = {}
    with TRAIN_ROWS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["target_operation"] == "replace" and (families is None or row["family"] in families):
                targets[row["id"]] = row["target_body_text"].split("\n")
    with TRAIN_SIDECAR.open(encoding="utf-8") as stream:
        for line in stream:
            side = json.loads(line)
            if side["row_id"] in targets:
                yield side["family"], side["context"], targets[side["row_id"]]


def synth_items(cases_path: Path) -> Iterable[tuple[str, Mapping, list[str]]]:
    with cases_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            yield record["family"], record["context"], record["region_new"]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    families = {"finish_block", "roxygen_drafting", "format_propagation"}
    result = {"train": profile(train_items(families))}
    for path in args.cases:
        result[str(path)] = profile(synth_items(path))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def train_finish_geometry() -> list[tuple[int, int]]:
    """(prefix lines, prompt tokens) of every TRAIN finish_block row."""
    tokens = {}
    with TRAIN_ROWS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["family"] == "finish_block":
                tokens[row["id"]] = row["prompt_token_count"]
    out = []
    with TRAIN_SIDECAR.open(encoding="utf-8") as stream:
        for line in stream:
            side = json.loads(line)
            if side["row_id"] in tokens:
                out.append((len(side["context"]["prefix"]), tokens[side["row_id"]]))
    return out
