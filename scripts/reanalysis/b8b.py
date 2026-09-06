#!/usr/bin/env python3
"""Paired B8b reconstruction and target-free token-preservation selector."""

import argparse
import hashlib
import re
import sys
from pathlib import Path

from common import Evidence, dump, paired

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/sepalith/src"))
from sepalith.evaluation import audit_pair


def split_group(name):
    return (
        "development"
        if hashlib.sha256(name.encode()).digest()[0] < 102
        else "evaluation"
    )


def r_tokens(text):
    """Conservative lexical screen, not an R parser or correctness validator."""
    pattern = re.compile(
        r"\s+|\#[^\n]*|[A-Za-z_.][A-Za-z0-9_.]*|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[Li]?|%[^%\n]*%|<<-|->>|<-|->|:::{0,1}|\|>|&&|\|\||==|!=|<=|>=|[^\s]"
    )
    tokens = []
    i = 0
    while i < len(text):
        if text[i] in ('"', "'", "`"):
            quote, start = text[i], i
            i += 1
            while i < len(text) and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
            if i >= len(text):
                return None
            i += 1
            tokens.append(text[start:i])
        else:
            m = pattern.match(text, i)
            if m is None:
                return None
            token = m.group()
            if not token.isspace():
                tokens.append(token)
            i = m.end()
    return tokens


def select(prompt, control, treatment):
    """All inputs available when choosing between two generated replacements."""
    event = prompt.split("<filename>edit_history\n", 1)[-1].split("<filename>", 1)[0]
    old_event = "\n".join(l[1:] for l in event.splitlines() if l.startswith("-"))
    new_event = "\n".join(l[1:] for l in event.splitlines() if l.startswith("+"))
    if (
        not old_event
        or not new_event
        or old_event == new_event
        or r_tokens(old_event) is None
        or r_tokens(old_event) != r_tokens(new_event)
    ):
        return "control"
    if "<<<<<<< CURRENT\n" not in prompt or "\n=======" not in prompt:
        return "control"
    old = prompt.split("<<<<<<< CURRENT\n", 1)[1].split("\n=======", 1)[0]
    old = old.replace("<|user_cursor|>", "")
    old_tokens = r_tokens(old)
    if old_tokens is None:
        return "control"
    valid_a = r_tokens(control) == old_tokens
    valid_b = r_tokens(treatment) == old_tokens
    return "treatment" if valid_b and not valid_a else "control"


def summary(rows):
    if not rows:
        return {"n": 0}
    out = {
        "n": len(rows),
        "control_exact": sum(r["control_exact"] for r in rows),
        "treatment_exact": sum(r["treatment_exact"] for r in rows),
        "oracle_exact": sum(
            max(r["control_exact"], r["treatment_exact"]) for r in rows
        ),
        "selected_exact": sum(r["selected_exact"] for r in rows),
        "switches": sum(r["selected"] == "treatment" for r in rows),
    }
    out["paired_exact"] = audit_pair(
        "B8b minus B4",
        [r["treatment_exact"] for r in rows],
        [r["control_exact"] for r in rows],
    )
    out["paired_valid"] = audit_pair(
        "B8b minus B4 valid",
        [r["treatment_valid"] for r in rows],
        [r["control_valid"] for r in rows],
    )
    out["selector_vs_control"] = audit_pair(
        "selector minus B4",
        [r["selected_exact"] for r in rows],
        [r["control_exact"] for r in rows],
    )
    return out


def run(e):
    renderer = e.module("/experiments/eval/run_eval.py")
    refs = {
        hashlib.sha1(r["prompt"].encode()).hexdigest()[:12]: r
        for r in e.rows("/sft_v3/eval.jsonl")
    }
    control_rows = e.rows("/results_scenarios_b4_qwen35_2b.jsonl")
    treatment_rows = e.rows("/results_scenarios_b8b_stacked_qwen35_2b.jsonl")
    rows, private = [], []
    for a, b in paired(control_rows, treatment_rows):
        ref = refs[a["id"]]
        if ref["family"] != a["family"] or ref["package_or_repo"] != a["package"]:
            raise ValueError("Prompt metadata mismatch")
        if "raw" not in a or "raw" not in b:
            raise ValueError("Missing raw response")
        pa, pb = ("\n".join(renderer.parse_pred("zeta2", r["raw"])) for r in (a, b))
        target = ref["target"].split("\n>>>>>>> UPDATED")[0]
        if any(int(pred == target) != r["exact"] for pred, r in [(pa, a), (pb, b)]):
            raise ValueError("Saved exact score disagrees with raw prediction")
        selected = select(ref["prompt"], pa, pb)
        r = {
            "id": a["id"],
            "family": a["family"],
            "package": a["package"],
            "split": split_group(a["package"]),
            "control_exact": a["exact"],
            "treatment_exact": b["exact"],
            "control_valid": a["valid_pass"],
            "treatment_valid": b["valid_pass"],
            "selected": selected,
            "selected_exact": b["exact"] if selected == "treatment" else a["exact"],
        }
        rows.append(r)
        if a["exact"] != b["exact"] or a["valid_pass"] != b["valid_pass"]:
            private.append(
                r
                | {
                    "prompt": ref["prompt"],
                    "target": target,
                    "control": pa,
                    "treatment": pb,
                    "reasons": [a.get("valid_reason"), b.get("valid_reason")],
                }
            )
    groups = {"all": summary(rows)}
    for family in sorted({r["family"] for r in rows}):
        groups[family] = summary([r for r in rows if r["family"] == family])
    for split in ("development", "evaluation"):
        groups[split] = summary([r for r in rows if r["split"] == split])
        groups["format_" + split] = summary(
            [
                r
                for r in rows
                if r["split"] == split and r["family"] == "format_propagation"
            ]
        )
    noop_rows = []
    for a, b in paired(
        e.rows("/results_noop_fp_b4_qwen35_2b.jsonl"),
        e.rows("/results_noop_fp_b8b_stacked_qwen35_2b.jsonl"),
        fields=(
            "cls",
            "kind",
            "source",
            "fn",
            "expectation",
            "prompt_chars",
            "truncated_prefix",
        ),
    ):
        noop_rows.append(
            {
                "id": a["id"],
                "expectation": a["expectation"],
                "control_proposal": a["proposal"],
                "treatment_proposal": b["proposal"],
            }
        )
    noop = {}
    for name, rs in [
        ("all", noop_rows),
        ("scored", [r for r in noop_rows if r["expectation"] == "no_proposal"]),
    ]:
        noop[name] = {
            "n": len(rs),
            "control_proposals": sum(r["control_proposal"] for r in rs),
            "treatment_proposals": sum(r["treatment_proposal"] for r in rs),
            "discordant": sum(
                r["control_proposal"] != r["treatment_proposal"] for r in rs
            ),
        }
    return {
        "retrospective": True,
        "groups": groups,
        "rows": rows,
        "noop": noop,
        "noop_rows": noop_rows,
    }, private


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--private-out", required=True)
    a = p.parse_args()
    result, private = run(Evidence(a.inventory, a.snapshot))
    dump(a.out, result)
    dump(a.private_out, private)


if __name__ == "__main__":
    main()
