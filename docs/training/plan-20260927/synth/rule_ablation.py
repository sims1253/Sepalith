"""How many retained SYNTH-v3 finish_block rows each generator-3.1 rule removes.

Re-assesses every retained finish row of a build (default SYNTH-v3) with the
3.1 ``common.assess`` and the 3.1 class rules, on the row's own prompt, and
attributes removals to rules. A row can trip several rules; ``rows`` counts
rows with at least one hit of the rule, ``only`` rows removed by that rule
alone. Test-file rows are listed separately (v3.1 excludes test files).

    PYTHONPATH=packages/sepalith/src python docs/training/plan-20260927/synth/rule_ablation.py \\
        --data /mnt/e/.../SYNTH-v3 --out docs/training/plan-20260927/receipts/SYNTH-v3.1-rule-ablation.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from sepalith.data.synth.common import STRING_RE, WORD_RE, assess, lexical_items, split_comment
from sepalith.data.synth.finish_block import SHORT_NOVELTY_MAX, SIBLING_NOVELTY_MAX, SIBLING_SUPPORT_MIN
from sepalith.data.synth.structure import describe

V3_IDENT_RE = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")  # generator 3.0 regex, to reproduce the v3 recount (31)


def package_functions(sources: Path) -> dict[str, set[str]]:
    files = sorted(sources.rglob("*.R"))
    structures = describe(files)
    out: dict[str, set[str]] = defaultdict(set)
    for path in files:
        parts = path.relative_to(sources).parts
        structure = structures[str(path)]
        if structure.ok:
            out[f"synth:{parts[1]}"].update(u.name for u in structure.units if u.kind == "assign_function" and u.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--strict-recount", type=Path, default=None,
                        help="optional determinacy-holes receipt for the v3 strict recount (for the overlap figure)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases_path = args.data / "cases.jsonl"
    rows = [json.loads(line) for line in cases_path.open(encoding="utf-8")]
    finish = [r for r in rows if r["family"] == "finish_block"]
    functions = package_functions(args.data / "sources")
    rule_rows: Counter = Counter()
    only: Counter = Counter()
    removed_ids: list[str] = []
    test_ids = [r["row_id"] for r in finish if r["meta"].get("test_file")]
    strict_ids = set()  # the v3 strict recount (determinacy_holes.py, "v3_would_reject": 31 on SYNTH-v3)
    for r in finish:
        prompt = r["context"]["prefix"] + r["context"]["region_old"]
        p_idents, p_words, _ = lexical_items(prompt)
        seen = set(p_words) | {i.lower() for i in p_idents}
        target = r["region_new"]
        comment_words = {w.lower() for line in target for w in WORD_RE.findall(split_comment(line)[1])}
        literals = [lit for line in target for lit in STRING_RE.findall(split_comment(line)[0])]
        extra = {"cw:" + w for w in comment_words - seen}
        extra |= {"si:" + i for lit in literals for i in V3_IDENT_RE.findall(lit[1:-1])
                  if len(i) >= 2 and i not in p_idents and i.lower() not in seen}
        novel = len(r["meta"]["novel_tokens"]) + len(extra)
        lines, support = r["meta"]["target_lines"], r["meta"]["structural_support"]
        if not ((lines <= 8 and novel <= SHORT_NOVELTY_MAX)
                or (support >= SIBLING_SUPPORT_MIN and novel <= SIBLING_NOVELTY_MAX)):
            strict_ids.add(r["row_id"])
    for r in finish:
        context = r["context"]
        prompt = context["prefix"] + context["region_old"]
        target = r["region_new"]
        novel, hard = assess(target, prompt, functions[r["package_id"]])
        meta = r["meta"]
        lines, support = meta["target_lines"], meta["structural_support"]
        reasons = {h.split(":", 1)[0] for h in hard}
        budget_ok = (lines <= 8 and len(novel) <= SHORT_NOVELTY_MAX) or (
            support >= SIBLING_SUPPORT_MIN and len(novel) <= SIBLING_NOVELTY_MAX)
        if not budget_ok:
            kinds = {n.split(":", 1)[0] for n in novel if ":" in n} & {"elt", "default"}
            reasons.add("novelty_budget" + ("(elt/default)" if kinds else ""))
        for reason in reasons:
            rule_rows[reason] += 1
        if len(reasons) == 1:
            only[next(iter(reasons))] += 1
        if reasons:
            removed_ids.append(r["row_id"])
    receipt = {
        "schema": "sepalith.synth.rule-ablation.v1",
        "cases": {"path": str(cases_path), "sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest()},
        "finish_block_rows": len(finish),
        "rows_removed_by_3_1_rules": len(removed_ids),
        "rows_per_rule": dict(rule_rows.most_common()),
        "rows_removed_by_this_rule_only": dict(only.most_common()),
        "strict_recount_rows": len(strict_ids),
        "strict_recount_rows_removed_by_3_1_rules": len(strict_ids & set(removed_ids)),
        "test_file_rows": len(test_ids),
        "test_file_rows_not_otherwise_removed": len(set(test_ids) - set(removed_ids)),
        "strict_recount_rows_not_removed": sorted(strict_ids - set(removed_ids)),
        "strict_recount_kept_justification": {
            "027e83fc8c4dcb7f4ccd80c6": "at_skew/gc_skew are visible as quoted choices in the prompt (`stats = c(..., \"gc_skew\", \"at_skew\", ...)`, also in @param); the recount counts each name twice (code identifier + string identifier). 3.1 novelty: at_skew, gc_skew, log2 = 3 <= sibling budget, support 0.54",
            "ca4d7aef7cc18938b4f64f8e": "goals_for/goals_against are visible as quoted strings in the prompt (`stats <- c(\"points\", \"won\", \"goals_for\", \"lost\", \"goals_against\")`); same double count. 3.1 novelty: 3 <= sibling budget, support 0.78",
            "039c9eec5d149b67457c7d27": "recount artefact: the sentence-final word `invalid.` was read as an identifier; the prompt shows the same message template (`Cursor field `%s` is invalid.`) twice",
            "0634dc9e9405040e40759722": "recount artefact: `once.` and `value.` are sentence-final words; the prompt shows the message template (`can only be given once.`) and `value` as a formal",
            "20ba943ca8ab31a01e0bd166": "recount artefact: `whitespace.` is a sentence-final word shown in the roxygen title (`Find lines with trailing whitespace`)",
            "3363132a42ada2d87e12848c": "recount artefact: `tallyhook.` is the option prefix the roxygen spells out (`Prefixes each name with `tallyhook.``)",
        },
        "note": "rows can trip several rules; v3.1 also excludes test-file finish_block cuts entirely",
    }
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "cases"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
