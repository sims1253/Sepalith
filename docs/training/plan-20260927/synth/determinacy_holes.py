"""Known holes of the v2 determinacy proxies, measured on the retained finish_block rows,
and the effect a stricter (v3) novelty would have. Writes a receipt; changes no data.

Holes measured:
- comment-bearing targets, and those whose comment words are absent from the prompt
  (``common.novelty`` ignores target comments, so this prose is unconstrained);
- targets with a string literal that does not occur verbatim in the prompt;
- glue-style strings whose ``{...}`` code uses identifiers absent from the prompt
  (identifiers inside strings are not counted as identifiers);
- non-``file`` packagings whose novel items include a called function name (helpers
  defined elsewhere in the file pass when the novelty budget allows).
v3 novelty = v2 novelty + novel target-comment words + novel identifiers inside strings.

    PYTHONPATH=packages/sepalith/src python docs/training/plan-20260927/synth/determinacy_holes.py \\
        --data /mnt/e/.../SYNTH-v2 --out docs/training/plan-20260927/receipts/SYNTH-v2-determinacy-holes.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from sepalith.data.synth.common import IDENT_RE, STRING_RE, WORD_RE, lexical_items, split_comment
from sepalith.data.synth.finish_block import SHORT_NOVELTY_MAX, SIBLING_NOVELTY_MAX, SIBLING_SUPPORT_MIN

CALL_RE = re.compile(r"([A-Za-z.][A-Za-z0-9._]*)\s*\(")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases_path = args.data / "cases.jsonl"
    rows = [json.loads(line) for line in cases_path.open()]
    finish = [r for r in rows if r["family"] == "finish_block"]
    counts: Counter = Counter()
    flips: Counter = Counter()
    for r in finish:
        context = r["context"]
        prompt = context["prefix"] + context["region_old"]
        target = r["region_new"]
        p_idents, p_words, _ = lexical_items(prompt)
        seen = set(p_words) | {i.lower() for i in p_idents}
        prompt_text = "\n".join(prompt)
        comments = [split_comment(line)[1] for line in target if split_comment(line)[1]]
        comment_words = {w.lower() for c in comments for w in WORD_RE.findall(c)}
        literals = [lit for line in target for lit in STRING_RE.findall(split_comment(line)[0])]
        glue_ids = {i for lit in literals for chunk in re.findall(r"\{([^}]*)\}", lit) for i in IDENT_RE.findall(chunk)}
        calls = {m for line in target for m in CALL_RE.findall(split_comment(line)[0])}
        counts["comment_bearing"] += bool(comments)
        counts["comment_bearing_with_novel_words"] += bool(comment_words - seen)
        counts["non_verbatim_string_literal"] += any(lit not in prompt_text for lit in literals)
        counts["glue_string_identifiers_absent"] += bool(glue_ids - p_idents)
        counts["non_file_variant_novel_call_name"] += (r["meta"]["variant"] != "file"
                                                       and any(c in r["meta"]["novel_tokens"] for c in calls))
        extra = {"cw:" + w for w in comment_words - seen}
        extra |= {"si:" + i for lit in literals for i in IDENT_RE.findall(lit[1:-1])
                  if len(i) >= 2 and i not in p_idents and i.lower() not in seen}
        novel = len(r["meta"]["novel_tokens"]) + len(extra)
        lines, support = r["meta"]["target_lines"], r["meta"]["structural_support"]
        v3 = ("constrained_short" if lines <= 8 and novel <= SHORT_NOVELTY_MAX else
              "sibling" if support >= SIBLING_SUPPORT_MIN and novel <= SIBLING_NOVELTY_MAX else "author_choice")
        flips[f"{r['meta']['determinacy']}->{v3}"] += 1
    receipt = {"schema": "sepalith.synth.determinacy-holes.v1",
               "cases": {"path": str(cases_path), "sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest()},
               "finish_block_rows": len(finish), "holes": dict(counts),
               "v3_novelty_reclassification": dict(flips),
               "v3_would_reject": sum(n for k, n in flips.items() if k.endswith("author_choice")),
               "decision": "not applied in v2 (would change the row set by more than 'a little'); listed as v3 work"}
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
