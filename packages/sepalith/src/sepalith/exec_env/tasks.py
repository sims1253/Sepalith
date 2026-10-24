"""Execution-graded tasks from gated pool functions (finish_block; rename_propagation optional).

    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.exec_env.tasks \\
        --root AUTHOR_ROOT --out OUT/tasks.jsonl [--families finish_block,rename_propagation]

For every function directory under ``--root`` whose ``gate.json`` passed:

- **finish_block**: the synth generator's cut logic (``finish_block.cuts_for``:
  statement starts at any depth, comment leads, nested closes; final-close and
  closer-only remainders skipped) and its case builder and prompt variants
  (``finish_block.build_case``: plain, file, strip, outline, diag; a full-body
  cut takes the signature line as its region with share 0.9), rendered with
  ``build_training_row`` (``zeta2-prm03-v1``), so prompts match TRAIN rows
  exactly. Unlike the synth pipeline, cuts are *not* filtered on determinacy:
  the reward accepts any behaviourally correct completion; determinacy and
  novelty stay as curriculum metadata.
- **rename_propagation**: the TRAIN geometry (a history event renames a local
  variable at its first assignment; the region is the next line using the old
  name; the cursor sits at the end of that line). For execution, every *other*
  later use is renamed as gold, so the region line alone decides behaviour, and
  a binding check requires the old name to be gone from the region.

Per-cut gate (observability and inferability), on top of the function gate:

1. gold passes the full suite;
2. the *truncated* completion (only the closing lines of the remainder) fails
   at least one assertion, so the tests reach the edited lines;
3. cut-local mutants (function mutants whose edit lies inside the remainder)
   are killed at a rate of at least ``CUT_MIN_KILL`` (all of them when fewer
   than ``CUT_MIN_LOCAL`` exist);
4. inferability: no string literal, ``$`` element or argument name that the
   remainder introduces (absent from the prompt) is referenced by the tests;
   otherwise the reward would teach guessing an arbitrary reference choice.

Each task carries what the reward needs (prompt IDs, PRM03 context, exec
splice, suite, expected per-category counts) and curriculum metadata (cut
kind, remainder lines, bucket, variant, determinacy, novelty, local kill rate).
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import re
import tempfile

from sepalith.data.synth import finish_block
from sepalith.data.synth.cases import Case, Event
from sepalith.data.synth.common import rng_for, split_comment
from sepalith.data.synth.structure import describe
from sepalith.exec_env import mutants as mutants_module
from sepalith.exec_env.gate import TIMEOUT, killed, suite_text
from sepalith.exec_env.harness import ExecHarness, ExecJob

TASK_SCHEMA = "sepalith.exec-env.task.v1"
CUT_MIN_KILL = 0.6
CUT_MIN_LOCAL = 3
CUT_MUTANT_LIMIT = 24
PROMPT_MAX = 3072
COMPLETION_MAX = 1024
CLOSER_RE = re.compile(r"^[\s})\],]*$")


def load_tokenizer():  # noqa: ANN201 - transformers type
    from sepalith.data.synth.build import load_tokenizer as synth_tokenizer
    return synth_tokenizer()


def render(case: Case, tokenizer: object, row_id: str) -> dict | None:
    from sepalith.data.synth.verify import render as synth_render
    row, _context, _error = synth_render(case, tokenizer, row_id)  # type: ignore[arg-type]
    return row


def suite_names(suite: str) -> set[str]:
    """Strings, $-elements and argument names the suite references."""
    out = set()
    for literal in re.findall(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'', suite):
        text = literal[0] or literal[1]
        if text:
            out.add(text)
    out |= set(re.findall(r"\$\s*`?([A-Za-z.][\w.]*)", suite))
    out |= set(re.findall(r"[\[(,]\s*([A-Za-z.][\w.]*)\s*=(?!=)", suite))
    return out


def unseen_choices(target: Sequence[str], prompt: Sequence[str], suite: str) -> list[str]:
    """Literals and names introduced by the remainder (absent from the prompt) that the tests reference."""
    from sepalith.data.synth.common import STRING_RE
    prompt_text = "\n".join(prompt)
    referenced = suite_names(suite)
    out = set()
    for line in target:
        code, _comment = split_comment(line)
        for literal in STRING_RE.findall(code):
            inner = literal[1:-1]
            if len(inner) < 2 or inner in prompt_text:
                continue
            # the suite references this new literal, or a piece of it (an error-message pattern, say)
            # that the prompt does not show
            hits = [s for s in referenced if (s == inner or (len(s) >= 4 and (s in inner or _regex_hit(s, inner))))
                    and s not in prompt_text]
            if hits:
                out.add("str:" + inner)
        for name in re.findall(r"(?:\$\s*|[(,]\s*)([A-Za-z.][\w.]*)\s*(?:=(?!=))?", code):
            if len(name) >= 2 and not re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w.])", prompt_text) \
                    and name in referenced and re.search(r"(?<![\w.])" + re.escape(name) + r"\s*=(?!=)|\$\s*" + re.escape(name), code):
                out.add("name:" + name)
    return sorted(out)


def _regex_hit(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text) is not None
    except re.error:
        return False


def truncated_body(target: Sequence[str]) -> list[str]:
    return [line for line in target if CLOSER_RE.match(line) and line.strip()]


def finish_cases(meta: dict, unit_text: str) -> list[tuple[Case, int, int]]:
    """(case, region start, region end) in unit-file coordinates (0-based, end exclusive)."""
    lines = unit_text.rstrip("\n").split("\n")
    with tempfile.TemporaryDirectory(prefix="sepalith-tasks-") as tmp:
        path = Path(tmp) / "unit.R"
        path.write_text(unit_text, encoding="utf-8")
        structure = describe([path])[str(path)]
    if not structure.ok:
        return []
    units = [u for u in structure.units if u.name == meta["name"]]
    if not units:
        return []
    unit = units[-1]
    if not finish_block.eligible(unit, lines):
        return []
    outer = unit.outer_block()
    assert outer is not None
    is_fun = unit.main_function() is not None and outer.open_line >= unit.start
    cuts = finish_block.cuts_for(unit, lines, structure)
    first_line = min(c.line for c in cuts)
    rng = rng_for("exec-env-fb", meta["id"])
    weights = {k: w for k, w in finish_block.VARIANT_WEIGHTS.items()
               if (k != "strip" or unit.roxygen) and (k not in ("outline", "diag") or is_fun)}
    package_id, rel_path = f"exec:{meta['package']}", meta["path"]
    out = []
    for cut in cuts:
        target = lines[cut.line - 1:unit.end]
        if cut.kind == "final_close" or finish_block.closers_only(target):
            continue
        variant = rng.choices(list(weights), weights=list(weights.values()))[0]
        signature = cut.line == first_line and rng.random() < finish_block.SIGNATURE_REGION_SHARE
        case = finish_block.build_case(unit, cut, variant, lines, structure, package_id, rel_path,
                                       signature_region=signature)
        case.meta["full_body"] = cut.line == first_line
        start = cut.line - 2 if signature else cut.line - 1
        out.append((case, start, unit.end))
    return out


RENAME_STYLES = ("{}2", "{}_new", "new_{}", "{}_val")


def rename_cases(meta: dict, unit_text: str, limit: int = 2) -> list[tuple[Case, int, int, dict]]:
    """TRAIN-geometry rename_propagation cases; returns (case, region start, region end, exec info)."""
    lines = unit_text.rstrip("\n").split("\n")
    with tempfile.TemporaryDirectory(prefix="sepalith-tasks-") as tmp:
        path = Path(tmp) / "unit.R"
        path.write_text(unit_text, encoding="utf-8")
        structure = describe([path])[str(path)]
    units = [u for u in structure.units if u.name == meta["name"]] if structure.ok else []
    if not units:
        return []
    unit = units[-1]
    formals = set(meta.get("formals") or [])
    rng = rng_for("exec-env-rename", meta["id"])
    # a use of the variable: not part of a longer name, not a call, not `$x`/`@x`, not an argument name `x =`
    word = r"(?<![A-Za-z0-9._$@]){}(?![A-Za-z0-9._(])(?!\s*=(?!=))"
    found = []
    seen: set[str] = set()
    for index in range(unit.start, unit.end - 1):  # 0-based line index inside the function body
        code, _ = split_comment(lines[index])
        match = re.match(r"^(\s*)([A-Za-z.][A-Za-z0-9._]*)\s*(<-|=)(?!=)", code)
        if not match or match.group(2) in formals or match.group(2) in seen or len(match.group(2)) < 2:
            continue
        old = match.group(2)
        seen.add(old)
        pattern = re.compile(word.format(re.escape(old)))
        uses = [j for j in range(index + 1, unit.end) if pattern.search(split_comment(lines[j])[0])
                and '"' not in lines[j] and "'" not in lines[j]]
        if len(uses) < 2 or '"' in lines[index] or "'" in lines[index]:
            continue
        new = rng.choice(RENAME_STYLES).format(old)
        if re.search(word.format(re.escape(new)), unit_text):
            continue
        found.append((index, old, new, uses))
    rng.shuffle(found)
    out = []
    package_id, rel_path = f"exec:{meta['package']}", meta["path"]
    for index, old, new, uses in found[:limit]:
        pattern = re.compile(word.format(re.escape(old)))
        renamed_def = re.sub(r"^(\s*)" + re.escape(old), lambda m: m.group(1) + new, lines[index], count=1)
        before = (*lines,)
        after_event = list(lines)
        after_event[index] = renamed_def
        region = uses[0]
        target_line = pattern.sub(new, lines[region])
        event = Event(rel_path, before, index, (lines[index],), (renamed_def,))
        start = (unit.roxygen[0] if unit.roxygen else unit.start) - 1
        doc = tuple(after_event)
        case = Case("rename_propagation", package_id, rel_path, doc, region, region + 1, (target_line,),
                    window_start=start, cursor=(0, len(lines[region])), events=(event,),
                    meta={"rule": "rename_local", "old": old, "new": new, "def_line": index + 1,
                          "region_line": region + 1, "later_uses": len(uses) - 1, "target_lines": 1,
                          "bucket": "short", "variant": "history"})
        exec_lines = list(after_event)
        for j in uses[1:]:
            exec_lines[j] = pattern.sub(new, exec_lines[j])
        out.append((case, region, region + 1, {"exec_lines": exec_lines, "old": old, "new": new}))
    return out


def task_record(meta: dict, case: Case, row: dict, exec_before: list[str], exec_after: list[str], suite: str,
                gate: dict, extra: dict) -> dict:
    expected = dict(gate["categories"])
    tid = hashlib.sha256(f"exec-env:{case.key()}".encode()).hexdigest()[:24]
    target_text = "\n".join(case.target)
    return {"schema": TASK_SCHEMA, "id": tid, "function_id": meta["id"], "package": meta["package"],
            "family": case.family, "prompt_ids": row["input_ids"][:row["target_start"]],
            "gold_completion_ids": row["input_ids"][row["target_start"]:],
            "context": case.to_context(), "target_operation": "replace", "target_body_text": target_text,
            "exec": {"before": "\n".join(exec_before), "after": "\n".join(exec_after),
                     "expect": [*meta["helpers"], meta["name"]], "timeout": TIMEOUT,
                     "gold_text": "\n".join([*exec_before, *case.target, *exec_after]),
                     **({"forbid_in_region": extra["forbid_in_region"]} if "forbid_in_region" in extra else {})},
            "suite": suite, "suite_sha256": hashlib.sha256(suite.encode()).hexdigest(),
            "expected_categories": expected, "expected_checks": int(gate["checks"]),
            "curriculum": {k: case.meta.get(k) for k in ("rule", "variant", "cut_line", "bucket", "target_lines",
                                                        "signature_region", "full_body", "determinacy", "novelty",
                                                        "structural_support") if k in case.meta}
                          | {"prompt_tokens": row["prompt_token_count"],
                             "completion_tokens": row["target_token_count"] + 1} | extra.get("curriculum", {})}


def build_for_function(directory: Path, harness: ExecHarness, tokenizer: object, families: set[str]) -> tuple[list[dict], Counter]:
    reasons: Counter = Counter()
    gate = json.loads((directory / "gate.json").read_text())
    if gate.get("verdict") != "pass":
        reasons["function_gate_not_passed"] += 1
        return [], reasons
    meta = json.loads((directory / "meta.json").read_text())
    unit_text = (directory / "source.R").read_text(encoding="utf-8")
    unit_lines = unit_text.rstrip("\n").split("\n")
    suite = suite_text(directory)
    expect = (*meta["helpers"], meta["name"])
    expected = int(gate["checks"])
    tokens = mutants_module.parse_tokens([unit_text])[0]
    assert tokens is not None
    all_mutants = mutants_module.generate(unit_text, tokens)
    candidates: list[tuple[Case, list[str], list[str], dict]] = []
    if "finish_block" in families:
        for case, start, end in finish_cases(meta, unit_text):
            before, after = unit_lines[:start], unit_lines[end:]
            if [*before, *case.target, *after] != unit_lines:
                reasons["splice_mismatch"] += 1
                continue
            candidates.append((case, before, after, {"region_unit": (start, end)}))
    if "rename_propagation" in families:
        for case, start, end, info in rename_cases(meta, unit_text):
            lines = info["exec_lines"]
            candidates.append((case, lines[:start], lines[end:], {"region_unit": (start, end),
                                                                   "forbid_in_region": info["old"],
                                                                   "curriculum": {"old": info["old"], "new": info["new"]}}))
    tasks = []
    for case, before, after, extra in candidates:
        reasons["cuts_seen"] += 1
        row = render(case, tokenizer, case.row_id())
        if row is None:
            reasons["render_failed"] += 1
            continue
        if row["prompt_token_count"] > PROMPT_MAX or row["target_token_count"] + 1 > COMPLETION_MAX:
            reasons["length_policy"] += 1
            continue
        prompt_lines = list(case.doc[case.window_start:])
        unseen = unseen_choices(case.target, prompt_lines, suite) if case.family == "finish_block" else []
        if unseen:
            reasons["tests_require_unseen_choice"] += 1
            continue
        start, end = extra["region_unit"]
        gold_src = "\n".join([*before, *case.target, *after]) + "\n"
        jobs = [ExecJob(gold_src, suite, expect, TIMEOUT, "gold")]
        if case.family == "finish_block":
            trunc = truncated_body(case.target)
            jobs.append(ExecJob("\n".join([*before, *trunc, *after]) + "\n", suite, expect, TIMEOUT, "truncated"))
            local = [m for m in all_mutants if m.line1 - 1 >= start + (1 if case.meta.get("signature_region") else 0)
                     and m.line2 <= end]
        else:
            # rename: a wrong region line is any line that keeps the old name or uses an unrelated one
            local = []
            jobs.append(ExecJob("\n".join([*before, case.region_old[0], *after]) + "\n", suite, expect, TIMEOUT, "truncated"))
        local = mutants_module.sample(local, CUT_MUTANT_LIMIT, case.key())
        # Local mutants are edits of the original unit; the remainder lies at the same lines, so
        # they apply to the gold splice unchanged for finish_block.
        jobs += [ExecJob(m.text, suite, expect, TIMEOUT, m.id) for m in local]
        results = harness.run_many(jobs)
        gold, trunc_result, mutant_results = results[0], results[1], results[2:]
        if killed(gold, expected):
            reasons["gold_fails"] += 1
            continue
        truncated_parses = bool(trunc_result.parse_ok)
        if truncated_parses and not killed(trunc_result, expected):
            reasons["truncated_passes_unobservable"] += 1
            continue
        valid = [r for r in mutant_results if r.parse_ok and r.structure_ok and r.sourced_ok is not False]
        dead = sum(killed(r, expected) for r in valid)
        rate = dead / len(valid) if valid else None
        if case.family == "finish_block":
            if len(valid) >= CUT_MIN_LOCAL and rate is not None and rate < CUT_MIN_KILL:
                reasons["cut_local_kill_rate"] += 1
                continue
            if 0 < len(valid) < CUT_MIN_LOCAL and dead < len(valid):
                reasons["cut_local_survivor"] += 1
                continue
        extra.setdefault("curriculum", {}).update({"local_mutants": len(valid), "local_killed": dead,
                                                    "truncated_parses": truncated_parses})
        tasks.append(task_record(meta, case, row, before, after, suite, gate, extra))
        reasons["tasks"] += 1
        reasons[f"tasks_{case.family}"] += 1
    return tasks, reasons


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", default="finish_block,rename_propagation")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    families = set(args.families.split(","))
    tokenizer = load_tokenizer()
    dirs = sorted(p.parent for p in args.root.rglob("gate.json"))
    total: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ExecHarness(workers=min(args.workers, 8)) as harness, args.out.open("w") as stream:
        for directory in dirs:
            tasks, reasons = build_for_function(directory, harness, tokenizer, families)
            total.update(reasons)
            for task in tasks:
                stream.write(json.dumps(task) + "\n")
    print(json.dumps(dict(sorted(total.items())), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
