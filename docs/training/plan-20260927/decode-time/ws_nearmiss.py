#!/usr/bin/env python3
"""Whitespace-only near-misses of arm A 536 and decode-time canonicalisation (measurement only).

CPU only. Nothing here changes the official scorer (``campaign_eval.classify``); the
baseline counts are reproduced first and every variant is scored alongside them.

Inputs
- TRAIN: the C12 census of arm A (8 samples per prompt at temperature 1.0, 15,006 TRAIN
  prompts). Gold bodies come from the census target ids, contexts from the DAT10 v4
  context sidecar. Exactness is the evaluator's raw region rule (body == region_new,
  no-op through ``parse_output``), not the census's air-canonical ``exact``.
- DEV: arm A 536's gate outputs on DEV250 and DEV75 (greedy, cap 1,024) and the panels.

Order: the variants below are fixed in code; the script processes TRAIN first, writes
the TRAIN section, then applies the same variants to DEV. No variant is chosen on DEV.

Variants (applied to the PREDICTION ONLY, from prompt-side information only):
- L0 extension-literal: drop leading and trailing whitespace-only body lines, as
  ``parsePrediction`` in extensions/vscode-sepalith/src/extension.ts already does.
- L1 drop trailing whitespace-only body lines only.
- L2 L1 plus strip trailing whitespace on every line.
- L2s L2 plus a product filter: a prediction equal to region_old up to whitespace is not
  shown (counted as a predicted no-op).
- L3 L2 plus re-indent to the document's indentation (unit and tabs/spaces detected
  from the prompt's prefix, region_old, scope and suffix lines, like VS Code's
  ``editor.detectIndentation``); the unit is rescaled relative to the block's first line.
- L3a L3 plus anchoring the block's first line to region_old's first-line indent when
  the first line re-emits region_old's first line.
Metric gaming (reported, not a product behaviour, because it uses the gold or treats
whitespace as irrelevant on both sides):
- G1 whitespace-insensitive comparison on both sides (strip lines, collapse runs, drop
  blank lines).
- G2 re-indent each line with the gold's indentation (after L2).
- G3 air formatter on both sides (window surface; formatting-only references compared
  raw, as reward v2.1 does). TRAIN reads the census's own air ``exact``.

A predicted body equal to region_old after canonicalisation counts as a predicted
no-op, as ``parse_output`` does. A body that canonicalises to nothing counts as a
no-op too (the extension shows nothing).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
RECIPE = PLAN / "c04/bound/sft-v2-lr1e-5.json"
GATES = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates/SFT-v2-r4-lr1e-5/results")
DEV_RESULTS = {
    "dev75": (GATES / "dev-dev75-step-536.20261014T021133-673743425.json",
              "c6e77e49f8fc6323244b576fe2d0ee730c7c524c844490872ca0b0f0c3796e67"),
    "dev250": (GATES / "dev-dev250-step-536.20261014T021133-673743425.json",
               "0a1dbbd903dedc4ec392b2848f8b2ea89641e5228a466ca3a79fe1c4936ab9d3"),
}
OFFICIAL = {"dev250": {"edit_exact": 79, "strict_noop_correct": 38, "protocol_valid": 234,
                       "strict_noop_false_suggestions": 12},
            "dev75": {"edit_exact": 27, "strict_noop_false_suggestions": 8, "protocol_valid": 74}}
CENSUS = Path("/home/m0hawk/.local/state/sepalith/c12/census-arm-a")
SIDECAR = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/context-sidecar.jsonl")
SIDECAR_SHA256 = "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a"
FORBIDDEN = ("DAT-07-final", "sealed-final", "DAT-08")
VARIANTS = ("raw", "L0", "L1", "L2", "L2s", "L3", "L3a", "G1", "G2", "G3")
LEGIT = ("L0", "L1", "L2", "L2s", "L3", "L3a")
STOPS = (1, 130073)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def guard(path):
    if any(token in str(path) for token in FORBIDDEN):
        raise SystemExit(f"refusing a sealed/final input: {path}")
    return Path(path)


# Whitespace classification ------------------------------------------------------------------------

def blank(line):
    return line.strip() == ""


def strip_outer(lines):
    lines = list(lines)
    while lines and blank(lines[0]):
        lines.pop(0)
    while lines and blank(lines[-1]):
        lines.pop()
    return lines


def ws_key(lines):
    """Whitespace-insensitive form: strip, collapse runs, drop blank lines."""
    return tuple(re.sub(r"\s+", " ", line.strip()) for line in lines if not blank(line))


def nows(lines):
    return re.sub(r"\s", "", "".join(lines))


def leading(line):
    return line[:len(line) - len(line.lstrip())]


def width(ws, tab=4):
    return sum(tab if ch == "\t" else 1 for ch in ws)


def ws_types(pred, gold):
    """Whitespace-only difference types between two different bodies (None if not ws-only)."""
    pred, gold = tuple(pred), tuple(gold)
    if pred == gold or nows(pred) != nows(gold):
        return None
    types = set()
    if any("\r" in line for line in pred + gold):
        types.add("line_endings")

    def outer(lines):
        lead = 0
        while lead < len(lines) and blank(lines[lead]):
            lead += 1
        trail = 0
        while trail < len(lines) - lead and blank(lines[len(lines) - 1 - trail]):
            trail += 1
        return lead, trail

    (pl, pt), (gl, gt) = outer(pred), outer(gold)
    if pl != gl:
        types.add("leading_blank_lines")
    if pt != gt:
        types.add("trailing_blank_lines")
    p1, g1 = strip_outer(pred), strip_outer(gold)
    pn, gn = [x for x in p1 if not blank(x)], [x for x in g1 if not blank(x)]
    if len(pn) != len(gn):
        types.add("line_wrapping")
        return types
    if [blank(x) for x in p1] != [blank(x) for x in g1]:
        types.add("interior_blank_lines")
    else:
        for p, g in zip(p1, g1, strict=False):
            if blank(p) and p != g:
                types.add("trailing_whitespace")  # whitespace on an otherwise blank line
    indent_pairs = []
    for p, g in zip(pn, gn, strict=False):
        if p == g:
            indent_pairs.append((leading(p), leading(g)))
            continue
        if p[len(p.rstrip()):] != g[len(g.rstrip()):]:
            types.add("trailing_whitespace")
        if p.strip() != g.strip():
            types.add("intra_line_whitespace")
        indent_pairs.append((leading(p), leading(g)))
    if any(a != b for a, b in indent_pairs):
        types.add("indentation")
        if any(("\t" in a) != ("\t" in b) for a, b in indent_pairs):
            types.add("indent:tabs_vs_spaces")
        diffs = {width(b) - width(a) for a, b in indent_pairs}
        nonzero = [(width(a), width(b)) for a, b in indent_pairs if width(a) or width(b)]
        if len(diffs) == 1:
            types.add("indent:uniform_offset")
        elif nonzero and all(wa and wb for wa, wb in nonzero) and len({wb / wa for wa, wb in nonzero}) == 1:
            types.add("indent:scale")
        else:
            types.add("indent:partial")
    return types


def primary(types):
    """One label per case for the summary table."""
    base = sorted(t for t in types if not t.startswith("indent:"))
    return "+".join(base) if base else "none"


# Canonicalisation (prediction only) ---------------------------------------------------------------

def detect_indent(lines):
    """VS Code-like detection: tabs or spaces, and the most common positive indent step."""
    tabs = spaces = 0
    steps = Counter()
    previous = 0
    for line in lines:
        if blank(line):
            continue
        lead = leading(line)
        if lead.startswith("\t"):
            tabs += 1
        elif lead.startswith(" "):
            spaces += 1
        w = len(lead.replace("\t", "    "))
        if w > previous and not lead.startswith("\t"):
            steps[w - previous] += 1
        previous = w
    use_tabs = tabs > spaces
    unit = None
    for step, _ in steps.most_common():
        if step in (2, 4, 8, 3):
            unit = step
            break
    return use_tabs, unit


def body_unit(lines):
    """Most common positive indent step between consecutive body lines (not from column 0)."""
    steps = Counter()
    previous = None
    for line in lines:
        if blank(line):
            continue
        w = len(leading(line).replace("\t", "    "))
        if previous is not None and w > previous:
            steps[w - previous] += 1
        previous = w
    for step, _count in steps.most_common():
        if step in (2, 4, 8, 3):
            return step, sum(steps.values())
    return None, 0


def reindent(body, context, anchored=True):
    """Re-indent ``body`` to the document's style (context only)."""
    ctx_lines = list(context.prefix) + list(context.scope_lines) + list(context.region_old) + list(context.suffix_lines)
    doc_tabs, doc_unit = detect_indent(ctx_lines)
    if doc_unit is None and not doc_tabs:
        return list(body)
    out = list(body)
    # tabs <-> spaces per the document
    if not doc_tabs and doc_unit:
        out = [(" " * doc_unit) * (len(leading(x)) - len(leading(x).lstrip("\t"))) + x.lstrip("\t")
               if leading(x).startswith("\t") else x for x in out]
    elif doc_tabs:
        pred_unit, _ = body_unit(out)
        if pred_unit:
            out = ["\t" * (len(leading(x)) // pred_unit) + " " * (len(leading(x)) % pred_unit) + x.lstrip(" ")
                   if leading(x).startswith(" ") else x for x in out]
        return anchor(out, context) if anchored else out
    # rescale the unit, relative to the block's base indent (the first line's indent is nesting, not a unit)
    pred_unit, steps = body_unit(out)
    nonblank = [x for x in out if not blank(x)]
    if nonblank and pred_unit and doc_unit and pred_unit != doc_unit and steps >= 2:
        base = len(leading(nonblank[0]))
        rel = [len(leading(x)) - base for x in nonblank]
        if all(r >= 0 and r % pred_unit == 0 for r in rel) and not any("\t" in leading(x) for x in nonblank):
            out = [x if blank(x) else " " * (base + (len(leading(x)) - base) // pred_unit * doc_unit) + x.lstrip(" ")
                   for x in out]
    return anchor(out, context) if anchored else out


def anchor(out, context):
    """Shift the block so its first line keeps region_old's first-line indent, when it re-emits that line."""
    old = [x for x in context.region_old if not blank(x)]
    first = next((i for i, x in enumerate(out) if not blank(x)), None)
    if not old or first is None or out[first].strip() != old[0].strip():
        return out
    shift = len(leading(old[0])) - len(leading(out[first]))
    if shift == 0 or "\t" in leading(old[0]) or any("\t" in leading(x) for x in out):
        return out
    if shift < 0 and any(not blank(x) and len(leading(x)) < -shift for x in out):
        return out
    return [x if blank(x) else (" " * shift + x if shift > 0 else x[-shift:]) for x in out]


def drop_trailing_blank(body):
    """Drop trailing whitespace-only lines (the 'final newline' of the PRM03 body)."""
    body = list(body)
    while body and blank(body[-1]):
        body.pop()
    return body


def canonical(variant, body, context):
    body = list(body)
    if variant == "raw":
        return body
    if variant == "L0":
        return strip_outer(body)
    body = drop_trailing_blank(body)
    if variant == "L1":
        return body
    body = [x.rstrip() for x in body]
    if variant == "L2":
        return body
    if variant == "L3":
        return reindent(body, context, anchored=False)
    if variant == "L3a":
        return reindent(body, context, anchored=True)
    raise ValueError(variant)


# Scoring one case under one variant ---------------------------------------------------------------

class Case:
    __slots__ = ("id", "family", "context", "gold", "expected_noop", "valid", "pred_noop", "body", "air_exact", "sample")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def outcome(case, variant, air=None):
    """Return (predicted_noop, correct) under a variant."""
    if not case.valid:
        return False, False
    old = tuple(case.context.region_old)
    gold = tuple(case.gold)
    if case.pred_noop:
        return True, case.expected_noop
    body = tuple(case.body)
    if variant in ("raw", *LEGIT):
        canon = tuple(canonical("L2" if variant == "L2s" else variant, body, case.context))
        noop = canon == old or (not canon and bool(body))
        if variant == "L2s" and old and ws_key(canon) == ws_key(old):
            noop = True  # product filter: suppress a suggestion that only changes whitespace
        if noop:
            return True, case.expected_noop
        return False, (not case.expected_noop) and canon == gold
    if variant == "G1":
        if ws_key(body) == ws_key(old):
            return True, case.expected_noop
        return False, (not case.expected_noop) and ws_key(body) == ws_key(gold)
    if variant == "G2":
        canon = [x.rstrip() for x in strip_outer(body)]
        g = list(gold)
        if len(canon) == len(g) and all(a.strip() == b.strip() for a, b in zip(canon, g, strict=True)):
            canon = list(g)
        canon = tuple(canon)
        if canon == old:
            return True, case.expected_noop
        return False, (not case.expected_noop) and canon == gold
    if variant == "G3":
        verdict = case.air_exact if air is None else air(case)
        if verdict == "noop":
            return True, case.expected_noop
        return False, (not case.expected_noop) and bool(verdict)
    raise ValueError(variant)


def tally(cases, variant, air=None):
    c = Counter()
    fam = defaultdict(Counter)
    for case in cases:
        noop, correct = outcome(case, variant, air)
        c["valid"] += case.valid
        if case.expected_noop:
            c["noop_cases"] += 1
            c["noop_correct"] += correct
            c["nofp"] += case.valid and not noop
        else:
            c["edit_cases"] += 1
            c["edit_exact"] += correct
            c["edit_missed_as_noop"] += case.valid and noop
        fam[case.family]["n"] += 1
        fam[case.family]["correct"] += correct
    c["score"] = c["edit_exact"] + c["noop_correct"]
    return dict(c), {k: dict(v) for k, v in sorted(fam.items())}


def flips(cases, variant, air=None):
    gained, lost = [], []
    for case in cases:
        before = outcome(case, "raw")[1]
        after = outcome(case, variant, air)[1]
        if before != after:
            types = None
            if case.valid and not case.pred_noop:
                ref = case.context.region_old if case.expected_noop else case.gold
                types = ws_types(case.body, ref)
            item = {"id": case.id, "family": case.family, "expected_noop": case.expected_noop,
                    "types": sorted(types) if types else None}
            if case.sample is not None:
                item["sample"] = case.sample
            (gained if after else lost).append(item)
    return gained, lost


def classify_misses(cases):
    """Per family and type: ws-only misses among valid wrong outputs."""
    per_type = Counter()
    per_primary = Counter()
    per_family = defaultdict(Counter)
    kinds = Counter()
    for case in cases:
        if not case.valid or case.pred_noop:
            continue
        body = tuple(case.body)
        old = tuple(case.context.region_old)
        if case.expected_noop:
            if body == old:
                continue
            types = ws_types(body, old)
            kind = "noop_case_ws_only_suggestion"
        else:
            if body == tuple(case.gold):
                continue
            types = ws_types(body, case.gold)
            kind = "edit_case_ws_only_vs_gold"
            if types is None and old and ws_types(body, old) is not None:
                kinds["edit_case_pred_equals_old_mod_ws"] += 1
                per_family[case.family]["pred_equals_old_mod_ws"] += 1
        if types is None:
            continue
        kinds[kind] += 1
        per_family[case.family][kind] += 1
        per_primary[primary(types)] += 1
        for t in types:
            per_type[t] += 1
    return {"kinds": dict(kinds), "per_type": dict(per_type.most_common()),
            "per_combination": dict(per_primary.most_common()),
            "per_family": {k: dict(v) for k, v in sorted(per_family.items())}}


# Inputs ------------------------------------------------------------------------------------------

def load_tokenizer(recipe):
    from tokenizers import Tokenizer

    from sepalith.campaign_protocol import TOKENIZER_JSON_SHA256

    path = Path(recipe["parent"]["path"]) / "tokenizer.json"
    if sha256(path) != TOKENIZER_JSON_SHA256:
        raise SystemExit("parent tokenizer.json is not the pinned tokenizer")
    tok = Tokenizer.from_file(str(path))
    return tok, str(path)


def parse_ids(tok, ids, context):
    """The evaluator's path: drop the final stop id, decode, parse, check tokens."""

    complete = bool(ids) and ids[-1] in STOPS
    raw = tok.decode(list(ids[:-1] if complete else ids), skip_special_tokens=False)
    return parse_raw(raw, ids, context)


def parse_raw(raw, ids, context):
    from sepalith.campaign_protocol import parse_output, valid_generation_tokens

    result = parse_output(raw, context)
    valid = valid_generation_tokens(list(ids)) and result.status == "accepted"
    noop = valid and result.operation == "no_op"
    return valid, noop, tuple(result.body)


def load_train(tok, limit=None):
    from sepalith.campaign_protocol import PromptContext

    guard(SIDECAR)
    summary = json.loads((CENSUS / "census-summary.json").read_text())
    if summary["data"]["context"]["sha256"] != SIDECAR_SHA256:
        raise SystemExit("census context sidecar identity differs")
    files = {name: summary["files"][name] for name in ("census-prompts.jsonl", "samples.jsonl", "scores.jsonl")}
    for name, record in files.items():
        if (CENSUS / name).stat().st_size != record["bytes"]:
            raise SystemExit(f"{name} size differs from the census summary")
    prompts = {}
    with (CENSUS / "census-prompts.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "train":
                raise SystemExit("census prompt is not TRAIN")
            prompts[row["id"]] = (row["family"], row["target_ids"])
            if limit and len(prompts) >= limit:
                break
    contexts = {}
    with SIDECAR.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["row_id"] in prompts:
                if row["split"] != "train":
                    raise SystemExit("sidecar row is not TRAIN")
                contexts[row["row_id"]] = PromptContext.from_mapping(row["context"])
    if set(contexts) != set(prompts):
        raise SystemExit("contexts missing for census prompts")
    golds = {}
    for rid, (family, target) in prompts.items():
        ctx = contexts[rid]
        valid, noop, body = parse_ids(tok, target, ctx)
        if not valid:
            raise SystemExit(f"gold target does not parse: {rid}")
        golds[rid] = (family, tuple(ctx.region_old) if noop else body, noop)
    air = {}
    with (CENSUS / "scores.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in prompts:
                air[row["sample_id"]] = (row["exact"], row["outcome"], row.get("similarity"), row.get("similarity_space"),
                                         row.get("exact_raw"))
    cases = []
    with (CENSUS / "samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            rid = row["id"]
            if rid not in prompts:
                continue
            family, gold, expected_noop = golds[rid]
            ctx = contexts[rid]
            valid, noop, body = parse_ids(tok, row["generated_ids"], ctx)
            sid = f"{rid}:{row['sample']}"
            exact, out, sim, space, exact_raw = air[sid]
            air_verdict = "noop" if out == "correct_no_edit" or (expected_noop and out == "exact") else bool(exact)
            cases.append(Case(id=rid, family=family, context=ctx, gold=gold, expected_noop=expected_noop,
                              valid=valid, pred_noop=noop, body=body, air_exact=air_verdict, sample=row["sample"]))
    nearmiss_canonical_equal = Counter()
    for sid, (_exact, out, sim, space, _) in air.items():
        if out == "near_miss" and sim == 1.0 and space and space.startswith("canonical"):
            nearmiss_canonical_equal[golds[sid.rsplit(":", 1)[0]][0]] += 1
    identity = {name: {"path": str(CENSUS / name), **files[name]} for name in files}
    identity["census-summary.json"] = {"path": str(CENSUS / "census-summary.json"), "sha256": sha256(CENSUS / "census-summary.json")}
    identity["context_sidecar"] = {"path": str(SIDECAR), "sha256": SIDECAR_SHA256,
                                   "note": "identity taken from census-summary.json; file read in full"}
    return cases, identity, air, nearmiss_canonical_equal


def load_dev(recipe, name):
    from sepalith.campaign_protocol import PromptContext

    panel = recipe["dev_gate"]["panels"][name]
    guard(panel["path"])
    if sha256(panel["path"]) != panel["sha256"]:
        raise SystemExit(f"{name} panel sha256 differs")
    path, digest = DEV_RESULTS[name]
    if sha256(path) != digest:
        raise SystemExit(f"{name} result sha256 differs")
    result = json.loads(path.read_text())
    cases_file = Path(result["cases"]["path"])
    if sha256(cases_file) != result["cases"]["sha256"]:
        raise SystemExit(f"{name} cases sha256 differs")
    outputs = {r["id"]: r for r in json.loads(cases_file.read_text())["results"]}
    cases = []
    for line in Path(panel["path"]).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] != "dev":
            raise SystemExit("panel row is not DEV")
        ctx = PromptContext.from_mapping(row["context"])
        out = outputs[row["id"]]
        valid, noop, body = parse_raw(out["raw_output"], out["generated_ids"], ctx)
        if (valid, noop) != (out["protocol_valid"], out["predicted_noop"]):
            raise SystemExit(f"{name} {row['id']}: re-parse differs from the gate's classification")
        gold = tuple(row["region_new"])
        expected_noop = tuple(ctx.region_old) == gold
        cases.append(Case(id=row["id"], family=row["family"], context=ctx, gold=gold, expected_noop=expected_noop,
                          valid=valid, pred_noop=noop, body=body, air_exact=None))
    identity = {"panel": {"path": panel["path"], "sha256": panel["sha256"]},
                "result": {"path": str(path), "sha256": digest},
                "cases": {"path": str(cases_file), "sha256": result["cases"]["sha256"]}}
    return cases, identity


def dev_air(formatter):
    """G3 for DEV: air on the context window (prefix + body + suffix) on both sides."""
    from sepalith.training.rl.air_canonical import normalize_lines, normalize_text

    cache = {}

    def window(case, body):
        return "\n".join((*case.context.prefix, *body, *case.context.suffix_lines)) + "\n"

    def fmt(text):
        if text not in cache:
            out = formatter.format(text)
            cache[text] = None if out is None else normalize_text(out)
        return cache[text]

    def verdict(case):
        body = tuple(case.body)
        if case.context.replacement_range.start.character != 0:
            raw_old = normalize_lines(body) == normalize_lines(case.context.region_old)
            return "noop" if raw_old else normalize_lines(body) == normalize_lines(case.gold)
        cand, old, gold = (fmt(window(case, body)), fmt(window(case, case.context.region_old)),
                           fmt(window(case, case.gold)))
        if gold is None or old is None or cand is None or gold == old:
            # air failure or a formatting-only reference: raw comparison, as reward v2.1 does
            if normalize_lines(body) == normalize_lines(case.context.region_old):
                return "noop"
            return normalize_lines(body) == normalize_lines(case.gold)
        if cand == old:
            return "noop"
        return cand == gold

    return verdict


def gold_stats(cases):
    """Gold-side whitespace conventions (one row per prompt): why a prediction-only trim can hurt."""
    seen, c = set(), Counter()
    for case in cases:
        if case.id in seen or case.expected_noop:
            continue
        seen.add(case.id)
        gold, rng = tuple(case.gold), case.context.replacement_range
        kind = "insert_mid_line" if rng.is_empty and rng.start.character > 0 else (
            "insert_col0" if rng.is_empty else "replace")
        c[f"{kind}:edits"] += 1
        if gold and blank(gold[0]):
            c[f"{kind}:gold_leading_blank"] += 1
        if gold and blank(gold[-1]):
            c[f"{kind}:gold_trailing_blank"] += 1
        if any(x != x.rstrip() for x in gold):
            c[f"{kind}:gold_trailing_whitespace"] += 1
        if any(leading(x).startswith("\t") for x in gold):
            c[f"{kind}:gold_tab_indent"] += 1
    return dict(sorted(c.items()))


def hurts(cases, variant):
    """Cases the raw output got right that the legit variant breaks (gold-side whitespace)."""
    return [c for c in cases if outcome(c, "raw")[1] and not outcome(c, variant)[1]]


def train_section(cases, air_counts):
    per_variant = {}
    for v in VARIANTS:
        counts, fam = tally(cases, v)
        prompts = defaultdict(list)
        for c in cases:
            prompts[c.id].append(outcome(c, v)[1])
        mixed = sum(1 for xs in prompts.values() if 0 < sum(xs) < len(xs))
        per_variant[v] = {"samples": len(cases), "correct": counts["score"],
                          "exact_rate": round(counts["score"] / len(cases), 6),
                          "edit_exact": counts["edit_exact"], "edit_samples": counts["edit_cases"],
                          "noop_correct": counts["noop_correct"], "noop_samples": counts["noop_cases"],
                          "nofp": counts["nofp"], "valid": counts["valid"],
                          "M_share_mixed_prompts": round(mixed / len(prompts), 6),
                          "per_family": {k: {**x, "rate": round(x["correct"] / x["n"], 6)} for k, x in fam.items()}}
        if v in LEGIT:
            h = hurts(cases, v)
            per_variant[v]["losses_vs_raw"] = len(h)
            per_variant[v]["losses_by_family"] = dict(Counter(c.family for c in h))
            g, _ = flips(cases, v)
            per_variant[v]["gains_vs_raw"] = len(g)
            per_variant[v]["gain_types"] = dict(Counter("+".join(x["types"] or ["n/a"]) for x in g).most_common(12))
    return per_variant


def dev_section(cases, air):
    out = {}
    for v in VARIANTS:
        counts, fam = tally(cases, v, air)
        gained, lost = flips(cases, v, air)
        out[v] = {**counts, "per_family": fam, "gained": gained, "lost": lost}
    return out


def markdown(receipt):
    t, d250, d75 = receipt["train"]["variants"], receipt["dev"]["dev250"]["variants"], receipt["dev"]["dev75"]["variants"]
    lines = ["# Whitespace near-misses and decode-time canonicalisation (arm A 536)", "",
             "Measurement only. The official scorer is unchanged. Variants are fixed in code, run on TRAIN first, then on DEV.",
             "L0 to L3a change only the prediction and use only prompt-side information. G1 to G3 game the metric.", "",
             "| Variant | TRAIN exact rate (census, T=1.0) | TRAIN nofp | DEV250 edit /200 | DEV250 no-op /50 | score250 | DEV250 nofp | DEV75 edit /43 | DEV75 nofp /32 | DEV75 valid |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in VARIANTS:
        a, b, c = t[v], d250[v], d75[v]
        lines.append(f"| {v} | {a['exact_rate']:.4f} ({a['correct']}/{a['samples']}) | {a['nofp']} | {b['edit_exact']} | "
                     f"{b['noop_correct']} | {b['score']} | {b['nofp']} | {c['edit_exact']} | {c['nofp']} | {c['valid']} |")
    lines += ["", "Whitespace-only misses (valid wrong outputs whose text matches after removing whitespace):", ""]
    for name in ("train", "dev250", "dev75"):
        m = receipt["train"]["misses"] if name == "train" else receipt["dev"][name]["misses"]
        lines.append(f"- {name}: {m['kinds']}; types {m['per_type']}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=HERE.parent / "receipts/DT-ws-nearmiss.json")
    p.add_argument("--markdown", type=Path, default=HERE.parent / "receipts/DT-ws-nearmiss.md")
    p.add_argument("--train-limit", type=int, default=0, help="first N census prompts only (smoke)")
    p.add_argument("--air", default="air")
    args = p.parse_args(argv)
    started = time.monotonic()
    recipe = json.loads(RECIPE.read_text())
    tok, tok_path = load_tokenizer(recipe)

    # TRAIN first.
    train_cases, train_identity, air_rows, nm_equal = load_train(tok, args.train_limit or None)
    edit_raw_check = Counter()
    for c in train_cases:
        if not c.expected_noop and c.valid and not c.pred_noop:
            sid = f"{c.id}:{c.sample}"
            edit_raw_check[(tuple(c.body) == tuple(c.gold)) == bool(air_rows[sid][4])] += 1
    train = {"identity": train_identity, "variants": train_section(train_cases, air_rows),
             "misses": classify_misses(train_cases), "gold_whitespace_conventions": gold_stats(train_cases),
             "census_exact_raw_agreement_on_valid_edit_samples": {"agree": edit_raw_check[True], "disagree": edit_raw_check[False]},
             "receipt_309_crosscheck": {"near_miss_with_canonical_region_line_f1_1": dict(nm_equal),
                                        "receipt": {"finish_block": 309, "pipe_rewrite": 1}}}
    # Which of the near-miss canonical-equal samples are raw whitespace-only misses?
    ws_of_nm = Counter()
    for c in train_cases:
        sid = f"{c.id}:{c.sample}"
        exact, out, sim, space, _ = air_rows[sid]
        if out == "near_miss" and sim == 1.0 and space and space.startswith("canonical") and c.valid and not c.pred_noop:
            t = ws_types(c.body, c.gold)
            ws_of_nm[primary(t) if t else "not_ws_only"] += 1
    train["receipt_309_crosscheck"]["raw_whitespace_types"] = dict(ws_of_nm.most_common())

    # DEV after.
    from sepalith.training.rl.air_canonical import AirFormatter

    formatter = AirFormatter((args.air,))
    air = dev_air(formatter)
    dev = {}
    for name in ("dev250", "dev75"):
        cases, identity = load_dev(recipe, name)
        raw, _ = tally(cases, "raw")
        official = OFFICIAL[name]
        reproduced = {"edit_exact": raw["edit_exact"], "protocol_valid": raw["valid"],
                      "strict_noop_false_suggestions": raw["nofp"]}
        if name == "dev250":
            reproduced["strict_noop_correct"] = raw["noop_correct"]
        if reproduced != official:
            raise SystemExit(f"{name}: reproduced {reproduced} differs from official {official}")
        dev[name] = {"identity": identity, "reproduced_official_counts": reproduced,
                     "variants": dev_section(cases, air), "misses": classify_misses(cases),
                     "gold_whitespace_conventions": gold_stats(cases)}
    receipt = {
        "schema": "sepalith.decode-time.ws-nearmiss.v1",
        "purpose": "measurement only: no model or selection decision on DEV; variants defined in code before DEV was read and run on TRAIN first",
        "final_set_access": False,
        "model": "arm A checkpoint 536 (DEV gate outputs, greedy cap 1024); TRAIN = C12 census of the arm-a upload (same weights), k=8 at T=1.0",
        "tokenizer": {"path": tok_path, "decode": "tokenizers.decode(skip_special_tokens=False), final stop id dropped"},
        "recipe": {"path": str(RECIPE), "sha256": sha256(RECIPE)},
        "exactness": "evaluator raw region rule (campaign_eval.classify); no-op via parse_output",
        "variants": {
            "raw": "official evaluator",
            "L0": "extension-literal: drop every leading and trailing whitespace-only body line (parsePrediction in the shipped extension)",
            "L1": "drop trailing whitespace-only body lines only (leading blank lines are kept: TRAIN golds open an insertion with a blank line in about 15% of finish_block rows, so L0's leading trim is gold-dependent)",
            "L2": "L1 + strip trailing whitespace per line",
            "L2s": "L2 + suppress (treat as no suggestion) any prediction that differs from region_old only in whitespace; prediction and context only, but it also suppresses correct formatting-only edits",
            "L3": "L2 + re-indent to the document's detected indentation (tabs/spaces; indent unit rescaled relative to the block's first line)",
            "L3a": "L3 + shift the block so its first line keeps region_old's first-line indent when it re-emits that line",
            "G1": "GAMING: whitespace-insensitive comparison on both sides",
            "G2": "GAMING: re-indent with the gold's per-line indentation",
            "G3": "GAMING: air formatter on both sides (TRAIN: census reward v2.1 exact; DEV: context window)",
        },
        "line_endings_note": "split_wire_lines strips a trailing CR from every wire line and region lines are LF-free by contract, so CRLF differences cannot reach the comparison; the document EOL is applied at insertion (map_wire_to_document_eol)",
        "product": {
            "extension": "extensions/vscode-sepalith/src/extension.ts parsePrediction: cuts at first '>>>>>>>', removes <|user_cursor|> and marker lines, strips trailing CR, drops leading and trailing whitespace-only lines, cuts at the third identical consecutive line; inserted as a plain-string vscode.InlineCompletionItem (VS Code inserts it verbatim: no re-indent, no format, no trailing-whitespace trim)",
            "prompt": "the extension still renders the older cursor-only zeta2 prompt (context_build.ts) and has no [NO_EDIT] handling; PRM03 is not wired into the client yet",
            "editor_settings": "files.trimTrailingWhitespace, files.trimFinalNewlines, files.insertFinalNewline act on save and are off by default; editor.detectIndentation sets the indent style but is not applied to inline-completion text",
        },
        "train": train,
        "dev": dev,
        "elapsed_seconds": None,
    }
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    args.markdown.write_text(markdown(receipt))
    print(json.dumps({"output": str(args.output), "elapsed": receipt["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
