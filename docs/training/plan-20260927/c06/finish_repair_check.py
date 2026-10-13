#!/usr/bin/env python3
"""C06: stratified spot-check of the repaired finish_block TRAIN rows.

Two subcommands:

``check``
    Draws the seed-3407 sample (80 repaired rows stratified by source variant
    and finish kind, plus 20 unrepaired finish_block controls), decodes every
    target with the parent tokenizer, reconstructs the gold applied document,
    runs the R parse checks (``check_parse.R``) and ``air format`` stability
    checks, and writes ``checks.jsonl`` plus a human review dump
    ``review.md`` into ``--work-dir``. The work dir holds row text and stays
    outside the repository.

``finalize``
    Joins ``checks.jsonl`` with the hand classifications in
    ``classifications.json`` and writes the receipt with counts, defective ids,
    Wilson 95% intervals and the C06 rule outcome.

Inputs are read-only. Only TRAIN rows from DAT10-finish-source-repair-v3 are
read; no DEV, final or upstream source pool is opened.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import difflib
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3")
TOKENIZER = Path(
    "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586/tokenizer.json"
)
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
PINS = {
    "train-token-rows.jsonl": "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e",
    "context-sidecar.jsonl": "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a",
    "repair-ledger.jsonl": "254d6ef418edec64c79047f18a546dc1af112a84e709b2fe6233d2453bc33cfa",
}
SEED = 3407
N_REPAIRED = 80
N_CONTROL = 20
# Small strata get a fixed floor so every repair source/kind is inspected; the
# rest of the 80 is split proportionally over the three large strata.
ALLOCATION = {
    ("finish_block_compound", "mid_body"): 32,
    ("finish_block_compound_random", "mid_body"): 16,
    ("finish_block_compound", "signature"): 7,
    ("finish_block_authored_agy", "mid_body"): 5,
    ("finish_block_authored_gpt56sol", "mid_body"): 5,
    ("finish_block_authored_zai", "mid_body"): 5,
    ("finish_block_authored_agy", "signature"): 5,
    ("finish_block_authored_gpt56sol", "signature"): 5,
}
CLASSES = ("correct", "cosmetic", "defective")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utf16_index(line: str, units: int) -> int:
    used = 0
    for i, ch in enumerate(line):
        if used == units:
            return i
        used += len(ch.encode("utf-16-le")) // 2
        if used > units:
            raise ValueError("utf16 split")
    if used != units:
        raise ValueError("utf16 overflow")
    return len(line)


def offsets(document: str, rr: dict) -> tuple[int, int]:
    lines = document.split("\n")

    def off(pos):
        return sum(len(x) + 1 for x in lines[: pos["line"]]) + utf16_index(lines[pos["line"]], pos["character"])

    return off(rr["start"]), off(rr["end"])


def stratum(ctx: dict) -> tuple[str, str]:
    ref = ctx["source_identity"].get("source_ref") or {}
    if ref:
        return (ref.get("source_variant"), ref.get("kind"))
    derivation = ctx["source_identity"].get("source_provenance", {}).get("derivation", "unknown")
    return ("r2_short_task", derivation)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


# --------------------------------------------------------------------------- check


def draw_sample(ledger_ids: set[str]) -> tuple[list[dict], dict, dict]:
    strata_rep: dict[tuple, list[str]] = collections.defaultdict(list)
    control_pool: list[str] = []
    control_strata: dict[str, tuple] = {}
    with (DATA / "context-sidecar.jsonl").open() as f:
        for line in f:
            ctx = json.loads(line)
            if ctx["family"] != "finish_block":
                continue
            key = stratum(ctx)
            if ctx["row_id"] in ledger_ids:
                strata_rep[key].append(ctx["row_id"])
            else:
                control_pool.append(ctx["row_id"])
                control_strata[ctx["row_id"]] = key
    sizes = {"|".join(k): len(v) for k, v in sorted(strata_rep.items())}
    if set(strata_rep) != set(ALLOCATION) or sum(ALLOCATION.values()) != N_REPAIRED:
        raise SystemExit(f"stratum mismatch: {sizes}")
    if sum(len(v) for v in strata_rep.values()) != len(ledger_ids):
        raise SystemExit("ledger/context join mismatch")
    rng = random.Random(SEED)
    picks = []
    for key in sorted(ALLOCATION):
        for rid in rng.sample(sorted(strata_rep[key]), ALLOCATION[key]):
            picks.append({"row_id": rid, "group": "repaired", "stratum": "|".join(key)})
    rng_c = random.Random(SEED)
    for rid in rng_c.sample(sorted(control_pool), N_CONTROL):
        picks.append({"row_id": rid, "group": "control", "stratum": "|".join(control_strata[rid])})
    control_sizes = collections.Counter("|".join(v) for v in control_strata.values())
    return picks, sizes, dict(sorted(control_sizes.items()))


def run_air(src: Path, workdir: Path, air: str) -> dict:
    once = workdir / (src.stem + ".air1.R")
    twice = workdir / (src.stem + ".air2.R")
    shutil.copyfile(src, once)
    r1 = subprocess.run([air, "format", str(once)], capture_output=True, text=True, timeout=120)
    if r1.returncode != 0:
        return {"air_ok": False, "air_error": (r1.stderr or r1.stdout).strip()[:300]}
    shutil.copyfile(once, twice)
    r2 = subprocess.run([air, "format", str(twice)], capture_output=True, text=True, timeout=120)
    return {
        "air_ok": r2.returncode == 0,
        "air_idempotent": once.read_text() == twice.read_text(),
        "formatted_text": once.read_text(),
    }


def changed_lines(before: list[str], after: list[str]) -> tuple[set[int], list[str]]:
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    changed: set[int] = set()
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag != "equal":
            changed.update(range(i1, max(i2, i1 + 1)))
    diff = list(difflib.unified_diff(before, after, lineterm="", n=0))
    return changed, diff


def brace_analysis(lines: list[str], pairs_field: str, t_first: int, t_last: int, region_old: str) -> dict:
    """Check that the target's final `}` closes a block the prompt opened and
    that each `}` on its own line in the target is indented like its block.

    ``pairs_field`` is the R helper's "open:close:owner" list (1-based lines).
    A closing brace counts as consistent when its indentation equals either the
    line holding the matching `{` or the line where the owning construct
    (function, if, call ...) starts.
    """
    pairs = []
    if pairs_field:
        for item in pairs_field.split(","):
            o, c, w = item.split(":")
            pairs.append((int(o) - 1, int(c) - 1, int(w) - 1))
    # Lines the prompt shows: the prefix plus any old region the target rewrites.
    visible_end = t_first + (len(region_old.split("\n")) if region_old.strip() else 0)
    target_pairs = [(o, c, w) for o, c, w in pairs if t_first <= c <= t_last]
    final = max(target_pairs, key=lambda x: x[1], default=None)

    def consistent(o, c, w):
        return leading_ws(lines[c]) in (leading_ws(lines[o]), leading_ws(lines[w]))

    mism = []
    for o, c, w in target_pairs:
        if lines[c].strip().startswith("}") and not consistent(o, c, w):
            mism.append(
                {
                    "owner_line": w,
                    "close_line": c,
                    "owner_indent": len(leading_ws(lines[w])),
                    "close_indent": len(leading_ws(lines[c])),
                    "final": (o, c, w) == final,
                }
            )
    own_line = final is not None and lines[final[1]].strip().startswith("}")
    return {
        "final_brace_opener_in_prompt": final is not None and final[0] < visible_end,
        "final_brace_opener_line": lines[final[2]].strip()[:120] if final else None,
        "prompt_open_blocks_closed_by_target": len([1 for o, c, w in target_pairs if o < visible_end]),
        "brace_indent_mismatches": mism,
        "final_brace_own_line": own_line,
        "final_brace_indent_matches_opener": own_line and consistent(*final),
    }


def cmd_check(args) -> None:
    from tokenizers import Tokenizer

    work = Path(args.work_dir)
    if work.exists():
        raise SystemExit(f"work dir must be fresh: {work}")
    (work / "docs").mkdir(parents=True)
    (work / "air").mkdir()

    actual_tok = sha256_file(TOKENIZER)
    if actual_tok != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer sha mismatch {actual_tok}")
    pins = {name: sha256_file(DATA / name) for name in PINS}
    if pins != PINS:
        raise SystemExit(f"data pin mismatch {pins}")
    tok = Tokenizer.from_file(str(TOKENIZER))

    ledger = {}
    with (DATA / "repair-ledger.jsonl").open() as f:
        for line in f:
            x = json.loads(line)
            ledger[x["row_id"]] = x
    picks, rep_sizes, control_sizes = draw_sample(set(ledger))
    wanted = {p["row_id"]: p for p in picks}

    rows, ctxs = {}, {}
    with (DATA / "train-token-rows.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in wanted:
                rows[r["id"]] = r
    with (DATA / "context-sidecar.jsonl").open() as f:
        for line in f:
            c = json.loads(line)
            if c["row_id"] in wanted:
                ctxs[c["row_id"]] = c
    if len(rows) != len(picks) or len(ctxs) != len(picks):
        raise SystemExit("sample join failed")

    manifest = ["key\tkind\tpath"]
    results = []
    for idx, p in enumerate(picks):
        rid = p["row_id"]
        row, ctx = rows[rid], ctxs[rid]
        res = dict(p)
        res["index"] = idx
        if row["family"] != "finish_block" or row["split"] != "train":
            raise SystemExit(f"not a TRAIN finish row: {rid}")
        # Tokenizer round trip on the parent tokenizer.
        body_ids = row["target_body_tokens"]
        tail_ids = row["input_ids"][row["target_start"] :]
        # Under PRM03 the body tokens carry the LF that separates the body from
        # the terminal `>>>>>>> UPDATED` line, so they decode to body + "\n".
        decoded = tok.decode(body_ids, skip_special_tokens=False)
        res["decode_body_exact"] = decoded == row["target_body_text"] + "\n"
        res["target_ids_layout_ok"] = tail_ids == body_ids + row["target_terminal_tokens"] + [row["eos_token_id"]]
        res["decode_full_target_exact"] = (
            tok.decode(tail_ids[:-1], skip_special_tokens=False) == row["target_text"]
        )
        res["target_tokens"] = len(tail_ids)
        if not res["decode_body_exact"]:
            raise SystemExit(f"decoded body differs from target_body_text: {rid}")
        target = decoded[:-1]
        # Reconstruct pre-edit and applied documents.
        sel = ctx["source_identity"]["source_provenance"]["selection_source"]
        document = sel["document_text"]
        rr = ctx["context"]["replacement_range"]
        if hashlib.sha256(document.encode()).hexdigest() != rr["content_sha256"]:
            raise SystemExit(f"document hash mismatch {rid}")
        left, right = offsets(document, rr)
        applied = document[:left] + target + document[right:]
        res["range_at_eof"] = right == len(document)
        res["post_cursor_chars"] = len(document) - right
        res["suffix_lines"] = len(ctx["context"]["suffix_lines"])
        region_old = "\n".join(ctx["context"]["region_old"])
        res["region_old_nonempty"] = bool(region_old.strip())
        res["target_keeps_region_old"] = target.startswith(region_old) if region_old.strip() else None
        res["target_ends_with_brace"] = target.rstrip().endswith("}")
        res["target_trailing_after_brace"] = target[len(target.rstrip()) :] != "" if target.rstrip().endswith("}") else None
        if p["group"] == "repaired":
            led = ledger[rid]
            res["ledger_target_sha_match"] = hashlib.sha256(target.encode()).hexdigest() == led["repaired_target_body_sha256"]
            res["ledger_applied_sha_match"] = hashlib.sha256(applied.encode()).hexdigest() == led["applied_document_sha256"]
        stem = f"{idx:03d}"
        files = {"applied": work / "docs" / f"{stem}.applied.R", "pre": work / "docs" / f"{stem}.pre.R"}
        files["applied"].write_text(applied)
        files["pre"].write_text(document)
        if res["target_ends_with_brace"]:
            stripped = target.rstrip()
            nobrace = document[:left] + stripped[:-1] + target[len(stripped) :] + document[right:]
            files["nobrace"] = work / "docs" / f"{stem}.nobrace.R"
            files["nobrace"].write_text(nobrace)
        for kind, path in files.items():
            manifest.append(f"{stem}\t{kind}\t{path}")
        # Target line span inside the applied document (0-based, inclusive).
        t_first = applied[:left].count("\n")
        t_last = t_first + target.count("\n")
        res["target_line_span"] = [t_first, t_last]
        # air format: idempotence plus which lines it would touch.
        air = run_air(files["applied"], work / "air", args.air)
        app_lines = applied.split("\n")
        if air.get("air_ok") and "formatted_text" in air:
            changed, diff = changed_lines(app_lines, air.pop("formatted_text").split("\n"))
            in_target = sorted(i for i in changed if t_first <= i <= t_last)
            res["air_changed_lines_prefix"] = len([i for i in changed if i < t_first])
            res["air_changed_lines_target"] = len(in_target)
            res["air_prefix_lines"] = t_first
            res["air_target_lines"] = t_last - t_first + 1
            res["_air_diff"] = diff
        res.update({k: v for k, v in air.items() if k != "formatted_text"})
        res["_applied_lines"] = app_lines
        res["_target"] = target
        res["_region_old"] = region_old
        res["_prompt_text"] = row["prompt_text"]
        results.append(res)

    (work / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    rcmd = ["Rscript", "--vanilla", str(HERE / "check_parse.R"), str(work / "manifest.tsv"), str(work / "parse.tsv")]
    subprocess.run(rcmd, check=True, timeout=1800)
    parse = collections.defaultdict(dict)
    for line in (work / "parse.tsv").read_text().splitlines()[1:]:
        key, kind, ok, pairs, err = (line.split("\t") + [""] * 5)[:5]
        parse[int(key)][kind] = {"ok": ok == "1", "pairs": pairs, "error": err}

    for res in results:
        pr = parse[res["index"]]
        res["parse_applied"] = pr["applied"]["ok"]
        res["parse_pre_edit"] = pr["pre"]["ok"]
        res["parse_without_final_brace"] = pr["nobrace"]["ok"] if "nobrace" in pr else None
        res["parse_applied_error"] = pr["applied"]["error"] or None
        res.update(
            brace_analysis(res["_applied_lines"], pr["applied"]["pairs"], *res["target_line_span"], res["_region_old"])
        )

    with (work / "checks.jsonl").open("w") as f:
        for res in results:
            f.write(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, sort_keys=True) + "\n")
    write_review(work / "review.md", results)
    meta = {
        "seed": SEED,
        "allocation": {"|".join(k): v for k, v in sorted(ALLOCATION.items())},
        "repaired_stratum_sizes": rep_sizes,
        "control_pool_size": sum(control_sizes.values()),
        "control_stratum_sizes": control_sizes,
        "tokenizer": {"path": str(TOKENIZER), "sha256": actual_tok},
        "data_pins": pins,
        "r_version": subprocess.run(["Rscript", "--version"], capture_output=True, text=True).stdout.strip()
        or subprocess.run(["Rscript", "--version"], capture_output=True, text=True).stderr.strip(),
        "air_version": subprocess.run([args.air, "--version"], capture_output=True, text=True).stdout.strip(),
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (work / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


def write_review(path: Path, results: list[dict], context_lines: int = 30) -> None:
    out = []
    for r in results:
        flags = {
            k: r.get(k)
            for k in (
                "parse_applied",
                "parse_pre_edit",
                "parse_without_final_brace",
                "air_ok",
                "air_idempotent",
                "air_changed_lines_target",
                "air_changed_lines_prefix",
                "final_brace_opener_in_prompt",
                "final_brace_indent_matches_opener",
                "prompt_open_blocks_closed_by_target",
                "region_old_nonempty",
                "target_keeps_region_old",
                "decode_body_exact",
                "range_at_eof",
                "target_tokens",
            )
        }
        out.append(f"## [{r['index']:03d}] {r['row_id']} {r['group']} {r['stratum']}")
        out.append("`" + json.dumps(flags) + "`")
        if r["brace_indent_mismatches"]:
            out.append("indent mismatches: " + json.dumps(r["brace_indent_mismatches"]))
        t_first, _ = r["target_line_span"]
        lines = r["_applied_lines"]
        out.append("```r")
        start = max(0, t_first - context_lines)
        if start > 0:
            out.append(f"# ... {start} earlier lines; outline header: " + " | ".join(lines[:5])[:300])
        for i in range(start, t_first):
            out.append(f"P{i:4d}| {lines[i]}")
        if r["_region_old"].strip():
            out.append(f"OLD    | {r['_region_old']!r}")
        out.append("------ TARGET ------")
        for j, tl in enumerate(r["_target"].split("\n")):
            out.append(f"T{t_first + j:4d}| {tl}")
        out.append("```")
        if r.get("air_changed_lines_target"):
            out.append("air diff (first 30 lines):")
            out.append("```diff")
            out.extend(r["_air_diff"][:30])
            out.append("```")
        out.append("")
    path.write_text("\n".join(out))


# -------------------------------------------------------------------------- census


def cmd_census(args) -> None:
    """Population-wide structural pass over all 7,785 finish_block rows:
    parent-tokenizer decode, R parse of the applied document, and the brace
    checks from ``brace_analysis``. It supports the sample but does not replace
    reading the sampled rows."""
    from tokenizers import Tokenizer

    work = Path(args.work_dir)
    out_dir = work / "census"
    if out_dir.exists():
        raise SystemExit(f"census dir must be fresh: {out_dir}")
    out_dir.mkdir(parents=True)
    tok = Tokenizer.from_file(str(TOKENIZER))
    ledger_ids = set()
    with (DATA / "repair-ledger.jsonl").open() as f:
        for line in f:
            ledger_ids.add(json.loads(line)["row_id"])
    ctxs = {}
    with (DATA / "context-sidecar.jsonl").open() as f:
        for line in f:
            c = json.loads(line)
            if c["family"] == "finish_block":
                ctxs[c["row_id"]] = {
                    "stratum": "|".join(stratum(c)),
                    "document": c["source_identity"]["source_provenance"]["selection_source"]["document_text"],
                    "range": c["context"]["replacement_range"],
                    "region_old": "\n".join(c["context"]["region_old"]),
                }
    recs = []
    manifest = ["key\tkind\tpath"]
    with (DATA / "train-token-rows.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["family"] != "finish_block":
                continue
            c = ctxs[r["id"]]
            decoded = tok.decode(r["target_body_tokens"], skip_special_tokens=False)
            target = r["target_body_text"]
            left, right = offsets(c["document"], c["range"])
            applied = c["document"][:left] + target + c["document"][right:]
            key = f"{len(recs):05d}"
            path = out_dir / f"{key}.R"
            path.write_text(applied)
            manifest.append(f"{key}\tapplied\t{path}")
            t_first = applied[:left].count("\n")
            recs.append(
                {
                    "row_id": r["id"],
                    "group": "repaired" if r["id"] in ledger_ids else "control",
                    "stratum": c["stratum"],
                    "decode_body_exact": decoded == target + "\n",
                    "range_at_eof": right == len(c["document"]),
                    "span": (t_first, t_first + target.count("\n")),
                    "region_old": c["region_old"],
                    "lines": applied.split("\n"),
                }
            )
    (work / "census-manifest.tsv").write_text("\n".join(manifest) + "\n")
    subprocess.run(
        ["Rscript", "--vanilla", str(HERE / "check_parse.R"), str(work / "census-manifest.tsv"), str(work / "census-parse.tsv")],
        check=True,
        timeout=3600,
    )
    parse = {}
    for line in (work / "census-parse.tsv").read_text().splitlines()[1:]:
        key, _kind, ok, pairs, _err = (line.split("\t") + [""] * 5)[:5]
        parse[int(key)] = (ok == "1", pairs)
    agg: dict = collections.defaultdict(collections.Counter)
    mismatch_ids = collections.defaultdict(list)
    for i, rec in enumerate(recs):
        ok, pairs = parse[i]
        b = brace_analysis(rec["lines"], pairs, *rec["span"], rec["region_old"]) if ok else {}
        for key in (rec["group"], f"{rec['group']}:{rec['stratum']}"):
            a = agg[key]
            a["n"] += 1
            a["decode_body_exact"] += rec["decode_body_exact"]
            a["range_at_eof"] += rec["range_at_eof"]
            a["parse_applied"] += ok
            a["final_brace_own_line"] += bool(b.get("final_brace_own_line"))
            a["final_brace_opener_in_prompt"] += bool(b.get("final_brace_opener_in_prompt"))
            a["final_brace_indent_mismatch"] += bool(b) and not b.get("final_brace_indent_matches_opener")
            a["any_brace_indent_mismatch_in_target"] += bool(b.get("brace_indent_mismatches"))
        if b and not b.get("final_brace_indent_matches_opener"):
            mismatch_ids[rec["group"]].append(rec["row_id"])
    shutil.rmtree(out_dir)
    census = {
        "rows": len(recs),
        "by_group": {k: dict(v) for k, v in sorted(agg.items()) if ":" not in k},
        "by_group_stratum": {k: dict(v) for k, v in sorted(agg.items()) if ":" in k},
        "final_brace_indent_mismatch_ids": {k: sorted(v) for k, v in mismatch_ids.items()},
    }
    (work / "census.json").write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
    print(json.dumps(census["by_group"], indent=2, sort_keys=True))


# ------------------------------------------------------------------------ finalize


def cmd_finalize(args) -> None:
    work = Path(args.work_dir)
    checks = [json.loads(x) for x in (work / "checks.jsonl").read_text().splitlines()]
    meta = json.loads((work / "meta.json").read_text())
    cls = json.loads(Path(args.classifications).read_text())
    by_id = {c["row_id"]: c for c in cls["rows"]}
    if set(by_id) != {c["row_id"] for c in checks}:
        raise SystemExit("classification ids do not match the sample")
    rows = []
    for c in checks:
        k = by_id[c["row_id"]]
        if k["class"] not in CLASSES:
            raise SystemExit(f"bad class {k}")
        rows.append({**c, "class": k["class"], "reason": k["reason"]})

    def summary(group):
        g = [r for r in rows if r["group"] == group]
        counts = collections.Counter(r["class"] for r in g)
        k = counts["defective"]
        lo, hi = wilson(k, len(g))
        return g, {
            "n": len(g),
            "counts": {c: counts[c] for c in CLASSES},
            "defect_rate": k / len(g),
            "defect_rate_wilson95": [lo, hi],
        }

    rep, rep_s = summary("repaired")
    ctl, ctl_s = summary("control")
    # Population-weighted repaired defect rate (strata weighted by size).
    sizes = meta["repaired_stratum_sizes"]
    total = sum(sizes.values())
    per_stratum = {}
    weighted = 0.0
    for key, size in sorted(sizes.items()):
        g = [r for r in rep if r["stratum"] == key]
        d = sum(r["class"] == "defective" for r in g)
        cos = sum(r["class"] == "cosmetic" for r in g)
        per_stratum[key] = {"population": size, "sampled": len(g), "defective": d, "cosmetic": cos}
        weighted += size / total * d / len(g)
    rep_s["defect_rate_population_weighted"] = weighted
    rep_s["per_stratum"] = per_stratum

    rate = rep_s["defect_rate"]
    if rate > 0.10:
        outcome = "stop-for-user"
    elif rate <= 0.05 and rate <= ctl_s["defect_rate"] + 0.03:
        outcome = "continue"
    else:
        # 5-10%, or within 5% but worse than control + 3 points (the card does
        # not name that case; it is treated as the middle band).
        outcome = "continue_list_defective_types"
    defective_types = sorted({r["stratum"] for r in rep if r["class"] == "defective"})

    def slim(r):
        keep = (
            "index", "row_id", "group", "stratum", "class", "reason", "target_tokens",
            "parse_applied", "parse_pre_edit", "parse_without_final_brace", "air_ok", "air_idempotent",
            "air_changed_lines_target", "air_changed_lines_prefix", "final_brace_opener_in_prompt",
            "final_brace_indent_matches_opener", "brace_indent_mismatches", "region_old_nonempty",
            "target_keeps_region_old", "decode_body_exact", "decode_full_target_exact",
            "target_ids_layout_ok", "range_at_eof", "post_cursor_chars", "ledger_target_sha_match",
            "ledger_applied_sha_match",
        )
        return {k: r.get(k) for k in keep}

    automated = {
        "decode_body_exact": sum(bool(r["decode_body_exact"]) for r in rows),
        "decode_full_target_exact": sum(bool(r["decode_full_target_exact"]) for r in rows),
        "target_ids_layout_ok": sum(bool(r["target_ids_layout_ok"]) for r in rows),
        "parse_applied": sum(bool(r["parse_applied"]) for r in rows),
        "parse_pre_edit": sum(bool(r["parse_pre_edit"]) for r in rows),
        "repaired_parse_without_final_brace": sum(bool(r["parse_without_final_brace"]) for r in rep),
        "air_ok": sum(bool(r.get("air_ok")) for r in rows),
        "air_idempotent": sum(bool(r.get("air_idempotent")) for r in rows),
        "air_touches_target_lines": sum(bool(r.get("air_changed_lines_target")) for r in rows),
        "final_brace_opener_in_prompt": sum(bool(r["final_brace_opener_in_prompt"]) for r in rows),
        "final_brace_indent_matches_opener": sum(bool(r["final_brace_indent_matches_opener"]) for r in rows),
        "rows_with_any_brace_indent_mismatch": sum(bool(r["brace_indent_mismatches"]) for r in rows),
        "range_at_eof": sum(bool(r["range_at_eof"]) for r in rows),
        "repaired_ledger_target_sha_match": sum(bool(r.get("ledger_target_sha_match")) for r in rep),
        "repaired_ledger_applied_sha_match": sum(bool(r.get("ledger_applied_sha_match")) for r in rep),
        "n": len(rows),
    }
    receipt = {
        "schema": "sepalith.plan-20260927.C06.finish-repair-check.v1",
        "card": "C06",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inputs": {
            "data_dir": str(DATA),
            "data_pins": meta["data_pins"],
            "tokenizer": meta["tokenizer"],
            "r_version": meta["r_version"],
            "air_version": meta["air_version"],
            "script": "docs/training/plan-20260927/c06/finish_repair_check.py",
            "script_sha256": sha256_file(Path(__file__)),
            "parse_helper_sha256": sha256_file(HERE / "check_parse.R"),
            "classifications_sha256": sha256_file(Path(args.classifications)),
        },
        "sampling": {
            "seed": meta["seed"],
            "repaired_allocation": meta["allocation"],
            "repaired_stratum_sizes": meta["repaired_stratum_sizes"],
            "control_pool_size": meta["control_pool_size"],
            "control_stratum_sizes": meta["control_stratum_sizes"],
            "method": cls.get("sampling_note"),
        },
        "class_definitions": cls["class_definitions"],
        "automated_checks": automated,
        "repaired": rep_s,
        "control": ctl_s,
        "defective": [
            {"row_id": r["row_id"], "group": r["group"], "stratum": r["stratum"], "reason": r["reason"]}
            for r in rows if r["class"] == "defective"
        ],
        "cosmetic": [
            {"row_id": r["row_id"], "group": r["group"], "stratum": r["stratum"], "reason": r["reason"]}
            for r in rows if r["class"] == "cosmetic"
        ],
        "rule": {
            "text": "repaired defect rate <= 5% and <= control + 3 points: continue; 5-10%: continue and list defective repair types; > 10%: stop-for-user",
            "inputs": {
                "repaired_defect_rate": rate,
                "control_defect_rate": ctl_s["defect_rate"],
                "repaired_wilson95": rep_s["defect_rate_wilson95"],
            },
            "outcome": outcome,
            "defective_repair_types": defective_types,
        },
        "caveats": cls.get("caveats", []),
        "rows": [slim(r) for r in rows],
    }
    census_path = work / "census.json"
    if census_path.exists():
        census = json.loads(census_path.read_text())
        receipt["population_census"] = {
            "what": "Structural checks on all 7,785 finish_block rows (no reading): parent-tokenizer decode, R parse of the applied document, final brace closing a prompt-opened block on its own line, and final-brace indentation against its opener or owner line.",
            "rows": census["rows"],
            "by_group": census["by_group"],
            "by_group_stratum": census["by_group_stratum"],
            "repaired_final_brace_indent_mismatch_ids": census["final_brace_indent_mismatch_ids"].get("repaired", []),
        }
        receipt["inputs"]["census_sha256"] = sha256_file(census_path)
    out = Path(args.receipt)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=False, ensure_ascii=False) + "\n")
    print(json.dumps({"repaired": rep_s, "control": ctl_s, "rule": receipt["rule"]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--work-dir", required=True)
    c.add_argument("--air", default=shutil.which("air") or os.path.expanduser("~/.local/bin/air"))
    s = sub.add_parser("census")
    s.add_argument("--work-dir", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--work-dir", required=True)
    f.add_argument("--classifications", default=str(HERE / "classifications.json"))
    f.add_argument(
        "--receipt", default=str(HERE.parent / "receipts" / "C06-finish-repair-check.json")
    )
    args = ap.parse_args()
    {"check": cmd_check, "census": cmd_census, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
