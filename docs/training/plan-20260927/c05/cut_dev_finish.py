"""C05: cut DEV finish_block cases from DEV package code.

The code-cut half of DEV250's finish_block quota. It runs the same rule
TRAIN's finish_block_compound wave used (rules_finish_block.derive_all, no
LLM) over the eligible dev_group parents from the C05 CPT-overlap receipt:
allowlisted, not used by DEV75, and not overlapping CPT. Any source file
whose sha256 is a CPT document is skipped.

    python3 cut_dev_finish.py OUT_DIR [--n 55]

Writes OUT_DIR/rows.jsonl (all derived rows), bases.jsonl, selected-N.jsonl
and stats.json.
"""

import argparse
import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "experiments/synthetic-data"))

import scenarios as S  # noqa: E402
import cases.corpus as C  # noqa: E402
import cases.validators as V  # noqa: E402
from cases.compound import BaseSample  # noqa: E402
import cases.rules.rules_finish_block as FB  # noqa: E402
from cases.rules import load_rules  # noqa: E402
import finish_block_compound as FBC  # noqa: E402

DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work")
ALLOWLIST = DATA / "DAT-07-dev-parent-allowlist.json"
CPT_INDEX = DATA / "CPT-prefix-extension-v1/combined-cache/index.sqlite3"
OVERLAP = REPO / "docs/training/plan-20260927/receipts/C05-cpt-overlap.json"

MAX_PER_GROUP = 3
BASES_PER_PACKAGE = 6
# Same body-size window as the TRAIN wave.
MIN_BODY, MAX_BODY = FBC.MIN_BODY_NB_WAVE, FBC.MAX_BODY_NB_WAVE
# Plain cuts only; fb_ctx_* add a synthetic comment to the file.
TRANSFORMS = {"fb_cut_signature", "fb_cut_after_first", "fb_cut_mid_nested",
              "fb_cut_before_return", "fb_cut_random", "fb_docstring_strip"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cpt_hashes() -> set:
    con = sqlite3.connect(f"file:{CPT_INDEX}?mode=ro", uri=True)
    return {h for (h,) in con.execute("select source_sha256 from documents")}


def version_dir(parent: dict) -> Path | None:
    # Highest version, as the TRAIN wave picks it.
    roots = {Path(v).parent for v in parent["normalized_versions"]}
    picked = [C.ASTFIM.pick_version_dir(r) for r in sorted(roots)]
    picked = [p for p in picked if p is not None]
    return picked[0] if picked else None


def harvest(parent, cpt, stats):
    pkg = parent["package"]
    vd = version_dir(parent)
    if vd is None:
        stats["no_version_dir"] += 1
        return [], []
    rdir = C.ASTFIM.src_root_for(vd, pkg)
    if rdir is None:
        stats["no_r_dir"] += 1
        return [], []
    prov = FBC.package_provenance(pkg, vd)
    rows, bases = [], []
    for f in sorted(list(rdir.glob("*.R")) + list(rdir.glob("*.r"))):
        if len(bases) >= BASES_PER_PACKAGE:
            break
        src = f.read_bytes()
        if not src or len(src) > S.MAX_FILE_BYTES:
            continue
        src_sha = sha256(src)
        if src_sha in cpt:
            stats["file_in_cpt"] += 1
            continue
        b = S.Bundle(pkg, f"R/{f.name}", src)
        for fn in (n for n in V._walk(b.tree.root_node)
                   if n.type == "function_definition"):
            if C._fn_body(b, fn) is None:
                continue
            try:
                bs = BaseSample(b, fn, len(bases))
            except ValueError:
                continue
            if FB._lhs_name(bs) is None or not MIN_BODY <= bs.nbody <= MAX_BODY:
                continue
            try:
                derived, st = FB.derive_all(bs, prov=dict(prov, package=pkg, path=b.rel))
            except Exception as e:  # noqa: BLE001
                stats[f"derive_exc {type(e).__name__}"] += 1
                continue
            derived = [r for r in derived if r["transform"] in TRANSFORMS]
            if not derived:
                continue
            base_id = derived[0]["derivation"]["base_sample_id"]
            for r in derived:
                r.update(group_id=parent["group_id"], base_id=base_id,
                         source_file=str(f), source_sha256=src_sha)
            rows.extend(derived)
            bases.append(dict(group_id=parent["group_id"], package=pkg,
                              path=b.rel, fn=derived[0]["fn"], base_id=base_id,
                              source_file=str(f), source_sha256=src_sha,
                              body_lines=bs.nbody, rows=len(derived)))
            break  # one base sample per file, as in the TRAIN wave
    return rows, bases


def select(rows_path: Path, bases, n: int, seed: int):
    by_base = defaultdict(list)
    for line_no, line in enumerate(rows_path.open("rb"), 1):
        r = json.loads(line)
        by_base[r["base_id"]].append((line_no, sha256(line), r))
    by_group = defaultdict(list)
    for b in bases:
        by_group[b["group_id"]].append(b)
    rng = random.Random(seed)
    order = sorted(by_group, key=lambda g: sha256(f"{seed}:{g}".encode()))
    picked = defaultdict(list)
    for _ in range(MAX_PER_GROUP):
        for g in order:
            if sum(len(v) for v in picked.values()) >= n:
                break
            pool = [b for b in by_group[g] if b not in picked[g]]
            if pool:
                picked[g].append(pool[0])
    out = []
    for g in order:
        for b in picked[g]:
            line_no, line_sha, r = rng.choice(by_base[b["base_id"]])
            out.append(dict(group_id=g, package=b["package"], path=b["path"],
                            fn=b["fn"], base_id=b["base_id"],
                            source_sha256=b["source_sha256"],
                            rows_file=str(rows_path), rows_line=line_no,
                            raw_line_sha256=line_sha, transform=r["transform"],
                            cut=r["cut"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=55)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()
    load_rules()
    args.out.mkdir(parents=True, exist_ok=False)

    eligible = set(json.loads(OVERLAP.read_text())["eligible_construction_groups"])
    parents = [p for p in json.loads(ALLOWLIST.read_text())["parents"]
               if p["group_id"] in eligible]
    cpt = cpt_hashes()
    stats = Counter()
    all_rows, all_bases = [], []
    for parent in sorted(parents, key=lambda p: p["group_id"]):
        rows, bases = harvest(parent, cpt, stats)
        stats["groups_with_bases"] += bool(bases)
        all_rows += rows
        all_bases += bases

    rows_path = args.out / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    (args.out / "bases.jsonl").write_text(
        "".join(json.dumps(b) + "\n" for b in all_bases))
    selected = select(rows_path, all_bases, args.n, args.seed)
    (args.out / f"selected-{args.n}.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in selected))
    summary = dict(
        eligible_groups=len(parents), bases=len(all_bases), rows=len(all_rows),
        funnel=dict(stats), selected=len(selected),
        selected_groups=len({x["group_id"] for x in selected}),
        per_transform=dict(Counter(x["transform"] for x in selected)),
        rows_sha256=sha256(rows_path.read_bytes()),
        bases_per_group_hist=dict(Counter(Counter(
            b["group_id"] for b in all_bases).values())))
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
