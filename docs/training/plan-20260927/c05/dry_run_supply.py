"""C05: dry run of DEV250 supply for no_op and the propagation families.

For every eligible dev_group parent (C05 CPT-overlap receipt), run the
constructors DEV75 used and count the groups that yield converted cases:

- no_op: campaign_dev_panel._derived_candidate (already-applied `|>` or
  `na.rm = TRUE` lines);
- pipe_rewrite, rename_propagation, na_rm_propagation: the canonical
  scenario extractors, and format_propagation: the raw-tarball format pairs,
  each through campaign_dev_propagation._local_derived_convert.

Source files whose sha256 is a CPT document are skipped. Nothing is built
for the panel; converted rows and snapshots go to a scratch directory.

    python3 dry_run_supply.py SCRATCH_DIR OUT.json
"""

import argparse
import hashlib
import importlib.util
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training")
DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work")
CPT_INDEX = DATA / "CPT-prefix-extension-v1/combined-cache/index.sqlite3"
OVERLAP = REPO / "docs/training/plan-20260927/receipts/C05-cpt-overlap.json"
PER_GROUP = 3  # count at most this many converted cases per group and family


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scratch", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=2307)
    args = ap.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(EXEC))
    prop = load("campaign_dev_propagation", EXEC / "campaign_dev_propagation.py")
    panel = load("campaign_dev_panel", EXEC / "campaign_dev_panel.py")
    assert prop.file_sha(prop.SCENARIOS_PATH) == prop.SCENARIO_SHA

    con = sqlite3.connect(f"file:{CPT_INDEX}?mode=ro", uri=True)
    cpt = {h for (h,) in con.execute("select source_sha256 from documents")}
    eligible = set(json.loads(OVERLAP.read_text())["eligible_construction_groups"])
    parents = [p for p in prop._parent_records() if p["group_id"] in eligible]

    supply = defaultdict(dict)  # family -> group -> converted count
    reasons = Counter()
    rows_path = args.scratch / "rows.jsonl"
    src_dir = args.scratch / "sources"
    src_dir.mkdir()

    for index, parent in enumerate(sorted(parents, key=lambda p: p["group_id"])):
        gid = parent["group_id"]
        r_dir = Path(parent["normalized_root"]) / parent["package"] / "R"
        files = sorted([*r_dir.glob("*.R"), *r_dir.glob("*.r")]) if r_dir.is_dir() else []

        # no_op: one candidate per file, at most PER_GROUP files.
        n = 0
        for path in files:
            if n >= PER_GROUP:
                break
            raw = path.read_bytes()
            if sha256(raw) in cpt:
                reasons["no_op:file_in_cpt"] += 1
                continue
            try:
                _, text, _ = panel._read_utf8(path)
            except (OSError, UnicodeError, ValueError):
                reasons["no_op:unreadable"] += 1
                continue
            lines = text.split("\n")
            if any(panel._derived_candidate("no_op", lines, i) for i in range(len(lines))):
                n += 1
        if n:
            supply["no_op"][gid] = n

        try:
            prop._parse_license(Path(parent["normalized_root"]) / parent["package"])
        except Exception:  # noqa: BLE001
            reasons["license_missing"] += 1
            continue
        for family in prop.FAMILIES:
            rng = random.Random(args.seed + index * 101 + prop.FAMILIES.index(family) * 10007)
            cands = (prop._format_candidates(parent) if family == "format_propagation"
                     else prop._scenario_candidates(parent, family, rng))
            n = 0
            for row, before_raw, extra in cands:
                if n >= PER_GROUP:
                    break
                if sha256(before_raw) in cpt:
                    reasons[f"{family}:file_in_cpt"] += 1
                    continue
                result, _ = prop._local_derived_convert(
                    row, parent, before_raw, rows_path, src_dir,
                    normalized_after_sha=extra.get("normalized_after_sha"),
                    raw_tarball_member=extra.get("raw_member"))
                if result["status"] == "converted":
                    n += 1
                else:
                    reasons[f"{family}:{result.get('reason')}"] += 1
            if n:
                supply[family][gid] = n

    need = {"no_op": (50, 17), "pipe_rewrite": (28, 10), "format_propagation": (24, 8),
            "rename_propagation": (23, 8), "na_rm_propagation": (4, 2)}
    report = {"eligible_groups": len(parents), "per_family": {}, "reasons": dict(reasons)}
    for fam, (cases, groups) in need.items():
        g = supply.get(fam, {})
        report["per_family"][fam] = {
            "groups_with_supply": len(g),
            "cases_at_cap_3": sum(g.values()),
            "required_cases": cases, "required_groups": groups,
            "pass": len(g) >= groups and sum(g.values()) >= cases,
            "groups": g,
        }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({f: {k: v for k, v in r.items() if k != "groups"}
                      for f, r in report["per_family"].items()}, indent=2))
    print(json.dumps(dict(reasons), indent=2))


if __name__ == "__main__":
    main()
