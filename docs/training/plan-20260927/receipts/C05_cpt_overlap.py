"""C05 step 2: intersect DEV identities with the CPT training cohort.

Reads only metadata: the CPT cohort rows (group, package, source path and
hash per row), the DAT-02 split groups, the DAT-07 dev parent allowlist and
source manifest, and DEV75. No DAT-07-final-* file is opened.

    uv run --no-project python C05_cpt_overlap.py OUT.json
"""

import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work")
COHORT = DATA / "CPT-prefix-extension-v1/cohort-manifest.json"
SPLIT = DATA / "DAT-02-global-split-v2.json"
ALLOWLIST = DATA / "DAT-07-dev-parent-allowlist.json"
DEV_SOURCES = DATA / "DAT-07-dev-source-manifest.json"
DEV75 = Path(
    "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/"
    "lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
)

FIELDS = {
    k: re.compile(rb'"%s":"([^"]*)"' % k.encode())
    for k in ("group_id", "package", "source_path", "source_sha256")
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def norm_name(value) -> str:
    # Same normalization as DAT-02-build-split-v2.py.
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^a-z0-9._/-]+", "", text)


def rel_path(source_path: str) -> str:
    # /mnt/h/sepalith/normalized/<pkg>/<ver>/<pkg>/R/x.R -> R/x.R
    parts = source_path.split("/")
    return "/".join(parts[8:]) if len(parts) > 8 else source_path


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def document_hashes(obj) -> set:
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k.endswith("sha256") and HEX64.match(v):
                found.add(v)
            else:
                found |= document_hashes(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= document_hashes(v)
    return found


def scan_cpt(rows_path: Path):
    h = hashlib.sha256()
    docs = {}
    with rows_path.open("rb") as f:
        for line in f:
            h.update(line)
            meta = {}
            for key, rx in FIELDS.items():
                m = rx.search(line)
                meta[key] = m.group(1).decode() if m else None
            docs.setdefault(meta["source_sha256"], meta)
    return h.hexdigest(), docs


def main(out: Path) -> None:
    cohort = json.loads(COHORT.read_text())
    rows_path = Path(cohort["rows"]["path"])
    rows_sha, cpt_docs = scan_cpt(rows_path)
    assert rows_sha == cohort["rows"]["sha256"], "CPT rows hash mismatch"
    assert len(cpt_docs) == cohort["counts"]["documents"], len(cpt_docs)

    cpt_by_group = defaultdict(set)
    cpt_by_pkg = defaultdict(set)
    cpt_by_pkg_path = defaultdict(set)
    for sha, m in cpt_docs.items():
        cpt_by_group[m["group_id"]].add(sha)
        cpt_by_pkg[norm_name(m["package"])].add(sha)
        cpt_by_pkg_path[(norm_name(m["package"]), rel_path(m["source_path"]))].add(sha)

    split = json.loads(SPLIT.read_text())
    split_of = {g["group_id"]: g["split"] for g in split["groups"]}
    dev_groups = {g["group_id"]: g for g in split["groups"] if g["split"] == "dev_group"}
    allow = json.loads(ALLOWLIST.read_text())["parents"]
    allow_by_group = {p["group_id"]: p for p in allow}
    dev_sources = json.loads(DEV_SOURCES.read_text())["entries"]
    dev75 = [json.loads(line) for line in DEV75.open()]

    # CPT groups by split: CPT should be train_group only.
    cpt_group_splits = defaultdict(int)
    for gid in cpt_by_group:
        cpt_group_splits[split_of.get(gid, "not_in_split")] += 1

    group_report = {}
    for gid, g in sorted(dev_groups.items()):
        pkgs = {f.split(":", 1)[1] for f in g["identity_forms"] if f.startswith("pkg:")}
        if gid in allow_by_group:
            pkgs.add(norm_name(allow_by_group[gid]["package"]))
        hits = {
            "group_id": len(cpt_by_group.get(gid, ())),
            "package": sorted(p for p in pkgs if p in cpt_by_pkg),
            "package_docs": sum(len(cpt_by_pkg[p]) for p in pkgs if p in cpt_by_pkg),
        }
        group_report[gid] = {
            "packages": sorted(pkgs),
            "in_parent_allowlist": gid in allow_by_group,
            "cpt_hits": hits,
            "overlaps_cpt": bool(hits["group_id"] or hits["package"]),
        }

    def doc_hits(pkg, path, hashes):
        by_hash = sorted({h for h in hashes if h and h in cpt_docs})
        by_path = sorted(cpt_by_pkg_path.get((norm_name(pkg), path), ()))
        return {"by_hash": by_hash, "by_package_path": by_path}

    dev75_report = []
    for r in dev75:
        # Provenance differs by family; take every document hash it carries.
        pkg = r["package_id"].removeprefix("pkg:")
        path = r["context"]["path"]
        hits = doc_hits(pkg, path, document_hashes(r["source_provenance"]))
        dev75_report.append({
            "id": r["id"],
            "family": r["family"],
            "group_id": r["group_id"],
            "package": pkg,
            "path": path,
            "group_overlaps_cpt": group_report.get(r["group_id"], {}).get("overlaps_cpt"),
            "document_in_cpt": bool(hits["by_hash"] or hits["by_package_path"]),
            "document_hits": hits,
        })

    snapshot_report = []
    for e in dev_sources:
        hits = doc_hits(e["package"], e.get("source_path") or "", [e.get("content_sha256"), e.get("post_edit_sha256")])
        if hits["by_hash"] or hits["by_package_path"]:
            snapshot_report.append({"id": e["id"], "package": e["package"], "document_hits": hits})

    dev75_groups = sorted({r["group_id"] for r in dev75})
    overlap_groups = sorted(g for g, v in group_report.items() if v["overlaps_cpt"])
    eligible = sorted(
        g for g, v in group_report.items()
        if v["in_parent_allowlist"] and g not in dev75_groups and not v["overlaps_cpt"]
    )

    receipt = {
        "card": "C05",
        "step": "2 CPT-overlap check",
        "final_files_opened": False,
        "inputs": {
            "cpt_cohort_manifest": {"path": str(COHORT), "sha256": sha256(COHORT)},
            "cpt_rows": {"path": str(rows_path), "sha256": rows_sha, "documents": len(cpt_docs)},
            "split": {"path": str(SPLIT), "sha256": sha256(SPLIT), "split_id": split["split_id"]},
            "dev_parent_allowlist": {"path": str(ALLOWLIST), "sha256": sha256(ALLOWLIST)},
            "dev_source_manifest": {"path": str(DEV_SOURCES), "sha256": sha256(DEV_SOURCES)},
            "dev75": {"path": str(DEV75), "sha256": sha256(DEV75), "cases": len(dev75)},
        },
        "method": (
            "Identity join on DAT-02 group_id and normalized package name "
            "(DAT-02 norm_name), and document join on source sha256 (pre- and "
            "post-edit) and on (package, package-relative path)."
        ),
        "cpt_groups_by_split": dict(sorted(cpt_group_splits.items())),
        "summary": {
            "dev_groups": len(dev_groups),
            "dev_groups_overlapping_cpt": len(overlap_groups),
            "dev75_groups": len(dev75_groups),
            "dev75_cases_with_group_overlap": sum(1 for r in dev75_report if r["group_overlaps_cpt"]),
            "dev75_cases_with_document_in_cpt": sum(1 for r in dev75_report if r["document_in_cpt"]),
            "dat07_dev_snapshots_in_cpt": len(snapshot_report),
            "parent_allowlist_groups": len(allow_by_group),
            "eligible_construction_groups": len(eligible),
        },
        "eligible_construction_groups": eligible,
        "dev_groups_overlapping_cpt": {g: group_report[g] for g in overlap_groups},
        "dev75": dev75_report,
        "dat07_dev_snapshots_in_cpt": snapshot_report,
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["summary"], indent=2))
    print(json.dumps(receipt["cpt_groups_by_split"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
