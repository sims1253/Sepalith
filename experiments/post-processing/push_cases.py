#!/usr/bin/env python3
"""Unified HF projection for the public scholzmx/sepalith dataset.

The NAS layout grew organically, so this script projects it into a clean
repository structure without moving local files:

  corpus/
    cran/licenses.jsonl, cran/manifest.jsonl
    cran/provenance/<shard>/<pkg>.json|.license.txt
                               per-package license/provenance, sharded by
                               first character (the Hub caps a folder at
                               10,000 entries; the era-1 flat `provenance/`
                               stopped exactly there)
    bioc/, stack_v2/, stack_v3/, pwc/
                               ledgers, manifests and reports only
  families/<family>/<source>.jsonl[.stats.json|.done.jsonl]
                               synthetic families, one file per author source
  mixtures/sft_vX/...          assembled train/eval (derived data)
  pretraining/                 the A2 training package

Eval protection (plan card C16). The projection never includes:

- the sealed final set or anything named after it (C00's forbidden tokens),
- `datasets/cases_v1/` and `datasets/edit_pairs_v1/`: the sealed final
  evaluation rows are lines inside these files (see
  docs/campaign/work/lead/seal_final_metadata.py), so they are no longer
  projected. Mined edit pairs are excluded for the same reason,
- the eval-protected git mirror (`git/`, `a2_code_mirror/`),
- raw CRAN package shards (`datasets/packages/`): source snapshots that
  predate the campaign split (CONTEXT.md: avoid upstream pools),
- any file whose name contains `final`.

`guard()` enforces these rules again on every (local, repo) pair and aborts
the run if one slips through, and on any repo path outside the documented
top-level folders.

Modes:

  push_cases.py --dry-run --root /mnt/nas/home/sepalith --plan-out plan.json
  push_cases.py --apply   --root /mnt/nas/home/sepalith --plan-out plan.json \
      [--readme README.md] [--receipt receipt.json]

`--root` (or SEPALITH_ROOT) defaults to /mnt/h/sepalith. The diff is taken
against the Hub's current tree (size, then sha256 for LFS files or the git
blob sha1 for small files), so reruns upload only what changed. Uploads are
batched `create_commit` calls (HF allows 128 commits per hour). Nothing on the
Hub is deleted unless `--delete-legacy` is given. The README is uploaded only
when `--readme` names a file.

HF_TOKEN is read from the environment, falling back to the cached `hf` login.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi

REPO = "scholzmx/sepalith"
STATE = Path(__file__).parent / ".push_cases_state.json"

# Documented top-level folders. `provenance/` and `datasets/` hold the era-1
# CRAN upload; this script no longer writes there but the guard allows them.
ALLOWED_TOP = {"corpus", "families", "mixtures", "pretraining", "provenance",
               "datasets", "README.md"}
# C00's forbidden tokens (docs/training/plan-20260927/receipts/upload_secure_20260927.py)
FORBIDDEN_TOKENS = ("dat-07-final", "dat-08", "sealed-final", "sealed_final",
                    "final-evaluator", "final_eval")
# Sealed-final source pools and eval-protected mirrors (path components).
FORBIDDEN_DIRS = ("cases_v1", "edit_pairs_v1", "edit_pairs", "edit-pairs",
                  "git", "a2_code_mirror", "packages", "campaign-20260915")
# CRAN packages whose names contain "final": license metadata records, not
# evaluation files. Exact root-relative paths only.
FINAL_NAME_EXEMPT = {"provenance/finalfit.json", "provenance/finalsize.json"}

# backend/model -> canonical source label
SOURCE = {
    "zai": "glm-5.3", "agy": "gemini-3.7-flash",
    "opencode-spark": "muse-spark-1.2", "opencode-spark-free": "muse-spark-1.2",
    "xpreview-free": "x-preview", "sparkfree": "muse-spark-1.2",
    "spark": "muse-spark-1.2",
}
OPENROUTER_MODEL_RE = re.compile(r"orfree_(.+)\.jsonl$")

# exact filenames whose author-source the suffix rules can't see
EXPLICIT_SOURCE = {
    "rewrite_lint_fix.jsonl": "glm-5.3",        # zai quota-burn wave
    "fix_issue_inject.jsonl": "glm-5.3",
    "analyst_direct.jsonl": "glm-5.3", "analyst_gemini.jsonl": "gemini-3.7-flash",
    "comment_to_code_gemini.jsonl": "gemini-3.7-flash",
}


def source_label(fname: str) -> str:
    if fname in EXPLICIT_SOURCE:
        return EXPLICIT_SOURCE[fname]
    m = OPENROUTER_MODEL_RE.search(fname)
    if m:
        return m.group(1).replace("--", "/")
    for key, lab in SOURCE.items():
        if fname.endswith(f"_{key}.jsonl") or f"_{key}." in fname:
            return lab
    return "corpus"


def fam_of(fname: str) -> str:
    """Wave names embed the author; strip it for the family."""
    stem = fname
    for key in ["_sparkfree", "_xpreview-free"] + [f"_{k}" for k in SOURCE]:
        stem = stem.replace(key, "")
    m = OPENROUTER_MODEL_RE.search(stem)
    if m:
        stem = stem[: m.start()] + stem[m.end():]
    return stem.replace(".jsonl", "").rstrip("_-") or "misc"


def shard_of(name: str) -> str:
    c = name[:1].lower()
    return c if c.isalnum() else "_"


def projection(root: Path) -> list[tuple[Path, str]]:
    """(local file, repo path) pairs: everything this script may upload."""
    nas = root / "datasets"
    out: list[tuple[Path, str]] = []

    # -- corpus: CRAN ledgers + complete, sharded provenance --
    out.append((nas / "license_texts.jsonl", "corpus/cran/licenses.jsonl"))
    out.append((nas / "manifest.jsonl", "corpus/cran/manifest.jsonl"))
    prov = root / "provenance"
    if prov.exists():
        for p in sorted(prov.iterdir()):
            if p.is_file() and p.name.endswith((".json", ".license.txt")):
                out.append((p, f"corpus/cran/provenance/{shard_of(p.name)}/{p.name}"))

    # -- families: scenarios_v1 (never final-capable in DAT-02) --
    # cases_v1 is deliberately absent: it holds the sealed final source rows.
    base = nas / "scenarios_v1"
    if base.exists():
        for f in sorted(base.iterdir()):
            if ("contaminated" in f.name or ".partial." in f.name
                    or f.name.endswith((".done.jsonl", ".stats.json"))
                    or f.name in ("stats.json", "suffix_scenarios.stats.json")):
                continue  # sidecars ride via their .jsonl twin; partials are transient
            if f.suffix != ".jsonl":
                continue
            fam, src = fam_of(f.name), source_label(f.name)
            out.append((f, f"families/{fam}/{src}.jsonl"))
            for suf in (".done.jsonl", ".stats.json"):
                side = base / (f.name + suf)
                if side.exists():
                    out.append((side, f"families/{fam}/{src}{suf}"))

    sa = nas / "synthetic_analyst_v1"
    if sa.exists():
        for f in sorted(sa.glob("*.jsonl")):
            out.append((f, f"families/synthetic_analyst/{f.name}"))

    # -- mixtures --
    for v in ("sft_v3", "sft_v4", "sft_v5", "sft_v6", "sft_v7",
              "sft_v8_1", "sft_v8_2"):
        d = nas / v
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl")) + sorted(d.glob("stats.json")):
            out.append((f, f"mixtures/{v}/{f.name}"))

    # -- ledgers/manifests for the other acquired corpora --
    for src_rel, dst in [
        ("bioc_staging/bioc_license_ledger.jsonl", "corpus/bioc/license_ledger.jsonl"),
        ("bioc_staging/manifest.jsonl", "corpus/bioc/manifest.jsonl"),
        ("stack_staging/license_ledger.jsonl", "corpus/stack_v2/license_ledger.jsonl"),
        ("stack_staging/measure_report.json", "corpus/stack_v2/measure_report.json"),
        ("stack_staging/logs/phase_c_report.json", "corpus/stack_v2/dedup_report.json"),
        ("stack_staging/logs/phase_e_report.json", "corpus/stack_v3/acceptance_report.json"),
        ("pwc_staging/pwc_license_ledger.jsonl", "corpus/pwc/license_ledger.jsonl"),
        ("pwc_staging/measure_report.json", "corpus/pwc/measure_report.json"),
        ("pwc_staging/manifest.jsonl", "corpus/pwc/manifest.jsonl"),
    ]:
        if (root / src_rel).exists():
            out.append((root / src_rel, dst))

    # -- pretraining/: the A2 training package (2026-08-26) --
    a2r = root / "a2" / "r"
    if a2r.exists():
        for npy in sorted(a2r.glob("*.npy")):
            out.append((npy, f"pretraining/a2/r/{npy.name}"))
        for j in ("stats.json", "contamination.json"):
            if (a2r / j).exists():
                out.append((a2r / j, f"pretraining/a2/r/{j}"))
    tokf = nas / "a2_tokenizer_v1" / "tokenizer.json"
    if tokf.exists():
        out.append((tokf, "pretraining/datasets/a2_tokenizer_v1/tokenizer.json"))
    transfers = root / "a2_transfers"
    if transfers.exists():
        for d in sorted(transfers.iterdir()):
            if not (d / "blocks.npy").exists():
                continue
            out.append((d / "blocks.npy", f"pretraining/a2_transfers/{d.name}/blocks.npy"))
            if (d / "stats.json").exists():
                out.append((d / "stats.json", f"pretraining/a2_transfers/{d.name}/stats.json"))
    if (root / "a2_mixture_manifest.json").exists():
        out.append((root / "a2_mixture_manifest.json", "pretraining/a2_mixture_manifest.json"))
    return out


def guard(pairs: list[tuple[Path, str]], root: Path) -> list[str]:
    """Return every eval-protection or layout violation (empty = clean)."""
    bad = []
    for local, rel in pairs:
        low_local = str(local).lower()
        low_rel = rel.lower()
        top = rel.split("/", 1)[0]
        try:
            parts = local.resolve().relative_to(root.resolve()).parts
        except ValueError:
            bad.append(f"outside root: {local}")
            continue
        if top not in ALLOWED_TOP or top in ("provenance", "datasets"):
            bad.append(f"repo path outside the projected folders: {rel}")
        if any(t in low_local or t in low_rel for t in FORBIDDEN_TOKENS):
            bad.append(f"sealed-final token: {local} -> {rel}")
        if any(p in FORBIDDEN_DIRS for p in parts[:-1]):
            bad.append(f"sealed-final source or eval-protected mirror: {local}")
        if (("final" in local.name.lower() or "final" in rel.rsplit("/", 1)[-1].lower())
                and "/".join(parts) not in FINAL_NAME_EXEMPT):
            bad.append(f"final-named file: {local} -> {rel}")
    return bad


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def hub_tree(api: "HfApi") -> tuple[str, dict[str, dict]]:
    sha = api.dataset_info(REPO).sha
    files = {}
    for it in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, revision=sha):
        if not hasattr(it, "size"):
            continue
        lfs = getattr(it, "lfs", None)
        files[it.path] = {"size": it.size, "blob_id": it.blob_id,
                          "sha256": (lfs.sha256 if lfs is not None else None)}
    return sha, files


def same_content(local: Path, remote: dict) -> bool:
    if remote["sha256"]:
        return sha256_file(local) == remote["sha256"]
    return git_blob_sha1(local) == remote["blob_id"]


def diff(pairs: list[tuple[Path, str]], remote: dict[str, dict], verify_all: bool) -> dict:
    """Classify each existing projected file as new, changed or unchanged."""
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    res = {"new": [], "changed": [], "unchanged": [], "missing_local": []}
    for local, rel in pairs:
        if not local.exists():
            res["missing_local"].append(rel)
            continue
        st = local.stat()
        r = remote.get(rel)
        entry = {"local": str(local), "repo": rel, "size": st.st_size}
        if r is None:
            res["new"].append(entry)
        elif r["size"] != st.st_size:
            res["changed"].append(entry)
        elif (not verify_all and state.get(rel) == {"size": st.st_size, "mtime": st.st_mtime}):
            res["unchanged"].append(entry)  # same size, same mtime as the last push
        elif same_content(local, r):
            res["unchanged"].append(entry)
        else:
            res["changed"].append(entry)
    return res


def summarize(entries: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for e in entries:
        parts = e["repo"].split("/")
        key = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
        s = out.setdefault(key, {"files": 0, "bytes": 0})
        s["files"] += 1
        s["bytes"] += e["size"]
    return dict(sorted(out.items()))


def batches(entries: list[dict], max_files: int, max_bytes: int) -> Iterator[list[dict]]:
    """Group uploads by top folder, then cap each commit by count and bytes."""
    cur, cur_bytes, cur_top = [], 0, None
    for e in sorted(entries, key=lambda e: e["repo"]):
        top = e["repo"].split("/", 1)[0]
        if cur and (top != cur_top or len(cur) >= max_files
                    or cur_bytes + e["size"] > max_bytes):
            yield cur
            cur, cur_bytes = [], 0
        cur.append(e)
        cur_bytes += e["size"]
        cur_top = top
    if cur:
        yield cur


def legacy_paths() -> list[str]:
    return ["datasets/scenarios_v1", "datasets/synthetic_analyst_v1", "datasets/sft_v3",
            "synthetic/scenarios_v1", "synthetic/synthetic_analyst_v1",
            "synthetic/cases_v1", "synthetic/sft_v5", "synthetic/sft_v6",
            "synthetic/analyst_direct.jsonl", "synthetic/analyst_scripts.jsonl",
            "synthetic/na_rm_contexts.jsonl", "synthetic/paper_to_r.jsonl",
            "scenarios", "edit-pairs", "finish-block", "hidden-r",
            "license_texts.jsonl", "provenance_manifest.jsonl"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="diff against the Hub, write the plan, upload nothing")
    mode.add_argument("--apply", action="store_true", help="upload new and changed files in batched commits")
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("SEPALITH_ROOT", "/mnt/h/sepalith")),
                    help="NAS sepalith root (holds datasets/, provenance/, a2/, ...)")
    ap.add_argument("--plan-out", type=Path, help="write the diff plan as JSON")
    ap.add_argument("--receipt", type=Path, help="write the upload receipt as JSON (--apply)")
    ap.add_argument("--readme", type=Path, help="upload this file as README.md in the last commit")
    ap.add_argument("--verify-all", action="store_true", help="hash every same-size file instead of trusting the state file")
    ap.add_argument("--batch-files", type=int, default=5000)
    ap.add_argument("--batch-bytes", type=int, default=8 << 30)
    ap.add_argument("--max-commits", type=int, default=20)
    ap.add_argument("--max-bytes", type=int, default=50 * 10**9, help="refuse plans above this (card C16: 50 GB)")
    ap.add_argument("--delete-legacy", action="store_true", help="delete era-1/2 legacy repo paths (needs the user's approval)")
    ap.add_argument("--message", default="Re-sync the dataset projection")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    root = args.root
    if not (root / "datasets").is_dir():
        sys.exit(f"--root {root} has no datasets/ folder")
    pairs = projection(root)
    bad = guard(pairs, root)
    if args.readme:
        bad += [b for b in guard([(args.readme.resolve(), "README.md")], args.readme.resolve().parent)
                if "outside" not in b]
    if bad:
        print("eval-protection guard FAILED:", *bad[:50], sep="\n  ")
        return 2
    print(f"projection: {len(pairs)} files; guard clean", flush=True)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    base_sha, remote = hub_tree(api)
    plan = diff(pairs, remote, args.verify_all)
    upload = plan["new"] + plan["changed"]
    total = sum(e["size"] for e in upload)
    report = {
        "repo": REPO, "base_revision": base_sha, "root": str(root),
        "projection_files": len(pairs), "guard_violations": 0,
        "counts": {k: len(v) for k, v in plan.items()},
        "upload_bytes": total,
        "new_by_folder": summarize(plan["new"]),
        "changed_by_folder": summarize(plan["changed"]),
        "missing_local": plan["missing_local"],
        "hub_files_outside_projection": len(set(remote) - {r for _, r in pairs}),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "missing_local"}, indent=1))
    if args.plan_out:
        args.plan_out.write_text(json.dumps({**report, "upload": upload}, indent=1))
    if total > args.max_bytes:
        print(f"refusing: plan uploads {total} bytes > {args.max_bytes}")
        return 3
    if args.dry_run:
        return 0

    groups = list(batches(upload, args.batch_files, args.batch_bytes))
    n_commits = len(groups) + (1 if args.readme and not groups else 0)
    if n_commits > args.max_commits:
        print(f"refusing: {n_commits} commits > --max-commits {args.max_commits}")
        return 3

    commits = []
    for i, group in enumerate(groups, 1):
        ops = [CommitOperationAdd(path_in_repo=e["repo"], path_or_fileobj=e["local"]) for e in group]
        last = i == len(groups)
        if last and args.readme:
            ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(args.readme)))
        top = group[0]["repo"].split("/", 1)[0]
        msg = f"{args.message}: {top}/ ({len(group)} files, batch {i}/{len(groups)})"
        info = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                                 commit_message=msg, parent_commit=commits[-1]["oid"] if commits else base_sha)
        commits.append({"oid": info.oid, "message": msg, "files": len(ops),
                        "bytes": sum(e["size"] for e in group), "by_folder": summarize(group)})
        print(f"committed {info.oid[:10]} {msg}", flush=True)
        time.sleep(2)
    if args.readme and not groups:
        info = api.create_commit(repo_id=REPO, repo_type="dataset", commit_message=f"{args.message}: README",
                                 operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(args.readme))],
                                 parent_commit=base_sha)
        commits.append({"oid": info.oid, "message": "README", "files": 1, "bytes": args.readme.stat().st_size})

    if args.delete_legacy:
        for path in legacy_paths():
            try:
                api.delete_folder(repo_id=REPO, repo_type="dataset", path_in_repo=path)
                print(f"deleted legacy {path}/", flush=True)
            except Exception as e:  # noqa: BLE001 - missing folders are expected
                if "not found" not in str(e).lower() and "404" not in str(e):
                    print(f"(skip {path}: {str(e)[:80]})", flush=True)

    # record what is now on the Hub so the next dry run can skip hashing
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for e in upload + plan["unchanged"]:
        st = Path(e["local"]).stat()
        state[e["repo"]] = {"size": st.st_size, "mtime": st.st_mtime}
    STATE.write_text(json.dumps(dict(sorted(state.items())), indent=1))
    if args.receipt:
        args.receipt.write_text(json.dumps({**report, "commits": commits}, indent=1))
    print(f"done: {len(commits)} commits, {len(upload)} files, {total} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
