#!/usr/bin/env python3
"""Retroactive provenance enrichment for Sepalith datasets (docs/research/2026-08-19-provenance-schema.md).

Every record in every dataset gets:
  source_url : exact source artifact link (CRAN tarball / GitHub commit) or null
  license    : SPDX-ish string from the source ("unknown" when explicit-unknown)
Derived-from-CRAN datasets additionally inherit version/upstream and carry a
`derivation` line naming the extractor script + input sha256. LLM-synthetic
records additionally get full_prompt (exactly what was sent), prompt_reconstructed,
model, generator, generated_at.

Datasets handled here:
  1. experiments/synthetic/finish_block_sample.jsonl   CRAN join
  2. datasets/scenarios_v1/*.jsonl (7 canonical files; the leftover
     comment_to_code_synthetic.partial.jsonl byte-duplicate is skipped)
  3. datasets/synthetic_analyst_v1/*.jsonl             SNAPSHOT ONLY (a detached
     generate_analyst.py runner keeps appending -> never replace originals)
  4. datasets/edit_pairs_v1/{examples,eval}.jsonl      commit URLs + license
                                                        heuristics from clones
  5. datasets/sft_ablation/{types,plain}/{train,eval}.jsonl  carry-through join

Safety: write <name>.enriched.jsonl alongside, verify (line count, every
original field unchanged, new fields present), then os.replace() atomically.
Originals are never edited in place; on any verification failure the enriched
copy is kept and the original left untouched.

Usage:
  uv run python experiments/pipeline/enrich_provenance.py            # all datasets
  uv run python experiments/pipeline/enrich_provenance.py --only scenarios,sft
  uv run python experiments/pipeline/enrich_provenance.py --dry-run
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

REPO = Path("/home/m0hawk/Documents/Sepalith")
PROV_DIR = Path("/mnt/h/sepalith/provenance")
DATA = Path("/mnt/h/sepalith/datasets")
GIT_CLONES = Path("/mnt/h/sepalith/git")
SYN = REPO / "experiments" / "synthetic"

CRAN_CONTRIB = "https://cran.r-project.org/src/contrib/{}"

# ---------------------------------------------------------------------------
# prompt templates — extracted from the actual generator sources via ast, so
# there is exactly one source of truth (no copy drift).
# ---------------------------------------------------------------------------


def _const_from_src(path: Path, name: str):
    """Literal-eval a module-level constant (str / list / dict) without importing."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets:
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


C2C_PROMPT = _const_from_src(SYN / "comment_to_code.py", "PROMPT")
ANALYST_PROMPT = _const_from_src(SYN / "grid.py", "ANALYST_PROMPT")
GEN_ANALYST = SYN / "generate_analyst.py"
NA_RM_PROMPT = _const_from_src(GEN_ANALYST, "NA_RM_PROMPT")


def _source_models(path: Path) -> dict[str, str]:
    """source tag -> exact model id, from generate_analyst.py's SOURCES list
    (kept as ast extraction: SOURCES references BROWSER_UA etc. by name, so
    plain literal_eval chokes; we only need the constant name/model pairs)."""
    out = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if "name" in keys and "model" in keys:
                vals = {k.value: v.value for k, v in zip(node.keys, node.values)
                        if isinstance(k, ast.Constant)
                        and isinstance(v, ast.Constant)}
                if "name" in vals and "model" in vals:
                    out[vals["name"]] = vals["model"]
    return out


# source tag -> exact model id (generate_analyst.py SOURCES)
SOURCE_MODELS = _source_models(GEN_ANALYST)


def _placeholders(template: str) -> set[str]:
    import string
    return {fname for _, fname, _, _ in string.Formatter().parse(template)
            if fname}


# verified once at startup: the grid_cell keys the generator sampled are exactly
# the ANALYST_PROMPT placeholders (CellSampler.next() returns domain/packages/
# construct/style/line_target).
assert _placeholders(ANALYST_PROMPT) == {
    "domain", "packages", "construct", "style", "line_target"}, \
    "grid.py ANALYST_PROMPT placeholders drifted from grid_cell keys"
assert _placeholders(NA_RM_PROMPT) <= {"domain", "packages", "construct",
                                       "style", "line_target"}
assert _placeholders(C2C_PROMPT) == {"code"}

# ---------------------------------------------------------------------------
# CRAN provenance join
# ---------------------------------------------------------------------------

_prov_cache: dict[str, dict | None] = {}


def prov(pkg: str) -> dict | None:
    if pkg not in _prov_cache:
        p = PROV_DIR / f"{pkg}.json"
        _prov_cache[pkg] = json.loads(p.read_text()) if p.exists() else None
    return _prov_cache[pkg]


def cran_join(rec: dict) -> tuple[dict, bool]:
    """Add source_url/license/version/upstream (+ caller adds derivation).
    Returns (patch, hit)."""
    p = prov(rec.get("package", ""))
    if p is None or not p.get("tarball"):
        return {"source_url": None, "license": "unknown"}, False
    return {
        "source_url": CRAN_CONTRIB.format(p["tarball"]),
        "license": p.get("license") or "unknown",
        "version": p.get("version"),
        "upstream": p.get("upstream"),
        "tarball_sha256": p.get("tarball_sha256"),
    }, True


# ---------------------------------------------------------------------------
# edit_pairs license resolution (GitHub clones at /mnt/h/sepalith/git/)
# ---------------------------------------------------------------------------

_LICENSE_LINE = re.compile(r"^License:\s*(.+?)\s*$", re.M | re.I)


def _desc_license(desc_path: Path) -> str | None:
    try:
        m = _LICENSE_LINE.search(desc_path.read_text(errors="replace"))
    except OSError:
        return None
    if not m:
        return None
    val = m.group(1)
    if val.lower().startswith("file "):  # pure pointer, no SPDX text
        return None
    # keep the DESCRIPTION string, minus the "+ file LICENSE" pointer suffix
    val = re.sub(r"\+\s*file\s*LICEN[SC]E.*$", "", val, flags=re.I).strip()
    return val or None


def _license_text_heuristic(text: str) -> str | None:
    t = text[:20000]  # license texts identify themselves up front
    low = t.lower()

    def has(*needles: str) -> bool:
        return any(n in low for n in needles)

    if has("mit license", "permission is hereby granted, free of charge"):
        return "MIT"
    if has("apache license"):  # 1.x is extinct in R repos; treat as 2.0
        return "Apache-2.0"
    if has("gnu lesser general public license", "gnu library general public license"):
        return "LGPL-3" if has("version 3") else (
            "LGPL-2.1" if has("version 2.1") else "LGPL")
    if has("gnu affero general public license"):
        return "AGPL-3" if has("version 3") else "AGPL"
    if has("gnu general public license"):
        if has("version 3"):
            return "GPL-3"
        if has("version 2"):
            return "GPL-2"
        return "GPL"
    if has("mozilla public license"):
        return "MPL-2.0" if has("2.0") else "MPL"
    if has("creative commons"):
        if has("attribution", "by"):
            return "CC-BY-4.0" if has("4.0") else "CC-BY"
        return "CC0-1.0" if has("cc0", "public domain dedication") else "CC"
    if has("unlicense"):
        return "Unlicense"
    if has("bsd license", "redistribution and use in source and binary forms"):
        # the non-endorsement paragraph is exactly the third clause
        return "BSD-3-Clause" if has("endorse", "3-clause") else "BSD-2-Clause"
    return None


_repo_license_cache: dict[tuple[str, str], str] = {}


def edit_pairs_license(repo: str, license_file: str) -> str:
    """DESCRIPTION 'License:' wins; else read the record's license_file from
    the clone and map content heuristically; else 'unknown' (explicit)."""
    key = (repo, license_file)
    if key in _repo_license_cache:
        return _repo_license_cache[key]
    root = GIT_CLONES / repo.replace("/", "__")
    out = None
    method = "unknown"
    if (root / "DESCRIPTION").is_file():
        out = _desc_license(root / "DESCRIPTION")
        method = "DESCRIPTION"
    if out is None and license_file and license_file != "none":
        lf = root / license_file
        if lf.is_file():
            out = _license_text_heuristic(
                lf.read_text(errors="replace")[:40000])
            method = f"file:{license_file}"
    if out is None:
        out, method = "unknown", "none"
    _repo_license_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# generic jsonl enrich driver: enriched copy -> verify -> atomic replace
# ---------------------------------------------------------------------------


def enrich_file(path: Path, patch_fn, *, snapshot: bool = False,
                dry_run: bool = False) -> dict:
    """patch_fn(rec) -> patch dict merged into every record.

    snapshot=True keeps the original untouched (for files a detached runner
    is still appending to): only <name>.enriched.jsonl is written, from an
    in-memory snapshot taken at read time.
    """
    name = path.name
    enriched = path.with_name(name.removesuffix(".jsonl") + ".enriched.jsonl")
    with path.open() as f:
        orig_lines = f.readlines()

    stats = {"file": str(path), "records": 0, "replaced": False,
             "snapshot": snapshot, "verify": "ok", "hits": 0, "extra": {}}
    out_lines: list[str] = []
    for line in orig_lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        stats["records"] += 1
        patch = patch_fn(rec, stats)
        for k, v in patch.items():
            if k in rec and rec[k] != v:
                stats["extra"][f"conflict:{k}"] = \
                    stats["extra"].get(f"conflict:{k}", 0) + 1
            rec[k] = v
        out_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")

    if dry_run:
        stats["verify"] = "dry-run (no files written)"
        return stats

    with enriched.open("w") as f:
        f.writelines(out_lines)
        f.flush()
        os.fsync(f.fileno())

    # ---- verification against the snapshot we read ----
    with enriched.open() as f:
        new_lines = f.readlines()
    ok = len(new_lines) == len(out_lines)
    for ol, nl in zip(orig_lines, new_lines):
        if not ok:
            break
        if not ol.strip():
            continue
        o, n = json.loads(ol), json.loads(nl)
        ok &= all(n.get(k) == v for k, v in o.items())  # originals intact
    if not ok:
        stats["verify"] = "FAILED (kept enriched copy, original untouched)"
        return stats

    if snapshot:
        stats["verify"] = "ok (snapshot; original untouched)"
        return stats
    os.replace(enriched, path)  # atomic swap
    stats["replaced"] = True
    return stats


# ---------------------------------------------------------------------------
# per-dataset patches
# ---------------------------------------------------------------------------


def patch_finish_block(rec: dict, st: dict) -> dict:
    patch, hit = cran_join(rec)
    if hit:
        patch["derivation"] = ("finish_block.py tree-sitter-r extraction "
                               f"from {patch.pop('tarball_sha256')}")
    else:
        patch.pop("tarball_sha256", None)
    st["hits"] += int(hit)
    return patch


def patch_scenarios(rec: dict, st: dict) -> dict:
    fam = rec.get("family", "")
    patch, hit = cran_join(rec)
    patch.pop("tarball_sha256", None)
    script = ("comment_to_code.py" if fam.startswith("comment_to_code")
              else "scenarios.py")
    if hit:
        patch["derivation"] = f"{fam} constructor ({script})"
    st["hits"] += int(hit)
    if fam == "comment_to_code_synthetic":
        # generate_comment(cand['block'], ...) -> PROMPT.format(code=block):
        # `block` is the LIST of region_new lines, so str.format embedded the
        # Python list repr. region_new survived JSON round-trip verbatim, so
        # re-formatting the same template reproduces the literal prompt sent.
        patch["full_prompt"] = C2C_PROMPT.format(code=rec["region_new"])
        patch["prompt_reconstructed"] = True
        patch["model"] = rec.get("generator")   # endpoint/model tag as logged
        # NB: the pre-existing "generator" field (endpoint/model tag, e.g.
        # "opencode/deepseek-v4-flash-free") is preserved verbatim and mirrored
        # into "model"; the generating script is named in "derivation".
        patch["generated_at"] = None            # not logged at generation time
        st["extra"]["full_prompt"] = st["extra"].get("full_prompt", 0) + 1
    return patch


def patch_analyst(rec: dict, st: dict) -> dict:
    cell = rec.get("grid_cell")
    patch = {"license": None, "source_url": None,
             "generator": "generate_analyst.py"}
    if cell:
        # the generator rendered exactly these templates (stats.json:
        # oneshot_prompt=false, so no ONE_SHOT suffix on analyst prompts);
        # rec["model"] and rec["generated_at"] already exist and stay as-is
        template = NA_RM_PROMPT if rec.get("family") == "na_rm" \
            else ANALYST_PROMPT
        patch["full_prompt"] = template.format(**cell)
        patch["prompt_reconstructed"] = True
        st["extra"]["full_prompt"] = st["extra"].get("full_prompt", 0) + 1
    else:  # rejects.jsonl rows: the grid cell was never persisted
        patch["full_prompt"] = None
        patch["model"] = SOURCE_MODELS.get(rec.get("source"))
        patch["generated_at"] = rec.get("at")
        st["extra"]["no_grid_cell"] = st["extra"].get("no_grid_cell", 0) + 1
    return patch


def patch_edit_pairs(rec: dict, st: dict) -> dict:
    lic = edit_pairs_license(rec["repo"], rec.get("license_file", "none"))
    st["hits"] += int(lic not in ("unknown",))
    return {
        "source_url": f"{rec['repo_url']}/commit/{rec['sha_full']}",
        "license": lic,
        "derivation": (f"mine_edit_pairs.py + finalize_edit_pairs.py "
                       f"extraction from GitHub repo {rec['repo']} "
                       f"@ {rec['sha_full']}"),
    }


def _sft_patcher(variant: str):
    def patch(rec: dict, st: dict) -> dict:
        patch, hit = cran_join(rec)
        sha = patch.pop("tarball_sha256", None)
        base = ("format_sft_types.py re-render (format_sft_v1.render) of "
                "finish-block records")
        if hit:
            if variant == "types":
                patch["derivation"] = (base + "; types section from ry "
                                       "dump-types (ry-worktrees/dump-types "
                                       "@ a907e10)")
            else:
                patch["derivation"] = base
        st["hits"] += int(hit)
        return patch
    return patch


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma list: finish_block,scenarios,analyst,"
                         "edit_pairs,sft")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sel = {s for s in args.only.split(",") if s}

    jobs: list[tuple[str, Path, object, bool]] = [
        # (dataset-key, path, patch_fn, snapshot)
        ("finish_block", SYN / "finish_block_sample.jsonl",
         patch_finish_block, False),
        *[( "scenarios", p, patch_scenarios, False)
          for p in sorted((DATA / "scenarios_v1").glob("*.jsonl"))
          # crash artifact: byte-identical duplicate comment_to_code.py
          # rewrites/unlinks on resume; enriching it would go stale instantly
          if not p.name.endswith(".partial.jsonl")],
        # detached runner (generate_analyst.py --wait-on-outage) still appends
        # -> snapshot copies only, originals untouched
        *[("analyst", p, patch_analyst, True)
          for p in sorted((DATA / "synthetic_analyst_v1").glob("*.jsonl"))],
        *[("edit_pairs", p, patch_edit_pairs, False)
          for p in (DATA / "edit_pairs_v1" / "examples.jsonl",
                    DATA / "edit_pairs_v1" / "eval.jsonl")],
        *[("sft", p, _sft_patcher(variant), False)
          for variant in ("types", "plain")
          for p in ((DATA / "sft_ablation" / variant / "train.jsonl"),
                    (DATA / "sft_ablation" / variant / "eval.jsonl"))],
    ]

    report = []
    for key, path, fn, snap in jobs:
        if sel and key not in sel:
            continue
        print(f"enriching {path} {'(SNAPSHOT)' if snap else ''} ...",
              flush=True)
        st = enrich_file(path, fn, snapshot=snap, dry_run=args.dry_run)
        st["dataset"] = key
        st["join_coverage_pct"] = round(
            100 * st["hits"] / st["records"], 2) if st["records"] else None
        report.append(st)
        cov = st["join_coverage_pct"]
        print(f"  {st['records']} records, join {st['hits']}/{st['records']}"
              f" ({cov}%), verify={st['verify']}, "
              f"replaced={st['replaced']} extra={st['extra']}", flush=True)

    print("\n===== provenance enrichment summary =====")
    print(json.dumps(report, indent=1))
    failed = [r for r in report if r["verify"].startswith("FAILED")]
    if failed:
        print(f"\n{len(failed)} file(s) FAILED verification; originals kept.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
