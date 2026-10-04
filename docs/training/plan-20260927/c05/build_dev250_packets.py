"""C05 DEV250, phase 1: candidate conversion packets.

Turns every DEV250 candidate into a conversion packet ({row_ref, result,
family, meta}) with the same converters DEV75 used. Phase 2
(build_dev250.py) applies the PRM05 source window, renders PRM03, allocates
under the 3-per-group cap and writes the panel.

Sources, per family:
- finish_block: author_dev_finish.py selections (55, fixed) and every
  cut_dev_finish.py base function, through the DAT-04B completion adapter
  (finish_block_v5_prefix) plus DEV75's outer-brace correction;
- no_op: campaign_dev_panel already-applied `|>` / `na.rm = TRUE` lines;
- pipe_rewrite, rename_propagation, format_propagation: the DAT-07
  propagation constructor and converter;
- na_rm_propagation: the same, in DEV75's na_rm groups, skipping DEV75's
  own cases (user-approved deviation, status file 2026-10-03);
- roxygen_drafting: existing dev rows from the DAT-03 audit.

Run with a Python that has tree_sitter_r (no tokenizer needed):

    python3 build_dev250_packets.py OUT_DIR
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
AUTHORED = DATA / "C05-dev-finish-authored-v1"
CUT = DATA / "C05-dev-finish-cut-v1"
DEV75 = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/"
             "lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl")
REPAIR = REPO / "docs/campaign/work/finish-boundary-repair-v2/finish_boundary_repair_v2.py"

PER_GROUP = 3      # candidates kept per (family, group); the panel cap is 3 per group
NA_RM_SEEDS = 40   # extractor seeds tried in DEV75's na_rm groups
CUT_SEED = 20261003

# The converters run against the pinned protocol DEV75 was built with; phase
# 2 renders with the repo's copy, which reproduces all 75 DEV75 hashes.
sys.path.insert(0, str(EXEC.parents[1] / "packages/sepalith/src"))
sys.path.insert(0, str(EXEC))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


completion = load("campaign_admission_completion", EXEC / "campaign_admission_completion.py")
prop = load("campaign_dev_propagation", EXEC / "campaign_dev_propagation.py")
panel = load("campaign_dev_panel", EXEC / "campaign_dev_panel.py")
repair = load("finish_boundary_repair_v2", REPAIR)
scenarios = prop.scenarios
protocol = sys.modules["sepalith.campaign_protocol"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes())


class Rows:
    """Derived constructor rows, referenced by packets through file/line."""

    def __init__(self, path: Path):
        self.path, self.lines = path, []

    def add(self, row) -> int:
        self.lines.append((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())
        return len(self.lines)

    def write(self):
        self.path.write_bytes(b"".join(self.lines))
        return file_sha(self.path)


# ---------------------------------------------------------------- finish_block

def finish_packet(rows_file: Path, line_no: int, group_id: str, package_id: str, meta: dict):
    with rows_file.open("rb") as f:
        for i, line in enumerate(f, 1):
            if i == line_no:
                break
    raw = json.loads(line)
    row_id = f"c05-finish-{sha256(f'{rows_file}:{line_no}'.encode())[:24]}"
    ref = {
        "file": str(rows_file), "line": line_no,
        "raw_line_sha256": sha256(line), "source_sha256": file_sha(rows_file),
        "split": "dev_group", "row_id": row_id, "group_id": group_id,
        "document_role": "source_derived_simulated_pre_edit",
        "source_constructor": "finish_block_v5_prefix",
        "target_convention": "suffix",
        "uri": f"file:///sepalith/dev/{row_id}/{raw['path']}",
        "document_version": 0,
    }
    result = completion.convert_completion(raw, ref)
    if result["status"] != "converted":
        return None, f"finish:{result['reason']}"
    # DEV75's correction: the label lacks the function's outer brace.
    target = "\n".join(result["target_body"])
    if "\n" not in target or target.rsplit("\n", 1)[-1].strip(" \t") != "":
        return None, "finish:target_not_brace_correctable"
    before = result["selection_source"]["document_text"]
    ctx = result["context"]

    def splice(text):
        return repair.apply_document_replacement(
            before, ctx["replacement_range"], text, region_old=ctx["region_old"],
            utf16_to_codepoint_column=protocol.utf16_to_codepoint_column)

    old_post, new_post = splice(target), splice(target + "}")
    if not scenarios.parser.parse(old_post.encode()).root_node.has_error:
        return None, "finish:uncorrected_document_already_parses"
    if scenarios.parser.parse(new_post.encode()).root_node.has_error:
        return None, "finish:corrected_document_does_not_parse"
    result["target_body"] = (target + "}").split("\n")
    result["provenance"]["finish_splice"] = {
        "literal_source_splice_verified": True, "outer_closing_brace_in_label": True}
    result["provenance"]["correction"] = {
        "rule": "DEV75 finish-boundary correction: append the outer brace",
        "old_target_body_sha256": sha256(target.encode()),
        "new_target_body_sha256": sha256((target + "}").encode()),
        "new_post_document_sha256": sha256(new_post.encode()),
    }
    ref = {k: ref[k] for k in ("row_id", "split", "group_id", "file", "line",
                               "raw_line_sha256", "source_sha256")}
    ref["package_id"] = package_id
    return {"family": "finish_block", "row_ref": ref, "result": result, "meta": meta}, None


def finish_candidates(out, reasons):
    packets = []
    for line in (AUTHORED / "selected-55.jsonl").open():
        s = json.loads(line)
        group = "c05-" + s["group"].replace(":", "-")
        p, why = finish_packet(Path(s["rows_file"]), s["rows_line"], group, s["group"], {
            "source": "authored", "backend": s["backend"], "model": s["model"],
            "transform": s["transform"], "base_id": s["base_id"], "domain": s["domain"]})
        if p:
            packets.append(p)
        else:
            reasons[why] += 1
    rows_file = CUT / "rows.jsonl"
    by_base = defaultdict(list)
    for line_no, line in enumerate(rows_file.open("rb"), 1):
        r = json.loads(line)
        by_base[r["base_id"]].append((line_no, r))
    rng = random.Random(CUT_SEED)
    per_group = Counter()
    for b in sorted((json.loads(x) for x in (CUT / "bases.jsonl").open()),
                    key=lambda b: (b["group_id"], b["source_file"])):
        if per_group[b["group_id"]] >= PER_GROUP:
            continue
        line_no, r = rng.choice(by_base[b["base_id"]])
        p, why = finish_packet(rows_file, line_no, b["group_id"], "pkg:" + b["package"].lower(), {
            "source": "code_cut", "transform": r["transform"], "base_id": b["base_id"],
            "source_file": b["source_file"], "source_file_sha256": b["source_sha256"]})
        if p:
            packets.append(p)
            per_group[b["group_id"]] += 1
        else:
            reasons[why] += 1
    return packets


# ---------------------------------------------------------------- no_op

def noop_candidates(parents, cpt, rows, staging, reasons):
    packets = []
    for parent in parents:
        license_info = panel._parent_license(Path(parent["normalized_parent_path"]), parent["package"])
        if license_info is None:
            reasons["no_op:license_missing"] += 1
            continue
        parent = dict(parent, license=license_info)
        base = Path(parent["normalized_parent_path"]) / parent["package"] / "R"
        files = sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".r") \
            if base.is_dir() else []
        found = []
        # One line per file first, then more lines from the same files.
        per_file = []
        for path in files:
            raw = path.read_bytes()
            if sha256(raw) in cpt:
                reasons["no_op:file_in_cpt"] += 1
                continue
            try:
                raw, text, eol = panel._read_utf8(path)
            except (OSError, UnicodeError, ValueError):
                reasons["no_op:unreadable"] += 1
                continue
            lines = text.split("\n")
            hits = [(i, c) for i in range(len(lines))
                    if (c := panel._derived_candidate("no_op", lines, i))]
            if hits:
                per_file.append((path, raw, text, eol, lines, hits))
        for round_ in range(PER_GROUP):
            for path, raw, text, eol, lines, hits in per_file:
                if len(found) < PER_GROUP and round_ < len(hits):
                    found.append((path, raw, text, eol, lines, *hits[round_]))
        for path, raw, text, eol, lines, index, (old, target, validator, basis) in found:
            rel = str(path.relative_to(Path(parent["normalized_parent_path"]) / parent["package"]))
            packet_id = "c05-noop-" + sha256(f"{parent['identity']}:{rel}:{index}:{basis}".encode())[:24]
            origin = {"kind": "source_derived_companion", "constructor": basis, "path": rel,
                      "source_line": index, "base_source_sha256": sha256(raw),
                      "source_parent_authority": "DAT-07-dev-parent-allowlist"}
            try:
                pk = panel._make_packet(
                    packet_id=packet_id, family="no_op", source_family="dat07_source_builder_no_op",
                    parent=parent, source_path=path, source_text_lf=text, source_raw=raw,
                    source_eol=eol, prefix=lines[max(0, index - panel.MAX_CONTEXT_LINES):index],
                    old=old, target=target,
                    suffix=lines[index + 1:index + 1 + panel.MAX_CONTEXT_LINES],
                    sentinel=False, source_kind="full_snapshot", origin=origin,
                    validator=validator, staging=staging, no_op_basis=basis,
                    cursor={"region_line_index": 0, "code_point_column": len(old[0]),
                            "utf16_column": len(old[0].encode("utf-16-le")) // 2,
                            "encoding": "derived_line_anchor"},
                    source_start=index)
            except (OSError, UnicodeError, ValueError) as e:
                reasons[f"no_op:{e}"] += 1
                continue
            packets.append(panel_packet(pk, parent, rows, "no_op", {"source": "derived", "constructor": basis}))
    return packets


def panel_packet(pk, parent, rows, family, meta):
    line = rows.add({"packet_id": pk["id"], "family": family, "group_id": parent["group_id"],
                     "origin": pk["origin"], "source": pk["source"]})
    # campaign_dev_panel predates the versioned PromptContext; DEV75's
    # roxygen cases carry the same context plus schema_version.
    context = {"schema_version": protocol.SCHEMA_VERSION, **pk["context"]}
    result = {"status": "converted", "context": context, "target_body": pk["target_body"],
              "operation": pk["operation"], "selection_source": pk["selection_source"],
              "provenance": {"verification": pk["verification"], "origin": pk["origin"],
                             "parent_identity": pk["parent_identity"], "source": pk["source"],
                             "validator": pk["validator"], "no_op_basis": pk["no_op_basis"]}}
    ref = {"row_id": pk["id"], "split": "dev_group", "group_id": parent["group_id"],
           "file": str(rows.path), "line": line, "package_id": parent["identity"]}
    return {"family": family, "row_ref": ref, "result": result, "meta": meta}


# ---------------------------------------------------------------- propagation

def propagation_candidates(parents, cpt, rows, src_dir, reasons, *, families, only_groups=None,
                           seeds=(None,), skip=frozenset()):
    packets = []
    for index, parent in enumerate(parents):
        if only_groups is not None and parent["group_id"] not in only_groups:
            continue
        try:
            prop._parse_license(Path(parent["normalized_root"]) / parent["package"])
        except Exception:  # noqa: BLE001
            reasons["propagation:license_missing"] += 1
            continue
        for family in families:
            n, seen = 0, set()
            for seed in seeds:
                if n >= PER_GROUP:
                    break
                rng = random.Random(2307 + index * 101 + prop.FAMILIES.index(family) * 10007
                                    if seed is None else seed)
                cands = (prop._format_candidates(parent) if family == "format_propagation"
                         else all_scenario_candidates(parent, family, rng))
                for row, before_raw, extra in cands:
                    if n >= PER_GROUP:
                        break
                    key = (parent["group_id"], row["path"], tuple(row["region_new"]))
                    if key in seen or key in skip:
                        continue
                    seen.add(key)
                    if sha256(before_raw) in cpt:
                        reasons[f"{family}:file_in_cpt"] += 1
                        continue
                    result, _ = prop._local_derived_convert(
                        row, parent, before_raw, rows.path, src_dir,
                        normalized_after_sha=extra.get("normalized_after_sha"),
                        raw_tarball_member=extra.get("raw_member"))
                    if result["status"] != "converted":
                        reasons[f"{family}:{result.get('reason')}"] += 1
                        continue
                    line = rows.add(row)
                    result["provenance"]["line"] = line
                    ref = {"row_id": row["row_id"], "split": "dev_group",
                           "group_id": parent["group_id"], "file": str(rows.path),
                           "line": line, "package_id": parent["identity"]}
                    packets.append({"family": family, "row_ref": ref, "result": result,
                                    "meta": {"source": "derived", "constructor": row["derivation"]}})
                    n += 1
    return packets


def all_scenario_candidates(parent, family, rng):
    """DAT-07's _scenario_candidates stops at the first file with a hit;
    DEV250 needs up to PER_GROUP cases per group, so scan every file."""
    root = Path(parent["normalized_root"]) / parent["package"] / "R"
    fn = {"rename_propagation": scenarios.extract_rename,
          "pipe_rewrite": scenarios.extract_pipe,
          "na_rm_propagation": scenarios.extract_na_rm}[family]
    name = fn.__name__
    out = []
    for path in sorted([*root.glob("*.R"), *root.glob("*.r")]) if root.is_dir() else []:
        try:
            raw = path.read_bytes()
            if not raw or len(raw) > scenarios.MAX_FILE_BYTES:
                continue
            examples = fn(scenarios.Bundle(parent["package"], f"R/{path.name}", raw), rng, cap=4)
        except Exception:  # noqa: BLE001
            continue
        for ex in examples:
            if ex.get("family") != family:
                continue
            try:
                scenarios.validate_example(ex)
            except Exception:  # noqa: BLE001
                continue
            out.append((prop._base_row(parent, ex, f"canonical_scenarios.{name}",
                                       str(path), sha256(raw)), raw, {}))
    return out


# ---------------------------------------------------------------- roxygen

def roxygen_candidates(parents_by_group, cpt, rows, staging, reasons):
    records, _ = panel.collect_dev_audit(panel.AUDIT, parents_by_group)
    rox = [r for r in records.get("scenario_roxygen_drafting", [])]
    raws, failures = panel._read_raw_records(rox)
    reasons.update({f"roxygen:{k}": v for k, v in failures.items()})
    packets, per_group, cache = [], Counter(), {}
    for record in rox:
        gid = record["group_id"]
        if per_group[gid] >= PER_GROUP or record["row_id"] not in raws:
            continue
        parent = dict(parents_by_group[gid])
        license_info = panel._parent_license(Path(parent["normalized_parent_path"]), parent["package"])
        if license_info is None:
            reasons["roxygen:license_missing"] += 1
            continue
        parent["license"] = license_info
        pk, why = panel._packet_from_existing(record, raws[record["row_id"]], parent, staging, cache)
        if pk is None:
            reasons[f"roxygen:{why}"] += 1
            continue
        if pk["source"]["content_sha256"] in cpt:
            reasons["roxygen:file_in_cpt"] += 1
            continue
        pk["id"] = "c05-" + pk["id"]
        packets.append(panel_packet(pk, parent, rows, "roxygen_drafting",
                                    {"source": "existing_dev_row", "raw_row_id": record["row_id"]}))
        per_group[gid] += 1
    return packets


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    staging = args.out / "sources"
    staging.mkdir()

    cpt = {h for (h,) in sqlite3.connect(f"file:{CPT_INDEX}?mode=ro", uri=True)
           .execute("select source_sha256 from documents")}
    eligible = set(json.loads(OVERLAP.read_text())["eligible_construction_groups"])
    _, parents_by_group = panel.load_allowlist()
    prop_parents = prop._parent_records()
    for p in prop_parents:
        p["normalized_parent_path"] = parents_by_group[p["group_id"]]["normalized_parent_path"]
    eligible_parents = [p for p in prop_parents if p["group_id"] in eligible]
    dev75 = [json.loads(line) for line in DEV75.open()]
    narm_groups = {c["group_id"] for c in dev75 if c["family"] == "na_rm_propagation"}
    narm_used = frozenset((c["group_id"], c["context"]["path"], tuple(c["region_new"]))
                          for c in dev75 if c["family"] == "na_rm_propagation")

    reasons = Counter()
    rows = Rows(args.out / "derived-rows.jsonl")
    packets = finish_candidates(args.out, reasons)
    packets += noop_candidates(eligible_parents, cpt, rows, staging, reasons)
    packets += propagation_candidates(
        eligible_parents, cpt, rows, staging, reasons,
        families=("pipe_rewrite", "rename_propagation", "format_propagation"))
    packets += propagation_candidates(
        prop_parents, cpt, rows, staging, reasons, families=("na_rm_propagation",),
        only_groups=narm_groups, seeds=range(NA_RM_SEEDS), skip=narm_used)
    packets += roxygen_candidates({g: parents_by_group[g] for g in eligible},
                                  cpt, rows, staging, reasons)

    rows_sha = rows.write()
    for p in packets:
        ref = p["row_ref"]
        if ref["file"] == str(rows.path):
            ref["source_sha256"] = rows_sha
            ref["raw_line_sha256"] = sha256(rows.lines[ref["line"] - 1])
    out = args.out / "packets.jsonl"
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in packets))
    summary = {
        "packets": len(packets),
        "by_family": dict(Counter(p["family"] for p in packets)),
        "groups_by_family": {f: len({p["row_ref"]["group_id"] for p in packets if p["family"] == f})
                             for f in sorted({p["family"] for p in packets})},
        "exclusions": dict(reasons),
        "packets_sha256": file_sha(out),
        "derived_rows_sha256": rows_sha,
    }
    (args.out / "phase1.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
