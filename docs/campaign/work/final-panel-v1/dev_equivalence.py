#!/usr/bin/env python3
"""Show that final_panel's constructors reproduce DEV250 and DEV75 cases from raw dev parents.

Reads only dev-split material: DEV250 and its C05 phase-1 and finish-cut
artifacts, the DEV75 candidate file's stored hashes and identity fields, the
DAT-07 dev propagation rows and snapshots, and the dev parents' normalized
sources and tarballs. Writes aggregate counts only.

    python dev_equivalence.py OUTPUT.json
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_panel as F
from sepalith.campaign_protocol import PromptContext, render_prompt, serialize_target

D = Path("/mnt/e/sepalith/campaign-20260915/data-work")
DEV250 = D / "C05-dev250-v1r3/dev250-v1.jsonl"
PHASE1 = D / "C05-dev250-v1r3-phase1"
DEV75 = D / "DAT-07-final-dev-candidates.jsonl"      # DEV75 (split dev); preflight sha256 bf278755...
DAT07_ROWS = D / "DAT-07-propagation-v2-derived-rows.jsonl"
CUT_ROWS = D / "C05-dev-finish-cut-v1/rows.jsonl"
EXTRACT = {"rename_propagation": "extract_rename", "pipe_rewrite": "extract_pipe", "na_rm_propagation": "extract_na_rm"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("rb") if line.strip()]


def line_at(path: Path, number: int) -> dict:
    with path.open("rb") as stream:
        for index, line in enumerate(stream, 1):
            if index == number:
                return json.loads(line)
    raise KeyError(number)


def before_bytes(path: str) -> bytes:
    if "::" in path:
        archive, member = path.split("::")
        with tarfile.open(archive) as tar:
            handle = tar.extractfile(member)
            assert handle is not None
            return handle.read()
    return Path(path.replace("/mnt/h/", "/mnt/nas/home/", 1)).read_bytes()


def hashes(context: dict, operation: str, region_new: list[str]) -> tuple[str, str]:
    return sha(render_prompt(PromptContext.from_mapping(context))), sha(serialize_target(operation, region_new))


def key(example: dict) -> tuple:
    return (example["path"], tuple(example["prefix"]), tuple(example["region_old"]),
            tuple(example["region_new"]), example["event_diff"], example["cursor_idx"])


def propagation(m: F.Modules, dev250: list[dict]) -> dict:
    out: Counter[str] = Counter()
    for case in dev250:
        family = case["family"]
        if family not in (*EXTRACT, "format_propagation"):
            continue
        prov = case["source_provenance"]
        row = line_at(Path(prov["file"]), prov["line"])
        raw = before_bytes(row["source_path"])
        if family == "format_propagation":
            norm = Path(row["normalized_after_path"].replace("/mnt/h/", "/mnt/nas/home/", 1)).read_bytes()
            examples = m.scenarios.format_pairs_from_lines(
                row["package"], row["path"], raw.decode("utf-8", "replace").splitlines(),
                norm.decode("utf-8", "replace").splitlines())
            regenerated = key(row) in {key(e) for e in examples}
        else:
            # C05 seeded rename per parent index (2307 + 101*index + 10007*family); the index order is lost.
            extract = getattr(m.scenarios, EXTRACT[family])
            seeds = [None] if family != "rename_propagation" else [
                2307 + 101 * k + 10007 * i for k in range(400) for i in range(4)]
            regenerated = False
            for seed in seeds:
                bundle = m.scenarios.Bundle(row["package"], row["path"], raw)
                if key(row) in {key(e) for e in extract(bundle, random.Random(seed or 0), cap=4)}:
                    regenerated = True
                    break
        result = F.convert_example(row, raw.decode(), "file:///sepalith/" + row["path"],
                                   case["context"]["history"][0]["event_id"])
        context, _ = F.selected_context(result)
        context = context.to_dict()
        out[f"{family}:cases"] += 1
        out[f"{family}:regenerated_from_raw"] += regenerated
        out[f"{family}:context_equal"] += context == case["context"]
        out[f"{family}:region_new_equal"] += result["target_body"] == case["region_new"]
        prompt, target = hashes(context, result["operation"], result["target_body"])
        out[f"{family}:prompt_and_target_hash_equal"] += (prompt, target) == (case["prompt_sha256"],
                                                                               case["target_sha256"])
    return dict(sorted(out.items()))


def dev75_propagation() -> dict:
    out: Counter[str] = Counter()
    rows = DAT07_ROWS.read_bytes().split(b"\n")
    for case in jsonl(DEV75):
        ref = case["source_ref"]
        if not ref["file"].endswith(DAT07_ROWS.name):
            continue
        family = case["row"]["family"]
        derived = json.loads(rows[ref["line"] - 1])
        raw = before_bytes(case["source_provenance"]["source_snapshot_path"])
        result = F.convert_example(derived, raw.decode(), "file:///sepalith/" + derived["path"], "x")
        context, _ = F.selected_context(result)
        prompt, target = hashes(context.to_dict(), result["operation"], result["target_body"])
        out[f"{family}:cases"] += 1
        out[f"{family}:prompt_and_target_hash_equal"] += (prompt, target) == (case["prompt_sha256"],
                                                                               case["target_sha256"])
        out[f"{family}:metadata_equal"] += (
            case["row"]["package_id"] == ref["package_id"] and ref["split"] == "dev_group"
            and case["row"]["target_operation"] == result["operation"])
    return dict(sorted(out.items()))


def noop(dev250: list[dict]) -> dict:
    out: Counter[str] = Counter()
    packets = [json.loads(line) for line in (PHASE1 / "packets.jsonl").open()]
    want: dict[str, set] = defaultdict(set)
    sources: dict[str, Path] = {}
    for packet in packets:
        if packet["family"] != "no_op":
            continue
        origin = packet["result"]["provenance"]["origin"]
        parent = packet["result"]["provenance"]["parent_identity"]
        want[packet["row_ref"]["group_id"]].add((origin["path"], origin["source_line"], origin["constructor"]))
        sources[packet["row_ref"]["group_id"]] = Path(
            parent["normalized_parent_path"].replace("/mnt/h/", "/mnt/nas/home/", 1)) / parent["package"] / "R"
    cpt = F.cpt_hashes(F.load_recipe())
    for group, r_dir in sorted(sources.items()):
        per_file = []
        for path in sorted(p for p in r_dir.iterdir() if p.is_file() and p.suffix.lower() == ".r"):
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() in cpt:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeError:
                continue
            sites = F.noop_sites(F._lines(text)[0])
            if sites:
                per_file.append((f"R/{path.name}", sites))
        found = []
        for round_ in range(3):
            for rel, sites in per_file:
                if len(found) < 3 and round_ < len(sites):
                    found.append((rel, *sites[round_]))
        out["groups"] += 1
        out["group_site_sets_equal"] += set(found) == want[group]
    for case in dev250:
        if case["family"] != "no_op":
            continue
        src = case["source_provenance"]["source"]
        text = Path(src["path"]).read_bytes().decode("utf-8")
        lines, eol = F._lines(text)
        result = F.noop_result(case["context"]["path"], case["context"]["replacement_range"]["uri"], lines, eol,
                               src["region_start_line"])
        context, _ = F.selected_context(result)
        context = context.to_dict()
        out["cases"] += 1
        out["context_equal"] += context == case["context"]
        out["prompt_and_target_hash_equal"] += hashes(context, "no_op", result["target_body"]) == (
            case["prompt_sha256"], case["target_sha256"])
    return dict(sorted(out.items()))


def finish(m: F.Modules, dev250: list[dict]) -> dict:
    out: Counter[str] = Counter()
    for case in dev250:
        if case["family"] != "finish_block":
            continue
        source = case["strata"]["source"]
        ref = case["source_ref"]
        line = Path(ref["file"]).read_bytes().split(b"\n")[ref["line"] - 1] + b"\n"
        raw = json.loads(line)
        row_id = "c05-finish-" + sha(f"{ref['file']}:{ref['line']}")[:24]
        result, why = F.convert_finish(m, raw, {
            "file": ref["file"], "line": ref["line"], "raw_line_sha256": hashlib.sha256(line).hexdigest(),
            "source_sha256": F.sha256_file(Path(ref["file"])), "split": "dev_group", "row_id": row_id,
            "group_id": ref["group_id"], "document_role": "source_derived_simulated_pre_edit",
            "source_constructor": "finish_block_v5_prefix", "target_convention": "suffix",
            "uri": f"file:///sepalith/dev/{row_id}/{raw['path']}", "document_version": 0})
        out[f"{source}:cases"] += 1
        if result is None:
            out[f"{source}:failed:{why}"] += 1
            continue
        context, _ = F.selected_context(result)
        context = context.to_dict()
        out[f"{source}:context_and_target_equal"] += (context == case["context"]
                                                      and result["target_body"] == case["region_new"])
        out[f"{source}:prompt_and_target_hash_equal"] += hashes(context, result["operation"],
                                                                 result["target_body"]) == (
            case["prompt_sha256"], case["target_sha256"])
    # The code-cut harvest itself: rerun C05's rule on every eligible dev parent.
    cut = m.cut
    eligible = set(json.loads(cut.OVERLAP.read_text())["eligible_construction_groups"])
    parents = sorted((p for p in json.loads(cut.ALLOWLIST.read_text())["parents"] if p["group_id"] in eligible),
                     key=lambda p: p["group_id"])
    cpt, stats, rows = cut.cpt_hashes(), Counter(), []
    for parent in parents:
        parent = {**parent, "normalized_versions": [v.replace("/mnt/h/", "/mnt/nas/home/", 1)
                                                    for v in parent["normalized_versions"]]}
        rows += cut.harvest(parent, cpt, stats)[0]
    stored = jsonl(CUT_ROWS)
    for row in rows + stored:
        row["source_file"] = row["source_file"].replace("/mnt/h/", "/mnt/nas/home/", 1)
    out["code_cut_harvest_rows"] = len(rows)
    out["code_cut_harvest_rows_stored"] = len(stored)
    out["code_cut_harvest_identical"] = int(rows == stored)
    return dict(sorted(out.items()))


def main() -> int:
    m = F.Modules.load()
    dev250 = jsonl(DEV250)
    receipt = {
        "schema": "sepalith.final-panel.v1.dev-equivalence",
        "constructor_sha256": F.sha256_file(Path(F.__file__)),
        "recipe_sha256": F.sha256_file(F.RECIPE_PATH),
        "inputs": {"dev250": {"path": str(DEV250), "sha256": F.sha256_file(DEV250)},
                   "dev75": {"path": str(DEV75), "sha256_from_preflight":
                             "bf278755c795b51ba58bc14c674edf76dcb7799475e5982adc7bd50f9c0d05df",
                             "fields_read": ["prompt_sha256", "target_sha256", "row.family", "row.package_id",
                                             "row.target_operation", "source_ref", "source_provenance"]},
                   "phase1": str(PHASE1), "dat07_propagation_rows": str(DAT07_ROWS),
                   "finish_cut_rows": str(CUT_ROWS)},
        "final_content_opened": False,
        "dev250_propagation": propagation(m, dev250),
        "dev75_propagation": dev75_propagation(),
        "dev250_no_op": noop(dev250),
        "dev250_finish_block": finish(m, dev250),
    }
    Path(sys.argv[1]).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
