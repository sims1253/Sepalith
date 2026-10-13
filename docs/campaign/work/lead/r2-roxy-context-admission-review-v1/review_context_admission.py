#!/usr/bin/env python3
"""Audit source-reference categories for DAT-10 roxygen context candidates.

The review replays only normalized TRAIN sources.  It records categories and
hashes, never source or target text, and emits recommendations rather than a
training admission.  The full 10,017-row scope is classified; a deterministic
stratified subset also receives the more detailed occurrence audit requested
for formal mismatches, pending rows, unresolved references, and the four old
full-file outliers.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
V1_OUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-supported-context-v1")
V1_PROFILES = V1_OUT / "selected-context-profiles.jsonl"
V1_EVIDENCE = V1_OUT / "source-context-evidence.json"
V1_MANIFEST = V1_OUT / "manifest.json"
V1_FULL_PROFILES = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-repair-materialization-v2-10017/repaired-token-profiles.jsonl")
ROOT_IDS = PLAN / "docs/campaign/work/lead/r2-roxy10017-root-context-review-v1/candidate-ids.json"
ROOT_RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-roxy10017-root-context-review.json"
SUPPORT_LEDGER = PLAN / "docs/campaign/work/lead/r2-source-walk-support-review-v1/support-ledger.jsonl"
PACKET_ROOT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1")
PACKET_REL = "structured-materialization-v1/candidate-packets.jsonl"
SCENARIO_SOURCE = Path("/mnt/h/sepalith/datasets/scenarios_v1/roxygen_drafting.jsonl")
SCENARIO_SHA = "0d70ccc40a716a8a27cb508e39a16c9a5119a9bc9d0b5b8ee8362aa4b67f6193"
OUT_DEFAULT = Path("docs/campaign/work/lead/r2-roxy-context-admission-review-v1/full")

TRAINING_DIR = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training")
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module_missing:" + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREP = load_module(
    "dat10_admission_review_prepare",
    PLAN / "docs/campaign/work/lead/r2-source-walk-support-repair-v1/prepare_repair.py",
)
SELECTOR = load_module(
    "dat10_admission_review_selector",
    PLAN / "docs/campaign/work/lead/r2-roxy-supported-context-v1/context_selector.py",
)
review = PREP.review
adapter = PREP.adapter
import campaign_structured_batch as structured_batch  # noqa: E402

from tree_sitter import Language, Parser  # noqa: E402
import tree_sitter_r  # noqa: E402

R_LANGUAGE = Language(tree_sitter_r.language())
R_PARSER = Parser(R_LANGUAGE)


_PARAM_TAG = re.compile(r"^\s*#'\s*@param\s+([^\s]+(?:\s*,\s*[^\s]+)*)")
_TAG = re.compile(r"^\s*#'\s*@([A-Za-z][A-Za-z0-9_.-]*)\b")

# Expanded only for audit classification. A name labelled builtin/operator is
# not a claim that its use is correct; it means the source need not define it.
R_BUILTINS = set(SELECTOR._COMMON_R_NAMES) | {
    "::", ":::", "%>%", "%||%", "%||NA%", "%in%", "|>", "<-", "<<-",
    "->", "->>", "=", "+", "-", "*", "/", "^", "%%", "%/%", "&", "&&",
    "|", "||", "!", "<", "<=", ">", ">=", "==", "!=", "$", "@", "[",
    "[[", "]", "]>", "baseenv", "globalenv", "emptyenv", "as.environment",
    "as.function", "as.name", "as.symbol", "as.call", "as.expression", "as.raw",
    "as.Date", "as.POSIXct", "as.POSIXlt", "Date", "POSIXct", "POSIXlt", "structure",
    "attr", "attributes", "mostattributes", "is.function", "is.list", "is.atomic",
    "is.vector", "is.numeric", "is.character", "is.logical", "is.matrix", "is.data.frame",
    "is.finite", "is.infinite", "is.nan", "is.package_version", "as.factor", "factor",
    "levels", "nlevels", "droplevels", "read.csv", "read.table", "saveRDS", "readRDS",
    "file", "close", "flush", "connection", "textConnection", "rawConnection",
    "options", "getOption", "setNames", "system.file", "packageVersion", "utils",
    "stats", "methods", "grDevices", "graphics", "datasets", "tools", "utils",
    "dnorm", "pnorm", "qnorm", "rnorm", "optim", "integrate", "mapply", "Map",
    "list2", "abort", "warn", "rlang", "convergence", "SIMPLIFY", "FUN", "X", "MARGIN",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    candidate_ids = set(json.loads(ROOT_IDS.read_text(encoding="utf-8")))
    root_receipt = json.loads(ROOT_RECEIPT.read_text(encoding="utf-8"))
    if len(candidate_ids) != 10017 or root_receipt.get("candidate_count") != 10017:
        raise ValueError("candidate_scope_mismatch")
    profiles: dict[str, dict[str, Any]] = {}
    with V1_PROFILES.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["row_id"] in candidate_ids:
                profiles[row["row_id"]] = row
    if set(profiles) != candidate_ids:
        raise ValueError("v1_profile_scope_mismatch")
    ledgers: dict[str, dict[str, Any]] = {}
    with SUPPORT_LEDGER.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("row_id") in candidate_ids:
                ledgers[row["row_id"]] = row
    if set(ledgers) != candidate_ids:
        raise ValueError("support_ledger_scope_mismatch")
    packets: dict[str, dict[str, Any]] = {}
    packet_pins = []
    for shard_no in range(5):
        path = PACKET_ROOT / f"shard-{shard_no:04d}" / PACKET_REL
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                packet = json.loads(line)
                count += 1
                row_id = str(packet["row_ref"]["row_id"])
                if row_id in candidate_ids:
                    if row_id in packets:
                        raise ValueError("duplicate_packet:" + row_id)
                    packets[row_id] = packet
        packet_pins.append({"path": str(path), "rows": count, "sha256": sha_file(path)})
    if set(packets) != candidate_ids:
        raise ValueError("packet_scope_mismatch")
    full_lengths: dict[str, dict[str, Any]] = {}
    with V1_FULL_PROFILES.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["row_id"] in candidate_ids:
                full_lengths[row["row_id"]] = row["context_profiles"][-1]["lengths"]
    if set(full_lengths) != candidate_ids:
        raise ValueError("full_profile_scope_mismatch")
    pins = {
        "root_ids": {"path": str(ROOT_IDS), "rows": len(candidate_ids), "sha256": sha_file(ROOT_IDS)},
        "root_context_receipt": {"path": str(ROOT_RECEIPT), "sha256": sha_file(ROOT_RECEIPT)},
        "v1_profiles": {"path": str(V1_PROFILES), "rows": len(profiles), "sha256": sha_file(V1_PROFILES)},
        "v1_evidence": {"path": str(V1_EVIDENCE), "sha256": sha_file(V1_EVIDENCE)},
        "v1_manifest": {"path": str(V1_MANIFEST), "sha256": sha_file(V1_MANIFEST)},
        "support_ledger": {"path": str(SUPPORT_LEDGER), "rows": len(ledgers), "sha256": sha_file(SUPPORT_LEDGER)},
        "candidate_packets": packet_pins,
        "prior_full_profiles": {"path": str(V1_FULL_PROFILES), "rows": len(full_lengths), "sha256": sha_file(V1_FULL_PROFILES)},
        "scenario_source": {"path": str(SCENARIO_SOURCE), "sha256": SCENARIO_SHA, "consumption": "TRAIN path/hash guard only"},
    }
    return profiles, ledgers, packets, {"pins": pins, "full_lengths": full_lengths}


def reconstruct(ledger: dict[str, Any], packet: dict[str, Any]) -> tuple[bytes, list[str], int, dict[str, Any], str]:
    raw, package = review.packet_raw(packet)
    source_path = Path(ledger["source_path"])
    source_bytes = source_path.read_bytes()
    if sha_bytes(source_bytes) != ledger["source_sha256"]:
        raise ValueError("normalized_after_source_hash_mismatch")
    text = source_bytes.decode("utf-8")
    if "\r" in text.replace("\r\n", ""):
        raise ValueError("mixed_or_lone_cr_source")
    eol = "crlf" if "\r\n" in text else "lf"
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    positions = review.locate_inverse_target(lines, raw["region_new"], raw["suffix"])
    if len(positions) != 1:
        raise ValueError("inverse_target_not_unique")
    pos = positions[0]
    target_line = int(pos["target_start_line"])
    removed = len(raw["region_new"]) + int(pos["blank_after_target"])
    before_lines = lines[:target_line] + [""] + lines[target_line + removed:]
    before_text = ("\r\n" if eol == "crlf" else "\n").join(before_lines)
    before_hash = sha_bytes(before_text.encode("utf-8", "surrogatepass"))
    if before_hash != ledger.get("context", {}).get("full_snapshot_sha256"):
        raise ValueError("derived_before_hash_mismatch")
    ref = copy.deepcopy(packet["row_ref"])
    ref.update({
        "package_id": package,
        "source_snapshot_text": before_text,
        "source_snapshot_sha256": before_hash,
        "source_snapshot_provenance_path": "derived:normalized-after/" + str(source_path),
        "cursor_encoding": "scenario_char_offset",
        "workspace_revision_before": before_hash,
        "target_line": target_line,
        "parent_identity": {
            "package": package, "path": raw["path"],
            "normalized_after_source_path": str(source_path),
            "normalized_after_source_sha256": ledger["source_sha256"],
            "pre_edit_derivation": "remove_exact_mined_roxygen_target_retain_physical_blank_anchor",
        },
    })
    converted = adapter.convert_structured(raw, ref)
    if converted.get("status") != "converted":
        raise ValueError("structured_conversion_failed:" + str(converted.get("reason")))
    # Re-run the reviewed byte/UTF-16 application check without writing the
    # resulting source snapshot. This validates the reconstructed geometry and
    # post-edit hash for every audited row.
    structured_batch.apply_result(converted)
    return before_text.encode("utf-8", "surrogatepass"), before_lines, target_line, converted, package


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace").strip("`")


def _line_span(source: bytes, node: Any) -> tuple[int, int]:
    starts = [0]
    starts.extend(index + 1 for index, value in enumerate(source) if value == 10)
    from bisect import bisect_right
    start = bisect_right(starts, int(node.start_byte)) - 1
    end = bisect_right(starts, max(int(node.start_byte), int(node.end_byte) - 1)) - 1
    return start, end


def _scope_local_names(function_node: Any, source: bytes) -> set[str]:
    """Return names bound in one function's lexical scope.

    The selector's historical local-name helper intentionally walks a whole
    function subtree.  That is sufficient for conservative span selection,
    but an admission audit must reset scope at nested functions; otherwise a
    nested ``i``/``x`` is reported as a free name of the outer function (or an
    outer assignment is incorrectly credited to the nested function).
    """
    result = set(SELECTOR._formal_names(function_node, source))

    def visit(current: Any) -> None:
        for child in current.named_children:
            if child.type == "function_definition":
                # A nested function has its own binding environment. Its
                # names are collected when that node is visited as a scope.
                continue
            if child.type == "binary_operator" and SELECTOR._operator(child, source) in SELECTOR._ASSIGNMENT_OPERATORS:
                lhs = child.child_by_field_name("lhs")
                name = SELECTOR._identifier_name(source, lhs)
                if name:
                    result.add(name)
            if child.type == "for_statement":
                binding = child.child_by_field_name("name") or child.child_by_field_name("left")
                name = SELECTOR._identifier_name(source, binding)
                if name:
                    result.add(name)
            visit(child)

    visit(function_node)
    return result


def _extract_operator_symbol(node: Any, source: bytes) -> str | None:
    """Return the actual R extraction operator for one AST node."""
    if node.type != "extract_operator":
        return None
    for child in node.children:
        text = _node_text(source, child).strip()
        if text in ("$", "@", "[", "[[", "[<-"):
            return text
    return None


def _walk_with_parent(
    node: Any,
    source: bytes,
    parent_type: str | None = None,
    parent_index: int = -1,
    ancestors: tuple[str, ...] = (),
    scope_local_names: frozenset[str] = frozenset(),
    parent_operator: str | None = None,
):
    """Yield immutable occurrence facts with nearest-function scope."""
    if node.type == "function_definition":
        scope_local_names = frozenset(_scope_local_names(node, source))
    if node.type == "identifier":
        yield {
            "name": _node_text(source, node),
            "parent_type": parent_type,
            "parent_index": parent_index,
            "parent_operator": parent_operator,
            "ancestors": ancestors,
            "scope_local_names": scope_local_names,
        }
    children = list(node.named_children)
    operator = _extract_operator_symbol(node, source)
    for index, child in enumerate(children):
        yield from _walk_with_parent(
            child,
            source,
            node.type,
            index,
            ancestors + (node.type,),
            scope_local_names,
            operator,
        )


def _find_definition_nodes(tree: Any, source: bytes, definitions: list[Any]) -> dict[tuple[str, tuple[int, int], str], Any]:
    """Map immutable selector definitions to nodes only within this call."""
    result: dict[tuple[str, tuple[int, int], str], Any] = {}
    for node in tree.root_node.named_children:
        if node.type != "binary_operator":
            continue
        lhs = node.child_by_field_name("lhs")
        rhs = node.child_by_field_name("rhs")
        if lhs is None or rhs is None or lhs.type != "identifier":
            continue
        name = _node_text(source, lhs)
        span = _line_span(source, node)
        kind = "function" if rhs.type == "function_definition" else "expression"
        key = (name, span, kind)
        result[key] = rhs
    return result


def _docs_evidence(target_lines: list[str], formals: set[str]) -> dict[str, Any]:
    documented: set[str] = set()
    tags: Counter[str] = Counter()
    for line in target_lines:
        tag = _TAG.match(line)
        if tag:
            tags[tag.group(1)] += 1
        param = _PARAM_TAG.match(line)
        if param:
            for name in re.split(r"\s*,\s*", param.group(1)):
                normalized = name.strip().strip("`")
                # Match the selector's documented-formal normalization. In
                # R documentation both TeX spellings denote the variadic
                # formal (...); otherwise this audit would manufacture holds.
                if normalized in {r"\dots", r"\ldots"}:
                    normalized = "..."
                if normalized:
                    documented.add(normalized)
    return {
        "tag_names": sorted(tags),
        "tag_count": sum(tags.values()),
        "documented_param_names": sorted(documented),
        "function_formal_names": sorted(formals),
        "missing_documented_formals": sorted(documented - formals),
        "undocumented_formals": sorted(formals - documented),
    }


def classify_row(profile: dict[str, Any], ledger: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    """Replay one source and classify every v1 unresolved name occurrence."""
    before_bytes, before_lines, target_line, converted, package = reconstruct(ledger, packet)
    tree = R_PARSER.parse(before_bytes)
    if tree.root_node.has_error:
        raise ValueError("before_state_R_parse_error")
    definitions = SELECTOR.top_level_definitions(tree, before_bytes, len(before_lines))
    by_key = _find_definition_nodes(tree, before_bytes, definitions)
    evidence = profile["context_evidence"]
    selected = evidence.get("selected_definitions", [])
    selected_keys = {(str(item["name"]), tuple(item["span"]), str(item["kind"])) for item in selected}
    # If a stale selector record misses a current definition, classify the
    # row as a replay discrepancy rather than silently declaring it supported.
    selected_nodes = []
    definition_by_key = {
        (item.name, item.span, item.kind): item
        for item in definitions
    }
    missing_selected = []
    for key in sorted(selected_keys):
        node = by_key.get(key)
        if node is None:
            missing_selected.append({"name": key[0], "span": list(key[1]), "kind": key[2]})
        else:
            selected_nodes.append((node, definition_by_key[key]))
    unresolved = set(str(name) for name in evidence.get("unresolved_reference_names", []))
    occurrences: dict[str, Counter[str]] = defaultdict(Counter)
    current_defs = {item.name for item in definitions}
    for node, definition in selected_nodes:
        for occurrence in _walk_with_parent(node, before_bytes):
            name = occurrence["name"]
            if name not in unresolved:
                continue
            ancestors = occurrence["ancestors"]
            parent = occurrence["parent_type"]
            # The selector deliberately removes each definition's local and
            # formal bindings from its unresolved set. Re-check that lexical
            # scope here so a local loop/temporary (for example ``i``) is not
            # misreported as omitted source evidence when another selected
            # definition contains the same spelling.
            if name in occurrence["scope_local_names"]:
                category = (
                    "formal_binding_reference"
                    if "parameter" in ancestors
                    else "local_binding_reference"
                )
            elif occurrence["parent_operator"] in {"$", "@"} and occurrence["parent_index"] == 1:
                # R's object/member extraction labels are names in the source
                # structure, not free variables requiring a definition in the
                # selected context (e.g. config$service or object@slot).
                category = "member_label_reference"
            elif name in current_defs:
                category = "same_file_definition_not_selected"
            # Namespace syntax is source evidence even when the namespace
            # component shares a name with a base-R API (for example
            # stats::filter). Check it before the lexical builtin list.
            elif "namespace_operator" in ancestors or parent == "namespace_operator":
                category = "package_namespace_reference"
            elif name in R_BUILTINS:
                category = "r_builtin_or_operator"
            elif parent == "argument" and occurrence["parent_index"] == 0:
                category = "call_argument_label"
            elif parent == "call" and occurrence["parent_index"] == 0:
                category = "external_callable_or_global"
            elif "parameter" in ancestors:
                category = "formal_binding_reference"
            else:
                category = "free_variable_or_external_global"
            occurrences[name][category] += 1
    # Names listed by v1 but absent from selected AST nodes are genuine audit
    # gaps, even if a lexical heuristic might otherwise call them builtins.
    for name in sorted(unresolved - set(occurrences)):
        if name in current_defs:
            category = "same_file_definition_not_selected"
        elif name in R_BUILTINS:
            category = "r_builtin_or_operator_unobserved"
        else:
            category = "unresolved_occurrence_not_replayed"
        occurrences[name][category] += 1
    category_counts: Counter[str] = Counter()
    for counts in occurrences.values():
        category_counts.update(counts)
    formals = set()
    for definition in definitions:
        if definition.name == evidence.get("target_definition_name") and definition.span == tuple(evidence.get("target_definition_span", [])):
            formals = set(definition.formal_names)
            break
    docs = _docs_evidence(list(converted["target_body"]), formals)
    formal_mismatch = bool(docs["missing_documented_formals"])
    same_file = sum(value for key, value in category_counts.items() if key.startswith("same_file_definition"))
    free = sum(value for key, value in category_counts.items() if key in {"free_variable_or_external_global", "unresolved_occurrence_not_replayed"})
    package_refs = sum(value for key, value in category_counts.items() if key == "package_namespace_reference")
    builtins = sum(value for key, value in category_counts.items() if key.startswith("r_builtin"))
    calls = sum(value for key, value in category_counts.items() if key == "external_callable_or_global")
    if missing_selected:
        recommendation = "hold_selector_replay_discrepancy"
    elif formal_mismatch:
        recommendation = "hold_documented_formal_mismatch"
    elif same_file:
        recommendation = "hold_missing_same_file_evidence"
    elif free:
        recommendation = "hold_true_omitted_or_external_global_evidence"
    else:
        recommendation = "recommend_context_admission_pending_root_semantic_review"
    return {
        "row_id": profile["row_id"],
        "family": profile["family"],
        "group_id": profile["group_id"],
        "package_id": package,
        "source_path": ledger["source_path"],
        "source_sha256": ledger["source_sha256"],
        "derived_before_sha256": ledger.get("context", {}).get("full_snapshot_sha256"),
        "target_definition_name": evidence.get("target_definition_name"),
        "target_definition_span": evidence.get("target_definition_span"),
        "target_line": target_line,
        "v1_status": evidence.get("status"),
        "full_file_sequence_tokens": profile["full_file_profile_comparison"]["sequence_tokens"],
        "selected_sequence_tokens": profile["lengths"]["sequence"],
        "target_body_tokens": profile["lengths"]["target_body_tokens"],
        "old_full_file_gt_131072": int(profile["full_file_profile_comparison"]["sequence_tokens"]) > 131072,
        "selected_context_gt_131072": int(profile["lengths"]["sequence"]) > 131072,
        "target_gt_1024": int(profile["lengths"]["target_body_tokens"]) > 1024,
        "docs_evidence": docs,
        "unresolved_names": sorted(unresolved),
        "unresolved_name_categories": {name: dict(sorted(counts.items())) for name, counts in sorted(occurrences.items())},
        "category_counts": dict(sorted(category_counts.items())),
        "category_totals": {
            "r_builtin_or_operator": builtins,
            "package_namespace_reference": package_refs,
            "external_callable_or_global": calls,
            "local_binding_reference": sum(
                value for key, value in category_counts.items()
                if key == "local_binding_reference"
            ),
            "member_label_reference": sum(
                value for key, value in category_counts.items()
                if key == "member_label_reference"
            ),
            "same_file_definition_not_selected": same_file,
            "free_variable_or_external_global": free,
        },
        "missing_selected_definitions": missing_selected,
        "recommendation": recommendation,
        "semantic_support_claim": False,
        "admission": "review_only_unadmitted",
        "source_text_written": False,
        "target_text_written": False,
    }


def sample_ids(profiles: dict[str, dict[str, Any]]) -> tuple[set[str], dict[str, list[str]]]:
    strata: dict[str, list[str]] = defaultdict(list)
    for row_id, profile in profiles.items():
        status = profile["context_evidence"]["status"]
        if "formal" in status:
            strata["formal_mismatch"].append(row_id)
        elif "pending_semantic" in status:
            strata["pending_semantic"].append(row_id)
        elif "unresolved" in status:
            strata["unresolved_reference"].append(row_id)
    selected: set[str] = set()
    details: dict[str, list[str]] = {}
    for name, ids in strata.items():
        ids.sort()
        if name == "unresolved_reference":
            chosen = sorted(ids, key=lambda value: hashlib.sha256(value.encode("ascii")).hexdigest())[:256]
        else:
            chosen = ids
        selected.update(chosen)
        details[name] = chosen
    full_outliers = [row_id for row_id, profile in profiles.items() if profile["full_file_profile_comparison"]["sequence_tokens"] > 131072]
    selected.update(full_outliers)
    details["full_file_gt_131072"] = sorted(full_outliers)
    return selected, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--all", action="store_true", help="classify full 10,017-row scope; default is requested sample")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output_must_be_fresh")
    args.output.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    try:
        os.nice(10)
    except OSError:
        pass
    started = time.monotonic()
    profiles, ledgers, packets, input_data = load_inputs()
    requested, strata = sample_ids(profiles)
    row_ids = sorted(profiles if args.all else requested)
    write_json(args.output / "status.json", {
        "schema": "sepalith.dat10.roxy_context_admission_review_status.v2",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope_rows": len(row_ids),
        "full_candidate_scope_rows": len(profiles),
        "sample_strata": {name: len(ids) for name, ids in strata.items()},
        "cpu_threads_max": 2,
        "cuda": False,
        "training_admission": False,
        "frozen_v1_unchanged": True,
        "input_pins": input_data["pins"],
    })
    detail_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    recommendation_by_id: dict[str, str] = {}
    category_census: Counter[str] = Counter()
    for count, row_id in enumerate(row_ids, 1):
        try:
            result = classify_row(profiles[row_id], ledgers[row_id], packets[row_id])
            detail_rows.append(result)
            recommendation_by_id[row_id] = result["recommendation"]
            category_census.update(result["category_counts"])
        except Exception as exc:
            errors.append({"row_id": row_id, "error": type(exc).__name__ + ":" + str(exc)})
            recommendation_by_id[row_id] = "hold_replay_error"
        if count == 1 or count % 100 == 0 or count == len(row_ids):
            write_json(args.output / "status.json", {
                "schema": "sepalith.dat10.roxy_context_admission_review_status.v2",
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "scope_rows": len(row_ids),
                "full_candidate_scope_rows": len(profiles),
                "rows_done": count,
                "detail_rows": len(detail_rows),
                "errors": len(errors),
                "sample_strata": {name: len(ids) for name, ids in strata.items()},
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "cpu_threads_max": 2,
                "cuda": False,
                "training_admission": False,
                "frozen_v1_unchanged": True,
            })
    detail_rows.sort(key=lambda row: row["row_id"])
    errors.sort(key=lambda row: row["row_id"])
    write_jsonl(args.output / "audited-rows.jsonl", detail_rows)
    write_jsonl(args.output / "audit-errors.jsonl", errors)
    audited = {row["row_id"]: row for row in detail_rows}
    if args.all:
        # Recommendations cover the full scope. Formal mismatches and actual
        # omitted evidence are held; external package calls/builtins remain
        # reasonable source evidence pending root's semantic/license/dedup gate.
        accepted = sorted(row_id for row_id, result in recommendation_by_id.items() if result == "recommend_context_admission_pending_root_semantic_review")
        held = sorted(set(profiles) - set(accepted))
    else:
        accepted = sorted(row_id for row_id, result in recommendation_by_id.items() if result == "recommend_context_admission_pending_root_semantic_review")
        held = sorted(set(row_ids) - set(accepted))
    write_json(args.output / "recommended-context-ids.json", {
        "schema": "sepalith.dat10.roxy_context_recommendation_ids.v1",
        "scope": "full_10017" if args.all else "requested_stratified_sample",
        "status": "recommendation_only_not_training_admission",
        "accepted_recommendation_count": len(accepted),
        "held_count": len(held),
        "accepted_recommendation_ids": accepted,
        "held_ids": held,
    })
    summary = {
        "schema": "sepalith.dat10.roxy_context_admission_review_summary.v1",
        "status": "complete" if not errors else "complete_with_replay_errors",
        "scope_rows_audited": len(row_ids),
        "full_candidate_scope_rows": len(profiles),
        "detail_rows": len(detail_rows),
        "error_rows": len(errors),
        "recommendation_only": True,
        "recommended_context_rows": len(accepted),
        "held_rows": len(held),
        "sample_strata": {name: len(ids) for name, ids in strata.items()},
        "old_full_file_gt_131072_ids": strata["full_file_gt_131072"],
        "category_census_occurrences": dict(sorted(category_census.items())),
        "recommendation_counts": dict(sorted(Counter(recommendation_by_id.values()).items())),
        "selected_context_gt_131072": sum(int(row["selected_context_gt_131072"]) for row in detail_rows),
        "old_full_file_outlier_recovered_by_source_spans": sum(int(row["old_full_file_gt_131072"]) and not int(row["selected_context_gt_131072"]) for row in detail_rows),
        "target_gt_1024_audited": sum(int(row["target_gt_1024"]) for row in detail_rows),
        "semantic_support_claims": 0,
        "source_text_written": False,
        "target_text_written": False,
        "cuda": False,
        "training_admission": False,
        "input_pins": input_data["pins"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.output / "review-summary.json", summary)
    write_json(args.output / "sample-strata.json", {
        "schema": "sepalith.dat10.roxy_context_admission_review_strata.v1",
        "strata": {name: ids for name, ids in strata.items()},
        "sample_policy": "all formal mismatches (172), all pending semantic rows (392), first 256 unresolved IDs ordered by SHA256(row_id), plus all four prior full-file >131072 rows",
    })
    write_json(args.output / "status.json", {
        "schema": "sepalith.dat10.roxy_context_admission_review_status.v2",
        "status": summary["status"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scope_rows": len(row_ids),
        "full_candidate_scope_rows": len(profiles),
        "detail_rows": len(detail_rows),
        "error_rows": len(errors),
        "recommended_context_rows": len(accepted),
        "held_rows": len(held),
        "elapsed_seconds": summary["elapsed_seconds"],
        "cpu_threads_max": 2,
        "cuda": False,
        "training_admission": False,
        "frozen_v1_unchanged": True,
        "input_pins": input_data["pins"],
    })


if __name__ == "__main__":
    main()
