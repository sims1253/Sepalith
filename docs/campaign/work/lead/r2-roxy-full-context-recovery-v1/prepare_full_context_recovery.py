#!/usr/bin/env python3
"""Prepare a source-only recovery ledger for held roxygen references.

The input denominator is the 5,942 rows held by the corrected prior
admission review.  This packet tests whether a full source file restores a
same-file definition and separately recognizes ordinary R APIs and package
imports.  It never emits source or target text and does not admit data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
REVIEW_PACKET = PLAN / "docs/campaign/work/lead/r2-roxy-context-admission-review-v1"
PRIOR_FULL = REVIEW_PACKET / "full"
SCOPE_CORRECTED_FULL = REVIEW_PACKET / "full-v3"
PRIOR_RECEIPT = PLAN / "docs/campaign/receipts/DAT-10-roxy-context-admission-review.json"
SUPPORT_LEDGER = PLAN / "docs/campaign/work/lead/r2-source-walk-support-review-v1/support-ledger.jsonl"
ROOT_IDS = PLAN / "docs/campaign/work/lead/r2-roxy10017-root-context-review-v1/candidate-ids.json"
OUT_DEFAULT = Path("docs/campaign/work/lead/r2-roxy-full-context-recovery-v1")

if str(REVIEW_PACKET) not in sys.path:
    sys.path.insert(0, str(REVIEW_PACKET))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module_missing:" + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADMISSION = load_module("dat10_roxy_recovery_admission", REVIEW_PACKET / "review_context_admission.py")
SELECTOR = ADMISSION.SELECTOR


def _fast_line_span(source: bytes, node: Any) -> tuple[int, int]:
    """Use the C-level byte counter instead of rescanning every source line.

    The prior audit helper builds a complete line-start array for every AST
    definition.  That is correct but quadratic for large source files when a
    held-only replay invokes it thousands of times.  This local runtime
    override is equivalent because R source is already normalized to LF by
    ``reconstruct``; it does not change the prior packet's source or outputs.
    """
    start_byte = int(node.start_byte)
    end_byte = max(start_byte, int(node.end_byte) - 1)
    return source.count(b"\n", 0, start_byte), source.count(b"\n", 0, end_byte)


ADMISSION._line_span = _fast_line_span

from tree_sitter import Language, Parser  # noqa: E402
import tree_sitter_r  # noqa: E402

R_LANGUAGE = Language(tree_sitter_r.language())
R_PARSER = Parser(R_LANGUAGE)

_IMPORT_FROM = re.compile(r"\bimportFrom\s*\(\s*([A-Za-z][A-Za-z0-9_.]*)\s*,(.*?)\)", re.S)
_IMPORT_PACKAGE = re.compile(r"\bimport\s*\(\s*([A-Za-z][A-Za-z0-9_.]*)\s*\)")
_IMPORT_SYMBOL = re.compile(r"(?:`([^`]+)`|([A-Za-z.][A-Za-z0-9_.]*))")


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


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row_id = str(row["row_id"])
            if row_id in rows:
                raise ValueError("duplicate_row:" + row_id)
            rows[row_id] = row
    return rows


def standard_r_symbols() -> tuple[set[str], dict[str, Any]]:
    """Read installed base/recommended namespace exports without source data."""
    expression = (
        "ns <- c('base','utils','stats','graphics','grDevices','methods','datasets','tools'); "
        "for (n in ns) { x <- tryCatch(getNamespaceExports(n), error=function(e) character()); "
        "cat(n, '\\t', paste(x, collapse='\\n'), '\\n', sep='') }"
    )
    try:
        completed = subprocess.run(
            ["Rscript", "--vanilla", "-e", expression],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return set(), {"available": False, "error": type(exc).__name__ + ":" + str(exc)}
    symbols: set[str] = set(ADMISSION.R_BUILTINS)
    namespace_counts: dict[str, int] = {}
    current_namespace = None
    for line in completed.stdout.splitlines():
        if "\t" in line:
            namespace, symbol = line.split("\t", 1)
            current_namespace = namespace
            namespace_counts.setdefault(namespace, 0)
            for name in symbol.splitlines():
                if name:
                    symbols.add(name)
                    namespace_counts[namespace] += 1
        elif current_namespace and line:
            symbols.add(line)
            namespace_counts[current_namespace] = namespace_counts.get(current_namespace, 0) + 1
    # Namespaces print one symbol per line after the tab. The fallback lexical
    # set keeps the audit deterministic if a local R installation changes.
    return symbols, {
        "available": True,
        "command": expression,
        "stdout_sha256": sha_bytes(completed.stdout.encode()),
        "namespace_export_counts": namespace_counts,
        "symbol_count": len(symbols),
    }


def package_root(source_path: Path) -> Path | None:
    for parent in source_path.parents:
        if (parent / "DESCRIPTION").is_file() and (parent / "NAMESPACE").is_file():
            return parent
    return None


def namespace_metadata(source_path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    root = package_root(source_path)
    key = str(root) if root else "missing:" + str(source_path)
    if key in cache:
        return cache[key]
    if root is None:
        result = {
            "package_root": None,
            "namespace_exists": False,
            "namespace_sha256": None,
            "description_sha256": None,
            "imported_symbols": {},
            "imported_packages": [],
        }
        cache[key] = result
        return result
    namespace_path = root / "NAMESPACE"
    description_path = root / "DESCRIPTION"
    namespace = namespace_path.read_text(encoding="utf-8", errors="replace")
    imported_symbols: dict[str, set[str]] = defaultdict(set)
    for match in _IMPORT_FROM.finditer(namespace):
        package = match.group(1)
        for symbol_match in _IMPORT_SYMBOL.finditer(match.group(2)):
            symbol = symbol_match.group(1) or symbol_match.group(2)
            if symbol:
                imported_symbols[symbol].add(package)
    imported_packages = sorted(set(_IMPORT_PACKAGE.findall(namespace)))
    result = {
        "package_root": str(root),
        "namespace_exists": True,
        "namespace_sha256": sha_file(namespace_path),
        "description_sha256": sha_file(description_path),
        "imported_symbols": {name: sorted(packages) for name, packages in sorted(imported_symbols.items())},
        "imported_packages": imported_packages,
    }
    cache[key] = result
    return result


def top_level_names(source_path: Path, expected_sha256: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(source_path)
    if key in cache:
        return cache[key]
    source = source_path.read_bytes()
    if sha_bytes(source) != expected_sha256:
        raise ValueError("source_hash_mismatch")
    tree = R_PARSER.parse(source)
    if tree.root_node.has_error:
        raise ValueError("source_parse_error")
    lines = source.decode("utf-8", "replace").splitlines()
    definitions = SELECTOR.top_level_definitions(tree, source, len(lines))
    result = {
        "source_sha256": expected_sha256,
        "top_level_names": sorted({definition.name for definition in definitions}),
        "top_level_definition_count": len(definitions),
    }
    cache[key] = result
    return result


def free_name_class(name: str, categories: dict[str, int], metadata: dict[str, Any], standard: set[str], top_names: set[str]) -> str:
    if name in top_names:
        return "same_file_top_level_definition"
    if "package_namespace_reference" in categories:
        return "package_namespace_reference"
    if "r_builtin_or_operator" in categories and name in standard:
        return "standard_r_api_or_operator"
    imported = metadata.get("imported_symbols", {})
    if name in imported:
        return "exact_namespace_import"
    # An unqualified call is evidence of a callable dependency, not proof that
    # the source omitted a definition. Keep it separate for root semantic/API
    # review rather than silently admitting it as base R.
    if "external_callable_or_global" in categories:
        return "unqualified_external_callable"
    return "residual_noncall_free_reference"


def classify_row(
    row: dict[str, Any],
    corrected: dict[str, Any],
    source_cache: dict[str, dict[str, Any]],
    metadata_cache: dict[str, dict[str, Any]],
    standard: set[str],
) -> dict[str, Any]:
    source_path = Path(row["source_path"])
    source = top_level_names(source_path, row["source_sha256"], source_cache)
    metadata = namespace_metadata(source_path, metadata_cache)
    top_names = set(source["top_level_names"])
    free_names = {
        name: categories
        for name, categories in corrected.get("unresolved_name_categories", {}).items()
        if "free_variable_or_external_global" in categories
    }
    classifications = {
        name: free_name_class(name, categories, metadata, standard, top_names)
        for name, categories in sorted(free_names.items())
    }
    class_counts = Counter(classifications.values())
    full_available = int(row["full_file_sequence_tokens"]) <= 131072
    same_file = sorted(name for name, value in classifications.items() if value == "same_file_top_level_definition")
    # A full-file fallback is useful only when it is within the shared serving
    # bound and restores a definition actually named by the selected target.
    fallback_useful = bool(full_available and same_file)
    residual = sorted(
        name for name, value in classifications.items()
        if value == "residual_noncall_free_reference"
    )
    nonlocal_known = all(
        value in {
            "standard_r_api_or_operator",
            "exact_namespace_import",
            "package_namespace_reference",
            "unqualified_external_callable",
        }
        for value in classifications.values()
    )
    if corrected.get("recommendation") == "recommend_context_admission_pending_root_semantic_review":
        recovery = "recoverable_by_scope_correction"
    elif fallback_useful:
        recovery = "recoverable_full_context_same_file_definition"
    elif nonlocal_known and classifications:
        recovery = "recoverable_ordinary_r_or_imported_reference_pending_root_api_review"
    else:
        recovery = "hold_residual_noncall_reference_repair_or_semantic_evidence"
    return {
        "row_id": row["row_id"],
        "family": row["family"],
        "group_id": row["group_id"],
        "source_path": row["source_path"],
        "source_sha256": row["source_sha256"],
        "target_definition_name": row["target_definition_name"],
        "target_definition_span": row["target_definition_span"],
        "target_body_tokens": row["target_body_tokens"],
        "selected_context_tokens": row["selected_sequence_tokens"],
        "full_file_tokens": row["full_file_sequence_tokens"],
        "full_file_within_131072": full_available,
        "full_file_fallback_useful": fallback_useful,
        "same_file_top_level_names": same_file,
        "docs_evidence": row["docs_evidence"],
        "prior_recommendation": row["recommendation"],
        "corrected_recommendation": corrected.get("recommendation"),
        "corrected_v1_status": corrected.get("v1_status"),
        "free_reference_names": sorted(free_names),
        "free_reference_classes": classifications,
        "free_reference_class_counts": dict(sorted(class_counts.items())),
        "residual_noncall_names": residual,
        "namespace_import_evidence": {
            "package_root": metadata.get("package_root"),
            "namespace_sha256": metadata.get("namespace_sha256"),
            "description_sha256": metadata.get("description_sha256"),
            "imported_symbol_names": sorted(metadata.get("imported_symbols", {})),
            "imported_packages": metadata.get("imported_packages", []),
        },
        "recovery_status": recovery,
        "semantic_support_claim": False,
        "admission": "review_only_unadmitted",
        "source_text_written": False,
        "target_text_written": False,
    }


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--id-file",
        type=Path,
        help="Optional JSON list selecting a disjoint subset of the pinned held denominator.",
    )
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
    started = datetime.now(timezone.utc)
    prior = load_rows(PRIOR_FULL / "audited-rows.jsonl")
    if len(prior) != 10017:
        raise ValueError("prior_scope_mismatch")
    held_ids = sorted(
        row_id for row_id, row in prior.items()
        if row.get("recommendation") == "hold_true_omitted_or_external_global_evidence"
    )
    if len(held_ids) != 5942:
        raise ValueError(f"held_denominator_mismatch:{len(held_ids)}")
    full_held_ids = held_ids
    if args.id_file is not None:
        selected_ids = json.loads(args.id_file.read_text(encoding="utf-8"))
        if not isinstance(selected_ids, list) or len(selected_ids) != len(set(selected_ids)):
            raise ValueError("id_file_must_be_unique_json_list")
        held_ids = sorted(str(row_id) for row_id in selected_ids)
        if not set(held_ids).issubset(set(full_held_ids)):
            raise ValueError("id_file_outside_held_denominator")
    scope_corrected = load_rows(SCOPE_CORRECTED_FULL / "audited-rows.jsonl")
    if len(scope_corrected) != 10017 or set(scope_corrected) != set(prior):
        raise ValueError("corrected_scope_mismatch")
    root_ids = set(json.loads(ROOT_IDS.read_text(encoding="utf-8")))
    if root_ids != set(prior):
        raise ValueError("root_id_scope_mismatch")
    # Replay exactly the held denominator through the current admission
    # classifier.  The historical full-v3 rows remain pinned for comparison,
    # but cannot be used as the current result: the classifier was subsequently
    # corrected for nearest-function scope and R `$`/`@` member labels.
    profiles, ledgers, packets, admission_inputs = ADMISSION.load_inputs()
    if set(profiles) != set(prior) or set(ledgers) != set(prior) or set(packets) != set(prior):
        raise ValueError("current_admission_scope_mismatch")
    standard, standard_meta = standard_r_symbols()
    write_json(args.output / "status.json", {
        "schema": "sepalith.dat10.roxy_full_context_recovery_status.v1",
        "status": "running",
        "started_at": started.isoformat(),
        "scope_rows": len(held_ids),
        "prior_held_denominator": len(full_held_ids),
        "selected_held_denominator": len(held_ids),
        "rows_done": 0,
        "cpu_threads_max": 2,
        "cuda": False,
        "training_admission": False,
        "source_text_written": False,
        "target_text_written": False,
        "prior_review_outputs_unchanged": True,
        "standard_r_inventory": standard_meta,
    })
    source_cache: dict[str, dict[str, Any]] = {}
    metadata_cache: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    recovery_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    source_hashes: dict[str, str] = {}
    for count, row_id in enumerate(held_ids, 1):
        try:
            corrected = ADMISSION.classify_row(profiles[row_id], ledgers[row_id], packets[row_id])
            result = classify_row(prior[row_id], corrected, source_cache, metadata_cache, standard)
            output_rows.append(result)
            recovery_counts[result["recovery_status"]] += 1
            class_counts.update(result["free_reference_class_counts"])
            source_hashes[result["source_path"]] = result["source_sha256"]
        except Exception as exc:
            errors.append({"row_id": row_id, "error": type(exc).__name__ + ":" + str(exc)})
        if count == 1 or count % 100 == 0 or count == len(held_ids):
            write_json(args.output / "status.json", {
                "schema": "sepalith.dat10.roxy_full_context_recovery_status.v1",
                "status": "running",
                "started_at": started.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "scope_rows": len(held_ids),
                "prior_held_denominator": len(full_held_ids),
                "selected_held_denominator": len(held_ids),
                "rows_done": count,
                "detail_rows": len(output_rows),
                "error_rows": len(errors),
                "source_files_seen": len(source_cache),
                "cpu_threads_max": 2,
                "cuda": False,
                "training_admission": False,
                "source_text_written": False,
                "target_text_written": False,
                "prior_review_outputs_unchanged": True,
            })
    output_rows.sort(key=lambda row: row["row_id"])
    errors.sort(key=lambda row: row["row_id"])
    write_jsonl(args.output / "recovery-ledger.jsonl", output_rows)
    write_jsonl(args.output / "recovery-errors.jsonl", errors)
    recoverable = sorted(
        row["row_id"] for row in output_rows
        if row["recovery_status"] != "hold_residual_noncall_reference_repair_or_semantic_evidence"
    )
    held = sorted(
        row["row_id"] for row in output_rows
        if row["recovery_status"] == "hold_residual_noncall_reference_repair_or_semantic_evidence"
    )
    write_json(args.output / "recoverable-ids.json", {
        "schema": "sepalith.dat10.roxy_full_context_recovery_ids.v1",
        "status": "recommendation_only_not_training_admission",
        "scope": "prior_5942_held_rows",
        "recoverable_count": len(recoverable),
        "recoverable_ids": recoverable,
    })
    write_json(args.output / "held-ids.json", {
        "schema": "sepalith.dat10.roxy_full_context_recovery_held_ids.v1",
        "status": "repair_or_semantic_evidence_required",
        "scope": "prior_5942_held_rows",
        "held_count": len(held),
        "held_ids": held,
    })
    representative = []
    for status in sorted(recovery_counts):
        candidates = [row for row in output_rows if row["recovery_status"] == status]
        representative.extend(candidates[: min(5, len(candidates))])
    write_json(args.output / "representative-evidence.json", {
        "schema": "sepalith.dat10.roxy_full_context_recovery_representatives.v1",
        "status": "review_only_no_source_or_target_text",
        "rows": representative,
    })
    summary = {
        "schema": "sepalith.dat10.roxy_full_context_recovery_summary.v1",
        "status": "complete" if not errors else "complete_with_errors",
        "prior_held_denominator": len(full_held_ids),
        "selected_held_denominator": len(held_ids),
        "rows_audited": len(output_rows),
        "error_rows": len(errors),
        "recoverable_count": len(recoverable),
        "residual_held_count": len(held),
        "recovery_status_counts": dict(sorted(recovery_counts.items())),
        "free_reference_class_counts": dict(sorted(class_counts.items())),
        "full_file_within_131072_rows": sum(int(row["full_file_within_131072"]) for row in output_rows),
        "full_file_fallback_useful_rows": sum(int(row["full_file_fallback_useful"]) for row in output_rows),
        "same_file_definition_recovery_rows": sum(bool(row["same_file_top_level_names"]) for row in output_rows),
        "source_files_seen": len(source_cache),
        "package_metadata_roots_seen": len(metadata_cache),
        "target_gt_1024_rows": sum(int(row["target_body_tokens"]) > 1024 for row in output_rows),
        "semantic_support_claims": 0,
        "source_text_written": False,
        "target_text_written": False,
        "training_admission": False,
        "prior_review_outputs_unchanged": True,
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "input_pins": {
            "prior_review_receipt": {"path": str(PRIOR_RECEIPT), "sha256": sha_file(PRIOR_RECEIPT)},
            "prior_full_audited_rows": {"path": str(PRIOR_FULL / "audited-rows.jsonl"), "rows": 10017, "sha256": sha_file(PRIOR_FULL / "audited-rows.jsonl")},
            "scope_corrected_audited_rows": {"path": str(SCOPE_CORRECTED_FULL / "audited-rows.jsonl"), "rows": 10017, "sha256": sha_file(SCOPE_CORRECTED_FULL / "audited-rows.jsonl")},
            "current_admission_classifier": {"path": str(REVIEW_PACKET / "review_context_admission.py"), "sha256": sha_file(REVIEW_PACKET / "review_context_admission.py"), "replayed_held_rows": len(held_ids), "scope": "exact held denominator only"},
            "root_candidate_ids": {"path": str(ROOT_IDS), "rows": 10017, "sha256": sha_file(ROOT_IDS)},
            "support_ledger": {"path": str(SUPPORT_LEDGER), "rows": 10017, "sha256": sha_file(SUPPORT_LEDGER)},
            "standard_r_inventory": standard_meta,
        },
    }
    write_json(args.output / "recovery-summary.json", summary)
    write_json(args.output / "status.json", {
        "schema": "sepalith.dat10.roxy_full_context_recovery_status.v1",
        "status": summary["status"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scope_rows": len(held_ids),
        "rows_audited": len(output_rows),
        "error_rows": len(errors),
        "recoverable_count": len(recoverable),
        "residual_held_count": len(held),
        "cpu_threads_max": 2,
        "cuda": False,
        "training_admission": False,
        "source_text_written": False,
        "target_text_written": False,
        "prior_review_outputs_unchanged": True,
        "summary_path": str(args.output / "recovery-summary.json"),
    })


if __name__ == "__main__":
    main()
