#!/usr/bin/env python3
"""Audit and materialize the bounded DAT-10 task-data recovery candidate.

This pass is deliberately independent of the existing accepted packets.  It
joins corrected rows to DAT-05 by exact row ID, applies the DAT-02 global
training split, and treats the CPT partition as a validation reservation only.
The resulting files are candidate artifacts.  They retain complete tokenized
rows and carry a separate current-task-cap gate; this script never truncates,
repairs, or admits a row and never reads DEV labels or prompt content.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    except OSError:
        pass
sys.dont_write_bytecode = True


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign")
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
OUT = PLAN / "work/lead/r2-data-expansion-audit-v1"
CORRECTED = PLAN / "work/lead/finish-corrected-train-v1/train-token-rows.jsonl"
SHORT = PLAN / "work/r2-short-task-preparation-v1/candidate-packets.jsonl"
DAT05_PROVENANCE = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-registry-v1/provenance.jsonl"
)
DAT02 = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
PARTITION = PLAN / "work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
TOKENIZER = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/"
    "SFT-primary-step1000-theta0/tokenizer.json"
)
PROTOCOL = EXEC / "packages/sepalith/src/sepalith/campaign_protocol.py"
SAMPLER = EXEC / "experiments/training/campaign_sampling.py"
TASK_TRAINER = PLAN / "work/r2-task-trainer-review-v2/source/experiments/training/campaign_task_sft.py"

RL_ELIGIBLE = PLAN / "work/corrected-rl-admission-audit-v1/candidate-data/eligible-train-rows.jsonl"
RL_SCHEDULE = PLAN / "work/corrected-rl-admission-audit-v1/candidate-data/source-row-draw-sequence.json"

FULL_ROWS = OUT / "expanded-corrected-short-token-rows.jsonl"
FULL_PROVENANCE = OUT / "expanded-corrected-short-provenance.jsonl"
CURRENT_ROWS = OUT / "expanded-current-192-token-rows.jsonl"
CURRENT_PROVENANCE = OUT / "expanded-current-192-provenance.jsonl"
EXCLUSION_LEDGER = OUT / "expanded-exclusion-ledger.jsonl"
SELECTION = OUT / "expanded-row-selection.json"
SCHEDULE_METADATA = OUT / "expanded-schedule-metadata.json"
RL_COVERAGE = OUT / "rl-coverage.json"
REPORT = OUT / "audit-report.json"


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def iter_jsonl(path: Path) -> Iterable[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for line_number, raw in enumerate(stream, 1):
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            yield line_number, raw, value


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_split_and_partition() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    split_doc = json.loads(DAT02.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, Any]] = {}
    for value in split_doc.get("groups", []):
        if not isinstance(value, dict) or not isinstance(value.get("group_id"), str):
            raise ValueError("DAT-02 contains a group without a string group_id")
        group_id = value["group_id"]
        if group_id in groups:
            raise ValueError(f"duplicate DAT-02 group {group_id}")
        groups[group_id] = value
    partition_doc = json.loads(PARTITION.read_text(encoding="utf-8"))
    partition = partition_doc.get("groups")
    if not isinstance(partition, dict):
        raise ValueError("CPT partition groups must be an object")
    if sum(value == "cpt_train" for value in partition.values()) != 10163:
        raise ValueError("unexpected CPT train group count")
    if sum(value == "cpt_validation" for value in partition.values()) != 556:
        raise ValueError("unexpected CPT validation group count")
    if split_doc.get("split_id") != partition_doc.get("split_id"):
        raise ValueError("DAT-02/CPT partition split identities differ")
    return split_doc, groups, {str(key): str(value) for key, value in partition.items()}


def classify_group(
    group_id: object,
    groups: Mapping[str, Mapping[str, Any]],
    partition: Mapping[str, str],
) -> str:
    """Return the repaired task binding status for one exact DAT-02 group."""
    if not isinstance(group_id, str) or not group_id:
        return "missing_group_id"
    group = groups.get(group_id)
    if group is None:
        return "unknown_dat02_group"
    split = group.get("split")
    if split in {"dev_group", "final_candidate_group"}:
        return f"global_{split}_rejected"
    if split not in {"train", "train_group"}:
        return f"global_nontrain_{split}"
    cpt_status = partition.get(group_id)
    if cpt_status == "cpt_validation":
        return "cpt_validation_reserved"
    if cpt_status is None:
        # Task-authored groups are valid DAT-02 train groups even when the
        # CPT-only source partition has no entry for them.
        return "task_train_unmapped_cpt"
    if cpt_status != "cpt_train":
        return "partition_not_cpt_train"
    return "task_train_cpt_train"


def old_binding_reason(status: str, target_labels: int, target_cap: int) -> str | None:
    """Reproduce the old builder's first exclusion reason for known rows."""
    if status == "task_train_unmapped_cpt":
        return "group_not_in_cpt_partition"
    if status == "cpt_validation_reserved":
        return "cpt_validation_reserved"
    if status == "task_train_cpt_train" and target_labels > target_cap:
        return "target_over_192_including_EOS"
    return None


def row_contract_status(row: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
    """Check the complete token boundary without changing row data."""
    ident = row.get("id")
    if not isinstance(ident, str) or not ident:
        return "row_id_missing", {}
    ids = row.get("input_ids")
    start = row.get("target_start")
    target_count = row.get("target_token_count")
    prompt_count = row.get("prompt_token_count")
    body = row.get("target_body_tokens")
    terminal = row.get("target_terminal_tokens")
    if (
        not isinstance(ids, list)
        or any(type(token) is not int for token in ids)
        or type(start) is not int
        or type(target_count) is not int
        or type(prompt_count) is not int
        or not isinstance(body, list)
        or not isinstance(terminal, list)
    ):
        return "row_contract_shape", {}
    target_labels = target_count + 1
    total = len(ids)
    if (
        total < 4
        or ids[0] != 0
        or ids[-1] != 1
        or ids.count(0) != 1
        or ids.count(1) != 1
        or start <= 1
        or start + target_labels != total
        or prompt_count + 1 != start
        or not terminal
        or target_count != len(body) + len(terminal)
        or ids[start : start + len(body)] != body
        or ids[start + len(body) : start + len(body) + len(terminal)] != terminal
    ):
        return "row_contract_boundary", {"target_labels": target_labels, "total_tokens": total}
    return "ok", {"target_labels": target_labels, "total_tokens": total}


def cap_status(metrics: Mapping[str, int], *, target_cap: int = 192, total_cap: int = 4096) -> str:
    if metrics.get("total_tokens", 0) > total_cap:
        return "total_over_4096"
    if metrics.get("target_labels", 0) > target_cap:
        return "target_over_192_including_EOS"
    return "within_current_task_cap"


def family_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("family")) for row in rows).items()))


def target_bins(values: Iterable[int]) -> dict[str, int]:
    bins = Counter()
    for value in values:
        if value <= 192:
            bins["<=192"] += 1
        elif value <= 256:
            bins["193-256"] += 1
        elif value <= 384:
            bins["257-384"] += 1
        elif value <= 512:
            bins["385-512"] += 1
        elif value <= 1024:
            bins["513-1024"] += 1
        else:
            bins[">1024"] += 1
    order = ("<=192", "193-256", "257-384", "385-512", "513-1024", ">1024")
    return {name: bins[name] for name in order if bins[name]}


def threshold_table(records: Iterable[Mapping[str, Any]], field: str, limits: Iterable[int]) -> list[dict[str, int]]:
    values = [int(record[field]) for record in records]
    return [
        {
            "cap": int(limit),
            "rows_within_cap": sum(value <= limit for value in values),
            "rows_over_cap": sum(value > limit for value in values),
        }
        for limit in limits
    ]


def source_fields(meta: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "source", "source_file", "source_line", "source_sha256", "raw_line_sha256",
        "source_snapshot_path", "source_snapshot_sha256", "source_snapshot_is_simulated",
        "source_verification", "input_row_id", "source_document_id", "source_role",
    )
    return {key: meta[key] for key in keys if key in meta and meta[key] is not None}


def make_provenance(
    row: Mapping[str, Any],
    *,
    source_class: str,
    line_number: int,
    source_meta: Mapping[str, Any],
    group_id: str,
    status: str,
    group: Mapping[str, Any],
    partition: Mapping[str, str],
    metrics: Mapping[str, int],
    current_target_cap: int,
) -> dict[str, Any]:
    old_reason = old_binding_reason(status, metrics["target_labels"], current_target_cap)
    current_gate = cap_status(metrics, target_cap=current_target_cap)
    source_ref = source_meta.get("source_ref") if isinstance(source_meta.get("source_ref"), Mapping) else {}
    license_state = (
        "direct_source_ref_evidence"
        if source_ref.get("license_evidence_present") is True
        else "no_row_level_license_evidence"
    )
    if status == "task_train_cpt_train":
        repaired_binding = "task_train_cpt_train"
    elif status == "task_train_unmapped_cpt":
        repaired_binding = "task_train_unmapped_cpt"
    else:
        repaired_binding = status
    value: dict[str, Any] = {
        "schema": "DAT-10-r2-expansion-audit-v1",
        "id": row.get("id"),
        "input_row_id": row.get("id"),
        "source_class": source_class,
        "source_line": line_number,
        "family": row.get("family"),
        "package_id": row.get("package_id"),
        "group_id": group_id,
        "global_split": group.get("split"),
        "global_split_flags": list(group.get("flags", [])) if isinstance(group.get("flags", []), list) else [],
        "cpt_partition": partition.get(group_id),
        "legacy_binding_status": (
            "group_not_in_cpt_partition" if status == "task_train_unmapped_cpt" else status
        ),
        "repaired_binding_status": repaired_binding,
        "target_label_tokens_including_protocol_EOS": metrics["target_labels"],
        "total_tokens": metrics["total_tokens"],
        "current_target_cap": current_target_cap,
        "current_cap_gate": current_gate,
        "candidate_status": "candidate_only_pending_root_quality_source_license_duplicate_checks",
        "license_evidence_state": license_state,
        "row_canonical_sha256": sha256_bytes(canonical(row)),
        "prompt_sha256": sha256_bytes(str(row.get("prompt_text", "")).encode("utf-8")),
        "target_sha256": sha256_bytes(str(row.get("target_text", "")).encode("utf-8")),
        "tokenizer_json_sha256": row.get("tokenizer_json_sha256"),
        "tokenizer_revision": row.get("tokenizer_revision"),
        "tokenization_policy": row.get("tokenization_policy"),
        "source_join": (
            "DAT-05 exact row id -> group_id; DAT-02 global split; CPT partition validation reservation only"
            if source_class == "corrected"
            else "short packet source_provenance.group_id -> DAT-02 global split; CPT partition validation reservation only"
        ),
    }
    if old_reason is not None:
        value["legacy_exclusion_reason"] = old_reason
    value.update(source_fields(source_meta))
    return {key: val for key, val in value.items() if val is not None}


def repaired_action(status: str, current_gate: str) -> str:
    if status == "cpt_validation_reserved":
        return "preserve_exclusion_global_cpt_validation"
    if status == "task_train_unmapped_cpt":
        if current_gate == "within_current_task_cap":
            return "recover_under_global_task_train_rebind"
        return "recover_as_cap_deferred_under_global_task_train_rebind"
    if current_gate == "target_over_192_including_EOS":
        return "retain_complete_candidate_defer_until_task_cap_identity_changes"
    if current_gate == "total_over_4096":
        return "exclude_total_sequence_over_geometry_cap"
    return "preserve_current_candidate"


def build_sampler_metadata(row: Mapping[str, Any], group_id: str, source_class: str) -> dict[str, Any]:
    total = len(row["input_ids"])
    target = int(row["target_token_count"]) + 1
    return {
        "row_id": row["id"],
        "family": str(row["family"]),
        "source_id": group_id,
        "package_id": str(row["package_id"]),
        "split": "train",
        "semantic_noop": row["target_operation"] == "no_op",
        "operation": row["target_operation"],
        "prompt_tokens": int(row["target_start"]),
        "target_tokens": target,
        "total_tokens": total,
        "length_bucket": "short" if total <= 2048 else "long",
        "naturally_long": total > 2048,
        "source_kind": "ordinary",
        "provenance": f"expansion-{source_class}:{group_id}",
    }


def audit_rl() -> dict[str, Any]:
    eligible: dict[str, str] = {}
    eligible_rows = 0
    for _line, _raw, row in iter_jsonl(RL_ELIGIBLE):
        ident = row.get("id")
        if not isinstance(ident, str) or ident in eligible:
            raise ValueError("RL eligible rows contain a missing or duplicate id")
        eligible[ident] = str(row.get("family"))
        eligible_rows += 1
    schedule_doc = json.loads(RL_SCHEDULE.read_text(encoding="utf-8"))
    draw_ids = schedule_doc.get("row_ids")
    if not isinstance(draw_ids, list) or any(not isinstance(item, str) for item in draw_ids):
        raise ValueError("RL schedule row_ids is not a string list")
    unknown = sorted(set(draw_ids) - set(eligible))
    if unknown:
        raise ValueError(f"RL schedule contains {len(unknown)} IDs outside eligible rows")
    distinct = set(draw_ids)
    missing = sorted(set(eligible) - distinct)
    candidate_count = int(schedule_doc.get("candidate_count", 0))
    buffer_reuse = int(schedule_doc.get("buffer_reuse", 0))
    source_draws_per_update = int(schedule_doc.get("prompt_groups_per_update", 0))
    family_draws = Counter(eligible[item] for item in draw_ids)
    family_unique = Counter(eligible[item] for item in distinct)
    omitted_families = Counter(eligible[item] for item in missing)
    prefix_updates = (1, 5, 25, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000)
    prefixes = []
    for updates in prefix_updates:
        prefix_count = min(len(draw_ids), updates * source_draws_per_update)
        prefix = draw_ids[:prefix_count]
        prefixes.append({
            "updates": updates,
            "source_draws": prefix_count,
            "distinct_source_ids": len(set(prefix)),
            "declared_generated_candidates": prefix_count * candidate_count,
            "declared_policy_row_exposures": prefix_count * candidate_count * buffer_reuse,
        })
    result = {
        "source": {
            "eligible_rows": eligible_rows,
            "eligible_file": str(RL_ELIGIBLE),
            "eligible_file_sha256": sha256_file(RL_ELIGIBLE)[0],
            "schedule_file": str(RL_SCHEDULE),
            "schedule_file_sha256": sha256_file(RL_SCHEDULE)[0],
        },
        "schedule": {
            "source_draws": len(draw_ids),
            "source_draws_per_update": source_draws_per_update,
            "candidate_count": candidate_count,
            "buffer_reuse": buffer_reuse,
            "completions_per_update": int(schedule_doc.get("completions_per_update", 0)),
            "no_truncation": schedule_doc.get("no_truncation"),
        },
        "coverage": {
            "distinct_source_ids": len(distinct),
            "source_id_coverage_fraction": len(distinct) / eligible_rows if eligible_rows else 0.0,
            "omitted_source_ids": len(missing),
            "omitted_families": dict(sorted(omitted_families.items())),
            "draws_by_family": dict(sorted(family_draws.items())),
            "distinct_source_ids_by_family": dict(sorted(family_unique.items())),
            "actual_rollout_coverage": "not evidenced; candidate-data contains schedule and source rows only",
            "coverage_unit": "source prompt IDs; each source draw declares four completions, buffer reuse is repeated policy exposure",
        },
        "prefixes": prefixes,
    }
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    outputs = (
        FULL_ROWS, FULL_PROVENANCE, CURRENT_ROWS, CURRENT_PROVENANCE,
        EXCLUSION_LEDGER, SELECTION, SCHEDULE_METADATA, RL_COVERAGE, REPORT,
    )
    if any(path.exists() for path in outputs):
        raise RuntimeError("refusing to overwrite an expansion audit artifact")
    inputs = (CORRECTED, SHORT, DAT05_PROVENANCE, DAT02, PARTITION, TOKENIZER, PROTOCOL, SAMPLER, TASK_TRAINER, RL_ELIGIBLE, RL_SCHEDULE)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing pinned input(s): " + ", ".join(missing))
    started = datetime.now(timezone.utc)
    initial_signatures = {str(path): file_signature(path) for path in inputs}
    input_hashes: dict[str, dict[str, Any]] = {}
    # Hashing these inputs pins the exact audit basis.  No raw source corpus is
    # opened here; DAT-05 is used only as an exact row-ID provenance registry.
    for path in inputs:
        digest, size = sha256_file(path)
        input_hashes[str(path)] = {"sha256": digest, "bytes": size}

    split_doc, groups, partition = load_split_and_partition()
    corrected_rows: list[tuple[int, dict[str, Any]]] = []
    corrected_ids: set[str] = set()
    for line, _raw, row in iter_jsonl(CORRECTED):
        ident = row.get("id")
        if not isinstance(ident, str) or not ident or ident in corrected_ids:
            raise ValueError(f"corrected rows contain missing or duplicate ID at line {line}")
        corrected_ids.add(ident)
        corrected_rows.append((line, row))
    corrected_provenance: dict[str, dict[str, Any]] = {}
    for _line, _raw, item in iter_jsonl(DAT05_PROVENANCE):
        ident = item.get("id")
        if ident not in corrected_ids:
            continue
        if ident in corrected_provenance:
            raise ValueError(f"duplicate DAT-05 provenance for corrected row {ident}")
        corrected_provenance[ident] = item
    if len(corrected_provenance) != len(corrected_rows):
        missing_ids = len(corrected_rows) - len(corrected_provenance)
        raise ValueError(f"{missing_ids} corrected IDs lack exact DAT-05 provenance")

    protocol = load_module("dat10_expansion_protocol", PROTOCOL)
    full_rows_out = FULL_ROWS.open("x", encoding="utf-8")
    full_prov_out = FULL_PROVENANCE.open("x", encoding="utf-8")
    current_rows_out = CURRENT_ROWS.open("x", encoding="utf-8")
    current_prov_out = CURRENT_PROVENANCE.open("x", encoding="utf-8")
    exclusion_out = EXCLUSION_LEDGER.open("x", encoding="utf-8")
    full_digest = hashlib.sha256()
    full_prov_digest = hashlib.sha256()
    current_digest = hashlib.sha256()
    current_prov_digest = hashlib.sha256()
    exclusion_digest = hashlib.sha256()
    all_candidate_metadata: list[dict[str, Any]] = []
    current_candidate_metadata: list[dict[str, Any]] = []
    prompt_seen: dict[str, tuple[str, str]] = {}
    status_counts = Counter()
    source_counts = Counter()
    family_all = Counter()
    family_full = Counter()
    family_current = Counter()
    target_by_status: dict[str, list[int]] = defaultdict(list)
    total_by_status: dict[str, list[int]] = defaultdict(list)
    all_global_metrics: list[dict[str, Any]] = []
    license_by_scope: defaultdict[str, Counter[str]] = defaultdict(Counter)
    full_ids: set[str] = set()
    current_ids: set[str] = set()
    exclusion_counts = Counter()
    contract_counts = Counter()
    all_rows_seen = 0
    protocol_errors: list[dict[str, Any]] = []

    def emit_candidate(
        row: dict[str, Any], provenance: dict[str, Any], metadata: dict[str, Any], *, current: bool
    ) -> None:
        nonlocal full_digest, full_prov_digest, current_digest, current_prov_digest
        row_line = canonical(row) + b"\n"
        prov_line = canonical(provenance) + b"\n"
        if current:
            current_rows_out.write(row_line.decode("utf-8"))
            current_prov_out.write(prov_line.decode("utf-8"))
            current_digest.update(row_line)
            current_prov_digest.update(prov_line)
            current_candidate_metadata.append(metadata)
            current_ids.add(str(row["id"]))
            family_current[str(row["family"])] += 1
        else:
            full_rows_out.write(row_line.decode("utf-8"))
            full_prov_out.write(prov_line.decode("utf-8"))
            full_digest.update(row_line)
            full_prov_digest.update(prov_line)
            all_candidate_metadata.append(metadata)
            full_ids.add(str(row["id"]))
            family_full[str(row["family"])] += 1
        license_by_scope["current_candidate" if current else "full_candidate"][
            str(provenance.get("license_evidence_state"))
        ] += 1

    def inspect_row(
        row: dict[str, Any], *, source_class: str, line: int, source_meta: Mapping[str, Any], group_id: object
    ) -> None:
        nonlocal all_rows_seen
        all_rows_seen += 1
        source_counts[source_class] += 1
        family_all[str(row.get("family"))] += 1
        status = classify_group(group_id, groups, partition)
        status_counts[status] += 1
        group = groups.get(group_id) if isinstance(group_id, str) else None
        if group is None:
            group = {}
        contract_status, metrics = row_contract_status(row)
        contract_counts[contract_status] += 1
        if contract_status != "ok":
            protocol_errors.append({"id": row.get("id"), "source_class": source_class, "line": line, "reason": contract_status})
            return
        try:
            protocol.validate_training_row(row)
        except Exception as error:
            protocol_errors.append({"id": row.get("id"), "source_class": source_class, "line": line, "reason": f"protocol:{type(error).__name__}:{str(error)[:160]}"})
            return
        target_by_status[status].append(metrics["target_labels"])
        total_by_status[status].append(metrics["total_tokens"])
        all_global_metrics.append({"status": status, **metrics})
        current_gate = cap_status(metrics)
        provenance = make_provenance(
            row, source_class=source_class, line_number=line, source_meta=source_meta,
            group_id=str(group_id) if isinstance(group_id, str) else "",
            status=status, group=group, partition=partition, metrics=metrics, current_target_cap=192,
        )
        license_state = provenance["license_evidence_state"]
        license_by_scope["all_global_train"][license_state] += 1
        prompt_hash = provenance["prompt_sha256"]
        target_hash = provenance["target_sha256"]
        prior = prompt_seen.get(prompt_hash)
        if prior is not None:
            duplicate_reason = "prompt_duplicate_same_target" if prior[0] == target_hash else "prompt_conflicting_target"
            status_counts[duplicate_reason] += 1
            protocol_errors.append({"id": row.get("id"), "source_class": source_class, "line": line, "reason": duplicate_reason, "prior_id": prior[1]})
            return
        prompt_seen[prompt_hash] = (target_hash, str(row["id"]))
        legacy_reason = old_binding_reason(status, metrics["target_labels"], 192)
        if legacy_reason is not None:
            exclusion_counts[legacy_reason] += 1
            exclusion = {
                "schema": "DAT-10-r2-expansion-audit-v1",
                "id": row.get("id"),
                "source_class": source_class,
                "source_line": line,
                "family": row.get("family"),
                "package_id": row.get("package_id"),
                "group_id": group_id,
                "legacy_exclusion_reason": legacy_reason,
                "repaired_binding_status": status,
                "global_split": group.get("split"),
                "cpt_partition": partition.get(str(group_id)) if isinstance(group_id, str) else None,
                "target_label_tokens_including_protocol_EOS": metrics["target_labels"],
                "total_tokens": metrics["total_tokens"],
                "current_cap_gate": current_gate,
                "repaired_action": repaired_action(status, current_gate),
            }
            exclusion_line = canonical({key: value for key, value in exclusion.items() if value is not None}) + b"\n"
            exclusion_out.write(exclusion_line.decode("utf-8"))
            exclusion_digest.update(exclusion_line)
        if status not in {"task_train_cpt_train", "task_train_unmapped_cpt"}:
            return
        if current_gate == "total_over_4096":
            return
        metadata = build_sampler_metadata(row, str(group_id), source_class)
        emit_candidate(row, provenance, metadata, current=False)
        if current_gate == "within_current_task_cap":
            emit_candidate(row, provenance, metadata, current=True)

    for line, row in corrected_rows:
        ident = str(row["id"])
        prov = corrected_provenance[ident]
        if prov.get("decision") != "admitted" or prov.get("row_split") != "train":
            status_counts["DAT05_provenance_not_admitted_train"] += 1
            continue
        if prov.get("family") not in (None, row.get("family")) and prov.get("semantic_family") not in (None, row.get("family")):
            status_counts["DAT05_family_binding_mismatch"] += 1
            continue
        if prov.get("package_id", prov.get("normalized_package_id")) not in (None, row.get("package_id")) and prov.get("normalized_package_id") != row.get("package_id"):
            status_counts["DAT05_package_binding_mismatch"] += 1
            continue
        inspect_row(row, source_class="corrected", line=line, source_meta=prov, group_id=prov.get("group_id"))

    short_count = 0
    for line, _raw, packet in iter_jsonl(SHORT):
        short_count += 1
        row = packet.get("row")
        source_meta = packet.get("source_provenance")
        if not isinstance(row, dict) or not isinstance(source_meta, dict):
            status_counts["short_packet_shape"] += 1
            continue
        if packet.get("candidate_status") != "source_and_token_checks_passed_not_training_admitted":
            status_counts["short_candidate_status_not_passed"] += 1
            continue
        inspect_row(row, source_class="short", line=line, source_meta=source_meta, group_id=source_meta.get("group_id"))
    if short_count != 411:
        raise ValueError(f"expected 411 short rows, got {short_count}")
    for handle in (full_rows_out, full_prov_out, current_rows_out, current_prov_out, exclusion_out):
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    if protocol_errors:
        raise ValueError(f"row validation failed for {len(protocol_errors)} rows; see in-memory audit")

    rl_coverage = audit_rl()
    write_json(RL_COVERAGE, rl_coverage)

    # The schedule builder consumes metadata only.  We write metadata and its
    # deterministic hash, leaving actual draw selection to the root admission
    # step so no candidate schedule can be mistaken for an accepted packet.
    schedule_metadata = {
        "schema": "DAT-10-r2-expansion-schedule-metadata-v1",
        "split_id": split_doc.get("split_id"),
        "source_policy": "DAT-02 global train_group; reserve explicit CPT cpt_validation; exact IDs; no truncation",
        "full_candidate": {
            "row_count": len(all_candidate_metadata),
            "metadata_sha256": sha256_bytes(canonical(sorted(all_candidate_metadata, key=lambda item: item["row_id"]))),
            "target_cap_gate": "contains complete rows over current 192 cap; requires an explicit task-stage cap identity before use",
            "metadata": sorted(all_candidate_metadata, key=lambda item: item["row_id"]),
        },
        "current_192_candidate": {
            "row_count": len(current_candidate_metadata),
            "metadata_sha256": sha256_bytes(canonical(sorted(current_candidate_metadata, key=lambda item: item["row_id"]))),
            "target_cap_gate": "all complete target tails including EOS <=192; still candidate-only",
            "metadata": sorted(current_candidate_metadata, key=lambda item: item["row_id"]),
        },
        "sampler_policy_to_apply_after_root_admission": {
            "max_steps": 500,
            "effective_batch": 16,
            "requested_draws": 8000,
            "seed": 3407,
            "noop_fraction": 0.10,
            "family_ceiling": 0.25,
            "small_pack_cap": 3,
            "ordinary_replay_cap": 8,
            "naturally_long_fraction": 0.20,
            "short_max_tokens": 2048,
            "long_max_tokens": 4096,
            "truncation": "forbidden",
        },
    }
    write_json(SCHEDULE_METADATA, schedule_metadata)

    full_hash, full_bytes = sha256_file(FULL_ROWS)
    full_prov_hash, full_prov_bytes = sha256_file(FULL_PROVENANCE)
    current_hash, current_bytes = sha256_file(CURRENT_ROWS)
    current_prov_hash, current_prov_bytes = sha256_file(CURRENT_PROVENANCE)
    exclusion_hash, exclusion_bytes = sha256_file(EXCLUSION_LEDGER)
    selection = {
        "schema": "DAT-10-r2-expansion-selection-v1",
        "status": "candidate_only_pending_root_admission",
        "full_bounded_candidate": {
            "path": str(FULL_ROWS), "rows": len(full_ids), "sha256": full_hash, "bytes": full_bytes,
            "provenance_path": str(FULL_PROVENANCE), "provenance_sha256": full_prov_hash,
            "policy": "global DAT-02 train_group + not cpt_validation + complete full sequence <=4096; retains overcap targets",
        },
        "current_192_candidate": {
            "path": str(CURRENT_ROWS), "rows": len(current_ids), "sha256": current_hash, "bytes": current_bytes,
            "provenance_path": str(CURRENT_PROVENANCE), "provenance_sha256": current_prov_hash,
            "policy": "same split binding, complete target including EOS <=192, full sequence <=4096",
        },
        "exclusion_ledger": {"path": str(EXCLUSION_LEDGER), "rows": sum(exclusion_counts.values()), "sha256": exclusion_hash, "bytes": exclusion_bytes},
        "legacy_exclusions_recovered_or_preserved": dict(sorted(exclusion_counts.items())),
        "counts": {
            "existing_primary_rows": 8526,
            "full_bounded_rows": len(full_ids),
            "current_192_rows": len(current_ids),
            "safe_current_cap_recovery_vs_existing_primary": len(current_ids) - 8526,
            "additional_cap_deferred_rows": len(full_ids) - len(current_ids),
            "full_candidate_gain_vs_existing_primary": len(full_ids) - 8526,
            "permanent_cpt_validation_rows": status_counts["cpt_validation_reserved"],
        },
        "family_counts": {"all_input": dict(sorted(family_all.items())), "full_bounded": dict(sorted(family_full.items())), "current_192": dict(sorted(family_current.items()))},
        "hashes": {"full_rows": full_hash, "full_provenance": full_prov_hash, "current_rows": current_hash, "current_provenance": current_prov_hash, "exclusions": exclusion_hash},
    }
    write_json(SELECTION, selection)

    final_signatures = {str(path): file_signature(path) for path in inputs}
    mutations = [str(path) for path in inputs if initial_signatures[str(path)] != final_signatures[str(path)]]
    if mutations:
        raise RuntimeError("input mutated during audit: " + ", ".join(mutations))
    ended = datetime.now(timezone.utc)
    task_metrics = [
        item for item in all_global_metrics
        if item["status"] in {"task_train_cpt_train", "task_train_unmapped_cpt"}
    ]
    # Keep target output length and full context geometry as independent
    # dimensions.  These tables let the root compare 4K/8K/16K context
    # profiles and target budgets without treating 192 as data eligibility.
    length_gate_comparison = {
        "target_tail_including_EOS": {
            "task_train_candidate_before_cpt_validation": threshold_table(
                task_metrics, "target_labels", (192, 256, 384, 512, 1024)
            ),
            "all_global_train_before_cpt_validation_reservation": threshold_table(
                all_global_metrics, "target_labels", (192, 256, 384, 512, 1024)
            ),
        },
        "full_sequence": {
            "task_train_candidate_before_cpt_validation": threshold_table(
                task_metrics, "total_tokens", (2048, 4096, 8192, 16384)
            ),
            "all_global_train_before_cpt_validation_reservation": threshold_table(
                all_global_metrics, "total_tokens", (2048, 4096, 8192, 16384)
            ),
        },
        "interpretation": "target-tail and full-sequence limits are independent; rows are never truncated; no cap is selected as a quality optimum by this CPU audit",
    }
    report = {
        "schema": "DAT-10-r2-expansion-audit-v1",
        "task": "DAT-10/SFT11",
        "status": "verified_candidate_materialized",
        "started": started.isoformat(),
        "ended": ended.isoformat(),
        "scope": {"cpu_only": True, "threads": 2, "model_or_weights": False, "cuda": False, "ssh_or_cloud": False, "dev_content_read": False, "dev_labels_into_train": False},
        "source_inputs": input_hashes,
        "split": {
            "dat02_path": str(DAT02), "split_id": split_doc.get("split_id"), "groups": len(groups),
            "split_counts": dict(sorted(Counter(str(item.get("split")) for item in groups.values()).items())),
            "cpt_partition_path": str(PARTITION),
            "cpt_partition_counts": dict(sorted(Counter(partition.values()).items())),
        },
        "input_counts": {"corrected_rows": len(corrected_rows), "short_rows": short_count, "rows_seen": all_rows_seen, "source_classes": dict(sorted(source_counts.items()))},
        "binding_status_counts": dict(sorted(status_counts.items())),
        "contract": {"row_contract_counts": dict(sorted(contract_counts.items())), "protocol_errors": protocol_errors, "target_bins_by_binding": {key: target_bins(value) for key, value in sorted(target_by_status.items())}, "total_tokens_by_binding": {key: {"rows": len(value), "max": max(value), "over_4096": sum(item > 4096 for item in value)} for key, value in sorted(total_by_status.items())}},
        "length_gate_comparison": length_gate_comparison,
        "legacy_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "license_provenance_gap": {
            "by_scope": {scope: dict(sorted(values.items())) for scope, values in sorted(license_by_scope.items())},
            "evidence_definition": "direct_source_ref_evidence means source_ref.license_evidence_present=true in the joined DAT-05 record; no_row_level_license_evidence is not a clean-license decision",
            "root_action": "repeat license and source-provenance review before admission, including candidate rows whose source_ref records synthetic-authored or package license values",
        },
        "duplicate_gap": {
            "prompt_hash_duplicates_observed": status_counts["prompt_duplicate_same_target"],
            "prompt_hash_conflicts_observed": status_counts["prompt_conflicting_target"],
            "row_id_duplicates_observed": 0,
            "upstream_dat05_exact_train_duplicates_removed": 75,
            "root_action": "repeat source-document/package collision and combined-pool duplicate checks; this audit only proves candidate prompt/target hashes and row IDs",
        },
        "repaired_pipeline": {
            "old_failure": {"code": str(PLAN / "work/r2-task-mixture-v1/prepare_task_mixture.py"), "binding_status_lines": "167-182", "first_failure_line": 173, "description": "Task mixture called a CPT-only partition a prerequisite, so task-authored DAT-02 train groups absent from that CPT map became group_not_in_cpt_partition."},
            "old_cap": {"code": str(PLAN / "work/r2-task-mixture-v1/prepare_task_mixture.py"), "lines": "289-300", "description": "CandidateWriter rejects complete target tails over 192 before task admission."},
            "repaired_join": "DAT-05 exact row id -> group_id; package/name joins never determine group membership",
            "repaired_global_guard": "require DAT-02 group split train_group/train; reject dev_group, final_candidate_group, quarantine, excluded_source_only, and unknown groups",
            "repaired_cpt_guard": "reserve only explicit cpt_validation; allow task_train_unmapped_cpt because the CPT map covers raw CPT groups, not task-authored rows",
            "repaired_geometry": "retain complete token rows; require full sequence <=4096; current trainer target gate remains <=192; overcap rows are separate cap-deferred candidates",
            "cap_decoupling": "a new training identity may declare train_target_limit independently of development_max_new_tokens; retain the paired 192-token DEV baseline and use 512/1024 generation only as separately measured diagnostics",
            "context_profiles": "profile 4096, 8192, and 16384 full-sequence limits independently; this candidate has no rows over 4096, so the CPU audit selects no context optimum",
            "repaired_dedup": "exact row IDs plus prompt SHA256 with conflicting target rejection; observed zero prompt duplicates/conflicts",
            "root_gates_pending": ["source quality and exact source-line verification", "license/provenance review", "tokenization and renderer identity", "duplicate/collision review", "training recipe identity and DEV gate for any target-cap change"],
        },
        "candidate_artifacts": {"selection": str(SELECTION), "schedule_metadata": str(SCHEDULE_METADATA), "full_rows": str(FULL_ROWS), "full_provenance": str(FULL_PROVENANCE), "current_rows": str(CURRENT_ROWS), "current_provenance": str(CURRENT_PROVENANCE), "exclusion_ledger": str(EXCLUSION_LEDGER), "rl_coverage": str(RL_COVERAGE)},
        "rl": rl_coverage,
        "novel_roster": {"status": "metadata_only_not_materialized_or_admitted_by_this_audit", "source": str(PLAN / "work/r2-corpus-preparation-v1/novel-roster-summary.json"), "quality_license_tokenization_duplicate_checks": "remain mandatory"},
        "acceptance": {"split_rebinding_audited": True, "explicit_validation_exclusion_preserved": True, "full_rows_materialized_without_truncation": True, "current_cap_subset_materialized": True, "rl_schedule_coverage_audited": True, "dev_content_not_copied": True, "accepted_packets_edited": False},
    }
    write_json(REPORT, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
