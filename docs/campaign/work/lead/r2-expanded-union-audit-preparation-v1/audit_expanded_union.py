#!/usr/bin/env python3
"""Fail-closed, provenance-joined audit for the expanded TRAIN union.

This preparation does not admit data. It streams each token/provenance pair,
checks its closure, and writes a decision ledger. A row can participate in
prompt checks while its geometry remains explicitly unresolved.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "sepalith.sft11.expanded-union-audit.v1"
GEOMETRY_SCHEMA = "sepalith.source-cursor-geometry.v1"
EXPECTED_COHORTS = {
    "accepted_current_20191": 20191,
    "finalized_semantic10948": 10682,
    "eventual_semantic9534": 9534,
    "noop4100": 4100,
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def position(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{label}:position_missing")
    require(set(value) == {"line", "character"}, f"{label}:position_fields")
    line, character = value["line"], value["character"]
    require(isinstance(line, int) and not isinstance(line, bool) and line >= 0,
            f"{label}:line")
    require(isinstance(character, int) and not isinstance(character, bool) and character >= 0,
            f"{label}:character")
    return {"line": line, "character": character}


def geometry_from_provenance(prov: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (digest, reason). Never creates geometry from incomplete fields."""
    geometry = prov.get("source_cursor_geometry")
    supplied = prov.get("source_cursor_geometry_sha256")
    if geometry is None:
        return None, "missing_source_cursor_geometry"
    if not isinstance(geometry, dict):
        return None, "malformed_source_cursor_geometry"
    try:
        require(geometry.get("schema") == GEOMETRY_SCHEMA, "geometry_schema")
        source_path = geometry.get("source_path")
        source_sha = geometry.get("source_sha256")
        preedit_sha = geometry.get("preedit_sha256")
        window_sha = geometry.get("window_sha256")
        require(isinstance(source_path, str) and source_path, "source_path")
        for name, value in (("source_sha256", source_sha), ("preedit_sha256", preedit_sha),
                            ("window_sha256", window_sha)):
            require(isinstance(value, str) and len(value) == 64, name)
        cursor = position(geometry.get("cursor"), "cursor")
        replacement = geometry.get("replacement_range")
        require(isinstance(replacement, dict), "replacement_range")
        require(set(replacement) == {"start", "end"}, "replacement_range_fields")
        start = position(replacement["start"], "replacement_start")
        end = position(replacement["end"], "replacement_end")
        require((end["line"], end["character"]) >= (start["line"], start["character"]),
                "replacement_range_reversed")
        normalized = {
            "schema": GEOMETRY_SCHEMA,
            "source_path": source_path,
            "source_sha256": source_sha,
            "preedit_sha256": preedit_sha,
            "cursor": cursor,
            "replacement_range": {"start": start, "end": end},
            "window_sha256": window_sha,
        }
        observed = digest_text(canonical(normalized))
        require(isinstance(supplied, str) and len(supplied) == 64,
                "missing_geometry_digest")
        require(observed == supplied, "geometry_digest_mismatch")
        identity = prov.get("source_identity", {})
        if isinstance(identity, dict):
            if identity.get("source_path") is not None:
                require(identity["source_path"] == source_path, "source_path_join")
            if identity.get("source_sha256") is not None:
                require(identity["source_sha256"] == source_sha, "source_sha_join")
        return observed, None
    except (AuditError, KeyError, TypeError) as exc:
        return None, f"ambiguous_source_cursor_geometry:{exc}"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{number}:not_object")
            yield row


def validate_file_pin(item: dict[str, Any], label: str) -> Path:
    require(isinstance(item, dict), f"{label}:pin_not_object")
    path = Path(item.get("path", ""))
    expected_sha = item.get("sha256")
    expected_rows = item.get("rows")
    require(path.is_file(), f"{label}:missing_file:{path}")
    require(isinstance(expected_sha, str) and len(expected_sha) == 64, f"{label}:sha_pin")
    require(isinstance(expected_rows, int) and not isinstance(expected_rows, bool) and expected_rows >= 0,
            f"{label}:row_pin")
    require(sha256(path) == expected_sha, f"{label}:sha_mismatch")
    return path


def validate_token_row(row: dict[str, Any], cohort: str) -> tuple[str, str, str]:
    rid = row.get("id")
    prompt = row.get("prompt_text")
    target = row.get("target_text")
    require(isinstance(rid, str) and rid, f"{cohort}:row_id")
    require(isinstance(prompt, str) and prompt, f"{cohort}:{rid}:prompt")
    require(isinstance(target, str) and target, f"{cohort}:{rid}:target")
    require(row.get("split") == "train", f"{cohort}:{rid}:split_not_train")
    return rid, digest_text(prompt), digest_text(target)


def preflight(spec: dict[str, Any]) -> list[dict[str, Any]]:
    require(spec.get("schema") == SCHEMA, "spec_schema")
    cohorts = spec.get("cohorts")
    require(isinstance(cohorts, list), "cohort_list")
    require([x.get("name") for x in cohorts] == list(EXPECTED_COHORTS), "cohort_order_or_names")
    issues: list[dict[str, Any]] = []
    for cohort in cohorts:
        name = cohort["name"]
        require(cohort.get("expected_rows") == EXPECTED_COHORTS[name], f"{name}:expected_rows")
        manifest = cohort.get("manifest")
        token_files = cohort.get("token_rows")
        provenance_files = cohort.get("provenance_rows")
        if manifest is None or token_files is None or provenance_files is None:
            issues.append({"cohort": name, "rows": EXPECTED_COHORTS[name],
                           "reason": "cohort_unbound", "missing_geometry_is_not_exclusion": True})
            continue
        validate_file_pin(manifest, f"{name}:manifest")
        require(isinstance(token_files, list) and isinstance(provenance_files, list),
                f"{name}:rowsets")
        require(len(token_files) == len(provenance_files) and len(token_files) > 0,
                f"{name}:token_provenance_rowset_pairing")
        require(sum(x.get("rows", -1) for x in token_files) == EXPECTED_COHORTS[name],
                f"{name}:token_denominator")
        require(sum(x.get("rows", -1) for x in provenance_files) == EXPECTED_COHORTS[name],
                f"{name}:provenance_denominator")
    return issues


def audit(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unbound = preflight(spec)
    decisions: dict[str, dict[str, Any]] = {}
    prompt_groups: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    pair_first: dict[tuple[str, str], str] = {}
    geometry_groups: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    cohort_counts: dict[str, int] = {}

    for cohort in spec["cohorts"]:
        name = cohort["name"]
        if cohort.get("manifest") is None or cohort.get("token_rows") is None or cohort.get("provenance_rows") is None:
            continue
        seen = 0
        for part, (token_pin, provenance_pin) in enumerate(zip(cohort["token_rows"], cohort["provenance_rows"])):
            token_path = validate_file_pin(token_pin, f"{name}:token:{part}")
            provenance_path = validate_file_pin(provenance_pin, f"{name}:provenance:{part}")
            local = 0
            token_iter, provenance_iter = iter_jsonl(token_path), iter_jsonl(provenance_path)
            while True:
                token = next(token_iter, None)
                provenance = next(provenance_iter, None)
                require((token is None) == (provenance is None), f"{name}:{part}:row_count_or_order_mismatch")
                if token is None:
                    break
                rid, prompt_sha, target_sha = validate_token_row(token, name)
                require(provenance.get("row_id") == rid, f"{name}:{part}:{rid}:provenance_row_join")
                require(rid not in decisions, f"duplicate_row_id:{rid}")
                if provenance.get("prompt_sha256") is not None:
                    require(provenance["prompt_sha256"] == prompt_sha, f"{rid}:provenance_prompt_join")
                if provenance.get("target_sha256") is not None:
                    require(provenance["target_sha256"] == target_sha, f"{rid}:provenance_target_join")
                geometry, geometry_reason = geometry_from_provenance(provenance)
                status = "retained_for_prompt_audit"
                reasons: list[str] = []
                if geometry is None:
                    reasons.append(geometry_reason or "missing_source_cursor_geometry")
                decisions[rid] = {"row_id": rid, "cohort": name, "status": status,
                                  "reasons": reasons, "prompt_sha256": prompt_sha,
                                  "target_sha256": target_sha,
                                  "source_cursor_geometry_sha256": geometry,
                                  "silent_drop": False}
                prompt_groups[prompt_sha][target_sha].append(rid)
                pair_first.setdefault((prompt_sha, target_sha), rid)
                if geometry is not None:
                    geometry_groups[geometry][target_sha].append(rid)
                local += 1
            require(local == token_pin["rows"] == provenance_pin["rows"],
                    f"{name}:{part}:pinned_row_count")
            seen += local
        require(seen == EXPECTED_COHORTS[name], f"{name}:observed_denominator")
        cohort_counts[name] = seen

    # Exact duplicates are duplicate prompt+target only. Target-only grouping is forbidden.
    for targets in prompt_groups.values():
        if len(targets) > 1:
            ids = sorted(rid for group in targets.values() for rid in group)
            for rid in ids:
                decisions[rid]["status"] = "hold"
                decisions[rid]["reasons"].append("prompt_contradiction:" + ",".join(ids))
        else:
            ids = next(iter(targets.values()))
            if len(ids) > 1:
                first = ids[0]
                for rid in ids[1:]:
                    decisions[rid]["status"] = "exact_duplicate"
                    decisions[rid]["reasons"].append("duplicate_prompt_target:" + first)

    for targets in geometry_groups.values():
        ids = sorted(rid for group in targets.values() for rid in group)
        if len(targets) > 1:
            for rid in ids:
                decisions[rid]["status"] = "hold"
                decisions[rid]["reasons"].append("source_cursor_replacement_conflict:" + ",".join(ids))
        # Same source/cursor/range and same target can still have legitimate
        # different history/evidence prompts. Exact prompt+target is the only
        # duplicate rule; geometry is used only to expose contradictory targets.

    rows = [decisions[rid] for rid in sorted(decisions)]
    counts = collections.Counter(x["status"] for x in rows)
    missing_geometry = sum(x["source_cursor_geometry_sha256"] is None for x in rows)
    complete = not unbound and missing_geometry == 0
    report = {
        "schema": "sepalith.sft11.expanded-union-audit.result.v1",
        "status": "complete_geometry_audit_review_only" if complete else "partial_fail_closed",
        "expected_rows": sum(EXPECTED_COHORTS.values()),
        "audited_rows": len(rows),
        "cohort_counts": cohort_counts,
        "unbound_cohorts": unbound,
        "missing_geometry_rows": missing_geometry,
        "decision_counts": dict(sorted(counts.items())),
        "target_only_dedup_forbidden": True,
        "same_file_different_cursor_retained": True,
        "training_admission": False,
        "admission_ready": complete and counts.get("hold", 0) == 0,
    }
    return rows, report


def write_atomic(output: Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    require(not output.exists(), f"fresh_output_required:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        ledger = temp / "decision-ledger.jsonl"
        with ledger.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        report["outputs"] = {"decision-ledger.jsonl": {
            "rows": len(rows), "bytes": ledger.stat().st_size, "sha256": sha256(ledger)}}
        manifest = temp / "manifest.json"
        with manifest.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.rename(temp, output)
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha256(args.spec) == args.spec_sha256, "spec_sha256")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    rows, report = audit(spec)
    write_atomic(args.output, rows, report)
    print(canonical(report))


if __name__ == "__main__":
    main()
