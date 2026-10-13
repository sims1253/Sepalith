#!/usr/bin/env python3
"""Combine the original and repaired DAT10 candidate token audits.

The output has an all-global-TRAIN review view and a separately marked
mechanical-dedup view.  CPT validation remains held out.  A missing CPT
partition entry is retained when the global registry says train_group, as
required by the all-eligible-data policy.  This script does not perform root
quality/license/duplicate admission.
"""
from __future__ import annotations
import hashlib
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[5]
E = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1")
IN_PATHS = [
    E / "structured-token-audit/candidate-token-rows.jsonl",
    E / "completion-token-audit/candidate-token-rows.jsonl",
    E / "completion-mixed-eol-repair-v1/token-audit/candidate-token-rows.jsonl",
]
EXISTING_PROVENANCE = ROOT / "docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-provenance.jsonl"
GLOBAL_SPLIT = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
CPT = ROOT / "docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json"
OUT = E / "candidate-gate-v3"


def sha_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(p: pathlib.Path):
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield json.loads(line)


def source_key(ref: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(ref.get("source") or ref.get("source_file") or ref.get("file") or ""),
        int(ref.get("source_line") or ref.get("line") or 0),
        str(ref.get("package") or ref.get("package_id") or ""),
    )


def metadata(rec: dict[str, Any], *, cpt: str, global_split: str, reasons: list[str], rank: int | None = None) -> dict[str, Any]:
    ref = rec["source_ref"]
    row = rec["row"]
    lengths = rec.get("lengths", {})
    prov = rec.get("source_provenance") or {}
    repair = prov.get("mixed_eol_repair")
    return {
        "row_id": ref.get("row_id"),
        "group_id": ref.get("group_id"),
        "family": row.get("family") or ref.get("family"),
        "source": ref.get("source"),
        "source_variant": ref.get("source_variant"),
        "source_file": ref.get("source_file") or ref.get("file"),
        "source_line": ref.get("source_line") or ref.get("line"),
        "package_id": ref.get("package") or ref.get("package_id"),
        "split": ref.get("split"),
        "global_split": global_split,
        "cpt_partition": cpt,
        "license": ref.get("license"),
        "license_evidence_present": ref.get("license_evidence_present"),
        "license_status": ref.get("license_status"),
        "canonical_row_sha256": ref.get("canonical_row_sha256"),
        "raw_line_sha256": ref.get("raw_line_sha256"),
        "target_sha256": rec.get("target_sha256") or ref.get("target_sha256"),
        "prompt_sha256": rec.get("prompt_sha256"),
        "target_body_tokens": lengths.get("target_body"),
        "target_tokens_including_eos": lengths.get("response_with_terminal_eos"),
        "prompt_tokens_with_bos": lengths.get("prompt_with_bos"),
        "sequence_tokens": lengths.get("sequence"),
        "length_gates": {
            "train_target_1024": lengths.get("response_with_terminal_eos", 10**9) <= 1024,
            "sft_sequence_4096": lengths.get("sequence", 10**9) <= 4096,
        },
        "repair": {
            "mixed_eol": isinstance(repair, dict),
            "policy": repair.get("policy") if isinstance(repair, dict) else None,
            "geometry_preserved": repair.get("geometry_preserved") if isinstance(repair, dict) else None,
            "source_bytes_unchanged": True if isinstance(repair, dict) else None,
            "target_bytes_unchanged": repair.get("target_bytes_unchanged") if isinstance(repair, dict) else None,
        },
        "gate_reasons": reasons,
        "mechanical_rank_for_prompt": rank,
        "candidate_status": "review_only_unadmitted",
    }


def main() -> int:
    for p in IN_PATHS + [EXISTING_PROVENANCE, GLOBAL_SPLIT, CPT]:
        if not p.exists():
            raise FileNotFoundError(p)
    OUT.mkdir(parents=True, exist_ok=True)

    global_doc = json.loads(GLOBAL_SPLIT.read_text(encoding="utf-8"))
    global_groups = {str(g["group_id"]): str(g.get("split")) for g in global_doc.get("groups", [])}
    cpt_doc = json.loads(CPT.read_text(encoding="utf-8"))
    cpt_groups = {str(k): str(v) for k, v in cpt_doc.get("groups", {}).items()}

    existing_prompt: set[str] = set()
    existing_source: set[tuple[str, int, str]] = set()
    existing_ids: set[str] = set()
    for p in rows(EXISTING_PROVENANCE):
        if p.get("prompt_sha256"):
            existing_prompt.add(str(p["prompt_sha256"]))
        if p.get("id"):
            existing_ids.add(str(p["id"]))
        existing_source.add((str(p.get("source") or ""), int(p.get("source_line") or 0), str(p.get("package_id") or "")))

    candidates: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    input_counts: dict[str, int] = {}
    for p in IN_PATHS:
        input_hashes[str(p)] = sha_file(p)
        count = 0
        for rec in rows(p):
            ref = rec.get("source_ref") or {}
            rid = str(ref.get("row_id") or "")
            if not rid:
                raise RuntimeError(f"missing row_id in {p}")
            if rid in by_id:
                raise RuntimeError(f"candidate row_id collision: {rid}")
            by_id[rid] = rec
            candidates.append(rec)
            count += 1
        input_counts[str(p)] = count

    prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in candidates:
        prompt_groups[str(rec.get("prompt_sha256") or "")].append(rec)
        source_groups[source_key(rec["source_ref"])].append(rec)

    exact_duplicate_extra: set[str] = set()
    conflicting_prompt: set[str] = set()
    prompt_rank: dict[str, int] = {}
    for prompt, group in prompt_groups.items():
        ordered = sorted(group, key=lambda r: str(r["source_ref"].get("row_id")))
        for idx, rec in enumerate(ordered, 1):
            prompt_rank[str(rec["source_ref"]["row_id"])] = idx
        targets = {str(rec.get("target_sha256") or rec["source_ref"].get("target_sha256") or "") for rec in group}
        if len(targets) > 1:
            conflicting_prompt.update(str(rec["source_ref"]["row_id"]) for rec in group)
        elif len(group) > 1:
            exact_duplicate_extra.update(str(rec["source_ref"]["row_id"]) for rec in ordered[1:])

    source_duplicate_extra: set[str] = set()
    for key, group in source_groups.items():
        if not key[0] or len(group) < 2:
            continue
        ordered = sorted(group, key=lambda r: str(r["source_ref"].get("row_id")))
        source_duplicate_extra.update(str(rec["source_ref"]["row_id"]) for rec in ordered[1:])

    ledger: list[dict[str, Any]] = []
    all_global: list[dict[str, Any]] = []
    mechanical: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    cpt_counts: Counter[str] = Counter()
    global_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for rec in sorted(candidates, key=lambda r: str(r["source_ref"].get("row_id"))):
        ref = rec["source_ref"]
        rid = str(ref["row_id"])
        group_id = str(ref.get("group_id") or "")
        gs = global_groups.get(group_id, str(ref.get("split") or "unknown"))
        cp = cpt_groups.get(group_id, "cpt_partition_missing")
        reasons: list[str] = []
        if gs != "train_group":
            reasons.append("global_split_not_train_group")
        if cp == "cpt_validation":
            reasons.append("cpt_validation_reserved")
        if rid in existing_ids:
            reasons.append("existing_row_id_duplicate")
        if str(rec.get("prompt_sha256") or "") in existing_prompt:
            reasons.append("existing_prompt_duplicate")
        if source_key(ref) in existing_source:
            reasons.append("existing_source_identity_duplicate")
        if rid in conflicting_prompt:
            reasons.append("within_candidate_conflicting_target")
        if rid in exact_duplicate_extra:
            reasons.append("within_candidate_exact_prompt_target_duplicate")
        if rid in source_duplicate_extra:
            reasons.append("within_candidate_source_identity_duplicate")
        if rec.get("lengths", {}).get("response_with_terminal_eos", 10**9) > 1024:
            reasons.append("train_target_over_1024")
        if rec.get("lengths", {}).get("sequence", 10**9) > 4096:
            reasons.append("sft_sequence_over_4096")
        cpt_counts[cp] += 1
        global_counts[gs] += 1
        family_counts[str(ref.get("family") or rec.get("row", {}).get("family"))] += 1
        for reason in reasons:
            reason_counts[reason] += 1
        m = metadata(rec, cpt=cp, global_split=gs, reasons=reasons, rank=prompt_rank.get(rid))
        ledger.append(m)
        if gs == "train_group" and cp != "cpt_validation":
            all_global.append(rec)
            # A mechanical view is only a deterministic review convenience.
            if not reasons:
                mechanical.append(rec)

    def dump_records(path: pathlib.Path, records: list[dict[str, Any]]) -> str:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        return sha_file(path)

    ledger_path = OUT / "candidate-gate-ledger-v3.jsonl"
    all_path = OUT / "all-global-train-token-rows-v3.jsonl"
    mech_path = OUT / "mechanical-global-train-token-rows-v3.jsonl"
    ledger_sha = dump_records(ledger_path, ledger)
    all_sha = dump_records(all_path, all_global)
    mech_sha = dump_records(mech_path, mechanical)
    summary = {
        "schema": "DAT-10-novel-candidate-gate-v3",
        "status": "review_only_unadmitted",
        "policy": "all_eligible_global_train_rows_retained;_cpt_validation_reserved;missing_cpt_partition_retained_if_global_train",
        "input_counts": input_counts,
        "input_sha256": input_hashes,
        "candidate_total": len(candidates),
        "all_global_train_excluding_cpt_validation": len(all_global),
        "mechanical_global_train_view": len(mechanical),
        "global_split_counts": dict(sorted(global_counts.items())),
        "cpt_partition_counts": dict(sorted(cpt_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "existing_parent": {
            "provenance_path": str(EXISTING_PROVENANCE),
            "provenance_sha256": sha_file(EXISTING_PROVENANCE),
            "prompt_hash_count": len(existing_prompt),
            "source_identity_count": len(existing_source),
            "row_id_count": len(existing_ids),
        },
        "within_candidate": {
            "prompt_groups": len(prompt_groups),
            "source_identity_groups": len(source_groups),
            "exact_prompt_target_duplicate_extra_rows": len(exact_duplicate_extra),
            "conflicting_prompt_rows": len(conflicting_prompt),
            "source_identity_duplicate_extra_rows": len(source_duplicate_extra),
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "outputs": {
            "ledger": {"path": str(ledger_path), "sha256": ledger_sha, "rows": len(ledger)},
            "all_global_train": {"path": str(all_path), "sha256": all_sha, "rows": len(all_global)},
            "mechanical_global_train": {"path": str(mech_path), "sha256": mech_sha, "rows": len(mechanical)},
        },
        "heldout_content_included": False,
        "target_truncation": False,
        "admission": "none; root quality/source/license/dedup review remains mandatory",
    }
    summary_path = OUT / "candidate-gate-summary-v3.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "DAT-10-novel-candidate-gate-v3-manifest",
        "summary": str(summary_path),
        "summary_sha256": sha_file(summary_path),
        "input_global_split": str(GLOBAL_SPLIT),
        "input_global_split_sha256": sha_file(GLOBAL_SPLIT),
        "input_cpt_partition": str(CPT),
        "input_cpt_partition_sha256": sha_file(CPT),
        "outputs": summary["outputs"],
    }
    (OUT / "candidate-gate-manifest-v3.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_total": len(candidates), "all_global_train": len(all_global), "mechanical": len(mechanical), "reason_counts": dict(reason_counts), "summary": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
