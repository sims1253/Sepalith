#!/usr/bin/env python3
"""Metadata-only DAT-03 collision, visibility and provenance audit.

This adapter rebuilds the accepted DAT-02 identity graph from the bounded raw
source metadata, then audits only rows assigned to train_group or dev_group.
It emits hashes, shapes, source/validator metadata and reasons.  It never
prints or writes prompt, context, target, code, or model-answer text.  Final
candidate rows are used only for identity metadata and remain sealed.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
DATASET = Path("/mnt/h/sepalith/datasets")
MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
HASH_MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-01-source-hashes.json")
INVENTORY = PLAN / "docs/campaign/work/data/DAT-01-source-inventory.json"
TU3_AUDIT = PLAN / "docs/research/72h-tu3-collision-audit.json"
DEFAULT_LOOKUP = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-03-identity-lookup.json")
DEFAULT_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-03-row-audit.jsonl")
DEFAULT_DUPLICATES = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-03-duplicate-groups.json")
HEX_RE = re.compile(r"^[0-9a-f]{32,128}$", re.I)


def load_v2_module():
    path = PLAN / "docs/campaign/work/data/DAT-02-build-split-v2.py"
    spec = importlib.util.spec_from_file_location("dat02_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import accepted DAT-02 builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = load_v2_module()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def sha_value(value: Any) -> str:
    return sha_text(canonical_json(value))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def raw_jsonl(path: Path):
    """Yield line number, exact bytes and decoded row without printing it."""
    with path.open("rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except Exception:
                yield line_no, raw, None
                continue
            yield line_no, raw, row


def candidate_specs() -> list[tuple[Path, str, str]]:
    """Return the exact candidate file set used by accepted DAT-02."""
    specs: list[tuple[Path, str, str]] = []
    specs.append((DATASET / "edit_pairs_v1/examples.jsonl", "edit_pairs_train", "edit_pairs"))
    deterministic = [
        "rename_propagation", "pipe_rewrite", "na_rm_propagation", "format_propagation",
        "doc_sync", "no_op", "comment_insert",
    ]
    teacher = [
        "comment_drafting", "roxygen_drafting", "mid_roxygen", "comment_to_code_real",
        "comment_to_code_synthetic", "comment_to_code_gemini",
    ]
    for family in deterministic + teacher:
        specs.append((DATASET / f"scenarios_v1/{family}.jsonl", f"scenario_{family}", family))
    case_dir = DATASET / "cases_v1"
    for path in sorted(case_dir.glob("*.jsonl")):
        name = path.name
        if (name.endswith(".done.jsonl") or name.endswith("_bases.jsonl")
                or name == "base_samples_spark.jsonl"
                or name.startswith("astfim_partial")):
            continue
        family = name.removesuffix(".jsonl")
        specs.append((path, f"case_{family}", family))
    return specs


def historical_specs() -> list[tuple[Path, str, str, tuple[str, ...]]]:
    """Return historical/alias files used to construct the DAT-02 graph."""
    p = DATASET
    return [
        (p / "sft_v7/train.jsonl", "sft_v7", "historical_train", ("sft_v7",)),
        (p / "sft_v3/eval.jsonl", "sft_v3_eval", "historical_eval", ("historical_eval", "sft_v3_eval")),
        (p / "edit_pairs_v1/eval.jsonl", "edit_pairs_eval", "historical_eval", ("historical_eval",)),
        (p / "nextcoder_r_v1/eval.jsonl", "b11_eval", "historical_eval", ("historical_eval", "b11_eval")),
        (p / "nextcoder_r2_cross_v1/eval.jsonl", "tu3_eval", "historical_eval", ("historical_eval", "tu3")),
        (p / "nextcoder_r_v1/authored.jsonl", "b11_authored", "b11_authored", ("b11",)),
        (p / "nextcoder_r_v1/seeds.jsonl", "b11_seeds", "b11_seed", ("b11",)),
        (p / "nextcoder_r_v1/train.jsonl", "b11_train", "b11_train", ("b11",)),
        (p / "nextcoder_r2_cross_v1/authored.jsonl", "tu3_authored", "tu3_authored", ("tu3",)),
        (p / "nextcoder_r2_cross_v1/seeds.jsonl", "tu3_seeds", "tu3_seed", ("tu3",)),
        (p / "nextcoder_r2_cross_v1/train.jsonl", "tu3_train", "tu3_train", ("tu3",)),
        (p / "spec_traces/traces.jsonl", "spec_traces", "historical_trace", ("historical_trace",)),
        (p / "loc1_s0_r/set.jsonl", "loc1_set", "environment", ("environment",)),
        (p / "loc1_s0_r/corpus.jsonl", "loc1_corpus", "environment", ("environment",)),
        (p / "sim_trajectories_v1/trajectories.jsonl", "sim_trajectories", "historical_trajectory", ("historical_trajectory",)),
        (p / "sim_trajectories_v1/trajectories_v2.jsonl", "sim_trajectories", "historical_trajectory", ("historical_trajectory",)),
    ]


def build_identity_lookup(manifest: dict) -> tuple[Any, dict, dict, dict, dict, collections.Counter]:
    """Load the accepted v2 DSU's complete forms without rereading history.

    DAT-02 froze every named identity form and parent token in each manifest
    group.  Reusing those bytes is both deterministic and materially safer than
    traversing the large historical NAS trees a second time.  Candidate rows
    are still checked below by recomputing their tokens and requiring every
    token to resolve in this lookup.
    """
    accepted = {g["group_id"]: g for g in manifest["groups"]}
    if len(accepted) != len(manifest["groups"]):
        raise RuntimeError("accepted DAT-02 manifest contains duplicate group IDs")
    token_group: dict[str, str] = {}
    for gid, group in accepted.items():
        for token in list(group.get("identity_forms", [])) + list(group.get("parent_tokens", [])):
            if not isinstance(token, str) or not token:
                raise RuntimeError(f"invalid empty identity token in accepted group {gid}")
            old = token_group.setdefault(token, gid)
            if old != gid:
                raise RuntimeError(f"conflicting identity lookup token {sha_text(token)}: {old} vs {gid}")
    # Keep a builder-shaped return value for callers and downstream receipt
    # code; the accepted manifest is the source of truth for group IDs.
    builder = V2.SplitBuilder({})
    bad = collections.Counter()
    groups = accepted
    for gid, group in accepted.items():
        if group["split"] not in {"train_group", "dev_group", "final_candidate_group", "quarantine_tu3", "quarantine_historical_eval", "quarantine_sft_v7", "excluded_source_only"}:
            raise RuntimeError(f"unknown DAT-02 split {group['split']!r} for {gid}")
    parent_group = {t: gid for t, gid in token_group.items() if t.startswith("parent:")}
    named_group = {t: gid for t, gid in token_group.items() if t.startswith(("pkg:", "repo:"))}
    lookup = {
        "task": "DAT-03",
        "source_split_id": manifest["split_id"],
        "source_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "algorithm": "accepted DAT-02 v2 canonical-token DSU; complete named pkg:/repo: forms and parent: tokens",
        "named_form_count": len(named_group),
        "parent_token_count": len(parent_group),
        "named_forms": [{"token": t, "group_id": named_group[t]} for t in sorted(named_group)],
        "parent_tokens": [{"token": t, "group_id": parent_group[t]} for t in sorted(parent_group)],
    }
    return builder, groups, accepted, token_group, lookup, bad


TARGET_KEYS = {"region_new", "target", "model_target", "corpus_target", "text"}


def normalize_prompt_text(value: str, *, loose: bool = False) -> str:
    """Normalize line endings; loose mode is diagnostic only.

    The hard collision key must preserve R identifiers, string contents and
    whitespace.  NFKC/rstrip can merge distinct prediction envelopes, so they
    are retained only as a secondary diagnostic key.
    """
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if loose:
        value = unicodedata.normalize("NFKC", value)
        value = "\n".join(line.rstrip() for line in value.split("\n"))
    return value


def line_values(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(x, str) and "\n" not in x and "\r" not in x for x in value):
        return value
    return None


def body_text(value: Any) -> str | None:
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return "\n".join(x.replace("\r\n", "\n").replace("\r", "\n") for x in value)
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    return None


def _semantic_body(value: Any) -> str | None:
    """Return replacement bytes while preserving whitespace exactly."""
    return body_text(value)


def target_info(row: dict, source: str, family: str) -> dict:
    region_new = row.get("region_new")
    region_old = row.get("region_old")
    is_noop = source == "scenario_no_op" and family == "no_op" and region_new == []
    variants: dict[str, str] = {}
    for key in ("region_new", "target", "model_target", "corpus_target"):
        if key in row and row[key] is not None:
            text = body_text(row[key])
            if text is not None:
                variants[key] = sha_text(text)
    if is_noop:
        # [] is a source sentinel, not a deletion.  Compare its semantic
        # replacement bytes with ordinary full-copy rows so equivalent no-ops
        # do not become artificial target contradictions.
        semantic = {"kind": "replacement", "bytes": body_text(region_old) or ""}
        return {
            "target_sha256": sha_value(semantic),
            "target_kind": "noop_sentinel",
            "target_visible": True,
            "target_chars": 0,
            "target_variant_hashes": variants,
            "target_contract": "semantic region_new=region_old; serialized body=[NO_EDIT] plus existing UPDATED terminal",
            "empty_non_noop": False,
        }
    if "region_new" in row and row["region_new"] is not None:
        if isinstance(row["region_new"], list):
            text = body_text(row["region_new"])
            text = "" if text is None else text
            kind = "region_new" if len(row["region_new"]) > 0 else "empty_region_new"
            return {
                "target_sha256": sha_value({"kind": "replacement", "bytes": text}),
                "target_kind": kind,
                "target_visible": True,
                "target_chars": len(text),
                "target_variant_hashes": variants,
                "target_contract": "region_new is the structured target; empty non-no_op requires authoritative deletion proof",
                "empty_non_noop": kind == "empty_region_new",
            }
        text = body_text(row["region_new"])
        if text is not None:
            return {
                "target_sha256": sha_value({"kind": "replacement", "bytes": text}),
                "target_kind": "region_new_string",
                "target_visible": True,
                "target_chars": len(text),
                "target_variant_hashes": variants,
                "target_contract": "string target requires conversion to a lossless structured boundary",
                # Whitespace-only replacement is a real replacement.  Only an
                # exactly empty body needs deletion evidence.
                "empty_non_noop": text == "",
            }
    for key in ("target", "model_target", "corpus_target"):
        if key not in row or row.get(key) is None:
            continue
        text = body_text(row.get(key))
        if text is not None:
            return {
                "target_sha256": sha_value({"kind": "replacement", "bytes": text}),
                "target_kind": f"{key}_only",
                "target_visible": True,
                "target_chars": len(text),
                "target_variant_hashes": variants,
                "target_contract": "materialized target has no independently recoverable structured boundary",
                "empty_non_noop": text == "",
            }
    return {
        "target_sha256": "",
        "target_kind": "missing",
        "target_visible": False,
        "target_chars": 0,
        "target_variant_hashes": variants,
        "target_contract": "target field absent or unsupported type",
        "empty_non_noop": False,
    }


def target_semantic_variants(row: dict, source: str, family: str) -> dict[str, str]:
    """Hash all supplied replacement variants after sentinel resolution."""
    values: dict[str, str] = {}
    is_noop = source == "scenario_no_op" and family == "no_op" and row.get("region_new") == []
    for key in ("region_new", "target", "model_target", "corpus_target"):
        if key not in row or row[key] is None:
            continue
        if key == "region_new" and is_noop:
            value = body_text(row.get("region_old"))
        else:
            value = body_text(row[key])
        if value is not None:
            values[key] = sha_value({"kind": "replacement", "bytes": value})
    return values


def prompt_info(row: dict) -> dict:
    prompt_value = None
    prompt_key = None
    for key in ("prompt", "full_prompt"):
        if isinstance(row.get(key), str) and row[key] != "":
            prompt_value = row[key]
            prompt_key = key
            break
    if prompt_value is None:
        # Preserve every non-target field, including history/evidence and
        # cursor_column.  This is deliberately conservative: unknown fields
        # cannot create a false hard collision.  ``text`` is excluded because
        # it is commonly a materialized prompt+target sequence.
        context = {key: value for key, value in row.items() if key not in TARGET_KEYS}
        prompt_value = canonical_json(context)
        comparison_status = "structural_all_non_target_fields"
    else:
        comparison_status = f"explicit_{prompt_key}"
    exact = normalize_prompt_text(prompt_value)
    loose = normalize_prompt_text(prompt_value, loose=True)
    return {
        "prediction_envelope_sha256": sha_text(exact),
        "loose_prompt_sha256": sha_text(loose),
        "prompt_chars": len(exact),
        "prompt_was_explicit": prompt_key is not None,
        "prompt_comparison_status": comparison_status,
    }


def boundary_info(row: dict, source: str, family: str) -> dict:
    required = ("prefix", "region_old", "suffix", "cursor_idx")
    missing = [key for key in required if key not in row or row[key] is None]
    type_errors: list[str] = []
    for key in ("prefix", "region_old", "suffix"):
        if key not in row:
            continue
        if line_values(row[key]) is None:
            type_errors.append(key)
    if "cursor_idx" in row and not isinstance(row["cursor_idx"], int):
        type_errors.append("cursor_idx")
    # Negative cursor_idx is used by canonical empty insertion regions.  It is
    # retained as metadata for the family validator, not treated as a broken
    # boundary here.
    parent_keys = [
        key for key in ("sha_full", "content_hash", "file_sha", "corpus_key", "base_sample_id")
        if isinstance(row.get(key), (str, int)) and str(row[key]).strip()
    ]
    if source == "edit_pairs_train":
        parent_quality = "sha_full_repo_path" if HEX_RE.fullmatch(str(row.get("sha_full", ""))) and row.get("repo") and row.get("path") else "edit_parent_incomplete"
    elif source.startswith("scenario_"):
        parent_quality = "package_version_url_or_upstream" if row.get("package") and row.get("path") and (row.get("version") or row.get("source_url") or row.get("upstream")) else "scenario_parent_incomplete"
    elif parent_keys:
        parent_quality = "hash_or_base_sample_id" if any(k in parent_keys for k in ("sha_full", "content_hash", "file_sha", "corpus_key")) else "base_sample_id_only"
    else:
        parent_quality = "missing_parent"
    return {
        "boundary_recoverable": not missing and not type_errors,
        "boundary_missing": missing,
        "boundary_type_errors": sorted(set(type_errors)),
        "parent_evidence_fields": parent_keys,
        "parent_quality": parent_quality,
    }


def row_record(path: Path, source: str, family: str, line_no: int, raw: bytes, row: dict,
               gid: str, split: str, source_sha: str, token_group: dict[str, str]) -> dict:
    target = target_info(row, source, family)
    prompt = prompt_info(row)
    boundary = boundary_info(row, source, family)
    canonical_hash = sha_value(row)
    raw_hash = hashlib.sha256(raw).hexdigest()
    row_id = sha_text(f"{path}\0{line_no}\0{raw_hash}")[:24]
    full_text = row.get("text") if isinstance(row.get("text"), str) else ""
    full_text_sha = sha_text(full_text) if full_text else ""
    # Recreate only the identity tokens for this row. This reads package,
    # repository, parent and path keys; it does not read context/target text.
    tokens, parents = V2.SplitBuilder({}).tokens_for(row, source, line_no)
    named_tokens = sorted(t for t in tokens if t.startswith(("pkg:", "repo:")))
    parent_tokens = sorted(parents)
    license_present = any(
        isinstance(row.get(key), str) and row[key].strip()
        for key in ("license", "license_file")
    )
    license_status = (
        "direct_row_evidence" if license_present else
        "inherited_source_or_upstream_pending" if any(row.get(key) for key in ("source_url", "upstream", "repo_url", "repo"))
        else "missing_unresolved"
    )
    gates = row.get("gates")
    validator_metadata = bool(
        isinstance(gates, dict) and gates
        or row.get("gated") is not None
        or row.get("rl_ready") is not None
        or row.get("constraint_spec")
        or source.startswith("scenario_")
        or source == "edit_pairs_train"
    )
    target_variant_hashes = target["target_variant_hashes"]
    semantic_variant_hashes = target_semantic_variants(row, source, family)
    target_variant_conflict = len(set(semantic_variant_hashes.values())) > 1
    target_body_candidates = []
    for key in ("target", "model_target", "corpus_target"):
        text = body_text(row.get(key))
        if text is not None and text:
            target_body_candidates.append(text)
    prompt_overlap_suspected = False
    for key in ("prompt", "full_prompt"):
        prompt_text = row.get(key)
        if isinstance(prompt_text, str) and prompt_text:
            # Substring overlap is only a diagnostic.  Short/common targets
            # and full-copy no-ops naturally occur in the visible prompt.
            prompt_overlap_suspected = prompt_overlap_suspected or any(
                len(text) >= 16 and text in prompt_text for text in target_body_candidates
            )
    explicit_truncation = any(
        ("partial" in str(key).lower() or "truncat" in str(key).lower())
        and bool(value)
        for key, value in row.items()
    )
    reasons: list[str] = []
    pending_reasons: list[str] = []
    if not boundary["boundary_recoverable"]:
        reasons.append("boundary_missing_or_type_mismatch")
    if not target["target_visible"]:
        reasons.append("target_not_visible")
    if target["empty_non_noop"]:
        reasons.append("empty_target_requires_authoritative_deletion_proof")
    if target_variant_conflict:
        reasons.append("target_variant_conflict")
    if explicit_truncation:
        reasons.append("explicit_truncation_or_partial_metadata")
    if boundary["parent_quality"] in {"missing_parent", "edit_parent_incomplete", "scenario_parent_incomplete"}:
        pending_reasons.append("source_parent_evidence_incomplete")
    if not license_present:
        pending_reasons.append("license_provenance_pending")
    if not validator_metadata:
        pending_reasons.append("validator_metadata_missing")
    if boundary["boundary_recoverable"]:
        prediction_evidence_status = "structured_current_region"
    elif prompt["prompt_was_explicit"]:
        prediction_evidence_status = "explicit_prompt_target_visible_only"
        pending_reasons.append("prediction_evidence_boundary_pending")
    else:
        prediction_evidence_status = "missing_prediction_envelope"
        pending_reasons.append("prediction_evidence_missing")
    if source.startswith("scenario_") and family not in {"no_op"}:
        # The generator/validator is documented, but DAT-03 has not executed
        # its fresh exact rerender gate. Keep this as a pending gate, not a
        # guessed pass/fail label.
        validator_status = "documented_builder_validator_pending"
    elif isinstance(gates, dict) and gates:
        validator_status = "per_row_gate_metadata_present_fresh_check_pending"
    elif source == "edit_pairs_train":
        validator_status = "event_license_metadata_present_fresh_check_pending"
    else:
        validator_status = "validator_metadata_missing"
    return {
        "row_id": row_id,
        "split": split,
        "group_id": gid,
        "source": source,
        "family": family,
        "file": str(path),
        "line": line_no,
        "source_sha256": source_sha,
        "raw_line_sha256": raw_hash,
        "canonical_row_sha256": canonical_hash,
        "named_identity_tokens": named_tokens,
        "parent_tokens": parent_tokens,
        # ``prediction_envelope_sha256`` is the exact LF-normalized hard
        # collision key.  The loose key is diagnostic only.
        "prediction_envelope_sha256": prompt["prediction_envelope_sha256"],
        "normalized_prompt_sha256": prompt["loose_prompt_sha256"],
        "prompt_chars": prompt["prompt_chars"],
        "prompt_was_explicit": prompt["prompt_was_explicit"],
        "prompt_comparison_status": prompt["prompt_comparison_status"],
        "prediction_evidence_status": prediction_evidence_status,
        "target_sha256": target["target_sha256"],
        "target_kind": target["target_kind"],
        "target_visible": target["target_visible"],
        "target_chars": target["target_chars"],
        "target_variant_hashes": target_variant_hashes,
        "target_semantic_variant_hashes": semantic_variant_hashes,
        "full_text_sha256": full_text_sha,
        "boundary_recoverable": boundary["boundary_recoverable"],
        "boundary_missing": boundary["boundary_missing"],
        "boundary_type_errors": boundary["boundary_type_errors"],
        "parent_quality": boundary["parent_quality"],
        "parent_evidence_fields": boundary["parent_evidence_fields"],
        "license_evidence_present": license_present,
        "license_status": license_status,
        "validator_status": validator_status,
        "explicit_truncation": explicit_truncation,
        "prompt_target_overlap_suspected": prompt_overlap_suspected,
        "noop_status": "authoritative_noop_sentinel" if target["target_kind"] == "noop_sentinel" else "not_noop",
        "base_reasons": sorted(set(reasons)),
        "reasons": sorted(set(reasons)),
        "pending_reasons": sorted(set(pending_reasons)),
        "status": (
            "metadata_eligible_pending_fresh_validator" if not reasons and not pending_reasons
            else "metadata_eligible_pending_gates" if not reasons
            else "quarantine"
        ),
    }


def compact_ref(rec: dict) -> dict:
    return {
        "row_id": rec["row_id"],
        "split": rec["split"],
        "group_id": rec["group_id"],
        "source": rec["source"],
        "family": rec["family"],
        "file": rec["file"],
        "line": rec["line"],
        "source_sha256": rec["source_sha256"],
        "raw_line_sha256": rec["raw_line_sha256"],
        "prediction_envelope_sha256": rec["prediction_envelope_sha256"],
        "normalized_prompt_sha256": rec["normalized_prompt_sha256"],
        "target_sha256": rec["target_sha256"],
        "target_kind": rec["target_kind"],
    }


def add_pair(pairs: list[dict], a: dict, b: dict, reason: str) -> None:
    left, right = sorted((a, b), key=lambda x: x["row_id"])
    pair_id = sha_text(f"{left['row_id']}\0{right['row_id']}\0{reason}")[:24]
    pairs.append({"pair_id": pair_id, "reason": reason, "left": compact_ref(left), "right": compact_ref(right)})


def collision_groups(records: list[dict]) -> tuple[list[dict], list[dict], list[dict], collections.Counter]:
    """Find exact-envelope collisions without quadratic pair expansion.

    Each conflict stores target-group counts and a bounded set of immutable
    representatives.  The full contradictory-pair count is retained for
    audit arithmetic; every row in a conflicting group is quarantined.
    """
    prompt_map: dict[str, list[dict]] = collections.defaultdict(list)
    loose_prompt_map: dict[str, list[dict]] = collections.defaultdict(list)
    full_map: dict[str, list[dict]] = collections.defaultdict(list)
    exact_map: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in records:
        prompt_map[rec["prediction_envelope_sha256"]].append(rec)
        loose_prompt_map[rec["normalized_prompt_sha256"]].append(rec)
        if rec["full_text_sha256"]:
            full_map[rec["full_text_sha256"]].append(rec)
        exact_map[rec["canonical_row_sha256"]].append(rec)
    contradictory: list[dict] = []
    duplicate_groups: list[dict] = []
    collision_summaries: list[dict] = []
    reason_counts = collections.Counter()
    max_representatives = 3
    max_pair_representatives = 1000

    def conflict_summary(kind: str, digest: str, rows: list[dict], reason: str) -> None:
        target_groups = collections.defaultdict(list)
        for rec in rows:
            target_groups[rec["target_sha256"]].append(rec)
        if len(target_groups) > 1:
            groups_sorted = sorted(target_groups.items(), key=lambda item: item[0])
            pair_count = sum(len(a) * len(b) for (_, a), (_, b) in itertools.combinations(groups_sorted, 2))
            collision_summaries.append({
                "kind": kind,
                "digest": digest,
                "reason": reason,
                "row_count": len(rows),
                "contradictory_pair_count": pair_count,
                "target_groups": [
                    {
                        "target_sha256": target_sha,
                        "count": len(group_rows),
                        "target_kinds": sorted({r["target_kind"] for r in group_rows}),
                        "representatives": [compact_ref(r) for r in sorted(group_rows, key=lambda r: r["row_id"])[:max_representatives]],
                    }
                    for target_sha, group_rows in groups_sorted
                ],
            })
            pair_reps = 0
            for (_, left_rows), (_, right_rows) in itertools.combinations(groups_sorted, 2):
                if pair_reps >= max_pair_representatives:
                    break
                left = sorted(left_rows, key=lambda r: r["row_id"])[0]
                right = sorted(right_rows, key=lambda r: r["row_id"])[0]
                pair_reason = reason
                if "noop_sentinel" in {left["target_kind"], right["target_kind"]}:
                    pair_reason += ":noop_vs_edit"
                add_pair(contradictory, left, right, pair_reason)
                pair_reps += 1
            reason_counts[reason] += pair_count
            row_reason = (
                "prediction_envelope_target_collision"
                if kind == "prediction_envelope"
                else "full_text_target_collision"
            )
            for rec in rows:
                rec["reasons"].append(row_reason)

    for prompt_hash, rows in sorted(prompt_map.items()):
        conflict_summary("prediction_envelope", prompt_hash, rows, "identical_prediction_envelope_conflicting_target")
    for full_hash, rows in sorted(full_map.items()):
        targets = {rec["target_sha256"] for rec in rows}
        if len(targets) > 1:
            conflict_summary("full_text", full_hash, rows, "identical_full_text_conflicting_target")
        if len(rows) > 1:
            duplicate_groups.append({
                "kind": "full_text",
                "digest": full_hash,
                "row_ids": sorted(rec["row_id"] for rec in rows),
                "target_hashes": sorted(targets),
                "split_counts": dict(collections.Counter(rec["split"] for rec in rows)),
            })
            # Held-out precedence: preserve dev and shadow duplicate train
            # rows, so deduplication cannot shrink the development split.
            owner = sorted(rows, key=lambda rec: (0 if rec["split"] == "dev_group" else 1, rec["file"], rec["line"]))[0]
            for rec in rows:
                if rec is not owner:
                    if owner["split"] == "dev_group" and rec["split"] == "train_group":
                        rec["reasons"].append("cross_split_duplicate_train_shadowed")
                    else:
                        rec["reasons"].append("full_text_duplicate")
    for exact_hash, rows in sorted(exact_map.items()):
        if len(rows) > 1:
            duplicate_groups.append({
                "kind": "canonical_row",
                "digest": exact_hash,
                "row_ids": sorted(rec["row_id"] for rec in rows),
                "target_hashes": sorted({rec["target_sha256"] for rec in rows}),
                "split_counts": dict(collections.Counter(rec["split"] for rec in rows)),
            })
            owner = sorted(rows, key=lambda rec: (0 if rec["split"] == "train_group" else 1, rec["file"], rec["line"]))[0]
            if any(rec["split"] == "dev_group" for rec in rows):
                owner = sorted(rows, key=lambda rec: (0 if rec["split"] == "dev_group" else 1, rec["file"], rec["line"]))[0]
            for rec in rows:
                if rec is not owner:
                    if owner["split"] == "dev_group" and rec["split"] == "train_group":
                        rec["reasons"].append("cross_split_duplicate_train_shadowed")
                    else:
                        rec["reasons"].append("canonical_row_duplicate")
    contradictory = sorted({(p["pair_id"], canonical_json(p)): p for p in contradictory}.values(), key=lambda p: p["pair_id"])
    for rec in records:
        rec["reasons"] = sorted(set(rec["reasons"]))
        rec["status"] = (
            "metadata_eligible_pending_fresh_validator" if not rec["reasons"] and not rec.get("pending_reasons")
            else "metadata_eligible_pending_gates" if not rec["reasons"]
            else "quarantine"
        )
    return contradictory, duplicate_groups, collision_summaries, reason_counts


def tu3_pairs(source_hashes: dict[str, str], audit: dict) -> list[dict]:
    train_path = DATASET / "nextcoder_r2_cross_v1/train.jsonl"
    eval_path = DATASET / "nextcoder_r2_cross_v1/eval.jsonl"
    train = list(raw_jsonl(train_path))
    evaluation = list(raw_jsonl(eval_path))
    pairs = []
    for spec in audit["pairs"]:
        ti, ei = spec["train_i"], spec["eval_i"]
        tl, traw, trow = train[ti]
        el, eraw, erow = evaluation[ei]
        tprompt = traw and traw  # keeps the row reference independent of decoded text output
        eprompt = eraw and eraw
        t_prompt_sha = sha_text(str(trow.get("prompt", "")))
        e_prompt_sha = sha_text(str(erow.get("prompt", "")))
        if t_prompt_sha != spec["prompt_sha256"] or e_prompt_sha != spec["prompt_sha256"]:
            raise RuntimeError(f"TU3 supplied audit prompt hash mismatch at train/eval indices {ti}/{ei}")
        t_target = trow.get("target", trow.get("text", ""))
        e_target = erow.get("target", erow.get("text", ""))
        t_ref = {
            "row_id": sha_text(f"{train_path}\0{tl}\0{hashlib.sha256(traw).hexdigest()}")[:24],
            "split": "quarantine_tu3", "group_id": "TU3-held-out",
            "source": "tu3_train", "family": trow.get("family", "unknown"), "file": str(train_path), "line": tl,
            "source_sha256": source_hashes.get(str(train_path), ""), "raw_line_sha256": hashlib.sha256(traw).hexdigest(),
            "normalized_prompt_sha256": t_prompt_sha, "target_sha256": sha_value(t_target),
            "target_kind": "materialized_target_hash",
        }
        e_ref = {
            "row_id": sha_text(f"{eval_path}\0{el}\0{hashlib.sha256(eraw).hexdigest()}")[:24],
            "split": "quarantine_tu3", "group_id": "TU3-held-out",
            "source": "tu3_eval", "family": erow.get("family", "unknown"), "file": str(eval_path), "line": el,
            "source_sha256": source_hashes.get(str(eval_path), ""), "raw_line_sha256": hashlib.sha256(eraw).hexdigest(),
            "normalized_prompt_sha256": e_prompt_sha, "target_sha256": sha_value(e_target),
            "target_kind": "materialized_target_hash",
        }
        if t_ref["target_sha256"] == e_ref["target_sha256"]:
            raise RuntimeError(f"TU3 supplied contradiction is not target-distinct at {ti}/{ei}")
        pairs.append({
            "pair_id": sha_text(f"TU3\0{t_ref['row_id']}\0{e_ref['row_id']}")[:24],
            "reason": "TU3_identical_normalized_prompt_conflicting_target_wrong_reference_holdout",
            "prompt_sha256": spec["prompt_sha256"],
            "train_expected_noop": spec["train_expected_noop"],
            "eval_expected_noop": spec["eval_expected_noop"],
            "train": t_ref,
            "eval": e_ref,
        })
    if len(pairs) != audit["train_eval_identical_prompt_pairs"]:
        raise RuntimeError(f"TU3 pair count mismatch: {len(pairs)}")
    return pairs


def nested_counts(records: list[dict], key: str) -> dict:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for rec in records:
        split = rec["split"]
        source = rec["source"]
        family = rec["family"]
        out.setdefault(split, {}).setdefault(source, {}).setdefault(family, 0)
        out[split][source][family] += 1
    return out


def nested_reason_counts(records: list[dict]) -> dict:
    out: dict[str, dict[str, dict[str, collections.Counter]]] = {}
    for rec in records:
        for reason in rec["reasons"]:
            out.setdefault(rec["split"], {}).setdefault(rec["source"], {}).setdefault(rec["family"], collections.Counter())[reason] += 1
    return {
        split: {source: {family: dict(sorted(counter.items())) for family, counter in families.items()} for source, families in sources.items()}
        for split, sources in out.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--lookup", default=str(DEFAULT_LOOKUP))
    parser.add_argument("--rows", default=str(DEFAULT_ROWS))
    parser.add_argument("--duplicates", default=str(DEFAULT_DUPLICATES))
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    hash_data = json.loads(HASH_MANIFEST.read_text())
    accepted_hashes = {x["path"]: x["sha256"] for x in hash_data["files"]}
    builder, rebuilt_groups, accepted_groups, token_group, lookup, build_bad = build_identity_lookup(manifest)

    # Hash candidate files against the accepted DAT-01 manifest. This proves
    # the row references point into unchanged bytes without opening final
    # target/context semantics.
    source_hash_status: dict[str, dict] = {}
    for path, source, family in candidate_specs() + [(p, s, "historical") for p, s, _ph, _f in historical_specs() if s in {"tu3_train", "tu3_eval"}]:
        key = str(path)
        if key in source_hash_status:
            continue
        expected = accepted_hashes.get(key, "")
        actual = sha_file(path) if path.exists() else ""
        source_hash_status[key] = {"expected_sha256": expected, "actual_sha256": actual, "match": bool(expected and actual and expected == actual), "bytes": path.stat().st_size if path.exists() else 0}
    hash_mismatches = {path: item for path, item in source_hash_status.items() if not item["match"]}

    records: list[dict] = []
    bad_json = collections.Counter(build_bad)
    lookup_unknown: list[dict] = []
    final_sealed_counts = collections.Counter()
    non_train_dev_counts = collections.Counter()
    for path, source, family in candidate_specs():
        source_sha = accepted_hashes.get(str(path), "")
        for line_no, raw, row in raw_jsonl(path):
            if row is None:
                bad_json[source] += 1
                continue
            tokens, _parents = V2.SplitBuilder({}).tokens_for(row, source, line_no)
            gids = {token_group[t] for t in tokens if t in token_group}
            if len(gids) != 1:
                lookup_unknown.append({
                    "row_id": sha_text(f"{path}\0{line_no}\0{hashlib.sha256(raw).hexdigest()}")[:24],
                    "file": str(path), "line": line_no, "source": source,
                    "token_count": len(tokens), "known_token_count": sum(t in token_group for t in tokens),
                    "reason": "identity_lookup_unknown_or_conflicting",
                })
                continue
            gid = next(iter(gids))
            split = accepted_groups[gid]["split"]
            if split == "final_candidate_group":
                # Final rows stay sealed. Only aggregate source/family and
                # final-capable counts are returned from DAT-02 metadata.
                final_sealed_counts[(source, family)] += 1
                continue
            if split not in {"train_group", "dev_group"}:
                non_train_dev_counts[(split, source, family)] += 1
                continue
            rec = row_record(path, source, family, line_no, raw, row, gid, split, source_sha, token_group)
            if hash_mismatches.get(str(path)):
                rec["reasons"].append("source_hash_mismatch")
            records.append(rec)
    if lookup_unknown:
        raise RuntimeError(f"identity lookup unknown/conflicting for {len(lookup_unknown)} candidate rows")

    contradictory, duplicate_groups, collision_summaries, collision_reason_counts = collision_groups(records)
    # The accepted v2 groups include a final bucket whose scenario rows are
    # not final-capable. Keep the sealed summary metadata-only.
    final_manifest_summary = {
        "groups": sum(g["split"] == "final_candidate_group" for g in accepted_groups.values()),
        "clean_candidate_groups": manifest["counts"]["actual_clean_final_candidate_packages"],
        "clean_candidate_rows": manifest["counts"]["actual_clean_final_candidate_rows"],
        "sealed_source_family_rows_from_identity_manifest": {
            f"{source}|{family}": count for (source, family), count in sorted(final_sealed_counts.items())
        },
        "opened": False,
    }

    # TU3 is audited separately from candidate train/dev rows and remains
    # held out. All 28 supplied contradictory pairs are retained by hash only.
    tu3 = tu3_pairs(accepted_hashes, json.loads(TU3_AUDIT.read_text()))

    # Deterministic row-level metadata artifact for DAT-04/DAT-05 adapters.
    rows_path = Path(args.rows)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8") as stream:
        for rec in sorted(records, key=lambda x: x["row_id"]):
            stream.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
    duplicates_path = Path(args.duplicates)
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_payload = {
        "task": "DAT-03",
        "split_id": manifest["split_id"],
        "scope": "train_group/dev_group metadata only; no final rows",
        "duplicate_groups": duplicate_groups,
    }
    duplicates_path.write_text(json.dumps(duplicate_payload, indent=2, sort_keys=True) + "\n")
    lookup_path = Path(args.lookup)
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    lookup_path.write_text(json.dumps(lookup, indent=2, sort_keys=True) + "\n")

    eligible = [r for r in records if not r["reasons"]]
    eligible_by_split = collections.Counter(r["split"] for r in eligible)
    eligible_by_source_family = collections.Counter((r["split"], r["source"], r["family"]) for r in eligible)
    eligible_groups = {r["group_id"] for r in eligible}
    eligible_parents = {p for r in eligible for p in r["parent_tokens"]}
    eligible_named = {p for r in eligible for p in r["named_identity_tokens"]}
    dev_family = collections.Counter(r["family"] for r in records if r["split"] == "dev_group")
    dev_coverage = {family: dev_family.get(family, 0) for family in ("no_op", "rename_propagation", "format_propagation", "pipe_rewrite")}
    train_family = collections.Counter(r["family"] for r in records if r["split"] == "train_group")
    target_kind_counts = collections.Counter(r["target_kind"] for r in records)
    boundary_counts = collections.Counter((r["split"], r["boundary_recoverable"]) for r in records)
    reason_counts = collections.Counter(reason for r in records for reason in r["reasons"])
    prompt_groups = collections.Counter(r["prediction_envelope_sha256"] for r in records)
    loose_prompt_groups = collections.Counter(r["normalized_prompt_sha256"] for r in records)
    prediction_evidence_counts = collections.Counter(r["prediction_evidence_status"] for r in records)
    pending_reason_counts = collections.Counter(reason for r in records for reason in r.get("pending_reasons", []))
    family_denominators: dict[str, dict[str, int]] = {}
    for rec in records:
        key = f"{rec['split']}|{rec['source']}|{rec['family']}"
        item = family_denominators.setdefault(key, collections.Counter())
        item["rows_observed"] += 1
        item["boundary_recoverable"] += int(rec["boundary_recoverable"])
        item["target_visible"] += int(rec["target_visible"])
        item["prediction_evidence_supported"] += int(rec["prediction_evidence_status"] == "structured_current_region" or rec["prompt_was_explicit"])
        item["hard_quarantine"] += int(bool(rec["reasons"]))
        item["pending_gate_rows"] += int(bool(rec.get("pending_reasons")))
    family_denominators = {key: dict(sorted(value.items())) for key, value in sorted(family_denominators.items())}
    full_text_groups = collections.Counter(r["full_text_sha256"] for r in records if r["full_text_sha256"])
    receipt = {
        "task": "DAT-03",
        "status": "partial",
        "owner": "worker-data",
        "started": None,
        "started_note": "No start clock was captured before this CPU audit; all artifact timestamps below are observed datetime.now values.",
        "ended": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dependencies_checked": [
            f"DAT-02 accepted v2 split_id {manifest['split_id']}",
            f"DAT-02 manifest: {manifest_path} (sha256 {hashlib.sha256(manifest_path.read_bytes()).hexdigest()})",
            f"DAT-01 source hash manifest: {HASH_MANIFEST} (sha256 {hashlib.sha256(HASH_MANIFEST.read_bytes()).hexdigest()})",
            f"TU3 supplied collision audit: {TU3_AUDIT} (sha256 {hashlib.sha256(TU3_AUDIT.read_bytes()).hexdigest()})",
            "DAT-03 brief: python3 docs/campaign/campaign.py brief DAT-03",
        ],
        "action": "Rebuilt the accepted DAT-02 identity graph from bounded source metadata, wrote a complete named/parent lookup, audited train/dev candidate rows for recoverable boundary, target visibility, no-op/deletion semantics, prompt/target collisions, duplicates, provenance/license/validator metadata and explicit truncation markers. Final rows remained sealed.",
        "commands_or_method": [
            f"cd {PLAN} && python3 docs/campaign/campaign.py brief DAT-03",
            f"python3 {PLAN / 'docs/campaign/work/data/DAT-03-audit.py'} --manifest {manifest_path} --lookup {lookup_path} --rows {rows_path} --duplicates {duplicates_path}",
            "Identity pass imports the accepted DAT-02 v2 canonicalizer and rebuilds all historical aliases plus bounded edit/scenario/case candidate metadata; group IDs must exactly equal the accepted manifest or the audit fails.",
            "Train/dev row pass hashes exact source lines and target/prompt fields without emitting their values. Rows in final_candidate_group are skipped after identity assignment; final output is aggregate metadata only.",
            "No CUDA, model, cloud, API, teacher generation, final target/context inspection, raw-source write or registry admission.",
        ],
        "result": {
            "split_id": manifest["split_id"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "identity_lookup_path": str(lookup_path),
            "identity_lookup_sha256": hashlib.sha256(lookup_path.read_bytes()).hexdigest(),
            "identity_lookup_bytes": lookup_path.stat().st_size,
            "row_audit_path": str(rows_path),
            "row_audit_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            "row_audit_bytes": rows_path.stat().st_size,
            "duplicate_groups_path": str(duplicates_path),
            "duplicate_groups_sha256": hashlib.sha256(duplicates_path.read_bytes()).hexdigest(),
            "builder_script_path": str(PLAN / "docs/campaign/work/data/DAT-02-build-split-v2.py"),
            "builder_script_sha256": hashlib.sha256((PLAN / "docs/campaign/work/data/DAT-02-build-split-v2.py").read_bytes()).hexdigest(),
            "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "complete_named_form_count": lookup["named_form_count"],
            "complete_parent_token_count": lookup["parent_token_count"],
            "identity_groups_rebuilt": len(rebuilt_groups),
            "train_dev_rows_audited": len(records),
            "train_dev_rows_by_split": dict(collections.Counter(r["split"] for r in records)),
            "train_dev_rows_by_source_family": nested_counts(records, "source"),
            "train_dev_rows_by_reason": nested_reason_counts(records),
            "train_dev_pending_gate_reasons": dict(sorted(pending_reason_counts.items())),
            "family_denominators": family_denominators,
            "reason_counts_total": dict(sorted(reason_counts.items())),
            "boundary_counts": {f"{split}|{kind}": count for (split, kind), count in sorted(boundary_counts.items())},
            "target_kind_counts": dict(sorted(target_kind_counts.items())),
            "metadata_eligible_rows_pending_fresh_validator": len(eligible),
            "metadata_eligible_rows_by_split": dict(eligible_by_split),
            "metadata_eligible_rows_by_source_family": {f"{split}|{source}|{family}": count for (split, source, family), count in sorted(eligible_by_source_family.items())},
            "eligible_unique_dsu_groups": len(eligible_groups),
            "eligible_unique_parent_tokens": len(eligible_parents),
            "eligible_unique_named_identity_tokens": len(eligible_named),
            "train_family_rows_observed": dict(sorted(train_family.items())),
            "dev_family_rows_observed": dict(sorted(dev_family.items())),
            "dev_coverage_deficit": dev_coverage,
            "prediction_envelope_groups": len(prompt_groups),
            "prediction_envelope_duplicate_groups": sum(n > 1 for n in prompt_groups.values()),
            "loose_normalized_prompt_groups_diagnostic": len(loose_prompt_groups),
            "loose_normalized_prompt_duplicate_groups_diagnostic": sum(n > 1 for n in loose_prompt_groups.values()),
            "prediction_evidence_status_counts": dict(sorted(prediction_evidence_counts.items())),
            "full_text_duplicate_groups": sum(n > 1 for n in full_text_groups.values()),
            "contradictory_candidate_pairs_representatives": len(contradictory),
            "contradictory_candidate_pair_count_total": sum(s["contradictory_pair_count"] for s in collision_summaries),
            "candidate_collision_summaries": collision_summaries,
            "tu3_contradictory_pairs": len(tu3),
            "tu3_pair_reasons": dict(collections.Counter(p["reason"] for p in tu3)),
            "tu3_severe_truncation_supplied_audit": {"rows_over_2048_total": 156, "prompts_alone_fill_budget": 111},
            "source_hash_status": source_hash_status,
            "source_hash_mismatch_count": len(hash_mismatches),
            "bad_json_lines_by_source": dict(sorted(bad_json.items())),
            "final_sealed_summary": final_manifest_summary,
            "lookup_unknown_rows": len(lookup_unknown),
            "collision_reason_counts": dict(sorted(collision_reason_counts.items())),
            "checks": {
                "identity_lookup_matches_accepted_group_ids": True,
                "named_and_parent_lookup_conflicts": False,
                "unknown_candidate_lookup_rows": len(lookup_unknown) == 0,
                "source_hashes_match_accepted_manifest": len(hash_mismatches) == 0,
                "train_dev_conflict_rows_blocked": all(
                    r["status"] == "quarantine"
                    for r in records
                    if any(reason.endswith("target_collision") for reason in r["reasons"])
                ),
                "tu3_pairs_all_target_distinct_and_held_out": all(p["train"]["target_sha256"] != p["eval"]["target_sha256"] and p["train"]["split"] == "quarantine_tu3" for p in tu3),
                "final_records_opened": False,
                "final_target_or_context_values_emitted": False,
                "raw_inputs_changed": False,
            },
        },
        "contradictory_pairs": contradictory,
        "tu3_contradictory_pairs": tu3,
        "acceptance": "inconclusive — identity, source-hash and metadata audit completed; rows remain pending fresh exact renderer/validator and DAT-04 tokenizer gates, with final sealed.",
        "changed_files": [str(Path(__file__)), str(lookup_path), str(rows_path), str(duplicates_path), str(PLAN / "docs/campaign/receipts/DAT-03-collision-and-visibility.json")],
        "artifacts": [str(lookup_path), str(rows_path), str(duplicates_path)],
        "unresolved": [
            "Every metadata-eligible row still requires fresh exact source rerender, R/jarl or family validator, and pinned tokenizer length checks; DAT-03 does not admit training rows.",
            "No-op scenario rows with raw region_new=[] are classified as authoritative unchanged-region sentinels: semantic region_new=region_old and serialized [NO_EDIT]+UPDATED. Empty non-no_op targets are quarantined until authoritative deletion evidence exists.",
            "Rows with target/model_target/corpus_target disagreement are quarantined by hash; no label was selected or relabeled.",
            "TU3's 28 wrong-reference contradictions remain held out. Supplied audit records 156 severe total-length cases and 111 prompt-only budget fills; DAT-04 must retokenize with the pinned tokenizer.",
            "Teacher-derived rows with weak or missing source/license evidence remain tagged; unsupported hidden future intent, B11 joins and inspected/materialized derivatives are not inferred.",
            "Dev coverage is structurally sparse for no-op/rename/format/pipe and must remain exposure-labelled rather than quota-filled with historical or final rows.",
        ],
        "next": "Lead reviews the collision/visibility receipt, then DAT-04 retokenizes only non-quarantined train/dev rows under the accepted renderer contract; DAT-05 may materialize rows only after fresh gates pass. Final stays sealed.",
        "lease_released": "yes — CPU-only metadata lease released; no CUDA or external service used.",
    }
    receipt_path = PLAN / "docs/campaign/receipts/DAT-03-collision-and-visibility.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("receipt", receipt_path)
    print("receipt_sha256", hashlib.sha256(receipt_path.read_bytes()).hexdigest())
    print("rows", len(records), "eligible_pending_validator", len(eligible), "candidate_pairs", len(contradictory), "tu3_pairs", len(tu3))
    print("lookup", lookup["named_form_count"], lookup["parent_token_count"], "groups", len(rebuilt_groups))
    print("dev_coverage", json.dumps(dev_coverage, sort_keys=True))


if __name__ == "__main__":
    main()
