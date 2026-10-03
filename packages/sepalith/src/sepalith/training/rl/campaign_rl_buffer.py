#!/usr/bin/env python3
"""Hash-checked 15,006-row reward-buffer binding with explicit syntax modes."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

HELD_CONTRADICTORY_IDS = frozenset({
    "8451310ff8e0c5d9f3e77bbc",
    "dd65bd2cd11f38e729a9c712",
})
VERIFIED_MODES = frozenset({"complete_document", "completion_prefix", "framed_fragment"})


class BindingError(ValueError):
    pass


def require(ok: bool, why: str) -> None:
    if not ok:
        raise BindingError(why)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RewardBufferIndex:
    root: Path
    records: Mapping[str, Mapping[str, Any]]
    sidecar_sha256: str
    source_row_count: int
    held_ids: frozenset[str]

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        expected_manifest_sha256: str,
        ordered_ids: Iterable[str],
        *,
        held_ids: frozenset[str] = HELD_CONTRADICTORY_IDS,
    ) -> "RewardBufferIndex":
        require(sha_file(manifest_path) == expected_manifest_sha256, "buffer_manifest_hash_mismatch")
        manifest = json.loads(manifest_path.read_text())
        require(manifest["status"] == "prepared_root_review_required", "buffer_status_not_reviewable")
        root = manifest_path.parent
        meta = manifest["artifacts"]["reward-buffer-sidecar.jsonl"]
        sidecar = root / "reward-buffer-sidecar.jsonl"
        require(sidecar.stat().st_size == meta["bytes"] and sha_file(sidecar) == meta["sha256"],
                "buffer_sidecar_identity_mismatch")
        wanted = list(ordered_ids)
        require(len(wanted) == 15006 and len(set(wanted)) == 15006, "eligible_15006_ids_invalid")
        require(not held_ids.intersection(wanted), "held_contradictory_id_selected")
        wanted_set = set(wanted)
        records: dict[str, Mapping[str, Any]] = {}
        filtered_order: list[str] = []
        all_ids: set[str] = set()
        held_seen: set[str] = set()
        source_rows = 0
        with sidecar.open() as stream:
            for position, line in enumerate(stream):
                row = json.loads(line)
                row_id = row.get("row_id")
                source_rows += 1
                require(row.get("position") == position, "buffer_position_mismatch")
                require(isinstance(row_id, str) and row_id not in all_ids, "buffer_id_invalid_or_duplicate")
                all_ids.add(row_id)
                if row_id in held_ids:
                    held_seen.add(row_id)
                    require(row.get("supported") is False and row.get("repair_reason") == "gold_applied_parse_failed",
                            f"held_id_evidence_differs:{row_id}")
                    continue
                require(row_id in wanted_set, f"unadmitted_sidecar_row:{row_id}")
                if row.get("supported"):
                    mode = row.get("buffer_mode")
                    require(mode in VERIFIED_MODES, "buffer_mode_invalid")
                    if mode == "framed_fragment":
                        require(row.get("framed_projection_parse_ok") is True and row.get("diagnostic_suffix"),
                                "framed_gold_parse_not_verified")
                    else:
                        require(row.get("gold_applied_parse_ok") is True, "gold_parse_not_verified")
                    if mode == "complete_document":
                        require(row.get("baseline_parse_ok") is True, "complete_baseline_parse_not_verified")
                else:
                    require(isinstance(row.get("repair_reason"), str) and row["repair_reason"],
                            "unverified_repair_reason_missing")
                records[row_id] = row
                filtered_order.append(row_id)
        require(source_rows == 15008, "source_sidecar_row_count_differs")
        require(held_seen == set(held_ids), "held_contradictory_ids_missing")
        require(filtered_order == wanted, "buffer_order_or_coverage_mismatch")
        return cls(root, records, meta["sha256"], source_rows, held_ids)

    def envelope_for(self, row_id: str) -> dict[str, Any]:
        require(row_id not in self.held_ids, f"held_contradictory_id_refused:{row_id}")
        require(row_id in self.records, "buffer_row_missing")
        row = self.records[row_id]
        base = {
            "row_id": row_id,
            "position": row["position"],
            "context_sha256": row["replacement_range"]["content_sha256"],
            "source_sidecar_sha256": self.sidecar_sha256,
        }
        if not row.get("supported"):
            return {
                **base,
                "syntax_evidence_mode": "unverified",
                "syntax_available": False,
                "repair_reason": row["repair_reason"],
            }
        mode = row["buffer_mode"]
        return {
            **base,
            "syntax_evidence_mode": mode,
            "syntax_available": True,
            "repair_reason": None,
            "baseline_sha256": row["baseline_sha256"],
            "baseline_bytes": row["baseline_bytes"],
            "baseline_blob": row["baseline_blob"],
            "baseline_parse_ok": row.get("baseline_parse_ok"),
            "gold_applied_parse_ok": row.get("gold_applied_parse_ok"),
            "diagnostic_suffix": row.get("diagnostic_suffix", ""),
            "framed_projection_parse_ok": row.get("framed_projection_parse_ok"),
            "parser_identity": row["parser_identity"],
        }

    def baseline_text(self, envelope: Mapping[str, Any]) -> str:
        row_id = envelope.get("row_id")
        require(row_id not in self.held_ids, f"held_contradictory_id_refused:{row_id}")
        require(envelope.get("syntax_available") is True, "unverified_buffer_load_refused")
        require(envelope.get("source_sidecar_sha256") == self.sidecar_sha256,
                "buffer_envelope_sidecar_identity_mismatch")
        rel = envelope.get("baseline_blob")
        require(isinstance(rel, str) and not Path(rel).is_absolute() and ".." not in Path(rel).parts,
                "baseline_blob_path_invalid")
        path = self.root / rel
        raw = path.read_bytes()
        require(len(raw) == envelope["baseline_bytes"], "baseline_blob_size_mismatch")
        require(hashlib.sha256(raw).hexdigest() == envelope["baseline_sha256"],
                "baseline_blob_hash_mismatch")
        return raw.decode("utf-8")
