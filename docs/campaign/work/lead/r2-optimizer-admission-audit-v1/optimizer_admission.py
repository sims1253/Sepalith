"""Pure admission helpers for the SFT-11 optimizer comparison.

This module does not construct the campaign model.  It creates and verifies the
ordered optimizer-dispatch manifest which a later measured GPU arm must bind.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


HIDDEN_PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)
AURORA_PROJECTIONS = ("gate_proj", "up_proj")


@dataclass(frozen=True)
class Dispatch:
    name: str
    shape: tuple[int, ...]
    dtype: str
    optimizer: str
    reason: str


def _projection(name: str) -> str | None:
    return next((p for p in HIDDEN_PROJECTIONS if f".{p}." in name or name.endswith(f".{p}.weight")), None)


def classify_full_weight(name: str, shape: Sequence[int], dtype: str = "unknown") -> Dispatch:
    """Aurora gate/up, Muon other hidden matrices, AdamW all side parameters."""
    shape = tuple(int(x) for x in shape)
    if ".lora_A." in name or ".lora_B." in name:
        raise ValueError("PEFT factor presented to full-weight optimizer dispatch")
    projection = _projection(name)
    if len(shape) == 2 and projection in AURORA_PROJECTIONS and shape[0] > shape[1]:
        return Dispatch(name, shape, dtype, "aurora", "tall MLP gate/up projection")
    if len(shape) == 2 and projection in HIDDEN_PROJECTIONS:
        return Dispatch(name, shape, dtype, "muon", "other two-dimensional hidden projection")
    return Dispatch(name, shape, dtype, "adamw", "embedding, output, norm, or non-2D side parameter")


def classify_lora(name: str, shape: Sequence[int], dtype: str = "unknown") -> Dispatch:
    """Experimental LoRA translation; original Aurora suffixes match no PEFT names.

    Only B factors of gate/up are tall in the same output-space orientation as
    the dense projection.  All other 2-D factors go to Muon for a paired probe.
    This is a proposed experiment, not evidence that factor-space optimization
    has the same behavior as dense weight optimization.
    """
    shape = tuple(int(x) for x in shape)
    projection = _projection(name)
    is_b = ".lora_B." in name
    if len(shape) == 2 and projection in AURORA_PROJECTIONS and is_b and shape[0] > shape[1]:
        return Dispatch(name, shape, dtype, "aurora", "translated tall gate/up LoRA-B factor")
    if len(shape) == 2 and projection in HIDDEN_PROJECTIONS:
        return Dispatch(name, shape, dtype, "muon", "other two-dimensional LoRA factor")
    return Dispatch(name, shape, dtype, "adamw", "non-2D side parameter")


def ordered_manifest(entries: Iterable[Dispatch]) -> dict:
    rows = [asdict(entry) for entry in entries]
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate parameter name in optimizer dispatch")
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "sepalith.optimizer-dispatch-manifest.v1",
        "ordered_entries": rows,
        "ordered_entries_sha256": hashlib.sha256(canonical).hexdigest(),
        "counts": {key: sum(r["optimizer"] == key for r in rows) for key in ("adamw", "muon", "aurora")},
    }


def verify_manifest(expected: dict, actual: dict) -> None:
    """Reject name/order/shape/dtype/dispatch drift before optimizer-state load."""
    for label, manifest in (("expected", expected), ("actual", actual)):
        rows = manifest.get("ordered_entries")
        if not isinstance(rows, list):
            raise ValueError(f"{label} optimizer dispatch manifest lacks entries")
        names = [row.get("name") for row in rows]
        if len(names) != len(set(names)):
            raise ValueError(f"{label} optimizer dispatch manifest contains duplicate names")
        canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        recomputed = hashlib.sha256(canonical).hexdigest()
        if recomputed != manifest.get("ordered_entries_sha256"):
            raise ValueError(f"{label} optimizer dispatch manifest self-hash differs")
    if expected.get("ordered_entries_sha256") != actual.get("ordered_entries_sha256"):
        raise ValueError("optimizer dispatch manifest differs; optimizer state load refused")
