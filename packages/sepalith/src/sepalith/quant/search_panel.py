"""TRAIN search panel for quantization search (card C09, distinct from DEV panels).

The quant gate's DEV250/DEV75 panels are scarce: DEV evaluations are reserved
for final confirmations.  This module builds a *search* panel from editing-SFT
TRAIN token rows so recipe iterations can measure greedy agreement with the
bf16 GGUF (identical generations, cap runaways, length drift) without touching
DEV data.

Rows are TRAIN-only and family-balanced.  A row enters only if it is in neither
the imatrix calibration set nor the held-out KLD slice:

- ``calibration.ids.json`` lists every row whose tokens contributed to
  ``calibration.txt`` (whole rows and filler prefixes);
- ``kld-heldout.ids.json`` lists the KLD window rows.

The panel stores the pre-tokenized prompt (``input_ids[:target_start]``) and
metadata per case; generation reuses ``sepalith.quant.gguf_dev`` request
helpers so a search-panel run and a DEV run share identical server semantics.
Scoring compares two result files case by case (agreement, cap transitions,
length delta) rather than scoring against gold.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from sepalith.quant import gguf_dev as gd
from sepalith.quant.gate import load_cases

SCHEMA = "sepalith.c09.quant-search-panel.v1"


def _row_ids(path: Path) -> set[str]:
    """Row ids recorded in a calibration/KLD ids file (nested lists tolerated)."""
    data = json.loads(path.read_text())
    flat: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, str):
            flat.add(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(data)
    return flat


def build_panel(rows: Sequence[Mapping[str, Any]], *, per_family: int, seed: int,
                calibration_ids: set[str], kld_ids: set[str]) -> list[dict[str, Any]]:
    """Family-balanced TRAIN panel excluding calibration and KLD rows.

    Rows must already be eligible TRAIN rows (``split == "train"``, DEV
    exclusion applied upstream by the caller).  Within a family, rows are
    ordered by id and shuffled with a seeded permutation.
    """
    excluded = calibration_ids | kld_ids
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("split")) != "train":
            raise ValueError("search panels admit TRAIN rows only")
        if str(row["id"]) in excluded:
            continue
        by_family.setdefault(str(row["family"]), []).append(row)
    panel: list[dict[str, Any]] = []
    for family in sorted(by_family):
        pool = sorted(by_family[family], key=lambda r: str(r["id"]))
        random.Random(f"{seed}:search:{family}").shuffle(pool)
        for row in pool[:per_family]:
            ids = list(row["input_ids"])
            start = int(row["target_start"])
            if not 0 < start < len(ids):
                raise ValueError(f"row {row['id']} has a degenerate target_start")
            panel.append({
                "id": str(row["id"]), "family": family, "package_id": str(row.get("package_id", "train")),
                "prompt_ids": ids[:start], "reference_tokens": len(ids) - start,
                "target_tokens": len(ids) - start,
            })
    return panel


def load_panel(path: Path) -> list[dict[str, Any]]:
    """Read a search panel written by :func:`build_panel` (with its manifest)."""
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError("not a quant search panel")
    cases, seen = data["cases"], set()
    for case in cases:
        if case["id"] in seen or not case["prompt_ids"]:
            raise ValueError("search panel ids must be unique with nonempty prompts")
        seen.add(case["id"])
    return cases


def panel_sha256(cases: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps([c["id"] for c in cases], separators=(",", ":")).encode()).hexdigest()


def completion_for(prompt_ids: Sequence[int], cap: int) -> dict[str, Any]:
    """A DEV-equivalent greedy request for one search-panel case."""
    return gd.completion_request(prompt_ids, cap)


def generated(response: Mapping[str, Any], prompt_len: int, cap: int) -> list[int]:
    return gd.generated_ids(response, prompt_len, cap)


def case_record(case: Mapping[str, Any], tokens: list[int], seconds: float, cap: int) -> dict[str, Any]:
    termination = gd.generation_termination(tokens, cap, gd.NATIVE_EOG_IDS)  # type: ignore[attr-defined]
    return {"id": case["id"], "family": case["family"], "prompt_tokens": len(case["prompt_ids"]),
            "generated_tokens": len(tokens), "generation_seconds": seconds,
            "cap_hit": bool(termination.get("cap_hit", len(tokens) >= cap)),
            "response_complete": bool(termination["response_complete"]),
            "generated_ids": tokens}


def agreement(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], cap: int) -> dict[str, Any]:
    """Greedy agreement of a candidate with the reference (bf16) run, per case."""
    ref, cand = load_cases(reference), load_cases(candidate)
    if ref.keys() != cand.keys():
        raise ValueError("case ids differ between reference and candidate")
    ids = sorted(ref)
    identical = sum(1 for i in ids if list(ref[i]["generated_ids"]) == list(cand[i]["generated_ids"]))
    done_to_cap = sum(1 for i in ids if not ref[i]["cap_hit"] and cand[i]["cap_hit"])
    cap_to_done = sum(1 for i in ids if ref[i]["cap_hit"] and not cand[i]["cap_hit"])
    deltas = sorted(cand[i]["generated_tokens"] - ref[i]["generated_tokens"] for i in ids)
    return {
        "cases": len(ids),
        "identical": identical,
        "identical_pct": 100.0 * identical / len(ids),
        "cap_reference": sum(1 for i in ids if ref[i]["cap_hit"]),
        "cap_candidate": sum(1 for i in ids if cand[i]["cap_hit"]),
        "net_cap_rise": done_to_cap - cap_to_done,
        "done_to_cap": done_to_cap,
        "cap_to_done": cap_to_done,
        "length_delta_median": deltas[len(deltas) // 2],
        "length_delta_p90": deltas[int(len(deltas) * 0.9)],
        "changed_case_ids": [i for i in ids if list(ref[i]["generated_ids"]) != list(cand[i]["generated_ids"])],
    }
