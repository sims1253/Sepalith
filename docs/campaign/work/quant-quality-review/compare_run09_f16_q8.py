#!/usr/bin/env python3
"""Compare the two known RUN-09 native DEV receipts without loading a model."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys


EXECUTION_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
RUN_ROOT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training")
PANEL_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl")
sys.path.insert(0, str(EXECUTION_ROOT / "packages" / "sepalith" / "src"))
sys.path.insert(0, str(EXECUTION_ROOT / "experiments" / "training"))

from campaign_eval import classify  # noqa: E402
from sepalith.campaign_protocol import PromptContext  # noqa: E402


def digest(path: Path) -> dict[str, int | str]:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return {"bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def main() -> int:
    panel = {}
    with PANEL_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            panel[row["id"]] = row
    runs = OrderedDict(
        (label, json.loads((RUN_ROOT / f"RUN-09-primary500-{label}-dev-a" / "quality.json").read_text()))
        for label in ("f16", "q8")
    )
    cases = {label: {row["id"]: row for row in value["cases"]} for label, value in runs.items()}
    ids = list(cases["f16"])
    if ids != list(cases["q8"]) or set(ids) != set(panel):
        raise SystemExit("case IDs differ from each other or the pinned panel")

    result: dict[str, object] = {
        "panel": {
            "rows": len(panel),
            "sha256": digest(PANEL_PATH)["sha256"],
            "case_ids_sha256": runs["f16"]["panel"]["case_ids_sha256"],
        },
        "prompt_exact_rows": {},
        "output_exact_rows": {},
        "terminal": {},
        "wire_hf": {},
        "classifier": {},
        "changed_outputs": [],
        "classification_changes": [],
        "families": {},
        "runs": {},
        "artifacts": {},
    }
    for name, getter in {
        "hf_prompt_ids": lambda row: row["hf_prompt_ids"],
        "native_prompt_ids_without_bos": lambda row: row["native_prompt_ids_without_bos"],
        "prompt_sha256": lambda row: row["prompt_sha256"],
        "request": lambda row: row["request"],
        "prompt_token_identity": lambda row: row["prompt_token_identity"],
    }.items():
        result["prompt_exact_rows"][name] = sum(
            getter(cases["f16"][row_id]) == getter(cases["q8"][row_id]) for row_id in ids
        )
    for name, getter in {
        "returned_token_ids": lambda row: row["returned_token_ids"],
        "returned_token_ids_sha256": lambda row: row["returned_token_ids_sha256"],
        "raw_text": lambda row: row["raw_text"],
        "decoded_body_text": lambda row: row["decoded_body_text"],
    }.items():
        result["output_exact_rows"][name] = sum(
            getter(cases["f16"][row_id]) == getter(cases["q8"][row_id]) for row_id in ids
        )

    for row_id in ids:
        f16, q8 = cases["f16"][row_id], cases["q8"][row_id]
        if f16["returned_token_ids"] != q8["returned_token_ids"] or f16["raw_text"] != q8["raw_text"]:
            result["changed_outputs"].append({
                "id": row_id,
                "family": f16["family"],
                "expected_noop": f16["expected_noop"],
                "f16_tokens": len(f16["returned_token_ids"]),
                "q8_tokens": len(q8["returned_token_ids"]),
                "f16_ids_sha256": f16["returned_token_ids_sha256"],
                "q8_ids_sha256": q8["returned_token_ids_sha256"],
                "f16_eos": f16["eos"]["status"],
                "q8_eos": q8["eos"]["status"],
                "f16_cap_hit": f16["cap"]["hit"],
                "q8_cap_hit": q8["cap"]["hit"],
            })
        for label, row in (("f16", f16), ("q8", q8)):
            outcome = classify(
                row["raw_text"],
                PromptContext.from_mapping(panel[row_id]["context"]),
                panel[row_id]["region_new"],
                row["returned_token_ids"],
            )
            quality = row["quality"]
            expected_noop = row["expected_noop"]
            checks = {
                "protocol_valid": (outcome["protocol_valid"], quality["protocol_valid"]),
                "predicted_noop": (outcome["predicted_noop"], quality["predicted_noop"]),
                "noop_false_positive": (
                    expected_noop and outcome["suggestion"], quality["noop_false_positive"],
                ),
                "strict_noop_correct": (
                    expected_noop and outcome["predicted_noop"], quality["strict_noop_correct"],
                ),
            }
            if not expected_noop:
                checks["exact_edit"] = (outcome["exact_region"], quality["exact_edit"])
            mismatch = {key: value for key, value in checks.items() if value[0] != value[1]}
            result["classifier"].setdefault(label, {"matched_rows": 0, "mismatches": []})
            if mismatch:
                result["classifier"][label]["mismatches"].append({"id": row_id, "fields": mismatch})
            else:
                result["classifier"][label]["matched_rows"] += 1

        f16_quality, q8_quality = f16["quality"], q8["quality"]
        fields = (
            "protocol_valid", "exact_edit", "edit_exact", "predicted_noop",
            "noop_false_positive", "strict_noop_correct",
        )
        delta = {
            key: (f16_quality.get(key), q8_quality.get(key))
            for key in fields if f16_quality.get(key) != q8_quality.get(key)
        }
        if delta or f16.get("failure_class") != q8.get("failure_class") or f16.get("status") != q8.get("status"):
            result["classification_changes"].append({
                "id": row_id,
                "family": f16["family"],
                "expected_noop": f16["expected_noop"],
                "delta": delta,
                "status": (f16.get("status"), q8.get("status")),
                "failure_class": (f16.get("failure_class"), q8.get("failure_class")),
            })

    result["terminal"] = {
        "eos_metadata_exact_rows": sum(cases["f16"][row_id]["eos"] == cases["q8"][row_id]["eos"] for row_id in ids),
        "cap_metadata_exact_rows": sum(cases["f16"][row_id]["cap"] == cases["q8"][row_id]["cap"] for row_id in ids),
        "canonical_eos": {label: sum(row["eos"]["canonical"] for row in cases[label].values()) for label in runs},
        "cap_hits": {label: sum(row["cap"]["hit"] for row in cases[label].values()) for label in runs},
    }
    result["wire_hf"] = {
        "wire_text_match": {
            label: sum(row["protocol"]["wire_hf_text_match"] for row in cases[label].values()) for label in runs
        },
        "decode_errors": {
            label: sum(row["protocol"]["wire_hf_decode_error"] is not None for row in cases[label].values()) for label in runs
        },
        "protocol_metadata_exact_rows": sum(cases["f16"][row_id]["protocol"] == cases["q8"][row_id]["protocol"] for row_id in ids),
    }
    for label, value in runs.items():
        result["families"][label] = {
            family: {
                key: stats[key] for key in (
                    "attempted_rows", "responses", "protocol_valid", "protocol_error_rows",
                    "mechanical_failures", "cap_hit", "exact_region", "edit_exact",
                    "strict_noop_correct", "noop_false_positive", "transport_failures",
                )
            }
            for family, stats in value["families"].items()
        }
        result["runs"][label] = {
            "schema_version": value["schema_version"],
            "status": value["status"],
            "evaluation_complete": value["evaluation_complete"],
            "denominators": value["denominators"],
            "model": value["model"],
            "tokenizer": value["tokenizer"],
            "panel": {key: item for key, item in value["panel"].items() if key != "case_ids"},
            "server": {
                "health": value["server"]["health"],
                "model_path_reported": value["server"]["model_path_reported"],
                "n_ctx": value["server"]["n_ctx"],
                "build_info": value["server"]["props"].get("build_info"),
            },
            "static_preflight": value["static_preflight"],
            "persistence": value["persistence"],
        }
        directory = RUN_ROOT / f"RUN-09-primary500-{label}-dev-a"
        result["artifacts"][label] = {
            name: digest(directory / name)
            for name in (
                "quality.json", "terminal.json", "launch.json", "remote-launch.json",
                "remote-live-device-audit.json", "remote-ready.json", "remote-terminal.json",
                "client.log", "remote-server.log", "ssh-stderr.log",
            )
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
