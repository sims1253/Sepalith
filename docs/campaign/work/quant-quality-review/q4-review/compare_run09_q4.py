#!/usr/bin/env python3
"""Independent CPU-only RUN-09 Q4 paired DEV review.

The three native quality receipts are the measurement inputs.  This script
recomputes protocol/no-op/edit classifications through the pinned
``campaign_eval.classify`` helper, checks stored denominators, compares all
prompt/request identities and output token streams, and records only bounded
hash/length diagnostics for changed outputs.  It never loads a model or
starts a server.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


EXECUTION_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
RUN_ROOT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training")
PANEL_PATH = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl")
EXPECTED_PANEL_SHA256 = "b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21"
EXPECTED_LABELS = ("f16", "q8", "q4")
ARTIFACT_NAMES = (
    "quality.json", "terminal.json", "launch.json", "remote-launch.json",
    "remote-live-device-audit.json", "remote-ready.json", "remote-terminal.json",
    "client.log", "remote-server.log", "ssh-stderr.log",
)

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


def bool_count(rows: list[Mapping[str, Any]], key: str) -> int:
    return sum(bool(row.get(key)) for row in rows)


def q(row: Mapping[str, Any], key: str) -> Any:
    value = row.get("quality")
    return value.get(key) if isinstance(value, Mapping) else None


def classify_rows(label: str, rows: list[Mapping[str, Any]], panel: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    mismatches = []
    recomputed = []
    for row in rows:
        case_id = row["id"]
        case = panel[case_id]
        outcome = classify(
            row["raw_text"],
            PromptContext.from_mapping(case["context"]),
            case["region_new"],
            row["returned_token_ids"],
        )
        expected_noop = bool(row["expected_noop"])
        checks = {
            "protocol_valid": (outcome["protocol_valid"], q(row, "protocol_valid")),
            "predicted_noop": (outcome["predicted_noop"], q(row, "predicted_noop")),
            "noop_false_positive": (
                expected_noop and outcome["suggestion"], q(row, "noop_false_positive"),
            ),
            "strict_noop_correct": (
                expected_noop and outcome["predicted_noop"], q(row, "strict_noop_correct"),
            ),
        }
        if not expected_noop:
            checks["exact_edit"] = (outcome["exact_region"], q(row, "exact_edit"))
        mismatch = {key: value for key, value in checks.items() if value[0] != value[1]}
        if mismatch:
            mismatches.append({"id": case_id, "fields": mismatch})
        recomputed.append({
            "id": case_id,
            "expected_noop": expected_noop,
            "protocol_valid": bool(outcome["protocol_valid"]),
            "predicted_noop": bool(outcome["predicted_noop"]),
            "noop_false_positive": bool(expected_noop and outcome["suggestion"]),
            "strict_noop_correct": bool(expected_noop and outcome["predicted_noop"]),
            "exact_edit": bool((not expected_noop) and outcome["exact_region"]),
        })
    return {"status": "pass" if not mismatches else "fail", "rows": len(rows),
            "mismatches": mismatches, "recomputed": recomputed}


def denominators(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    attempted = len(rows)
    response_rows = [row for row in rows if row.get("response_received")]
    expected_noops = [row for row in rows if row.get("expected_noop")]
    edits = [row for row in rows if not row.get("expected_noop")]
    protocol_valid = sum(bool(q(row, "protocol_valid")) for row in rows)
    exact_edits = sum(bool(q(row, "exact_edit")) for row in edits)
    strict_noop = sum(bool(q(row, "strict_noop_correct")) for row in expected_noops)
    return {
        "attempted_rows": attempted,
        "cap_hit_rows": sum(bool(row.get("cap", {}).get("hit")) for row in rows),
        "edit_cases": len(edits),
        "edit_exact_rows": exact_edits,
        "exact_region_rows": exact_edits + strict_noop,
        "mechanical_failures": sum(row.get("failure_class") == "mechanical" for row in rows),
        "noop_false_positive_rows": sum(bool(q(row, "noop_false_positive")) for row in expected_noops),
        "panel_rows": attempted,
        "partial_responses": sum(bool(row.get("response_received")) and not bool(row.get("response_complete")) for row in rows),
        "protocol_error_rows": attempted - protocol_valid,
        "protocol_rows": protocol_valid,
        "responses": len(response_rows),
        "strict_noop_cases": len(expected_noops),
        "strict_noop_correct_rows": strict_noop,
        "transport_failures": attempted - len(response_rows),
    }


def denominator_review(label: str, value: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    actual = denominators(rows)
    stored = value.get("denominators")
    mismatch = {}
    if isinstance(stored, Mapping):
        mismatch = {
            key: {"recomputed": actual[key], "stored": stored.get(key)}
            for key in actual if actual[key] != stored.get(key)
        }
    else:
        mismatch = {"denominators": {"recomputed": actual, "stored": stored}}
    return {"status": "pass" if not mismatch else "fail", "recomputed": actual,
            "stored": stored, "mismatches": mismatch}


def identity_review(cases: Mapping[str, dict[str, Mapping[str, Any]]], ids: list[str]) -> dict[str, Any]:
    fields = (
        "hf_prompt_ids", "hf_prompt_ids_sha256", "native_prompt_ids_without_bos",
        "native_prompt_ids_sha256", "prompt_sha256", "prompt_token_identity", "request",
    )
    result: dict[str, Any] = {"against_f16": {}, "against_q8": {}, "all_three": {}, "mismatched_ids": {}}
    for field in fields:
        values = {label: [cases[label][case_id].get(field) for case_id in ids] for label in EXPECTED_LABELS}
        all_equal = values["f16"] == values["q8"] == values["q4"]
        result["all_three"][field] = sum(values["f16"][i] == values["q8"][i] == values["q4"][i] for i in range(len(ids)))
        for reference in ("f16", "q8"):
            result[f"against_{reference}"][field] = sum(values["q4"][i] == values[reference][i] for i in range(len(ids)))
        result["mismatched_ids"][field] = [
            case_id for i, case_id in enumerate(ids) if not (values["f16"][i] == values["q8"][i] == values["q4"][i])
        ]
        if all_equal and not result["mismatched_ids"][field]:
            continue
    result["panel_case_ids_exact"] = len(ids)
    return result


def output_review(cases: Mapping[str, dict[str, Mapping[str, Any]]], ids: list[str]) -> dict[str, Any]:
    fields = ("returned_token_ids", "returned_token_ids_sha256", "raw_text", "decoded_body_text")
    result: dict[str, Any] = {"exact_rows": {"against_f16": {}, "against_q8": {}}, "changed": {"against_f16": [], "against_q8": []}}
    for field in fields:
        for reference in ("f16", "q8"):
            result["exact_rows"][f"against_{reference}"][field] = sum(
                cases["q4"][case_id].get(field) == cases[reference][case_id].get(field) for case_id in ids
            )
    for reference in ("f16", "q8"):
        for case_id in ids:
            q4, ref = cases["q4"][case_id], cases[reference][case_id]
            if q4.get("returned_token_ids") == ref.get("returned_token_ids") and q4.get("raw_text") == ref.get("raw_text"):
                continue
            result["changed"][f"against_{reference}"].append({
                "id": case_id,
                "family": q4.get("family"),
                "expected_noop": q4.get("expected_noop"),
                "q4_tokens": len(q4.get("returned_token_ids", [])),
                "reference_tokens": len(ref.get("returned_token_ids", [])),
                "q4_ids_sha256": q4.get("returned_token_ids_sha256"),
                "reference_ids_sha256": ref.get("returned_token_ids_sha256"),
                "q4_eos": q4.get("eos", {}).get("status"),
                "reference_eos": ref.get("eos", {}).get("status"),
                "q4_cap_hit": q4.get("cap", {}).get("hit"),
                "reference_cap_hit": ref.get("cap", {}).get("hit"),
            })
    result["changed_counts"] = {key: len(value) for key, value in result["changed"].items()}
    return result


def classification_changes(cases: Mapping[str, dict[str, Mapping[str, Any]]], ids: list[str]) -> dict[str, Any]:
    fields = ("protocol_valid", "exact_edit", "edit_exact", "predicted_noop", "noop_false_positive", "strict_noop_correct")
    result = {}
    for reference in ("f16", "q8"):
        changes = []
        for case_id in ids:
            q4, ref = cases["q4"][case_id], cases[reference][case_id]
            delta = {field: (q(q4, field), q(ref, field)) for field in fields if q(q4, field) != q(ref, field)}
            status = (q4.get("status"), ref.get("status"))
            failure = (q4.get("failure_class"), ref.get("failure_class"))
            if delta or status[0] != status[1] or failure[0] != failure[1]:
                changes.append({"id": case_id, "family": q4.get("family"), "expected_noop": q4.get("expected_noop"),
                                "delta": delta, "status": status, "failure_class": failure})
        result[f"against_{reference}"] = {"count": len(changes), "rows": changes}
    return result


def terminal_review(cases: Mapping[str, dict[str, Mapping[str, Any]]], ids: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"metadata_exact_rows": {"against_f16": {}, "against_q8": {}}, "counts": {}, "cap_outcome_separation": {}}
    for field in ("eos", "cap"):
        for reference in ("f16", "q8"):
            result["metadata_exact_rows"][f"against_{reference}"][field] = sum(
                cases["q4"][case_id].get(field) == cases[reference][case_id].get(field) for case_id in ids
            )
    for label in EXPECTED_LABELS:
        rows = list(cases[label].values())
        result["counts"][label] = {
            "canonical_eos": sum(bool(row.get("eos", {}).get("canonical")) for row in rows),
            "cap_hits": sum(bool(row.get("cap", {}).get("hit")) for row in rows),
            "non_eos_terminal": sum(row.get("eos", {}).get("status") == "non_eos_terminal" for row in rows),
        }
        result["cap_outcome_separation"][label] = {
            "cap_hit_rows": sum(bool(row.get("cap", {}).get("hit")) for row in rows),
            "cap_hit_expected_noop": sum(bool(row.get("cap", {}).get("hit")) and bool(row.get("expected_noop")) for row in rows),
            "cap_hit_edit": sum(bool(row.get("cap", {}).get("hit")) and not bool(row.get("expected_noop")) for row in rows),
            "cap_hit_noop_false_positive": sum(bool(row.get("cap", {}).get("hit")) and bool(q(row, "noop_false_positive")) for row in rows),
            "cap_hit_strict_noop_correct": sum(bool(row.get("cap", {}).get("hit")) and bool(q(row, "strict_noop_correct")) for row in rows),
            "noop_false_positive_rows": sum(bool(q(row, "noop_false_positive")) for row in rows),
            "strict_noop_correct_rows": sum(bool(q(row, "strict_noop_correct")) for row in rows),
        }
    return result


def family_review(values: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("protocol_valid", "exact_edit", "predicted_noop", "noop_false_positive", "strict_noop_correct")
    result = {}
    for label, value in values.items():
        by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in value["cases"]:
            by_family[str(row.get("family"))].append(row)
        result[label] = {}
        for family, rows in sorted(by_family.items()):
            result[label][family] = {
                "rows": len(rows),
                "cap_hit": sum(bool(row.get("cap", {}).get("hit")) for row in rows),
                "mechanical_failures": sum(row.get("failure_class") == "mechanical" for row in rows),
                **{field: sum(bool(q(row, field)) for row in rows) for field in fields},
            }
    return result


def runtime_review(values: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for label, value in values.items():
        root = RUN_ROOT / f"RUN-09-primary500-{label}-dev-a"
        terminal = json.loads((root / "terminal.json").read_text())
        remote_launch = json.loads((root / "remote-launch.json").read_text())
        remote_terminal = json.loads((root / "remote-terminal.json").read_text())
        result[label] = {
            "quality_status": value.get("status"),
            "evaluation_complete": value.get("evaluation_complete"),
            "model": {key: value.get("model", {}).get(key) for key in ("label", "provenance", "artifact_selection")},
            "panel": {key: value.get("panel", {}).get(key) for key in ("rows", "sha256", "case_ids_sha256", "path")},
            "tokenizer": {key: value.get("tokenizer", {}).get(key) for key in ("revision", "vocab_sha256", "path")},
            "server": {key: value.get("server", {}).get(key) for key in ("n_ctx", "model_path_reported")},
            "remote": {key: remote_launch.get(key) for key in ("candidate", "model_sha256", "backend", "threads", "batch_threads", "openblas_threads", "build_receipt_sha256")},
            "terminal": {key: terminal.get(key) for key in ("client_exit_code", "ssh_exit_code", "seconds", "error")},
            "remote_terminal": {key: remote_terminal.get(key) for key in ("server_exit_code", "seconds", "peak_server_rss_kib")},
            "artifact_hashes": {name: digest(root / name) for name in ARTIFACT_NAMES},
        }
    return result


def main() -> int:
    panel: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with PANEL_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            panel[row["id"]] = row
    values: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for label in EXPECTED_LABELS:
        path = RUN_ROOT / f"RUN-09-primary500-{label}-dev-a" / "quality.json"
        values[label] = json.loads(path.read_text(encoding="utf-8"))
    cases = {label: OrderedDict((row["id"], row) for row in value["cases"]) for label, value in values.items()}
    ids = list(cases["q4"])
    failures = []
    if len(panel) != 75 or digest(PANEL_PATH)["sha256"] != EXPECTED_PANEL_SHA256:
        failures.append("pinned 75-case DEV panel row count or SHA256 mismatch")
    if any(list(cases[label]) != ids for label in EXPECTED_LABELS) or set(ids) != set(panel):
        failures.append("case IDs/order differ between Q4, F16, Q8, or the pinned panel")
    if any(value.get("status") != "complete" or value.get("evaluation_complete") is not True for value in values.values()):
        failures.append("one or more native quality receipts is incomplete")

    recomputed = {label: classify_rows(label, list(cases[label].values()), panel) for label in EXPECTED_LABELS}
    denominator = {label: denominator_review(label, values[label], list(cases[label].values())) for label in EXPECTED_LABELS}
    identity = identity_review(cases, ids)
    outputs = output_review(cases, ids)
    changes = classification_changes(cases, ids)
    terminal = terminal_review(cases, ids)
    families = family_review(values)
    runtime = runtime_review(values)
    if any(item["status"] != "pass" for item in recomputed.values()):
        failures.append("stored per-case classification differs from pinned classifier")
    if any(item["status"] != "pass" for item in denominator.values()):
        failures.append("stored denominator differs from recomputed denominator")
    if any(identity[f"against_{ref}"][field] != 75 for ref in ("f16", "q8") for field in identity[f"against_{ref}"]):
        failures.append("Q4 prompt/request identity differs from F16 or Q8")
    if any(runtime[label]["terminal"]["client_exit_code"] != 0 or runtime[label]["terminal"]["ssh_exit_code"] != 0 or runtime[label]["remote_terminal"]["server_exit_code"] != 0 for label in EXPECTED_LABELS):
        failures.append("one or more paired DEV process terminals has a nonzero exit")

    receipt = {
        "schema_version": "sepalith.run09.q4-native-dev-independent-review.v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "reviewer": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": digest(Path(__file__).resolve())["sha256"],
            "existing_f16_q8_verifier_path": str(EXECUTION_ROOT.parent / "t3code-a8153bbb" / "docs" / "campaign" / "work" / "quant-quality-review" / "compare_run09_f16_q8.py"),
            "method": "campaign_eval.classify plus stored denominator/identity/output recomputation; standard-library artifact hashing; no model or server load",
        },
        "recommendation": {
            "mechanically_admit_q4_for_root_review": not failures,
            "quality_promotion": "not_claimed",
            "basis": "75/75 native DEV responses, exact shared prompt/request identity, recomputed classifications and denominators, and zero transport/process failures across F16/Q8/Q4.",
        },
        "panel": {"path": str(PANEL_PATH), "rows": len(panel), "sha256": digest(PANEL_PATH)["sha256"],
                  "case_ids_sha256": values["q4"].get("panel", {}).get("case_ids_sha256")},
        "failures": failures,
        "runs": {label: {
            "quality_path": str(RUN_ROOT / f"RUN-09-primary500-{label}-dev-a" / "quality.json"),
            "denominators": denominator[label],
            "classifier_recompute": {key: value for key, value in recomputed[label].items() if key != "recomputed"},
        } for label in EXPECTED_LABELS},
        "prompt_request_identity": identity,
        "outputs": outputs,
        "classification_changes": changes,
        "terminal_and_caps": terminal,
        "families": families,
        "runtime_and_artifacts": runtime,
    }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
