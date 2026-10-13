#!/usr/bin/env python3
"""Independent, read-only audit of a completed RUN-09 theta0-Q8 CUDA run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


EXEC_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
PLAN_ROOT = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
DEFAULT_RUN = Path(
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
    "RUN-09-theta0-q8-cuda-dev-a"
)
PANEL = PLAN_ROOT / "docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
PROTOCOL = EXEC_ROOT / "packages/sepalith/src/sepalith/campaign_protocol.py"
TOKENIZER_DIR = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")

EXPECTED = {
    "model_sha256": "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559",
    "server_sha256": "e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee",
    "panel_sha256": "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035",
    "protocol_sha256": "5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156",
    "tokenizer_json_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config_sha256": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
}
PANEL_ROWS = 75
EDIT_ROWS = 43
NOOP_ROWS = 32
CAP = 512
CONTEXT = 4096
EOS = 1
NATIVE_EOG = {1, 130073}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_protocol():
    if sha256_file(PROTOCOL) != EXPECTED["protocol_sha256"]:
        raise RuntimeError("pinned protocol hash mismatch")
    package_src = EXEC_ROOT / "packages" / "sepalith" / "src"
    sys.path.insert(0, str(package_src))
    import sepalith.campaign_protocol as protocol
    return protocol


def load_panel() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if sha256_file(PANEL) != EXPECTED["panel_sha256"]:
        raise RuntimeError("corrected DEV panel hash mismatch")
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with PANEL.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise RuntimeError(f"blank panel line {line_number}")
            row = json.loads(line)
            row_id = row.get("id")
            if row.get("split") != "dev" or not isinstance(row_id, str):
                raise RuntimeError(f"invalid DEV row at line {line_number}")
            if row_id in by_id:
                raise RuntimeError(f"duplicate DEV row {row_id}")
            rows.append(row)
            by_id[row_id] = row
    if len(rows) != PANEL_ROWS:
        raise RuntimeError(f"panel has {len(rows)} rows")
    return rows, by_id


def load_tokenizer() -> tuple[Any, dict[str, Any]]:
    tokenizer_json = TOKENIZER_DIR / "tokenizer.json"
    tokenizer_config = TOKENIZER_DIR / "tokenizer_config.json"
    if sha256_file(tokenizer_json) != EXPECTED["tokenizer_json_sha256"]:
        raise RuntimeError("pinned tokenizer.json hash mismatch")
    if sha256_file(tokenizer_config) != EXPECTED["tokenizer_config_sha256"]:
        raise RuntimeError("pinned tokenizer_config.json hash mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR), local_files_only=True, use_fast=True,
        trust_remote_code=False,
    )
    identity = {
        "vocab_size": len(tokenizer),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    expected_identity = {"vocab_size": 130560, "bos_token_id": 0,
                         "eos_token_id": 1, "pad_token_id": 1}
    if identity != expected_identity:
        raise RuntimeError(f"tokenizer identity mismatch: {identity}")
    return tokenizer, {"path": str(TOKENIZER_DIR), "files": {
        "tokenizer.json": EXPECTED["tokenizer_json_sha256"],
        "tokenizer_config.json": EXPECTED["tokenizer_config_sha256"],
    }, "identity": identity}


def independent_case(protocol: Any, tokenizer: Any, case: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    token_ids = case.get("returned_token_ids")
    if not isinstance(token_ids, list) or any(type(x) is not int for x in token_ids):
        raise RuntimeError(f"{case.get('id')}: returned IDs are not integer list")
    context = protocol.PromptContext.from_mapping(row["context"])
    raw_text = case.get("raw_text")
    if not isinstance(raw_text, str):
        raw_text = ""
    parsed = protocol.parse_output(raw_text, context)
    server_timing = case.get("server_timing")
    if not isinstance(server_timing, Mapping):
        server_timing = {}
    saved_protocol = case.get("protocol")
    if not isinstance(saved_protocol, Mapping):
        saved_protocol = {}
    canonical_tokens = bool(token_ids) and protocol.valid_generation_tokens(token_ids)
    terminal = token_ids[-1] if token_ids else None
    stop_type = server_timing.get("stop_type")
    final_sse_tokens = saved_protocol.get("final_sse_tokens")
    malformed = saved_protocol.get("malformed_sse_frames")
    client_wire_match = saved_protocol.get("wire_hf_text_match")
    body_ids = token_ids[:-1] if terminal == EOS else token_ids
    try:
        decoded_body = tokenizer.decode(
            body_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        wire_match = decoded_body == raw_text
        decoded_body_sha256 = sha256_text(decoded_body)
    except Exception:
        wire_match = False
        decoded_body_sha256 = None
    # The byte-level native/HF decode, core protocol, parser, cap, and
    # exact/no-op decisions are recomputed from the saved IDs and raw wire
    # body, pinned tokenizer/protocol, and panel context.
    protocol_valid = bool(
        canonical_tokens
        and wire_match
        and len(token_ids) <= CAP
        and saved_protocol.get("saw_stop") is True
        and final_sse_tokens == 0
        and not malformed
        and stop_type == "eos"
        and parsed.status == "accepted"
    )
    cap_hit = len(token_ids) > CAP or (
        terminal != EOS
        and (
            stop_type in {"limit", "length"}
            or server_timing.get("truncated") is True
            or (isinstance(server_timing.get("tokens_predicted"), int)
                and server_timing["tokens_predicted"] >= CAP)
        )
    )
    if parsed.status == "accepted" and parsed.operation == "no_op":
        actual_region = list(context.region_old)
    elif parsed.status == "accepted":
        actual_region = list(parsed.body)
    else:
        actual_region = []
    expected_region = list(row.get("region_new", []))
    exact = bool(protocol_valid and actual_region == expected_region)
    predicted_noop = bool(protocol_valid and parsed.operation == "no_op")
    return {
        "id": case.get("id"),
        "family": row.get("family"),
        "operation": row.get("operation"),
        "returned_tokens": len(token_ids),
        "terminal_id": terminal,
        "stop_type": stop_type,
        "canonical_generation_tokens": canonical_tokens,
        "cap_hit": bool(cap_hit),
        "parser_status": parsed.status,
        "parser_operation": parsed.operation,
        "parser_reason": parsed.reason,
        "protocol_valid_recomputed": protocol_valid,
        "exact_recomputed": exact,
        "predicted_noop_recomputed": predicted_noop,
        "wire_hf_text_match_saved": client_wire_match,
        "wire_hf_text_match_recomputed": wire_match,
        "decoded_body_sha256": decoded_body_sha256,
        "wire_hf_text_match_client": client_wire_match,
        "output_chars": len(raw_text),
        "output_sha256": sha256_text(raw_text),
        "parsed_body_lines": len(parsed.body),
        "saved_protocol_valid": bool(case.get("quality", {}).get("protocol_valid")),
        "saved_exact": bool(case.get("quality", {}).get("exact_edit")),
        "saved_noop_correct": bool(case.get("quality", {}).get("strict_noop_correct")),
    }


def compact_props(launch: Mapping[str, Any]) -> dict[str, Any]:
    props = launch.get("server_preflight", {}).get("props", {})
    if not isinstance(props, Mapping):
        props = {}
    generation = props.get("default_generation_settings", {})
    return {
        "build_info": props.get("build_info"),
        "model_path": props.get("model_path"),
        "n_ctx": generation.get("n_ctx") if isinstance(generation, Mapping) else None,
        "bos_token": props.get("bos_token"),
    }


def run_review(run: Path) -> dict[str, Any]:
    launch_path, terminal_path, quality_path, server_log = [run / name for name in
                                                             ("launch.json", "terminal.json", "quality.json", "server.log")]
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    rows, by_id = load_panel()
    protocol = load_protocol()
    tokenizer, tokenizer_audit = load_tokenizer()
    cases = quality.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("quality cases is not an array")
    ordered_ids = [row["id"] for row in rows]
    case_ids = [case.get("id") for case in cases]
    if case_ids != ordered_ids:
        raise RuntimeError("quality case order/ID sequence differs from corrected DEV panel")
    independent = [independent_case(protocol, tokenizer, case, by_id[case["id"]]) for case in cases]
    counts = Counter()
    family_counts: dict[str, Counter[str]] = {}
    for result in independent:
        family = str(result["family"])
        family_counts.setdefault(family, Counter())
        counts["protocol_rows"] += int(result["protocol_valid_recomputed"])
        counts["protocol_error_rows"] += int(not result["protocol_valid_recomputed"])
        counts["cap_hit_rows"] += int(result["cap_hit"])
        if result["operation"] == "no_op":
            counts["strict_noop_correct_rows"] += int(result["protocol_valid_recomputed"] and result["predicted_noop_recomputed"])
            counts["noop_false_positive_rows"] += int(result["protocol_valid_recomputed"] and not result["predicted_noop_recomputed"])
        else:
            counts["edit_exact_rows"] += int(result["exact_recomputed"])
        counts["exact_region_rows"] += int(result["exact_recomputed"])
        family_counts[family]["rows"] += 1
        family_counts[family]["protocol_rows"] += int(result["protocol_valid_recomputed"])
        family_counts[family]["cap_hit_rows"] += int(result["cap_hit"])
        family_counts[family]["exact_rows"] += int(result["exact_recomputed"])
    saved_denoms = quality.get("denominators", {})
    fields_to_compare = ["protocol_rows", "protocol_error_rows", "cap_hit_rows",
                         "edit_exact_rows", "strict_noop_correct_rows",
                         "noop_false_positive_rows", "exact_region_rows"]
    denominator_comparison = {
        field: {"independent": counts[field], "saved": saved_denoms.get(field),
                "match": counts[field] == saved_denoms.get(field)}
        for field in fields_to_compare
    }
    finish = [result for result in independent if result["family"] == "finish_block"]
    inputs = launch.get("inputs", {})
    input_checks = {
        "model": inputs.get("model", {}).get("sha256") == EXPECTED["model_sha256"],
        "server": inputs.get("server", {}).get("sha256") == EXPECTED["server_sha256"],
        "panel": inputs.get("panel", {}).get("sha256") == EXPECTED["panel_sha256"],
        "panel_rows": quality.get("panel", {}).get("rows") == PANEL_ROWS,
        "panel_case_ids_sha256": quality.get("panel", {}).get("case_ids_sha256") == canonical_sha(ordered_ids),
        "protocol_source": quality.get("static_preflight", {}).get("protocol", {}).get("sha256") == EXPECTED["protocol_sha256"],
        "server_n_ctx": compact_props(launch).get("n_ctx") == CONTEXT,
        "server_build": compact_props(launch).get("build_info") == "b10453-3cb7ffb1a",
        "client_returncode": terminal.get("client_returncode") == 0,
        "evaluation_complete": quality.get("evaluation_complete") is True and quality.get("status") == "complete",
        "all_rows_attempted": saved_denoms.get("attempted_rows") == PANEL_ROWS,
        "no_transport_or_partial": saved_denoms.get("transport_failures") == 0 and saved_denoms.get("partial_responses") == 0,
        "quality_sha_matches_terminal": sha256_file(quality_path) == terminal.get("client_output_sha256"),
    }
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    offload_match = re.search(r"offloaded (\d+/\d+) layers", log_text)
    cuda_match = re.search(r"using device CUDA0 \(([^)]+)\)", log_text)
    model_match = re.search(r"loading model '([^']+)'", log_text)
    input_checks.update({
        "server_log_hash_present": bool(sha256_file(server_log)),
        "server_log_offload_43_43": bool(offload_match and offload_match.group(1) == "43/43"),
        "server_log_cuda_device": cuda_match.group(1) if cuda_match else None,
        "server_log_model_path": model_match.group(1) if model_match else None,
    })
    saved_quality = {
        "edit_exact_rows": saved_denoms.get("edit_exact_rows"),
        "strict_noop_correct_rows": saved_denoms.get("strict_noop_correct_rows"),
        "noop_false_positive_rows": saved_denoms.get("noop_false_positive_rows"),
        "protocol_rows": saved_denoms.get("protocol_rows"),
        "cap_hit_rows": saved_denoms.get("cap_hit_rows"),
    }
    return {
        "schema_version": "sepalith.run09.theta0-q8-cuda-dev-independent-review.v1",
        "task": "RUN-09/SFT-08",
        "status": "independent_review_complete",
        "run": {
            "path": str(run),
            "launch_sha256": sha256_file(launch_path),
            "terminal_sha256": sha256_file(terminal_path),
            "quality_sha256": sha256_file(quality_path),
            "server_log_sha256": sha256_file(server_log),
        },
        "input_checks": input_checks,
        "runtime": {
            "props": compact_props(launch),
            "env_contract": launch.get("env_contract"),
            "server_argv": launch.get("server_argv"),
            "server_cleanup_returncode": terminal.get("server", {}).get("returncode"),
            "binary_version_field": launch.get("binary_version"),
            "binary_version_field_note": "empty in launch.json; server props/log independently identify b10453-3cb7ffb1a",
        },
        "panel": {
            "path": str(PANEL), "sha256": EXPECTED["panel_sha256"],
            "rows": PANEL_ROWS, "edit_rows": EDIT_ROWS, "strict_noop_rows": NOOP_ROWS,
            "ordered_ids_sha256": canonical_sha(ordered_ids),
            "case_order_matches_quality": case_ids == ordered_ids,
        },
        "tokenizer": tokenizer_audit,
        "independent_denominators": dict(sorted(counts.items())),
        "saved_denominators": saved_quality,
        "denominator_comparison": denominator_comparison,
        "family_counts": {name: dict(sorted(value.items())) for name, value in sorted(family_counts.items())},
        "finish_cases": finish,
        "baseline_comparison_only": {
            "accepted_sft1000_baseline": {
                "edit_exact_rows": 26, "edit_rows": 43,
                "strict_noop_correct_rows": 25, "strict_noop_rows": 32,
                "noop_false_positive_rows": 5, "protocol_rows": 68, "cap_hit_rows": 7,
                "source": "parent-supplied accepted HF baseline scalars"
            },
            "cuda_native": saved_quality,
            "delta_cuda_minus_baseline": {
                "edit_exact_rows": counts["edit_exact_rows"] - 26,
                "strict_noop_correct_rows": counts["strict_noop_correct_rows"] - 25,
                "noop_false_positive_rows": counts["noop_false_positive_rows"] - 5,
                "protocol_rows": counts["protocol_rows"] - 68,
                "cap_hit_rows": counts["cap_hit_rows"] - 7,
            },
            "interpretation": "descriptive backend comparison only; CUDA native and HF baseline are not a promotion or equivalence claim"
        },
        "limits": [
            "Wire/HF text parity is independently checked only through the saved native client flag; this review does not rerun the tokenizer/client.",
            "No model bytes, generation, server, CUDA, SSH, network, final, or campaign state were accessed by this review.",
            "The six finish rows are reported mechanically; no R code is executed and no semantic quality intent is inferred.",
            "No NLL is available from this native endpoint."
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_review(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(args.out),
                      "denominators": result["independent_denominators"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
