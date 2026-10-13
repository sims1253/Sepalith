#!/usr/bin/env python3
"""Bounded provider regression checks; reads only the two approved row fixtures."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]
PLAN = PACKET.parents[4]
LEAD = PLAN / "docs/campaign/work/lead"
NOOP = LEAD / "r2-noop4100-provider-preparation-v1"
FROZEN = LEAD / "r2-semantic10948-provider-materialization-v2"
PYTHON = "/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python"
TOKENIZER = "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json"


def load_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.loads(next(stream))


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def env() -> dict[str, str]:
    value = dict(os.environ)
    value.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return value


def run_renderer(renderer: Path, source_helper: Path, input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    command = [
        "taskset",
        "-c",
        "8",
        "nice",
        "-n",
        "10",
        "ionice",
        "-c",
        "3",
        "node",
        "--no-warnings=ExperimentalWarning",
        "--experimental-strip-types",
        str(renderer),
        str(input_path),
        str(output_path),
        PYTHON,
        str(FROZEN / "tokenize_bridge.py"),
        TOKENIZER,
        str(source_helper),
        "16384",
        "2048",
    ]
    return subprocess.run(command, capture_output=True, text=True, timeout=120, env=env(), check=False)


def invoke_source_helper(helper: Path, text: str, cursor_line: int) -> tuple[subprocess.CompletedProcess[str], dict]:
    with tempfile.TemporaryDirectory(dir="/mnt/e/sepalith/campaign-20260915/tmp") as raw_tmp:
        tmp = Path(raw_tmp)
        request = tmp / "request.json"
        output = tmp / "output.json"
        digest = sha_text(text)
        request.write_text(
            json.dumps(
                {
                    "source_base64": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                    "source_sha256": digest,
                    "max_bytes": 16 << 20,
                    "cursor_line": cursor_line,
                    "explicit_import_symbols": [],
                }
            )
            + "\n"
        )
        result = subprocess.run(["Rscript", str(helper), str(request), str(output)], capture_output=True, text=True, timeout=30, check=False)
        parsed = json.loads(output.read_text()) if output.exists() else {}
        return result, parsed


def main() -> None:
    e677 = load_row(PACKET / "inputs/e677-preedit.jsonl")
    valid = load_row(PACKET / "inputs/valid-control.jsonl")
    assert e677["row_id"] == "e677ee6a8da38436f4bdb6b6"
    assert valid["row_id"] == "43b24d15b32aae89fdf72245"
    assert sha_text(e677["preedit_text"]) == e677["preedit_sha256"]
    assert sha_text(valid["preedit_text"]) == valid["preedit_sha256"]

    # The helper itself must carry only explicit absence evidence for the exact
    # malformed snapshot: source hash plus empty dependency/evidence fields.
    source_result, source_record = invoke_source_helper(PACKET / "source/source_imports.R", e677["preedit_text"], e677["cursor"]["line"])
    assert source_result.returncode == 0, source_result.stderr
    assert source_record["status"] == "source_import_evidence_unavailable"
    assert source_record["source_sha256"] == e677["preedit_sha256"]
    assert source_record["import_dependencies"] == []
    assert source_record["unresolved_nonimport"] == []
    assert source_record["target_name"] == {}
    assert source_record["target_span"] == {}
    print("PASS exact e677 structured source-evidence absence")

    # Exact malformed TRAIN/editor snapshot: parse failure is explicit evidence
    # absence, so the policy can use existing context fallback.
    with tempfile.TemporaryDirectory(dir="/mnt/e/sepalith/campaign-20260915/tmp") as raw_tmp:
        tmp = Path(raw_tmp)
        e_input = tmp / "e677.jsonl"
        e_output = tmp / "e677.out.jsonl"
        e_input.write_text(json.dumps(e677, sort_keys=True, separators=(",", ":")) + "\n")
        result = run_renderer(PACKET / "render_shard.ts", PACKET / "source/namespace_evidence.R", e_input, e_output)
        assert result.returncode == 0, result.stderr
        rendered = json.loads(e_output.read_text())
        assert rendered["status"] == "supported"
        assert rendered["mode"] == "full_document"
        assert rendered["source_inventory_status"] == "source_import_evidence_unavailable"
        assert rendered["source_import_evidence"] == "unavailable"
        assert rendered["observed_dependencies"] == []
        assert rendered["selection_target_or_gold_used"] is False
    print("PASS exact e677 parse-unavailable fallback")

    # Missing Rscript stays a provider-infrastructure failure.
    command = [
        "node",
        "--no-warnings=ExperimentalWarning",
        "--experimental-strip-types",
        str(PACKET / "tests/missing_rscript.ts"),
        str(PACKET / "inputs/e677-preedit.jsonl"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env(), check=False)
    assert result.returncode == 0, result.stderr
    print("PASS missing Rscript negative")

    # Valid control must remain byte-semantic equivalent to the existing Noop
    # wrapper; this compares the actual wrapper whose source-import metadata is
    # being repaired, rather than the older Semantic wrapper.
    with tempfile.TemporaryDirectory(dir="/mnt/e/sepalith/campaign-20260915/tmp") as raw_tmp:
        tmp = Path(raw_tmp)
        valid_input = tmp / "valid.jsonl"
        patched_output = tmp / "patched.jsonl"
        frozen_output = tmp / "frozen.jsonl"
        valid_input.write_text(json.dumps(valid, sort_keys=True, separators=(",", ":")) + "\n")
        patched = run_renderer(PACKET / "render_shard.ts", PACKET / "source/namespace_evidence.R", valid_input, patched_output)
        frozen = run_renderer(NOOP / "render_shard.ts", FROZEN / "source/namespace_evidence.R", valid_input, frozen_output)
        assert patched.returncode == 0, patched.stderr
        assert frozen.returncode == 0, frozen.stderr
        rendered = json.loads(patched_output.read_text())
        baseline = json.loads(frozen_output.read_text())
        assert rendered["source_inventory_status"] == "source_import_inventory_complete"
        assert rendered["selection_target_or_gold_used"] is False
        assert rendered == baseline
    print("PASS valid control output parity")

    # Cursor geometry/no-following-function status is unchanged byte-for-byte
    # between the copied helper and the frozen Noop helper.
    source = "a <- function() {\n  x <- 1\n}\n"
    patched_result, patched_source = invoke_source_helper(PACKET / "source/source_imports.R", source, 2)
    frozen_result, frozen_source = invoke_source_helper(NOOP / "source/source_imports.R", source, 2)
    assert patched_result.returncode == frozen_result.returncode == 0
    assert patched_source == frozen_source
    assert patched_source["status"] == "source_import_inventory_no_following_function"
    assert patched_source["import_dependencies"] == []
    # jsonlite serializes NULL as {} under the frozen helper's existing schema.
    assert patched_source["target_name"] == {}
    assert patched_source["target_span"] == {}
    print("PASS cursor geometry/no-following parity")


if __name__ == "__main__":
    main()
