#!/usr/bin/env python3
"""Build the RUN-02 shared ID/prompt digest from pinned existing fixtures.

The four rows are the accepted native-probe TRAIN fixture.  Contexts come
from the RL-02 sidecar, and both renderers are invoked directly: Python's
``sepalith.campaign_protocol.render_prompt`` and the current extension
``campaign_protocol.ts`` renderer.  No prompt or context is reconstructed
from target text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
EXT = EXEC / "extensions/vscode-sepalith"
OUT = PLAN / "docs/campaign/work/run02-closure-v1"
FIXTURE = PLAN / "docs/campaign/work/serving-readiness/native-probe-train-fixture.jsonl"
FIXTURE_MANIFEST = PLAN / "docs/campaign/work/serving-readiness/native-probe-train-fixture.manifest.json"
SIDECAR = Path("/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/context-sidecar.jsonl")
TS_DRIVER = OUT / "render_shared_fixture.ts"
CONTEXT_INPUT = OUT / "shared-context-input.json"
TS_OUTPUT = OUT / "typescript-render-results.json"
MANIFEST_OUTPUT = OUT / "shared-id-prompt-manifest.json"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_pin(path: Path, *, role: str) -> dict[str, object]:
    return {"path": str(path), "role": role, "bytes": path.stat().st_size, "sha256": digest_file(path)}


def load_fixture_rows() -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    selected = manifest["selected_rows"]
    if [row["id"] for row in rows] != [row["row_id"] for row in selected]:
        raise AssertionError("native fixture order differs from its accepted manifest")
    return rows, {row["row_id"]: row for row in selected}


def load_selected_sidecar(ids: set[str]) -> dict[str, dict]:
    if not SIDECAR.is_file():
        raise FileNotFoundError(SIDECAR)
    found: dict[str, dict] = {}
    with SIDECAR.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            row_id = value.get("row_id")
            if row_id in ids:
                found[row_id] = {"line": line_number, "value": value}
                if len(found) == len(ids):
                    break
    if set(found) != ids:
        raise AssertionError(f"sidecar missing selected IDs: {sorted(ids - set(found))}")
    return found


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, manifest_rows = load_fixture_rows()
    ids = {row["id"] for row in rows}
    sidecar = load_selected_sidecar(ids)

    # This is the actual Python training renderer from the execution tree.
    sys.path.insert(0, str(EXEC / "packages" / "sepalith" / "src"))
    from sepalith.campaign_protocol import (  # type: ignore[import-not-found]
        BOS_ID,
        EOS_ID,
        RENDERER_ID,
        TOKENIZATION_POLICY,
        PromptContext,
        render_prompt,
    )

    input_cases: list[dict] = []
    python_results: dict[str, dict] = {}
    boundary_rows: list[dict] = []
    for row in rows:
        row_id = row["id"]
        sidecar_value = sidecar[row_id]["value"]
        context = PromptContext.from_mapping(sidecar_value["context"])
        prompt = render_prompt(context)
        expected = manifest_rows[row_id]
        expected_sha = expected["prompt_sha256"]
        if digest_text(prompt) != expected_sha:
            raise AssertionError(f"Python prompt hash mismatch for {row_id}")
        if prompt != row["prompt_text"]:
            raise AssertionError(f"Python prompt bytes differ from stored row for {row_id}")
        if sidecar_value.get("prompt_sha256") != expected_sha:
            raise AssertionError(f"sidecar prompt hash mismatch for {row_id}")
        if sidecar_value.get("context_has_target_or_reward_keys") is not False:
            raise AssertionError(f"context label guard missing for {row_id}")
        input_cases.append({"id": row_id, "context": sidecar_value["context"], "expected_prompt_sha256": expected_sha})
        prompt_ids = row["input_ids"][: row["target_start"]]
        target_ids = row["input_ids"][row["target_start"] :]
        terminal_ids = row["target_terminal_tokens"]
        boundary = {
            "id": row_id,
            "split": row["split"],
            "prompt_token_count": row["prompt_token_count"],
            "target_start": row["target_start"],
            "target_token_count": row["target_token_count"],
            "manual_bos_id": row["bos_token_id"],
            "canonical_eos_id": row["eos_token_id"],
            "prompt_prefix_length_including_bos": len(prompt_ids),
            "target_suffix_length": len(target_ids),
            "one_bos_in_prompt_prefix": prompt_ids.count(BOS_ID) == 1 and prompt_ids[0] == BOS_ID,
            "one_eos_at_sequence_end": row["input_ids"].count(EOS_ID) == 1 and row["input_ids"][-1] == EOS_ID,
            "target_start_is_prompt_count_plus_bos": row["target_start"] == row["prompt_token_count"] + 1,
            # The stored terminal segment excludes the separately appended
            # protocol EOS.  The complete target is body + terminal segment +
            # one EOS, as required by the PRM03 row contract.
            "terminal_suffix_matches_stored_target": target_ids[:-1][-len(terminal_ids) :] == terminal_ids,
            "terminal_suffix_ends_canonical_eos": target_ids[-1] == EOS_ID,
        }
        boundary_checks = (
            "one_bos_in_prompt_prefix",
            "one_eos_at_sequence_end",
            "target_start_is_prompt_count_plus_bos",
            "terminal_suffix_matches_stored_target",
            "terminal_suffix_ends_canonical_eos",
        )
        if not all(boundary[key] for key in boundary_checks):
            raise AssertionError(f"token boundary check failed for {row_id}: {boundary}")
        boundary_rows.append(boundary)
        python_results[row_id] = {
            "id": row_id,
            "ordinal": expected["ordinal"],
            "family": row["family"],
            "package_id": row["package_id"],
            "split": row["split"],
            "prompt_sha256": digest_text(prompt),
            "prompt_bytes": len(prompt.encode("utf-8")),
            "prompt_token_count": row["prompt_token_count"],
            "row_sha256": expected["row_sha256"],
            "target_operation": row["target_operation"],
            "sidecar_line": sidecar[row_id]["line"],
        }

    CONTEXT_INPUT.write_text(json.dumps({"cases": input_cases}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    command = [
        "node",
        "--experimental-strip-types",
        "--no-warnings",
        str(TS_DRIVER),
        str(CONTEXT_INPUT),
        str(TS_OUTPUT),
    ]
    completed = subprocess.run(command, cwd=EXT, text=True, capture_output=True, timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(f"TypeScript renderer failed:\n{completed.stdout}\n{completed.stderr}")
    ts = json.loads(TS_OUTPUT.read_text(encoding="utf-8"))
    ts_by_id = {item["id"]: item for item in ts["results"]}
    if set(ts_by_id) != ids:
        raise AssertionError("TypeScript renderer returned the wrong fixture IDs")
    for row_id, result in ts_by_id.items():
        if not result["matches_expected"]:
            raise AssertionError(f"TypeScript prompt hash mismatch for {row_id}")
        if result["prompt_sha256"] != python_results[row_id]["prompt_sha256"]:
            raise AssertionError(f"Python/TypeScript prompt mismatch for {row_id}")
        python_results[row_id]["typescript_prompt_sha256"] = result["prompt_sha256"]
        python_results[row_id]["typescript_prompt_bytes"] = result["prompt_bytes"]
        python_results[row_id]["python_typescript_match"] = True

    shared_receipt = PLAN / "docs/campaign/receipts/PRM-04-shared-fixtures.json"
    prm05_fixture = PLAN / "docs/campaign/work/lead/PRM-05-selection-v3-fixture.json"
    source_pins = [
        file_pin(EXEC / "packages/sepalith/src/sepalith/campaign_protocol.py", role="actual Python training renderer"),
        file_pin(EXT / "src/campaign_protocol.ts", role="actual current TypeScript serving renderer"),
        file_pin(EXT / "scripts/check-campaign-protocol.ts", role="accepted PRM03 TypeScript fixture checks"),
        file_pin(FIXTURE, role="accepted native-probe TRAIN rows"),
        file_pin(FIXTURE_MANIFEST, role="accepted native-probe fixture manifest"),
        file_pin(SIDECAR, role="RL-02 context-only sidecar; streamed selected rows"),
        file_pin(shared_receipt, role="PRM04 shared Python/TypeScript fixture receipt"),
        file_pin(prm05_fixture, role="PRM05 synthetic selection fixture provenance"),
    ]
    output = {
        "schema_version": "run-02.shared-id-prompt-manifest.v1",
        "status": "mechanical_shared_fixture_pass",
        "source_pins": source_pins,
        "contract": {
            "renderer_id": RENDERER_ID,
            "tokenization_policy": TOKENIZATION_POLICY,
            "manual_bos_id": BOS_ID,
            "canonical_eos_id": EOS_ID,
            "native_noncanonical_eog_id": 130073,
            "prompt_boundary": "renderPrompt output is UTF-8 and ends with exactly one LF before generation.",
        },
        "fixture": {
            "path": str(FIXTURE),
            "manifest_path": str(FIXTURE_MANIFEST),
            "row_count": len(rows),
            "row_order": [row["id"] for row in rows],
            "rows": [python_results[row["id"]] for row in rows],
        },
        "token_boundary_evidence": {
            "rows": boundary_rows,
            "all_rows_pass": all(
                all(row[key] for key in boundary_checks) for row in boundary_rows
            ),
            "accepted_PRM04_joint_boundary_proof": "4/4 rows: joint_ids == prompt_ids + target_ids",
            "accepted_PRM04_terminal_suffix_proof": "4/4 rows: target IDs end in independently encoded terminal IDs",
            "no_model_or_server": True,
        },
        "renderer_comparison": {
            "python_renderer": "sepalith.campaign_protocol.render_prompt",
            "typescript_renderer": "current extensions/vscode-sepalith/src/campaign_protocol.ts::renderPrompt",
            "python_typescript_exact_matches": len(rows),
            "expected_matches": len(rows),
            "all_prompt_hashes_match": True,
        },
        "execution": {
            "typescript_command": "node --experimental-strip-types --no-warnings render_shared_fixture.ts shared-context-input.json typescript-render-results.json",
            "python_renderer_invoked": True,
            "sidecar_rows_scanned_until_all_selected": max(sidecar[row_id]["line"] for row_id in ids),
        },
    }
    MANIFEST_OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "rows": len(rows), "exact_matches": len(rows), "manifest": str(MANIFEST_OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
