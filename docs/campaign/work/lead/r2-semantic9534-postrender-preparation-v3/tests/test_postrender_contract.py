#!/usr/bin/env python3
"""Small contract tests; all fixtures are synthetic and target-free."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "source"))

from verify_render16 import (  # noqa: E402
    ContractError,
    INPUT_ROOT,
    INPUT_MANIFEST_SHA256,
    accounting,
    load_json,
    sha256_file,
    validate_terminal,
    validate_input_manifest,
)


ACTUAL_MANIFEST = INPUT_ROOT / "manifest.json"
ACTUAL_RENDER16 = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Semantic9535-provider-preparation-v1/render-16k-retry-rich-v1"
)


class PostrenderContractTest(unittest.TestCase):
    def test_actual_input_manifest_binds_upstream_supported_denominator(self) -> None:
        manifest = load_json(ACTUAL_MANIFEST)
        result = validate_input_manifest(manifest, ACTUAL_MANIFEST)
        self.assertEqual(sha256_file(ACTUAL_MANIFEST), INPUT_MANIFEST_SHA256)
        self.assertEqual(result["provider_rows"], 9534)
        self.assertEqual(result["geometry_preparation_holds"], 1)
        self.assertEqual(result["supported_denominator"], 9535)
        self.assertEqual(manifest["upstream"]["semantic_supported"], 9535)
        self.assertNotIn("supported_denominator", manifest)

        wrong = {**manifest, "upstream": {**manifest["upstream"], "semantic_supported": 9534}}
        with self.assertRaisesRegex(ContractError, "upstream semantic supported denominator"):
            validate_input_manifest(wrong, ACTUAL_MANIFEST)

    def test_first_real_shard_uses_manifest_provider_hash_and_exact_ids(self) -> None:
        manifest = load_json(ACTUAL_MANIFEST)
        bound = manifest["shards"][0]
        input_path = ACTUAL_MANIFEST.parent / bound["provider"]["path"]
        output_path = ACTUAL_RENDER16 / "shard-0027.jsonl"
        terminal_path = ACTUAL_RENDER16 / "shard-0027.terminal.json"
        terminal = load_json(terminal_path)
        result = validate_terminal(
            terminal,
            expected_shard=27,
            terminal_path=terminal_path,
            input_path=input_path,
            expected_input_sha256=bound["provider"]["sha256"],
            output_path=output_path,
            expected_rows=bound["provider_rows"],
        )
        self.assertEqual(result["input_rows"], 746)
        self.assertEqual(result["output_rows"], 746)
        self.assertEqual(result["input_sha256"], bound["provider"]["sha256"])
        self.assertEqual(result["output_sha256"], terminal["output"]["sha256"])
        self.assertNotEqual(result["input_ids_sha256"], "")
        self.assertEqual(result["input_ids_sha256"], result["output_ids_sha256"])

    def test_denominator_keeps_provider_and_geometry_hold_separate(self) -> None:
        self.assertEqual(
            accounting(9534, 1),
            {
                "provider_rows": 9534,
                "geometry_preparation_holds": 1,
                "supported_denominator": 9535,
            },
        )
        with self.assertRaises(ContractError):
            accounting(9533, 1)
        with self.assertRaises(ContractError):
            accounting(9534, 0)

    def test_complete_terminal_preserves_rows_and_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semantic9534-contract-") as directory:
            root = Path(directory)
            input_path = root / "shard-0027.jsonl"
            output_path = root / "render-0027.jsonl"
            log_path = root / "shard-0027.log"
            terminal_path = root / "shard-0027.terminal.json"
            input_path.write_text('{"row_id":"r0"}\n{"row_id":"r1"}\n', encoding="utf-8")
            output_path.write_text(
                '{"row_id":"r0","status":"supported","selection_target_or_gold_used":false}\n'
                '{"row_id":"r1","status":"hold","selection_target_or_gold_used":false}\n',
                encoding="utf-8",
            )
            log_path.write_text("synthetic complete\n", encoding="utf-8")

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            terminal = {
                "schema": "sepalith.dat10.semantic9534.render_shard_rich_terminal.v1",
                "status": "complete",
                "shard": 27,
                "context_size": 16384,
                "generation_reserve": 2048,
                "exit_code": 0,
                "input": {"sha256": digest(input_path)},
                "output": {
                    "path": str(output_path),
                    "sha256": digest(output_path),
                    "bytes": output_path.stat().st_size,
                    "rows": 2,
                },
                "log": {"path": str(log_path), "sha256": digest(log_path)},
            }
            terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
            expected_input_sha = digest(input_path)
            result = validate_terminal(
                terminal,
                expected_shard=27,
                terminal_path=terminal_path,
                input_path=input_path,
                expected_input_sha256=expected_input_sha,
                output_path=output_path,
                expected_rows=2,
            )
            self.assertEqual(result["output_rows"], 2)
            self.assertEqual(result["output_ids_sha256"], digest_ids(["r0", "r1"]))
            self.assertEqual(result["terminal_sha256"], sha256_file(terminal_path))

            with self.assertRaisesRegex(ContractError, "partial"):
                validate_terminal(
                    terminal,
                    expected_shard=27,
                    terminal_path=terminal_path,
                    input_path=input_path,
                    expected_input_sha256=expected_input_sha,
                    output_path=output_path,
                    expected_rows=3,
                )
            failed = {**terminal, "status": "failed", "exit_code": 1}
            with self.assertRaisesRegex(ContractError, "infrastructure failure"):
                validate_terminal(
                    failed,
                    expected_shard=27,
                    terminal_path=terminal_path,
                    input_path=input_path,
                    expected_input_sha256=expected_input_sha,
                    output_path=output_path,
                    expected_rows=2,
                )

            substituted_path = root / "render-substituted.jsonl"
            substituted_path.write_text(
                '{"row_id":"r0","status":"supported","selection_target_or_gold_used":false}\n'
                '{"row_id":"different","status":"hold","selection_target_or_gold_used":false}\n',
                encoding="utf-8",
            )
            substituted = {
                **terminal,
                "output": {
                    **terminal["output"],
                    "path": str(substituted_path),
                    "sha256": digest(substituted_path),
                    "bytes": substituted_path.stat().st_size,
                },
            }
            with self.assertRaisesRegex(ContractError, "row-ID set mismatch"):
                validate_terminal(
                    substituted,
                    expected_shard=27,
                    terminal_path=terminal_path,
                    input_path=input_path,
                    expected_input_sha256=expected_input_sha,
                    output_path=substituted_path,
                    expected_rows=2,
                )

            with self.assertRaisesRegex(ContractError, "terminal shard identity mismatch"):
                validate_terminal(
                    {**terminal, "shard": 28},
                    expected_shard=27,
                    terminal_path=terminal_path,
                    input_path=input_path,
                    expected_input_sha256=expected_input_sha,
                    output_path=output_path,
                    expected_rows=2,
                )

            input_path.write_text('{"row_id":"r0"}\n{"row_id":"mutated"}\n', encoding="utf-8")
            mutated_terminal = {
                **terminal,
                "input": {"sha256": digest(input_path)},
            }
            with self.assertRaisesRegex(ContractError, "terminal input binding mismatch"):
                validate_terminal(
                    mutated_terminal,
                    expected_shard=27,
                    terminal_path=terminal_path,
                    input_path=input_path,
                    expected_input_sha256=expected_input_sha,
                    output_path=output_path,
                    expected_rows=2,
                )


def digest_ids(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
