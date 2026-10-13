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
    accounting,
    sha256_file,
    validate_terminal,
)


class PostrenderContractTest(unittest.TestCase):
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
