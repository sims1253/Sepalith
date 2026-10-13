#!/usr/bin/env python3
"""CPU-only contract tests for the SFT-08/RUN-03 preparation client."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run03_b4_dev_client", HERE / "run03_b4_dev_client.py")
assert SPEC is not None and SPEC.loader is not None
CLIENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLIENT
SPEC.loader.exec_module(CLIENT)


class B4DevClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = CLIENT.load_legacy_renderer(CLIENT.DEFAULT_REPO_ROOT)
        cls.rows = CLIENT.load_cases(CLIENT.DEFAULT_CASES)

    def test_import_is_framework_free(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertTrue(callable(CLIENT.parse_prediction))

    def test_all_frozen_rows_have_explicit_legacy_projection(self):
        adapted = [CLIENT.adapt_case(row, self.renderer) for row in self.rows]
        self.assertEqual(len(adapted), 75)
        self.assertEqual(len({item["legacy_prompt_sha256"] for item in adapted}), 75)
        self.assertEqual(
            {row["family"] for row in self.rows},
            {
                "finish_block",
                "format_propagation",
                "na_rm_propagation",
                "no_op",
                "pipe_rewrite",
                "rename_propagation",
                "roxygen_drafting",
            },
        )
        self.assertEqual(sum(row["operation"] == "replace" for row in self.rows), 43)
        self.assertEqual(sum(row["operation"] == "no_op" for row in self.rows), 32)
        self.assertEqual(
            sum(
                item["geometry"]["mode"]
                in {
                    "legacy_line_end_cursor_after_codepoint_split",
                    "legacy_line_end_cursor",
                }
                for item in adapted
            ),
            61,
        )
        self.assertEqual(
            sum(
                item["geometry"]["mode"]
                == "legacy_line_end_cursor_after_codepoint_split"
                for item in adapted
            ),
            32,
        )
        self.assertEqual(
            sum(item["geometry"]["mode"] == "legacy_line_end_cursor" for item in adapted),
            29,
        )
        self.assertEqual(
            sum(item["geometry"]["mode"] == "insertion_or_unpositioned_empty_region" for item in adapted),
            14,
        )

    def test_noop_is_empty_proposal_and_delete_is_rejected(self):
        no_op = next(row for row in self.rows if row["operation"] == "no_op")
        adapted = CLIENT.adapt_case(no_op, self.renderer)
        self.assertEqual(no_op["target_body_text"], "[NO_EDIT]")
        self.assertNotIn("[NO_EDIT]", adapted["legacy_prompt"])
        self.assertEqual(CLIENT.parse_prediction("\n>>>>>>> UPDATED"), [])
        self.assertEqual(CLIENT.parse_prediction("[NO_EDIT]\n>>>>>>> UPDATED"), ["[NO_EDIT]"])

        deletion = dict(no_op)
        deletion["id"] = "fixture-delete"
        deletion["operation"] = "delete"
        deletion["region_new"] = []
        with self.assertRaises(CLIENT.UnsupportedCase) as ctx:
            CLIENT.adapt_case(deletion, self.renderer)
        self.assertIn("operation 'delete'", str(ctx.exception))

    def test_projection_ignores_prm03_prompt_and_unsupported_context_fields(self):
        replace = next(row for row in self.rows if row["operation"] == "replace")
        no_op = next(row for row in self.rows if row["operation"] == "no_op")
        baseline = CLIENT.adapt_case(replace, self.renderer)
        no_op_baseline = CLIENT.adapt_case(no_op, self.renderer)
        replace_payload = CLIENT.make_completion_payload(baseline["legacy_prompt"])
        no_op_payload = CLIENT.make_completion_payload(no_op_baseline["legacy_prompt"])
        self.assertEqual(
            {k: v for k, v in replace_payload.items() if k != "prompt"},
            {k: v for k, v in no_op_payload.items() if k != "prompt"},
        )
        self.assertEqual(replace_payload["n_predict"], CLIENT.PRODUCTION_MAX_TOKENS)
        self.assertEqual(replace_payload["stop"], list(CLIENT.PRODUCTION_STOPS))
        projected = json.loads(json.dumps(replace))
        projected["prompt_sha256"] = "PRM03_PROMPT_MUST_NOT_BE_SENT"
        projected["context"]["schema_version"] = "PRM03"
        for field in (
            "replacement_range",
            "diagnostics",
            "retrieval",
            "selected_references",
            "scope_lines",
            "scope_mode",
            "source_provenance",
        ):
            projected["context"][field] = {"unsupported": "MUST_NOT_BE_SENT"}
        adapted = CLIENT.adapt_case(projected, self.renderer)
        self.assertEqual(adapted["legacy_prompt"], baseline["legacy_prompt"])
        self.assertNotIn("PRM03_PROMPT_MUST_NOT_BE_SENT", adapted["legacy_prompt"])
        self.assertNotIn("MUST_NOT_BE_SENT", adapted["legacy_prompt"])

        gold_mutated = json.loads(json.dumps(replace))
        gold_mutated["family"] = "mutated-gold-family"
        gold_mutated["target_body_text"] = "MUTATED_GOLD_TARGET"
        gold_mutated["target_sha256"] = "MUTATED_GOLD_HASH"
        gold_mutated["source_ref"] = "MUTATED_SOURCE_REF"
        gold_mutated["source_provenance"] = {"mutated": True}
        target = list(gold_mutated["region_new"])
        target[-1] = target[-1] + "  # MUTATED_GOLD_LINE"
        gold_mutated["region_new"] = target
        gold_adapted = CLIENT.adapt_case(gold_mutated, self.renderer)
        self.assertEqual(gold_adapted["legacy_prompt"], baseline["legacy_prompt"])
        gold_payload = CLIENT.make_completion_payload(gold_adapted["legacy_prompt"])
        self.assertEqual(
            {k: v for k, v in gold_payload.items() if k != "prompt"},
            {k: v for k, v in replace_payload.items() if k != "prompt"},
        )

        no_op_with_region = next(
            row
            for row in self.rows
            if row["operation"] == "no_op" and row["context"]["region_old"]
        )
        operation_mutated = json.loads(json.dumps(no_op_with_region))
        operation_mutated["operation"] = "replace"
        operation_mutated["target_body_text"] = "MUTATED_OPERATION_LABEL"
        operation_adapted = CLIENT.adapt_case(operation_mutated, self.renderer)
        self.assertEqual(
            operation_adapted["legacy_prompt"],
            CLIENT.adapt_case(no_op_with_region, self.renderer)["legacy_prompt"],
        )
        operation_payload = CLIENT.make_completion_payload(
            operation_adapted["legacy_prompt"]
        )
        self.assertEqual(
            {k: v for k, v in operation_payload.items() if k != "prompt"},
            {k: v for k, v in replace_payload.items() if k != "prompt"},
        )

    def test_utf16_cursor_is_verified_before_lossless_split(self):
        row = {
            "id": "fixture-utf16",
            "family": "fixture",
            "operation": "replace",
            "split": "dev",
            "target_body_text": "😀 <- y",
            "region_new": ["😀 <- y"],
            "context": {
                "path": "R/utf16.R",
                "prefix": [],
                "region_old": ["😀 <- x"],
                "suffix_lines": ["tail()"],
                "history": [],
                "document_eol": "lf",
                "cursor": {
                    "region_line_index": 0,
                    "code_point_column": 1,
                    "utf16_column": 2,
                },
            },
        }
        adapted = CLIENT.adapt_case(row, self.renderer)
        self.assertEqual(adapted["legacy"]["region_old"], ["😀"])
        self.assertEqual(adapted["legacy"]["suffix"], [" <- x", "tail()"])
        self.assertEqual(adapted["geometry"]["source_utf16_column"], 2)

        mismatch = json.loads(json.dumps(row))
        mismatch["context"]["cursor"]["utf16_column"] = 1
        with self.assertRaises(CLIENT.UnsupportedCase) as ctx:
            CLIENT.adapt_case(mismatch, self.renderer)
        self.assertIn("UTF-16/code-point", str(ctx.exception))

    def test_native_request_and_stop_metadata_are_complete(self):
        replace = next(row for row in self.rows if row["operation"] == "replace")
        adapted = CLIENT.adapt_case(replace, self.renderer)
        payload = CLIENT.make_completion_payload(adapted["legacy_prompt"])
        self.assertEqual(payload["n_predict"], CLIENT.PRODUCTION_MAX_TOKENS)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["stop"], list(CLIENT.PRODUCTION_STOPS))
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["cache_prompt"])
        self.assertTrue(payload["return_tokens"])
        self.assertTrue(payload["timings_per_token"])
        self.assertEqual(CLIENT.make_tokenize_payload("x")["add_special"], True)
        self.assertEqual(CLIENT.make_tokenize_payload("x")["parse_special"], True)

        response = {
            "content": "replacement\n",
            "tokens": [12, 34],
            "stop": True,
            "stop_type": "limit",
            "stopping_word": "",
            "tokens_predicted": 640,
            "tokens_evaluated": 99,
            "tokens_cached": 0,
            "truncated": False,
            "has_new_line": True,
        }
        native = CLIENT.native_stop_metadata(response, CLIENT.PRODUCTION_MAX_TOKENS)
        self.assertEqual(native["stop_type"], "limit")
        self.assertEqual(native["last_generated_token_id"], 34)
        self.assertEqual(native["cap_hit"], 1)
        eos = dict(response, stop_type="eos", tokens_predicted=12)
        eos_native = CLIENT.native_stop_metadata(eos, CLIENT.PRODUCTION_MAX_TOKENS)
        self.assertEqual(eos_native["stop_type"], "eos")
        self.assertEqual(eos_native["cap_hit"], 0)
        scored = CLIENT.classify_response(adapted, response, self.renderer)
        self.assertEqual(scored["parser_valid"], 1)
        self.assertIn("prediction_lines", scored)

    def test_noop_classification_uses_proposal_not_no_edit_text(self):
        no_op = next(row for row in self.rows if row["operation"] == "no_op")
        adapted = CLIENT.adapt_case(no_op, self.renderer)
        empty = CLIENT.classify_response(
            adapted,
            {"content": "\n>>>>>>> UPDATED", "tokens": [], "stop_type": "word"},
            self.renderer,
        )
        self.assertEqual(empty["no_op_correct"], 1)
        self.assertEqual(empty["no_op_false_suggestion"], 0)
        marker = CLIENT.classify_response(
            adapted,
            {"content": "[NO_EDIT]\n>>>>>>> UPDATED", "tokens": [4], "stop_type": "word"},
            self.renderer,
        )
        self.assertEqual(marker["no_op_correct"], 0)
        self.assertEqual(marker["no_op_false_suggestion"], 1)

    def test_deadline_policy_reserves_time_for_durable_output(self):
        args = type("Args", (), {"soft_deadline_s": 840.0})()
        old = {
            key: CLIENT.os.environ.get(key)
            for key in (
                "SEPALITH_B4_DEV_SOFT_DEADLINE_S",
                "SEPALITH_B4_DEV_HARD_DEADLINE_S",
                "SEPALITH_B4_DEV_CHECKPOINT_RESERVE_S",
            )
        }
        try:
            CLIENT.os.environ["SEPALITH_B4_DEV_HARD_DEADLINE_S"] = "900"
            CLIENT.os.environ["SEPALITH_B4_DEV_CHECKPOINT_RESERVE_S"] = "60"
            settings = CLIENT._deadline_settings(args)
            self.assertEqual(settings["soft_s"], 840)
            self.assertEqual(settings["hard_s"], 900)
            self.assertEqual(settings["checkpoint_reserve_s"], 60)
            args.soft_deadline_s = 2340.0
            CLIENT.os.environ["SEPALITH_B4_DEV_HARD_DEADLINE_S"] = "2400"
            settings = CLIENT._deadline_settings(args)
            self.assertEqual(settings, {"soft_s": 2340, "hard_s": 2400, "checkpoint_reserve_s": 60})
            CLIENT.os.environ["SEPALITH_B4_DEV_CHECKPOINT_RESERVE_S"] = "61"
            with self.assertRaises(ValueError):
                CLIENT._deadline_settings(args)
        finally:
            for key, value in old.items():
                if value is None:
                    CLIENT.os.environ.pop(key, None)
                else:
                    CLIENT.os.environ[key] = value

    def _resume_static_fixture(self):
        """Stable identity fixture; resume validation compares it byte-for-byte."""
        return {
            "cases": {"path": "/frozen/cases.jsonl", "rows": 75, "sha256": "cases"},
            "model": {"path": "/frozen/model.gguf", "bytes": 1, "sha256": "model"},
            "runtime": {"path": "/frozen/llama-server", "sha256": "runtime"},
            "legacy_renderer": {"path": "/frozen/run_eval.py", "sha256": "renderer", "implementation": "run_eval.render_zeta2"},
            "legacy_baseline": {"scenario_path": "/frozen/scenarios.py", "scenario_sha256": "scenario", "noop_eval_path": "/frozen/noop.py", "noop_eval_sha256": "noop"},
            "legacy_parser": {"path": "/frozen/extension.ts", "sha256": "extension", "implementation": "extension.ts parsePrediction"},
        }

    def _resume_record(self, item, response_ok=1):
        case = item["case"]
        payload = CLIENT.make_completion_payload(item["legacy_prompt"])
        tokenize = CLIENT.make_tokenize_payload(item["legacy_prompt"])
        return {
            "task": CLIENT.TASK,
            "case": {
                "id": case["id"],
                "family": case["family"],
                "operation": case["operation"],
                "package_id": case.get("package_id"),
                "group_id": case.get("group_id"),
                "split": case.get("split"),
                "source_line_number": case["_source_line_number"],
                "source_line_sha256": case["_source_line_sha256"],
                "source_prompt_sha256_prm03_ignored": case.get("prompt_sha256"),
                "source_target_sha256": case.get("target_sha256"),
            },
            "legacy_renderer": {
                "name": "zeta2",
                "implementation": "run_eval.render_zeta2",
                "prompt_sha256": item["legacy_prompt_sha256"],
                "prompt_chars": len(item["legacy_prompt"]),
                "geometry": item["geometry"],
                "history_count": item["history_count"],
            },
            "legacy_prompt": item["legacy_prompt"],
            "legacy_target_sha256": item["legacy_target_sha256"],
            "legacy_target_line_count": len(self.renderer.norm(item["target_lines"])),
            "native_tokenization": {
                "request": tokenize,
                "request_sha256": CLIENT.canonical_sha256(tokenize),
                "response": {"tokens": [1]},
                "token_count": 1,
                "token_ids": [1],
            },
            "request": {
                "endpoint": "/completion",
                "payload": payload,
                "payload_sha256": CLIENT.canonical_sha256(payload),
                "max_tokens": CLIENT.PRODUCTION_MAX_TOKENS,
                "stops": list(CLIENT.PRODUCTION_STOPS),
            },
            "response_ok": response_ok,
        }

    def _write_resume_fixture(self, tempdir, items, failed_ids=()):
        root = Path(tempdir)
        path = root / "per_request.jsonl"
        records = [
            self._resume_record(item, 0 if item["case"]["id"] in set(failed_ids) else 1)
            for item in items
        ]
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
        static = self._resume_static_fixture()
        families = dict(sorted(Counter(item["case"]["family"] for item in self.adapted).items()))
        operations = dict(sorted(Counter(item["case"]["operation"] for item in self.adapted).items()))
        coverage = {
            "requested_rows": 75,
            "adapted_rows": 75,
            "excluded_rows": 0,
            "legacy_prompt_hashes_unique": True,
            "families": families,
            "operations": operations,
        }
        summary_path = root / "summary.json"
        summary = {
            "task": CLIENT.TASK,
            "status": "deadline",
            "static_identities": static,
            "renderer_contract": CLIENT.renderer_contract(),
            "coverage": coverage,
            "denominators": {"completed_rows": len(records)},
            "artifacts": {
                "per_request_jsonl": {
                    "path": str(path.resolve()),
                    "rows": len(records),
                    "sha256": CLIENT.sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return path, summary_path, static, records

    def setUp(self):
        self.adapted = [CLIENT.adapt_case(row, self.renderer) for row in self.rows]

    def test_resume_validates_actual_partial_and_retains_failed_row(self):
        path = Path(
            "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
            "SFT-08-b4-dev-v2/per_request.jsonl"
        )
        summary_path = path.with_name("summary.json")
        prior_summary = json.loads(summary_path.read_text())
        resume = CLIENT.load_resume_successful_jsonl(
            path,
            summary_path,
            self.adapted,
            prior_summary["static_identities"],
            self.renderer,
        )
        self.assertEqual(resume["rows"], 26)
        self.assertEqual(resume["already_complete"], 25)
        self.assertEqual(len(resume["successful_ids"]), 25)
        self.assertEqual(len(resume["failed_ids"]), 1)
        self.assertEqual(len(resume["remaining_ids"]), 50)
        self.assertEqual(resume["replayed_failed_ids"], resume["failed_ids"])
        requested = CLIENT.remaining_adapted(self.adapted, resume)
        self.assertEqual(len(requested), 50)
        self.assertEqual(
            [item["case"]["id"] for item in requested], resume["remaining_ids"]
        )
        state = CLIENT.build_resume_state(self.adapted, resume, [])
        self.assertEqual(state["total_population"], 75)
        self.assertEqual(state["already_complete"], 25)
        self.assertEqual(state["requested"], 50)
        self.assertEqual(state["replayed_failed_ids"], resume["failed_ids"])

    def test_resume_rejects_mismatched_static_identity(self):
        with TemporaryDirectory() as tempdir:
            items = self.adapted[:2]
            path, summary_path, static, _ = self._write_resume_fixture(tempdir, items)
            summary = json.loads(summary_path.read_text())
            summary["static_identities"]["model"]["sha256"] = "changed"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            with self.assertRaises(RuntimeError) as ctx:
                CLIENT.load_resume_successful_jsonl(path, summary_path, self.adapted, static, self.renderer)
            self.assertIn("static source/model/runtime identity", str(ctx.exception))

    def test_resume_rejects_duplicate_ids(self):
        with TemporaryDirectory() as tempdir:
            path, summary_path, static, _ = self._write_resume_fixture(
                tempdir, [self.adapted[0], self.adapted[0]]
            )
            with self.assertRaises(RuntimeError) as ctx:
                CLIENT.load_resume_successful_jsonl(path, summary_path, self.adapted, static, self.renderer)
            self.assertIn("duplicate resume case ID", str(ctx.exception))

    def test_resume_rejects_mismatched_request_policy(self):
        with TemporaryDirectory() as tempdir:
            path, summary_path, static, records = self._write_resume_fixture(
                tempdir, [self.adapted[0]]
            )
            records[0]["request"]["max_tokens"] = 319
            path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
            summary = json.loads(summary_path.read_text())
            summary["artifacts"]["per_request_jsonl"]["sha256"] = CLIENT.sha256_file(path)
            summary["artifacts"]["per_request_jsonl"]["bytes"] = path.stat().st_size
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            with self.assertRaises(RuntimeError) as ctx:
                CLIENT.load_resume_successful_jsonl(path, summary_path, self.adapted, static, self.renderer)
            self.assertIn("request", str(ctx.exception))

    def test_resume_union_requires_root_acceptance_even_when_complete(self):
        previous = {"successful_ids": [item["case"]["id"] for item in self.adapted[:25]], "failed_ids": []}
        current = [self._resume_record(item, 1) for item in self.adapted[25:]]
        state = CLIENT.build_resume_state(self.adapted, previous, current)
        self.assertEqual(state["total_population"], 75)
        self.assertEqual(state["already_complete"], 25)
        self.assertEqual(state["requested"], 50)
        self.assertEqual(state["current_complete"], 50)
        self.assertEqual(state["missing_ids"], [])
        self.assertTrue(state["union"]["complete"])
        self.assertTrue(state["union"]["one_result_per_id"])
        self.assertTrue(state["union"]["root_review_required"])
        self.assertFalse(state["union"]["accepted_by_root"])
        self.assertFalse(state["union"]["one_accepted_result_per_id"])

    def test_mocked_resume_run_requests_only_exact_remaining_segment(self):
        prior_path = Path(
            "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/"
            "SFT-08-b4-dev-v2/per_request.jsonl"
        )
        prior_summary_path = prior_path.with_name("summary.json")
        prior_summary = json.loads(prior_summary_path.read_text())
        expected_remaining = [
            item["case"]["id"]
            for item in self.adapted
            if item["case"]["id"]
            not in {
                "dat07-existing-c54ed09efae0ca38cff22981",
                "dat07-existing-152d62f7a57472f8abb30ca9",
                "dat07-existing-2838b904f1e44b7440e6da9f",
                "dat07-existing-06847c759fd32bdf16f2e598",
                "dat07-existing-81ddc4c6ddd226b04dcf0910",
                "dat07-existing-5b5e844d3a8795c52149f848",
                "dat07-existing-06097c12b0328d475be857e1",
                "dat07-existing-d38a792699b4cccded445dc0",
                "dat07-derived-92f6d09aad6ff785964e2165",
                "dat07-derived-d566409e128abd081376e6fe",
                "dat07-derived-78fef7317afd1406ca5f1e66",
                "dat07-existing-4a64017b69fe23f5b553fa27",
                "dat07-derived-cd55b7ebb2f82f37cf26c07c",
                "dat07-derived-ac3860bc0cf085337a0206d7",
                "dat07-derived-9494058d0f2c32bd40e17d3a",
                "dat07-derived-d9011e8269f9e109350ffbcb",
                "dat07-derived-070d33653640c346278add08",
                "dat07-existing-a1b63fee525fbc1373bb82c3",
                "dat07-existing-7e51067ae755fcd8dcda1765",
                "dat07-derived-cc5f7cc76a4d8f143a7e8ef9",
                "dat07-derived-4366811a0192d0049c0eeacf",
                "dat07-existing-719cd49683667d0fb86fb2fa",
                "dat07-derived-d7da75be93bab6d10111259b",
                "dat07-derived-f6a8a2051155c52a8d33e635",
                "dat07-derived-a6adbbecf7f6bfdc07070036",
            }
        ]
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            output = root / "resume" / "per_request.jsonl"
            summary = root / "resume" / "summary.json"
            seen_payloads = []

            def fake_tokenize(_base_url, prompt, _timeout):
                request = CLIENT.make_tokenize_payload(prompt)
                return {
                    "request": request,
                    "request_sha256": CLIENT.canonical_sha256(request),
                    "response": {"tokens": [1]},
                    "token_count": 1,
                    "token_ids": [1],
                }

            def fake_completion(_base_url, endpoint, payload, _timeout):
                self.assertEqual(endpoint, "/completion")
                seen_payloads.append(payload)
                return 200, {
                    "content": "",
                    "tokens": [],
                    "stop": True,
                    "stop_type": "eos",
                    "tokens_predicted": 0,
                }

            args = CLIENT.parser().parse_args(
                [
                    "--url",
                    "http://127.0.0.1:18099",
                    "--cases",
                    str(CLIENT.DEFAULT_CASES),
                    "--model",
                    "/unused/model.gguf",
                    "--server-binary",
                    "/unused/server",
                    "--repo-root",
                    str(CLIENT.DEFAULT_REPO_ROOT),
                    "--output",
                    str(output),
                    "--summary",
                    str(summary),
                    "--resume-successful-jsonl",
                    str(prior_path),
                    "--resume-summary",
                    str(prior_summary_path),
                    "--soft-deadline-s",
                    "30",
                ]
            )
            with patch.object(CLIENT, "verify_static_identities", return_value=prior_summary["static_identities"]), patch.object(
                CLIENT, "server_preflight", return_value={"n_ctx": CLIENT.EXPECTED_CONTEXT_SIZE}
            ), patch.object(CLIENT, "native_tokenize", side_effect=fake_tokenize), patch.object(
                CLIENT, "http_json", side_effect=fake_completion
            ), redirect_stdout(io.StringIO()):
                result, rc = CLIENT.run_benchmark(args)

            self.assertEqual(rc, 0)
            self.assertEqual(len(seen_payloads), 50)
            self.assertEqual(result["resume"]["total_population"], 75)
            self.assertEqual(result["resume"]["already_complete"], 25)
            self.assertEqual(result["resume"]["requested"], 50)
            self.assertEqual(result["resume"]["current_complete"], 50)
            self.assertEqual(result["resume"]["request_order"], expected_remaining)
            self.assertEqual(result["resume"]["replayed_failed_ids"], [expected_remaining[0]])
            self.assertTrue(result["resume"]["union"]["complete"])
            self.assertFalse(result["resume"]["union"]["accepted_by_root"])
            written = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([record["case"]["id"] for record in written], expected_remaining)


if __name__ == "__main__":
    unittest.main(verbosity=2)
