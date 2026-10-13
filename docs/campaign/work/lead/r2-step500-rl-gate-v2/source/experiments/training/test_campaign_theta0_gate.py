"""CPU-only unit checks for the label-free theta0 diagnostic."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import campaign_theta0_gate as gate


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(value, **_kwargs):
        return FakeTensor(value)

    @staticmethod
    def ones_like(value):
        return FakeTensor([[1] * len(value.value[0])])


class FakeTokenizer:
    def decode(self, _ids, **_kwargs):
        return "[NO_EDIT]\n>>>>>>> UPDATED"


class FakeModel:
    device = "cpu"
    accepts_python_ids = False

    def __init__(self, response=None, error=None):
        self.response = list(response or [7, 1])
        self.error = error
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        prompt = kwargs["input_ids"].tolist()[0]
        return [prompt + self.response]


def _parsed():
    return SimpleNamespace(status="accepted", operation="no_op", body=("[NO_EDIT]",), reason=None)


def _case(response=None, kind="evidence_supported"):
    return {
        "variant_id": "base::evidence_supported",
        "base_row_id": "base",
        "kind": kind,
        "context": object(),
        "prompt": "prompt",
        "prompt_ids": [0, 11, 12],
        "context_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "prompt_ids_sha256": "c" * 64,
        "prompt_tokens_with_bos": 3,
        "row": {"id": "base", "target_operation": "no_op", "target_body_text": "[NO_EDIT]"},
        "fixture_contract": {},
        "parent": {"manifest_sha256": "d" * 64},
    }


class CaseRecordTests(unittest.TestCase):
    def setUp(self):
        self.parse = mock.patch.object(gate, "parse_output", return_value=_parsed())
        self.valid = mock.patch.object(gate, "valid_generation_tokens", return_value=True)
        self.parse.start()
        self.valid.start()
        self.addCleanup(self.parse.stop)
        self.addCleanup(self.valid.stop)

    def _record(self, model, case=None, use_cache=True):
        return gate._case_record(
            case or _case(), "arm", use_cache, model, FakeTokenizer(), object(),
            SimpleNamespace(for_inference=lambda _model: None),
            torch_module=FakeTorch(), case_guard=lambda _model: nullcontext(),
            identity_checker=lambda *_args: {"status": "verified"},
        )

    def test_payload_has_no_target_labels_and_greedy_cap_is_fixed(self):
        model = FakeModel(response=[7] * 191 + [1])
        record = self._record(model)
        kwargs = model.calls[0]
        self.assertEqual(kwargs["max_new_tokens"], 192)
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["eos_token_id"], [1, 130073])
        self.assertEqual(kwargs["bos_token_id"], 0)
        self.assertEqual(kwargs["pad_token_id"], 1)
        self.assertEqual(kwargs["use_cache"], True)
        self.assertFalse({"target", "target_text", "target_body", "target_operation", "region_new", "reward"}
                         & set(kwargs))
        self.assertEqual(record["cap"]["status"], "within_cap")
        self.assertTrue(record["eos"]["canonical_eos"])
        self.assertTrue(record["timing"]["synchronized_timing"] is False)
        self.assertEqual(record["peak_allocation"], {"device": "cpu", "cuda": False})
        self.assertEqual(record["parent"]["manifest_sha256"], "d" * 64)
        self.assertIsNone(record["counterfactual_target_score"])

    def test_cache_control_is_decode_kv_parity_only(self):
        cached = self._record(FakeModel(response=[7, 1]), use_cache=True)
        control = self._record(FakeModel(response=[7, 1]), use_cache=False)
        self.assertTrue(cached["fresh_request"])
        self.assertEqual(cached["cache_scope"], "per-request-decode-only")
        self.assertFalse(control["use_cache"])
        self.assertEqual(control["cache_scope"], "decode-KV-cache-disabled")

    def test_cap_without_eos_is_reported_inclusive(self):
        record = self._record(FakeModel(response=[7] * 192))
        self.assertEqual(record["cap"]["status"], "hit_without_terminal")
        self.assertTrue(record["cap"]["inclusive_terminal_policy"])
        self.assertIsNone(record["eos"]["terminal_token"])


class BoundaryTests(unittest.TestCase):
    def test_module_import_does_not_load_live_frameworks(self):
        code = (
            "import sys, campaign_theta0_gate; "
            "bad={'torch','transformers','trl','unsloth','datasets'} & set(sys.modules); "
            "assert not bad, bad"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_context_variant_fails_closed(self):
        with self.assertRaisesRegex(gate.Theta0GateError, "unknown context variant"):
            gate._derive_context(
                {"base": {"context": {}}},
                {"id": "bad", "base_row_id": "base", "kind": "made_up"},
                {},
            )

    def test_partial_receipt_is_durable_when_case_raises(self):
        fixture = {
            "schema_version": gate.FIXTURE_SCHEMA_VERSION,
            "inputs": {
                "rows_sha256": "a" * 64, "sidecar_sha256": "b" * 64,
                "selected_ids_sha256": "c" * 64, "ordered_ids_sha256": "d" * 64,
                "row_identity_sha256": "e" * 64,
            },
            "contract": {"renderer_id": "zeta2-prm03-v1", "tokenization_policy": "p"},
        }
        prepared = {
            "fixture": fixture,
            "rows": {"base": {"id": "base", "prompt_text": "prompt", "input_ids": [0], "target_start": 1}},
            "sidecars": {}, "selected_ids": ["base"], "base_ids": ["base"],
            "variant_defs": [{"id": "base::evidence_supported"}],
        }
        case = _case()
        model = FakeModel(error=RuntimeError("controlled fake failure"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "theta0.json"
            with mock.patch.object(gate, "validate_fixture_inputs", return_value=prepared), \
                 mock.patch.object(gate, "_verify_protocol_source", return_value={"sha256": "p"}), \
                 mock.patch.object(gate, "_verify_tokenizer_files", return_value={"path": "fake"}), \
                 mock.patch.object(gate, "build_variant_cases", return_value=[case]):
                result = gate.run_gate(
                    fixture_path=Path("fixture.json"), parent_manifest=Path("parent.json"),
                    parent_manifest_sha256="f" * 64, output=output, deadline_seconds=120,
                    require_admission=False, output_checker=lambda path: path,
                    parent_checker=lambda *_args: {"status": "verified", "manifest_sha256": "f" * 64},
                    occupancy_checker=lambda: {"gpu_count": 1},
                    model_loader=lambda *_args: (model, FakeTokenizer(), object(), None, {}),
                    torch_override=FakeTorch(), case_guard=lambda _model: nullcontext(),
                    identity_checker=lambda *_args: {"status": "verified"},
                )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["cases"]), 1)
            self.assertEqual(result["cases"][0]["status"], "case_failed")
            self.assertIn("controlled fake failure", result["failure"]["message"])
            persisted = json.loads(output.read_text())
            self.assertEqual(persisted["status"], "failed")


if __name__ == "__main__":
    unittest.main()
