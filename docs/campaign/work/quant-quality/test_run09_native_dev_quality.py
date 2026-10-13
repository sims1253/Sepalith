"""CPU-only contract tests for the RUN-09 native quality client."""
from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

import run09_native_dev_quality as quality


class FakeResponse:
    def __init__(self, lines, status=200):
        self._lines = list(lines)
        self._status = status
        self.fp = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self._status

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return b""

    def read(self):
        return b"{}"


class FakeTokenizer:
    def decode(self, _ids, **_kwargs):
        return "[NO_EDIT]\n>>>>>>> UPDATED"


class FakeProtocol:
    @staticmethod
    def valid_generation_tokens(tokens):
        return bool(tokens) and tokens[-1] == quality.EOS_ID and all(
            type(token) is int and 0 <= token < quality.VOCAB_SIZE for token in tokens[:-1]
        )

    @staticmethod
    def parse_output(_raw, _context):
        return SimpleNamespace(status="accepted", operation="no_op", body=("[NO_EDIT]",), reason=None)


def fake_case():
    return {
        "id": "dev-1",
        "family": "no_op",
        "package_id": "pkg:test",
        "operation": "no_op",
        "expected_noop": True,
        "expected_region": ["[NO_EDIT]"],
        "context": SimpleNamespace(region_old=("[NO_EDIT]",)),
        "tokenizer": FakeTokenizer(),
    }


class PayloadTests(unittest.TestCase):
    def test_exact_no_special_and_integer_prompt_payload(self):
        tokenize = quality.make_tokenize_payload("hello")
        self.assertEqual(tokenize, {
            "content": "hello", "add_special": False,
            "parse_special": False, "with_pieces": False,
        })
        completion = quality.make_completion_payload([0, 11, 12])
        self.assertEqual(completion, {
            "prompt": [0, 11, 12], "n_predict": 512,
            "temperature": 0, "stream": True,
            "cache_prompt": False, "return_tokens": True,
        })

    def test_model_artifact_is_not_selected_by_client(self):
        args = SimpleNamespace(
            url="http://127.0.0.1:18401", model_label="Q8",
            model_provenance="root://manifest/q8",
            panel=Path("panel.jsonl"),
            case_deadline_seconds=120, deadline_seconds=7200, reserve_seconds=60,
        )
        receipt = quality._initial_receipt(Path("/tmp/out.json"), args)
        self.assertEqual(receipt["model"]["artifact_selection"], "outer-root-wrapper-only")
        self.assertEqual(receipt["model"]["provenance"], "root://manifest/q8")


class SseAndResponseTests(unittest.TestCase):
    def test_sse_eos_is_kept_from_partial_chunk_and_final_chunk_is_not_duplicated(self):
        lines = [
            b'data: {"content":"[NO_EDIT]","tokens":[17]}\n',
            b'data: {"content":"\\n>>>>>>> UPDATED","tokens":[1],"stop":true,"stop_type":"eos"}\n',
        ]
        # b10453's final SSE frame should carry an empty token array.  The
        # nonempty-final case above must remain visible as a protocol issue.
        lines[-1] = b'data: {"content":"\\n>>>>>>> UPDATED","tokens":[],"stop":true,"stop_type":"eos"}\n'
        with mock.patch.object(quality, "urlopen", return_value=FakeResponse(lines)):
            streamed = quality._stream_completion(
                "http://127.0.0.1:18401", quality.make_completion_payload([0, 9]),
                quality.time.monotonic() + 2,
            )
        self.assertEqual(streamed["returned_token_ids"], [17])
        self.assertEqual(streamed["final_token_count"], 0)
        self.assertTrue(streamed["saw_stop"])
        self.assertEqual(streamed["malformed_sse_frames"], 0)

    def test_malformed_sse_is_recorded_and_invalidates_protocol(self):
        lines = [
            b'data: {"content":"[NO_EDIT]","tokens":[17]}\n',
            b'data: definitely-not-json\n',
            b'data: {"content":"\\n>>>>>>> UPDATED","tokens":[],"stop":true,"stop_type":"eos"}\n',
        ]
        with mock.patch.object(quality, "urlopen", return_value=FakeResponse(lines)):
            streamed = quality._stream_completion(
                "http://127.0.0.1:18401", quality.make_completion_payload([0, 9]),
                quality.time.monotonic() + 2,
            )
        self.assertEqual(streamed["malformed_sse_frames"], 1)
        result = quality._completion_validation(streamed, fake_case(), FakeProtocol())
        self.assertFalse(result["quality"]["protocol_valid"])
        self.assertEqual(result["protocol"]["malformed_sse_frames"], 1)

    def test_noncanonical_eog_and_cap_without_eos_fail_closed(self):
        native_eog = {
            "raw_text": "[NO_EDIT]\n>>>>>>> UPDATED", "returned_token_ids": [17, 130073],
            "final": {"stop_type": "eos"}, "saw_stop": True,
            "final_token_count": 0, "malformed_sse_frames": 0,
        }
        native_result = quality._completion_validation(native_eog, fake_case(), FakeProtocol())
        self.assertEqual(native_result["eos"]["status"], "noncanonical_native_eog")
        self.assertFalse(native_result["quality"]["protocol_valid"])
        capped = {
            "raw_text": "", "returned_token_ids": [17] * 512,
            "final": {"stop_type": "limit", "tokens_predicted": 512},
            "saw_stop": True, "final_token_count": 0, "malformed_sse_frames": 0,
        }
        capped_result = quality._completion_validation(capped, fake_case(), FakeProtocol())
        self.assertEqual(capped_result["cap"]["status"], "hit_without_eos")
        self.assertFalse(capped_result["quality"]["protocol_valid"])
        at_cap = {
            "raw_text": "[NO_EDIT]\\n>>>>>>> UPDATED", "returned_token_ids": [17] * 511 + [1],
            "final": {"stop_type": "eos", "tokens_predicted": 512},
            "saw_stop": True, "final_token_count": 0, "malformed_sse_frames": 0,
        }
        at_cap_result = quality._completion_validation(at_cap, fake_case(), FakeProtocol())
        self.assertEqual(at_cap_result["cap"]["status"], "within_cap")
        self.assertFalse(at_cap_result["cap"]["hit"])

    def test_parser_failure_and_final_token_duplication_are_visible(self):
        invalid_protocol = SimpleNamespace(
            valid_generation_tokens=FakeProtocol.valid_generation_tokens,
            parse_output=lambda _raw, _context: SimpleNamespace(
                status="invalid", operation=None, body=(), reason="missing_exact_terminal"
            )
        )
        streamed = {
            "raw_text": "broken", "returned_token_ids": [17, 1],
            "final": {"stop_type": "eos"}, "saw_stop": True,
            "final_token_count": 0, "malformed_sse_frames": 0,
        }
        result = quality._completion_validation(streamed, fake_case(), invalid_protocol)
        self.assertEqual(result["protocol"]["parser_status"], "invalid")
        self.assertFalse(result["quality"]["protocol_valid"])
        duplicated = dict(streamed, final_token_count=1)
        duplicate_result = quality._completion_validation(duplicated, fake_case(), FakeProtocol())
        self.assertFalse(duplicate_result["quality"]["protocol_valid"])
        self.assertEqual(duplicate_result["protocol"]["final_sse_tokens"], 1)


class BoundaryTests(unittest.TestCase):
    def test_fresh_output_uses_real_native_ext4_temp_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            self.assertEqual(quality._require_fresh_ext4(output), output)
            output.write_text("used", encoding="utf-8")
            with self.assertRaisesRegex(quality.QualityError, "fresh"):
                quality._require_fresh_ext4(output)

    def test_negative_and_out_of_vocab_ids_are_rejected(self):
        with self.assertRaisesRegex(quality.QualityError, "out-of-range"):
            quality._integer_ids([-1], "ids")
        with self.assertRaisesRegex(quality.QualityError, "out-of-range"):
            quality._integer_ids([quality.VOCAB_SIZE], "ids")

    def test_timeout_is_typed_and_does_not_retry(self):
        with mock.patch.object(quality, "urlopen", side_effect=socket.timeout("slow")) as opener:
            with self.assertRaises(quality.CaseTimeout):
                quality._request_json("http://127.0.0.1:18401", "/health", None, 1)
            opener.assert_called_once()

    def test_tokenizer_identity_mismatch(self):
        with self.assertRaisesRegex(quality.QualityError, "identity mismatch"):
            quality._validate_tokenizer_identity({
                "vocab_size": 130560, "bos_token_id": 0,
                "eos_token_id": 1, "pad_token_id": 130559,
            })

    def test_server_context_identity_is_exact(self):
        def fake_request(_url, endpoint, _body, _timeout):
            if endpoint == "/health":
                return 200, {"status": "ok"}, 0.001
            return 200, {"model_path": "outer-root", "default_generation_settings": {"n_ctx": 8192}}, 0.001
        with mock.patch.object(quality, "_request_json", side_effect=fake_request):
            with self.assertRaisesRegex(quality.QualityError, "exactly 4096"):
                quality._server_preflight("http://127.0.0.1:18401", 1)

    def test_receipt_separates_edit_and_strict_noop_denominators(self):
        args = SimpleNamespace(
            url="http://127.0.0.1:18401", model_label="Q8",
            model_provenance="root://manifest/q8", panel=Path("panel.jsonl"),
            case_deadline_seconds=120, deadline_seconds=7200, reserve_seconds=60,
        )
        receipt = quality._initial_receipt(Path("/tmp/out.json"), args)
        self.assertEqual(receipt["denominators"]["edit_cases"], 43)
        self.assertEqual(receipt["denominators"]["strict_noop_cases"], 32)
        self.assertIn("edit_exact_rows", receipt["denominators"])


class MechanicsTests(unittest.TestCase):
    def test_whole_sse_deadline_cancels_slow_stream(self):
        class SlowResponse(FakeResponse):
            def readline(self):
                raise socket.timeout("slow stream")

        with mock.patch.object(quality, "urlopen", return_value=SlowResponse([])):
            with self.assertRaises(quality.CaseTimeout):
                quality._stream_completion(
                    "http://127.0.0.1:18401", quality.make_completion_payload([0, 9]),
                    quality.time.monotonic() + 2,
                )

    def test_partial_stream_timeout_preserves_evidence_without_completion(self):
        class PartialThenTimeout(FakeResponse):
            def readline(self):
                if self._lines:
                    return self._lines.pop(0)
                raise socket.timeout("slow after first token")

        case = fake_case()
        case.update({
            "prompt": "hello", "prompt_sha256": "prompt", "target_sha256": "target",
            "hf_prompt_ids": [0, 9], "hf_prompt_tokens_with_bos": 2,
            "hf_prompt_ids_sha256": quality.canonical_sha256([0, 9]),
        })

        def fake_request(_url, endpoint, _body, _timeout):
            self.assertEqual(endpoint, "/tokenize")
            return 200, {"tokens": [9]}, 0.001

        response = PartialThenTimeout([b'data: {"content":"[NO_EDIT]","tokens":[17]}\n'])
        with mock.patch.object(quality, "_request_json", side_effect=fake_request):
            with mock.patch.object(quality, "urlopen", return_value=response):
                record = quality._run_case(
                    "http://127.0.0.1:18401", case, FakeProtocol(), FakeTokenizer(), 2
                )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_class"], "transport")
        self.assertTrue(record["response_received"])
        self.assertFalse(record["response_complete"])
        self.assertEqual(record["raw_text"], "[NO_EDIT]")
        self.assertEqual(record["returned_token_ids"], [17])
        self.assertFalse(record["quality"]["protocol_valid"])
        self.assertFalse(record["partial_stream"]["stream_complete"])

    def test_wire_text_mismatch_is_mechanical_quality_failure(self):
        case = fake_case()
        case["tokenizer"] = SimpleNamespace(
            decode=lambda _ids, **_kwargs: "different wire body"
        )
        streamed = {
            "raw_text": "[NO_EDIT]\n>>>>>>> UPDATED", "returned_token_ids": [17, quality.EOS_ID],
            "final": {"stop_type": "eos"}, "saw_stop": True,
            "final_token_count": 0, "malformed_sse_frames": 0,
        }
        result = quality._completion_validation(streamed, case, FakeProtocol())
        self.assertFalse(result["protocol"]["wire_hf_text_match"])
        self.assertEqual(result["failure_class"], "mechanical")
        self.assertEqual(result["failure"]["message"], "wire_hf_text_mismatch")
        self.assertFalse(result["quality"]["exact_edit"])


if __name__ == "__main__":
    unittest.main()
