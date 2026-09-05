"""Tests for spec_bench.py (S1 rig).

Run:  cd /home/m0hawk/Documents/Sepalith && \
      uv run --with pytest python -m pytest experiments/eval/test_spec_bench.py -q

Covers the three load-bearing pieces of the rig per the S1 task contract:
trace loading (MANIFEST invariants), acceptance math (tok/step semantics —
in particular that NO speculative activity is None, not 1.0), and TTFT
measurement (first CONTENT chunk, stop-chunk timings capture). Plus the
arm flag builder (single-mode arms, A2 §4.5).
"""

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spec_bench import (  # noqa: E402
    MAX_TOKENS,
    STOP,
    acceptance_from_delta,
    acceptance_from_response,
    arm_flags,
    load_traces,
    paired_smoke_traces,
    parse_prometheus,
    sample_traces,
    spec_deltas,
    stream_completion,
)

TRACE_OK = dict(
    trace_id="deadbeef00-cafe1234-2k", ctx_class="2k",
    prompt="<[fim-suffix]>x\n<<<<<<< CURRENT\n<|user_cursor|>\n=======\n<[fim-middle]>",
    target="y <- 1\n>>>>>>> UPDATED",
    prompt_tokens=1811, target_tokens=36,
)


def write_traces(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


class TestTraceLoader:
    def test_load_and_classify(self):
        r8 = dict(TRACE_OK, trace_id="deadbeef00-cafe1234-8k", ctx_class="8k")
        p = write_traces([TRACE_OK, r8])
        by = load_traces(p)
        assert [len(by["2k"]), len(by["8k"])] == [1, 1]
        os.unlink(p)

    def test_target_must_end_with_updated_marker(self):
        bad = dict(TRACE_OK, target="no marker here")
        p = write_traces([bad])
        with pytest.raises(AssertionError, match="terminal"):
            load_traces(p)
        os.unlink(p)

    def test_bad_ctx_class_rejected(self):
        bad = dict(TRACE_OK, ctx_class="4k")
        p = write_traces([bad])
        with pytest.raises(AssertionError):
            load_traces(p)
        os.unlink(p)

    def test_sample_deterministic_and_stable(self):
        rows = [dict(TRACE_OK, trace_id=f"t{i}-abcd1234-2k") for i in range(50)]
        s1 = [r["trace_id"] for r in sample_traces(rows, 10)]
        s2 = [r["trace_id"] for r in sample_traces(rows, 10)]
        assert s1 == s2, "sample must be deterministic across calls"
        assert len(s1) == 10
        # n >= len -> full set, order preserved
        assert sample_traces(rows, 100) == rows

    def test_paired_smoke_traces_same_edit(self):
        a = dict(TRACE_OK, trace_id="aaaabbbb-11112222-2k")
        b = dict(TRACE_OK, trace_id="aaaabbbb-11112222-8k", ctx_class="8k")
        by = {"2k": [dict(TRACE_OK, trace_id="zzzz-9999-2k"), a],
              "8k": [b]}
        pair = paired_smoke_traces(by)
        assert pair["2k"][0]["trace_id"] == "aaaabbbb-11112222-2k"
        assert pair["8k"][0]["trace_id"] == "aaaabbbb-11112222-8k"


class TestAcceptanceMath:
    def test_no_spec_activity_is_none_not_one(self):
        """The trap: 1 + 0/0 must not silently become 1.0 (== baseline ==
        'perfect'). No drafts -> (None, None)."""
        assert acceptance_from_delta(0, 0, 0) == (None, None)
        assert acceptance_from_delta(0, 5, 0) == (None, None)
        assert acceptance_from_delta(10, 0, -1) == (None, None)

    def test_tok_per_step_semantics(self):
        # 2 accepted drafts over 4 verify steps -> every step emits its
        # sampled token (4) + 2 accepted = 6 tokens / 4 steps = 1.5
        tps, rate = acceptance_from_delta(10, 2, 4)
        assert tps == pytest.approx(1.5)
        assert rate == pytest.approx(0.2)

    def test_perfect_half(self):
        # n_max=1 equivalent: every step accepts its single draft -> 2.0
        tps, rate = acceptance_from_delta(100, 100, 100)
        assert tps == pytest.approx(2.0)
        assert rate == pytest.approx(1.0)

    def test_zero_accepted_is_baseline_pace(self):
        tps, rate = acceptance_from_delta(50, 0, 50)
        assert tps == pytest.approx(1.0)
        assert rate == pytest.approx(0.0)


class TestAcceptanceFromResponse:
    """PRIMARY path: b10453 final-chunk timings carry draft_n /
    draft_n_accepted / predicted_n per request."""

    def test_baseline_no_draft_fields_is_none(self):
        t = {"prompt_ms": 100.0, "predicted_n": 8, "predicted_ms": 200.0}
        assert acceptance_from_response(t) == (None, None)
        assert acceptance_from_response({}) == (None, None)

    def test_zero_accepted_is_baseline_pace(self):
        t = {"predicted_n": 8, "draft_n": 15, "draft_n_accepted": 0}
        tps, rate = acceptance_from_response(t)
        assert tps == pytest.approx(1.0)
        assert rate == pytest.approx(0.0)

    def test_half_accepted_doubles_pace(self):
        # 8 emitted, 4 accepted -> 4 steps -> 2.0 tok/step
        t = {"predicted_n": 8, "draft_n": 6, "draft_n_accepted": 4}
        tps, rate = acceptance_from_response(t)
        assert tps == pytest.approx(2.0)
        assert rate == pytest.approx(4 / 6)

    def test_degenerate_accepted_ge_predicted(self):
        t = {"predicted_n": 4, "draft_n": 9, "draft_n_accepted": 4}
        assert acceptance_from_response(t) == (None, None)


class TestPrometheus:
    def test_parse_and_deltas(self):
        text = (
            '# HELP llamacpp:spec_decode_num_accepted_tokens_total x\n'
            '# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter\n'
            'llamacpp:spec_decode_num_accepted_tokens_total 7\n'
            'llamacpp:spec_decode_num_draft_tokens_total 20\n'
            'llamacpp:spec_decode_num_drafts_total 10\n'
            'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 5\n'
            'llamacpp:tokens_predicted_total 99\n'
        )
        m = parse_prometheus(text)
        assert m["llamacpp:spec_decode_num_accepted_tokens_total"] == 7
        assert m["llamacpp:spec_decode_num_drafts_total"] == 10
        # labeled series collapse to the bare name (last wins) — the rig
        # only uses the three unlabeled counters
        before = dict(m)
        after = dict(m, **{"llamacpp:spec_decode_num_accepted_tokens_total": 9.0,
                           "llamacpp:spec_decode_num_draft_tokens_total": 26.0,
                           "llamacpp:spec_decode_num_drafts_total": 13.0})
        d_tok, d_acc, d_ver = spec_deltas(before, after)
        assert (d_tok, d_acc, d_ver) == (6.0, 2.0, 3.0)
        tps, rate = acceptance_from_delta(d_tok, d_acc, d_ver)
        assert tps == pytest.approx(1 + 2 / 3)

    def test_missing_counters_treated_as_zero(self):
        d = spec_deltas({}, {"llamacpp:spec_decode_num_drafts_total": 4.0})
        assert d == (0.0, 0.0, 4.0)


class FakeResp(io.BytesIO):
    """urlopen() result stand-in: iterates lines like a socket file."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def sse(chunks):
    out = b""
    for c in chunks:
        out += b"data: " + json.dumps(c).encode() + b"\n\n"
    return out


class TestStreamTTFT:
    def test_ttft_first_content_and_stop_timings(self, monkeypatch):
        chunks = [
            {"content": "", "stop": False},              # open event — NOT ttft
            {"content": "y <- ", "stop": False},         # first content
            {"content": "1", "stop": False},
            {"content": "", "stop": True, "stop_type": "eos",
             "tokens_predicted": 12,
             "timings": {"prompt_ms": 100.0, "predicted_ms": 240.0,
                         "prompt_n": 1811}},
        ]
        seen = {}

        def fake_urlopen(req, timeout=None):
            time.sleep(0.01)
            seen["t"] = time.time()
            return FakeResp(sse(chunks))

        monkeypatch.setattr("spec_bench.urllib.request.urlopen", fake_urlopen)
        res = stream_completion(1, "prompt")  # type: ignore[arg-type]
        assert res.error is None
        assert res.stop_hit is True
        assert res.text == "y <- 1"
        assert res.ttft_ms is not None
        assert res.ttft_ms > 5  # measured after the open event arrived
        assert res.timings["prompt_ms"] == 100.0
        assert res.n_predicted == 12

    def test_error_recorded_not_raised(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            raise OSError("connection reset")

        monkeypatch.setattr("spec_bench.urllib.request.urlopen", fake_urlopen)
        res = stream_completion(1, "prompt")  # type: ignore[arg-type]
        assert res.error is not None
        assert res.text == ""

    def test_done_terminates(self, monkeypatch):
        body = sse([{"content": "hi", "stop": False}]) + b"data: [DONE]\n\n"

        def fake_urlopen(req, timeout=None):
            return FakeResp(body)

        monkeypatch.setattr("spec_bench.urllib.request.urlopen", fake_urlopen)
        res = stream_completion(1, "p")  # type: ignore[arg-type]
        assert res.text == "hi"
        assert res.stop_hit is False


class TestArmFlags:
    def test_baseline_has_no_spec_flags(self):
        model, flags, label = arm_flags("baseline")
        assert flags == [] and label == "baseline"

    def test_ngram_depth_is_size_m(self):
        _, flags, label = arm_flags("ngram-simple@4")
        assert "--spec-type" in flags and "ngram-simple" in flags
        i = flags.index("--spec-ngram-simple-size-m")
        assert flags[i + 1] == "4"
        assert label == "ngram-simple@4"

    def test_draft_mtp_uses_mtp_model_and_n_max(self):
        model, flags, _ = arm_flags("draft-mtp@2")
        assert model.name == "mtp-b4_qwen35_2b-Q8_0.gguf"
        assert "--spec-draft-n-max" in flags and \
            flags[flags.index("--spec-draft-n-max") + 1] == "2"
        assert "--spec-type" in flags and "draft-mtp" in flags

    def test_model_draft_single_mode(self):
        model, flags, _ = arm_flags("model-draft@3")
        assert model.name == "b4_qwen35_2b-Q8_0.gguf"
        assert "draft-simple" in flags
        assert any(f.endswith("b2_qwen35_08b-Q8_0.gguf") for f in flags)
        # single-mode rule: exactly one --spec-type value
        types = [flags[i + 1] for i, f in enumerate(flags)
                 if f == "--spec-type"]
        assert types == ["draft-simple"]

    def test_baseline_rejects_depth(self):
        with pytest.raises(AssertionError):
            arm_flags("baseline@2")

    def test_unknown_arm(self):
        with pytest.raises(ValueError):
            arm_flags("dspark@1")

    def test_constants_match_manifest(self):
        assert MAX_TOKENS == 64
        assert STOP == ">>>>>>> UPDATED"
