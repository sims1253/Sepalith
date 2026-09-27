#!/usr/bin/env python3
"""Offline contract tests; they never launch a server or read final data."""

from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> int:
    binder = load("binder", "bind_packet.py")
    runner = load("runner", "run_arm.py")
    six = load("six", "prepare_six_thread.py")
    template = json.loads((HERE / "packet.template.json").read_text())
    probe_path = (HERE / template["panel"]["probe"]).resolve()
    spec = importlib.util.spec_from_file_location("probe", probe_path)
    probe = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(probe)
    assert template["resource"]["maximum_threads"] == 2
    assert template["resource"]["suite_max_seconds"] <= 22 * 60
    assert template["remote"]["port"] != template["preserve"]["editor_port"]
    assert template["v1"]["requests_per_arm"] == template["v1"]["rows"] * 2
    assert template["v2_later"]["authorized_now"] is False
    rows, manifest = probe.load_panel((HERE / template["panel"]["path"]).resolve(),
                                      (HERE / template["panel"]["manifest"]).resolve(), 8)
    assert len(rows) == 8
    assert sum(row["target_operation"] == "no_op" for row in rows) == 4
    assert {row["family"] for row in rows if row["target_operation"] != "no_op"} == {
        "format_propagation", "na_rm_propagation", "pipe_rewrite", "rename_propagation"
    }
    synthetic = []
    for row in rows:
        synthetic.append({"row_id": row["id"], "phase": "cold", "raw_text": row["target_text"],
                          "parsed_output": {"operation": row["target_operation"]}, "protocol_status": "accepted"})
        synthetic.append({"row_id": row["id"], "phase": "warm", "raw_text": "different",
                          "parsed_output": {"operation": row["target_operation"]}, "protocol_status": "accepted"})
    score = runner.quality(rows, synthetic)
    assert score["denominator"] == 8
    assert score["strict_exact_target_text"] == 8
    assert score["operation_match"] == 8
    assert score["protocol_accepted"] == 8
    assert score["strict_no_edit_target_denominator"] == 4
    assert score["strict_no_edit_target_exact"] == 4
    assert score["context_aware_false_suggestion_rate"] is None
    assert signal.SIGHUP in runner.OWNED_STOP_SIGNALS
    owned = subprocess.Popen(["sh", "-c", "sleep 60 & wait"], start_new_session=True)
    runner.cleanup_owned_group(owned)
    assert owned.poll() is not None
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "bound.json"
        binder.bind(HERE / "packet.template.json", out, verify_files=False)
        bound = json.loads(out.read_text())
        assert bound["status"] == "root_hashes_bound_no_model_load"
        assert all(len(model["sha256"]) == 64 for model in bound["models"].values())
        try:
            binder.bind(HERE / "packet.template.json", out, verify_files=False)
            raise AssertionError("existing binding output was overwritten")
        except ValueError as exc:
            assert "refusing existing output" in str(exc)
        six_out = Path(directory) / "six.json"
        six_packet = six.prepare(out, six_out)
        assert six_packet["authorized_arms"] == ["q8", "q4_calibrated", "iq3"]
        assert six_packet["resource"]["maximum_threads"] == 6
        assert six_packet["resource"]["suite_max_seconds"] == 990
        assert six_packet["v1"]["root_admission_required"] is True
    print(json.dumps({"status": "pass", "tests": 12, "rows": 8, "final_or_dev_opened": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
