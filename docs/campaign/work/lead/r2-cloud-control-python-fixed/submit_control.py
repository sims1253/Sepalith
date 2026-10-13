"""Submit one root-admitted job. Credentials stay in process/provider memory."""
import contextlib
import datetime
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    import anyscale
    from anyscale.job.models import JobConfig

    payload = ROOT / "payload"
    binding = json.loads((payload / "payload-binding.json").read_text())
    admission = json.loads((ROOT / "admission.json").read_text())
    assert admission["status"] == "admitted"
    assert binding["admitted"] is True
    assert admission["run_id"] == binding["run_id"]
    for relative, expected in admission["payload_hashes"].items():
        assert hashlib.sha256((payload / relative).read_bytes()).hexdigest() == expected
    armed_path = ROOT / "watchdog" / "armed.json"
    assert hashlib.sha256(armed_path.read_bytes()).hexdigest() == binding["watchdog_armed_receipt_sha256"]
    armed = json.loads(armed_path.read_text())
    assert Path(f'/proc/{armed["pid"]}').exists()
    assert not (ROOT / "watchdog" / "terminal.json").exists()
    deadline = datetime.datetime.fromisoformat(binding["absolute_deadline_utc"]).timestamp()
    assert time.time() + 1800 < deadline
    marker = ROOT / "submission-started.json"
    if marker.exists():
        resolution = json.loads((ROOT / "submission-resolution.json").read_text())
        assert resolution["allow_one_diagnostic_retry"] is True
        assert resolution["observed_matching_jobs"] == 0
        assert resolution["run_id"] == binding["run_id"]
        assert time.time() - resolution["observed_epoch"] < 120
        marker = ROOT / "submission-diagnostic-retry-started.json"
    with marker.open("x") as stream:
        json.dump({"at": stamp(), "pid": os.getpid(), "run_id": binding["run_id"]}, stream)
    token = os.environ["HF_TOKEN"]
    config = JobConfig(
        name="sepalith-r2-control-" + binding["run_id"],
        cloud="Anyscale Cloud",
        working_dir=str(payload),
        image_uri=binding["image_uri"],
        ray_version="2.57.0",
        entrypoint="bash root-bound-entry.sh",
        compute_config={"head_node": {"instance_type": "g5.2xlarge"}, "worker_nodes": []},
        max_retries=0,
        timeout_s=6900,
        env_vars={"HF_TOKEN": token, "PYTHONDONTWRITEBYTECODE": "1"},
        excludes=["__pycache__", "*.pyc", ".env", ".git"],
    )
    logging.disable(logging.CRITICAL)
    # SDK output can contain remote request diagnostics. Retain no raw output.
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            job_id = anyscale.job.submit(config)
        assert isinstance(job_id, str) and job_id.startswith("prodjob_")
        receipt = {"at": stamp(), "status": "submitted_not_runtime_accepted", "job_id": job_id,
                   "name": config.name, "run_id": binding["run_id"], "provider_timeout_s": 6900,
                   "absolute_deadline_utc": binding["absolute_deadline_utc"]}
        (ROOT / "submission-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        receipt = {"at": stamp(), "status": "submission_error_outcome_requires_name_lookup",
                   "error_type": type(error).__name__, "run_id": binding["run_id"],
                   "message": str(error).replace(token, "[REDACTED]")[:1500]
                              if isinstance(error, ValueError) else "Provider exception; raw request suppressed",
                   "stack": [{"file": f.filename, "line": f.lineno, "function": f.name}
                             for f in traceback.extract_tb(error.__traceback__)]}
        destination = "submission-diagnostic-error.json" if "diagnostic" in marker.name else "submission-error.json"
        (ROOT / destination).write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "pre_submission_rejected", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)
