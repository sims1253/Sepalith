#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("root_submit_diagnostic", HERE / "root_submit_diagnostic.py")
submit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(submit)


class MockConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DiagnosticSubmitTest(unittest.TestCase):
    now = 2_000_000_000.0
    run_id = "a" * 32
    token = "hf_PrivateToken123"

    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix="diagnostic-submit-test-")
        root = Path(temporary.name)
        manifest = json.loads((HERE / "payload-manifest.json").read_text())
        for record in manifest["files"]:
            source = HERE / record["path"]
            destination = root / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(HERE / "payload-manifest.json", root / "payload-manifest.json")
        shutil.copyfile(HERE / "root-review.json", root / "root-review.json")
        availability = {"observed_epoch": self.now - 1, "balance_usd": 94.04391707,
                        "spent_usd": 5.95608293, "fleet_epoch": self.now - 1,
                        "nodes_total": 0, "groups_count": 0}
        (root / "root-availability.json").write_text(json.dumps(availability) + "\n")
        deadline = submit.stamp(self.now + 895)
        armed_at = submit.stamp(self.now - 5)
        armed = {"name": "sepalith-cpt-" + self.run_id, "deadline": deadline, "pid": 12345}
        (root / "watchdog-armed.json").write_text(json.dumps(armed) + "\n")
        admission = {
            "schema": "sepalith.cloud-cpt.bootstrap-diagnostic-root-admission.v1",
            "status": "admitted_bootstrap_diagnostic", "admitted": True, "paid_launch_authorized": True,
            "at": submit.stamp(self.now - 1), "run_id": self.run_id,
            "payload_manifest_sha256": submit.PAYLOAD_MANIFEST_SHA256,
            "root_review_sha256": submit.ROOT_REVIEW_SHA256,
            "root_availability_sha256": submit.sha(root / "root-availability.json"),
            "max_incremental_charge_usd": 28.0, "campaign_ceiling_usd": 60.0,
            "resource": {"provider": "Anyscale Hosted", "instance_type": "g5.2xlarge",
                         "head_nodes": 1, "worker_nodes": 0, "provider_seconds": 600,
                         "watchdog_seconds": 900, "max_retries": 0},
            "watchdog_identity": {"pid": 12345, "start_tick": "987654", "name": "sepalith-cpt-" + self.run_id,
                                  "deadline": deadline},
        }
        (root / "root-diagnostic-admission.json").write_text(json.dumps(admission) + "\n")
        binding = {
            "schema": "sepalith.cloud-cpt.bootstrap-diagnostic.v2", "admitted": True,
            "run_id": self.run_id, "image_uri": "docker.io/anyscale/ray@sha256:3b904e7cbb1736a8a17359708197d62333daa0bc8285d7de66f1d22aa78754f9",
            "instance_type": "g5.2xlarge", "artifact_repo": "scholzmx/sepalith-lora",
            "artifact_prefix": "r2-cpt-bootstrap-diagnostic/" + self.run_id,
            "provider_timeout_seconds": 600, "watchdog_timeout_seconds": 900, "max_retries": 0,
            "training_authorized": False, "input_staging_authorized": False,
            "payload_manifest_sha256": submit.PAYLOAD_MANIFEST_SHA256,
            "root_admission_relative_path": "root-diagnostic-admission.json",
            "root_admission_sha256": submit.sha(root / "root-diagnostic-admission.json"),
            "watchdog_armed_receipt_relative_path": "watchdog-armed.json",
            "watchdog_armed_receipt_sha256": submit.sha(root / "watchdog-armed.json"),
            "watchdog_armed_at_utc": armed_at, "absolute_deadline_utc": deadline,
        }
        (root / "binding.root.json").write_text(json.dumps(binding) + "\n")
        live = dict(availability)
        probe = lambda pid: {"pid": pid, "start_tick": "987654",
                             "argv": ["python", "deadline_watchdog.py", "--name", "sepalith-cpt-" + self.run_id,
                                      "--deadline-utc", deadline]}
        return temporary, root, live, probe

    def test_positive_config_and_mocked_submit(self):
        temporary, root, live, probe = self.fixture()
        with temporary:
            admission = json.loads((root / "root-diagnostic-admission.json").read_text())
            self.assertTrue(admission["admitted"] and "diagnostic" in admission["status"].lower())  # Remote bootstrap admission contract.
            captured = []
            receipt = submit.submit_once(root, self.token, self.now, live, MockConfig,
                                         lambda config: captured.append(config) or "prodjob_mock123", probe)
            self.assertEqual(receipt["provider_timeout_seconds"], 600)
            config = captured[0].kwargs
            self.assertEqual(config["entrypoint"], "bash payload/diagnostic-root-bound-entry.sh")
            self.assertEqual(config["env_vars"], {"HF_TOKEN": self.token, "PYTHONDONTWRITEBYTECODE": "1",
                                                   "SEPALITH_CPT_DIAGNOSTIC_BINDING": "binding.root.json"})
            self.assertEqual(config["compute_config"], {"head_node": {"instance_type": "g5.2xlarge"}, "worker_nodes": []})
            self.assertEqual((config["timeout_s"], config["max_retries"]), (600, 0))
            self.assertTrue((root / submit.MARKER_NAME).exists())
            self.assertNotIn(self.token, (root / submit.MARKER_NAME).read_text())
            self.assertNotIn(self.token, (root / "diagnostic-submission-receipt.json").read_text())

    def test_payload_hash_change_rejected_before_submit(self):
        temporary, root, live, probe = self.fixture()
        with temporary:
            with (root / "payload/bootstrap_diagnostic.py").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "payload file hash"):
                submit.prepare(root, self.now, live, probe)

    def test_spend_plus_reservation_ceiling_rejected(self):
        temporary, root, live, probe = self.fixture()
        with temporary:
            live["spent_usd"] = 33.0
            with self.assertRaisesRegex(ValueError, "USD60"):
                submit.prepare(root, self.now, live, probe)

    def test_nonempty_fleet_rejected(self):
        temporary, root, live, probe = self.fixture()
        with temporary:
            live["nodes_total"] = 1
            with self.assertRaisesRegex(ValueError, "fleet"):
                submit.prepare(root, self.now, live, probe)

    def test_stale_live_billing_rejected(self):
        temporary, root, live, probe = self.fixture()
        with temporary:
            live["observed_epoch"] = self.now - 121
            with self.assertRaisesRegex(ValueError, "billing observation"):
                submit.prepare(root, self.now, live, probe)

    def test_watchdog_start_tick_rejected(self):
        temporary, root, live, probe = self.fixture()
        with temporary:
            bad = lambda pid: {**probe(pid), "start_tick": "987655"}
            with self.assertRaisesRegex(ValueError, "PID/start tick"):
                submit.prepare(root, self.now, live, bad)

    def test_prior_submission_marker_rejected(self):
        temporary, root, live, probe = self.fixture()
        with temporary:
            (root / submit.MARKER_NAME).write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "already exists"):
                submit.prepare(root, self.now, live, probe)

    def test_binding_scope_expansion_rejected(self):
        temporary, root, live, probe = self.fixture()
        with temporary:
            path = root / "binding.root.json"
            binding = json.loads(path.read_text())
            binding["max_retries"] = 1
            path.write_text(json.dumps(binding) + "\n")
            with self.assertRaisesRegex(ValueError, "lifecycle bound"):
                submit.prepare(root, self.now, live, probe)

    def test_provider_failure_does_not_record_token_or_retry(self):
        temporary, root, live, probe = self.fixture()
        with temporary:
            def fail(_):
                print(self.token)
                raise RuntimeError(self.token)
            with self.assertRaises(RuntimeError):
                submit.submit_once(root, self.token, self.now, live, MockConfig, fail, probe)
            error = (root / "diagnostic-submission-error.json").read_text()
            self.assertNotIn(self.token, error)
            self.assertEqual(json.loads(error)["status"], "submission_failed_no_retry_authorized")
            self.assertTrue((root / submit.MARKER_NAME).exists())


if __name__ == "__main__":
    unittest.main()
