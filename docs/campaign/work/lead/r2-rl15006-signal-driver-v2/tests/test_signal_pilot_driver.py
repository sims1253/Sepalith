#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

PACKET = Path(__file__).resolve().parents[1]
SRC = PACKET / "source" / "experiments" / "training"
sys.path.insert(0, str(SRC))

import signal_pilot_driver as d
import signal_pilot_harness as h
from campaign_r_parse_probe import ParseProbeError


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = d.load_config(PACKET / "driver-config.template.json")
        cls.spec, cls.spec_sha, cls.groups = h.load_spec(Path(cls.config["pilot_spec"]["path"]))
        cls.prepared, cls.buffer = d.load_pilot_data(cls.spec, cls.groups)
        cls.identity = {"identity_sha256": "a" * 64, "manifest_sha256": "b" * 64}

    @staticmethod
    def generator(prompt, seed):
        del seed
        return [*prompt, 1, 1, 1]  # canonical EOS plus framework padding

    def reward(self, item, ids):
        mode = item["group"]["syntax_evidence_mode"]
        return 0.0, {
            "output_ids_sha256": d._ids_hash(ids), "protocol_valid": False,
            "failure": "parse_error:fixture", "exact_region": False,
            "false_noop_edit": False,
            "repetition": {"detected": False, "block_lines": 0, "copies": 0, "covered_lines": 0},
            "candidate_parse": "unavailable_unverified_buffer" if mode == "unverified" else "not_checked",
            "syntax_evidence_mode": mode,
        }

    def run_pilot(self, output, generator=None, reward=None):
        return d.run_groups(
            spec=self.spec, spec_sha=self.spec_sha, groups=self.groups,
            prepared=self.prepared, output=Path(output), model_identity=self.identity,
            generator=generator or self.generator, rewarder=reward or self.reward,
        )

    def test_full_112_group_448_candidate_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            summary = self.run_pilot(root)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["denominators"]["prompt_groups"], 112)
            self.assertEqual(summary["denominators"]["raw_outputs"], 448)
            self.assertEqual(len(list((Path(root) / "groups").glob("[0-9]*"))), 112)

    def test_interruption_resumes_at_first_uncommitted_group_with_same_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def interrupted(prompt, seed):
                calls.append(seed)
                if len(calls) == 5:
                    raise RuntimeError("interrupted")
                return self.generator(prompt, seed)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_pilot(root, generator=interrupted)
            self.assertTrue((Path(root) / "groups" / "000000").is_dir())
            self.assertFalse((Path(root) / "groups" / "000001").exists())
            resumed = []
            def resume(prompt, seed):
                resumed.append(seed)
                return self.generator(prompt, seed)
            self.run_pilot(root, generator=resume)
            self.assertEqual(resumed[0], d.candidate_seed(self.spec["selection"]["seed"], 1, 0))

    def test_reward_ids_must_match_generation_before_atomic_commit(self):
        with tempfile.TemporaryDirectory() as root:
            def wrong(item, ids):
                _, record = self.reward(item, ids)
                record["output_ids_sha256"] = "0" * 64
                return 0.0, record
            with self.assertRaisesRegex(h.PilotError, "reward output IDs differ"):
                self.run_pilot(root, reward=wrong)
            self.assertFalse((Path(root) / "groups" / "000000").exists())

    def test_parser_infrastructure_failure_emits_no_reward_group(self):
        with tempfile.TemporaryDirectory() as root:
            def failed(item, ids):
                raise ParseProbeError("R unavailable")
            with self.assertRaises(ParseProbeError):
                self.run_pilot(root, reward=failed)
            self.assertFalse((Path(root) / "groups" / "000000").exists())
            failure = json.loads(next((Path(root) / "infrastructure-failures").glob("*.json")).read_text())
            self.assertFalse(failure["reward_emitted"])

    def test_completion_over_cap_fails_without_group(self):
        with tempfile.TemporaryDirectory() as root:
            def too_long(prompt, seed):
                return [*prompt, *([2] * 1025)]
            with self.assertRaisesRegex(h.PilotError, "exceeded completion cap"):
                self.run_pilot(root, generator=too_long)
            self.assertFalse((Path(root) / "groups" / "000000").exists())

    def test_model_binding_rehashes_real_artifacts_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value); model = root / "model"; model.mkdir()
            (root / "manifest.json").write_text('{"status":"complete"}\n')
            payloads = {
                "config.json": json.dumps({"vocab_size": 130560}).encode(),
                "generation_config.json": b"{}\n", "tokenizer.json": b"{}\n",
                "tokenizer_config.json": b"{}\n", "model.safetensors": b"weights",
            }
            artifacts = []
            for name, raw in payloads.items():
                (model / name).write_bytes(raw)
                artifacts.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            weights = [x for x in artifacts if x["path"].endswith(".safetensors")]
            binding = {
                "manifest_path": str(root / "manifest.json"),
                "manifest_sha256": h.sha256(root / "manifest.json"), "model_path": str(model),
                "artifacts": artifacts,
                "merged_weights_sha256": hashlib.sha256(h.canonical(weights)).hexdigest(),
                "tokenizer_json_sha256": next(x["sha256"] for x in artifacts if x["path"] == "tokenizer.json"),
            }
            self.assertEqual(d.verify_model(binding)["merged_weights_sha256"], binding["merged_weights_sha256"])
            (model / "tokenizer.json").write_bytes(b"tampered")
            with self.assertRaisesRegex(h.PilotError, "model artifact differs"):
                d.verify_model(binding)


if __name__ == "__main__":
    unittest.main()
