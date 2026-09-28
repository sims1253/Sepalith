#!/usr/bin/env python3
import hashlib
import json
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

PACKET = Path(__file__).resolve().parents[1]
SRC = PACKET / "source" / "experiments" / "training"
sys.path.insert(0, str(SRC))

import signal_pilot_driver as d
import signal_pilot_harness as h
from campaign_r_parse_probe import ParseProbeError
from campaign_tokenizer_contract import restore_pinned_tokenizer_contract


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

    def test_repaired_finish_modes_overlay_exactly_eight_stale_selection_labels(self):
        overlays = [g for g in self.groups if g.get("selection_syntax_evidence_mode") == "framed_fragment"]
        self.assertEqual(len(overlays), 8)
        self.assertTrue(all(g["family"] == "finish_block" and g["syntax_evidence_mode"] == "completion_prefix"
                            for g in overlays))

    def test_live_framework_import_order_is_unsloth_then_torch(self):
        calls = []
        def importer(name, *args, **kwargs):
            calls.append(name)
            if name == "unsloth": return SimpleNamespace(FastLanguageModel=object())
            if name == "torch": return object()
            raise AssertionError(name)
        d.import_live_frameworks(importer)
        self.assertEqual(calls, ["unsloth", "torch"])

    def test_non_none_wrong_pad_is_repaired_without_vocab_change(self):
        vocab = {f"t{i}": i for i in range(130560)}
        del vocab["t0"]; del vocab["t1"]
        vocab["<s>"] = 0; vocab["</s>"] = 1
        class Tokenizer:
            def __init__(self, pad):
                self.vocab = vocab; self.bos_token_id = 0; self.eos_token_id = 1
                self.eos_token = "</s>"; self._pad_token = next(k for k, v in vocab.items() if v == pad)
                self.pad_token_id = pad
            def __len__(self): return len(self.vocab)
            def get_vocab(self): return dict(self.vocab)
            def convert_ids_to_tokens(self, value): return next(k for k, v in self.vocab.items() if v == value)
            @property
            def pad_token(self): return self._pad_token
            @pad_token.setter
            def pad_token(self, value): self._pad_token = value; self.pad_token_id = self.vocab[value]
        class Embedding:
            padding_idx = 2
        class Model:
            config = SimpleNamespace(bos_token_id=0, eos_token_id=[1, 130073], pad_token_id=2)
            generation_config = SimpleNamespace(bos_token_id=0, eos_token_id=[1, 130073], pad_token_id=2)
            def modules(self): return [Embedding()]
        loaded, reference, model = Tokenizer(2), Tokenizer(1), Model()
        audit = restore_pinned_tokenizer_contract(model, loaded, reference_tokenizer=reference)
        self.assertEqual((loaded.bos_token_id, loaded.eos_token_id, loaded.pad_token_id), (0, 1, 1))
        self.assertEqual((model.config.pad_token_id, model.generation_config.pad_token_id), (1, 1))
        self.assertTrue(audit["vocab_mapping_unchanged"])
        self.assertFalse(audit["added_tokens"])

    def test_config_rejects_old_unrepaired_rows_binding(self):
        bad = copy.deepcopy(self.config)
        bad["training_data_binding"]["rows_sha256"] = "65b2feb2e53970f02e7cfbe8947d628d584c204f741dad3a5a8b3254af5c1cd7"
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bad.json"; path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(h.PilotError, "repaired TRAIN/reward binding differs"):
                d.load_config(path)

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
