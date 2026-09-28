import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "source" / "prepare_extension.py"
SPEC = importlib.util.spec_from_file_location("prepare_extension", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value) -> str:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExtensionScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base_rows = self.root / "base.jsonl"
        self.recovery_rows = self.root / "recovery.jsonl"
        self.combined_rows = self.root / "combined.jsonl"
        self.base_schedule = self.root / "base-schedule.json"
        self.base_cache = self.root / "base-cache-manifest.json"
        self.recovery_result = self.root / "recovery-result.json"
        self.checkpoint = self.root / "checkpoint.json"
        self.output = self.root / "extension-schedule.json"

        base_ids = [f"base-{i}" for i in range(28)]
        old_replay = base_ids[:4]
        self.base_rows.write_text("".join(json.dumps({"schema": 1, "row_id": rid, "cpt_partition": "cpt_train"}) + "\n" for rid in base_ids), encoding="utf-8")
        recovery_ids = [f"new-{i}" for i in range(3)]
        self.recovery_rows.write_text("".join(json.dumps({"schema": 1, "row_id": rid, "cpt_partition": "cpt_train"}) + "\n" for rid in recovery_ids), encoding="utf-8")
        self.combined_rows.write_bytes(self.base_rows.read_bytes() + self.recovery_rows.read_bytes())
        self.base_rows_sha = hashlib.sha256(self.base_rows.read_bytes()).hexdigest()
        self.recovery_rows_sha = hashlib.sha256(self.recovery_rows.read_bytes()).hexdigest()
        self.combined_rows_sha = hashlib.sha256(self.combined_rows.read_bytes()).hexdigest()

        base_schedule = {
            "schema": "sepalith.sft11.cpt-draw-schedule.v1",
            "status": "complete_candidate_pending_root_admission",
            "method": "one_pass_plus_named_replay_v1",
            "seed": 3407,
            "effective_batch": 16,
            "max_steps": 2,
            "token_rows_sha256": self.base_rows_sha,
            "row_ids": base_ids + old_replay,
            "replay_count": 4,
            "replay_row_ids": old_replay,
        }
        self.base_schedule_sha = write_json(self.base_schedule, base_schedule)
        base_cache = {
            "schema": "sepalith.sft11.cpt-streaming-cache.v1",
            "status": "complete",
            "source": {"rows": {"sha256": self.base_rows_sha}},
            "counts": {"rows": 28, "documents": 1},
        }
        self.base_cache_sha = write_json(self.base_cache, base_cache)
        recovery_result = {
            "schema": MODULE.RESULT_SCHEMA,
            "status": "complete",
            "artifacts": {"cpt_train_ctx16384.jsonl": {"sha256": self.recovery_rows_sha}},
            "outputs": {"16384": {"rows": 3}},
        }
        write_json(self.recovery_result, recovery_result)
        checkpoint = {
            "checkpoint_step": 2,
            "source_cursor": 16,
            "checkpoint_id": "synthetic-step-2",
            "source_schedule_sha256": self.base_schedule_sha,
            "global_optimizer_step_offset": 1,
            "source_cursor_kind": "draw_position_exclusive",
        }
        write_json(self.checkpoint, checkpoint)

        # This function binds campaign pins; swap them with the synthetic pins.
        MODULE.BASE_SCHEDULE_SHA256 = self.base_schedule_sha
        MODULE.BASE_CACHE_MANIFEST_SHA256 = self.base_cache_sha
        MODULE.BASE_ROWS_SHA256 = self.base_rows_sha
        MODULE.BASE_ROWS_COUNT = 28
        MODULE.BASE_DOCUMENTS = 1
        MODULE.BASE_REPLAY_COUNT = 4

    def tearDown(self):
        self.temp.cleanup()

    def test_preserves_prefix_drops_old_tail_and_appends_new_tail(self):
        result = MODULE.make_extension_schedule(
            self.base_schedule, self.base_cache, self.base_rows,
            self.recovery_result, self.recovery_rows, self.combined_rows,
            self.checkpoint, self.output,
        )
        schedule = result["schedule"]
        self.assertEqual(schedule["row_ids"], [f"base-{i}" for i in range(28)] + ["new-0", "new-1", "new-2", "new-0"])
        self.assertEqual(schedule["replay_count"], 1)
        self.assertEqual(schedule["source_checkpoint"]["source_cursor"], 16)
        self.assertEqual(schedule["coverage"]["old_alignment_rows_removed"], 4)
        self.assertEqual(schedule["coverage"]["remaining_original_draws_at_cursor"], 12)
        self.assertEqual(schedule["row_ids"][16:28], [f"base-{i}" for i in range(16, 28)])
        self.assertNotIn("base-0", schedule["row_ids"][16:])
        self.assertTrue(schedule["coverage"]["no_consumed_base_draw_replayed"])
        self.assertEqual(schedule["token_rows_sha256"], self.combined_rows_sha)

    def test_checkpoint_must_bind_base_schedule(self):
        value = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        value["source_schedule_sha256"] = "0" * 64
        self.checkpoint.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.ExtensionError, "checkpoint_must_bind_base_schedule_sha256"):
            MODULE.make_extension_schedule(
                self.base_schedule, self.base_cache, self.base_rows,
                self.recovery_result, self.recovery_rows, self.combined_rows,
                self.checkpoint, self.output,
            )

    def test_cursor_after_old_tail_is_rejected(self):
        value = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        value["source_cursor"] = 32
        self.checkpoint.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.ExtensionError, "checkpoint_cursor_must_be_aligned"):
            MODULE.make_extension_schedule(
                self.base_schedule, self.base_cache, self.base_rows,
                self.recovery_result, self.recovery_rows, self.combined_rows,
                self.checkpoint, self.output,
            )

    def test_duplicate_recovery_row_is_rejected(self):
        self.recovery_rows.write_text(json.dumps({"schema": 1, "row_id": "base-1", "cpt_partition": "cpt_train"}) + "\n", encoding="utf-8")
        self.combined_rows.write_bytes(self.base_rows.read_bytes() + self.recovery_rows.read_bytes())
        self.recovery_rows_sha = hashlib.sha256(self.recovery_rows.read_bytes()).hexdigest()
        self.combined_rows_sha = hashlib.sha256(self.combined_rows.read_bytes()).hexdigest()
        write_json(self.recovery_result, {
            "schema": MODULE.RESULT_SCHEMA, "status": "complete",
            "artifacts": {"cpt_train_ctx16384.jsonl": {"sha256": self.recovery_rows_sha}},
            "outputs": {"16384": {"rows": 1}},
        })
        with self.assertRaisesRegex(MODULE.ExtensionError, "recovery_row_id_overlaps_base_schedule"):
            MODULE.make_extension_schedule(
                self.base_schedule, self.base_cache, self.base_rows,
                self.recovery_result, self.recovery_rows, self.combined_rows,
                self.checkpoint, self.output,
            )

    def test_cache_compatibility_adds_canonical_coverage_and_rebinds_metadata(self):
        schedule = self.root / "built-schedule.json"
        compatible = self.root / "compatible-schedule.json"
        cache_manifest = self.root / "cache" / "manifest.json"
        schedule_value = {
            "schema": MODULE.SCHEMA_EXTENSION,
            "token_rows_sha256": "r" * 64,
            "effective_batch": 16,
            "max_steps": 1,
            "row_ids": [f"row-{i}" for i in range(16)],
            "replay_count": 0,
            "replay_row_ids": [],
            "coverage": {"combined_unique_rows": 16},
        }
        schedule_sha = write_json(schedule, schedule_value)
        cache_manifest.parent.mkdir()
        write_json(cache_manifest, {
            "schema": "sepalith.sft11.cpt-streaming-cache.v1",
            "status": "complete",
            "source": {
                "rows": {"sha256": "r" * 64},
                "draw_schedule": {"sha256": schedule_sha},
            },
            "counts": {"rows": 16, "draws": 16},
            "files": {},
        })
        result = MODULE.finalize_cache_schedule_compatibility(schedule, compatible, cache_manifest)
        final_schedule = json.loads(compatible.read_text(encoding="utf-8"))
        final_cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
        self.assertEqual(final_schedule["coverage"]["unique_rows"], 16)
        self.assertEqual(final_schedule["coverage"]["draws"], 16)
        self.assertEqual(final_schedule["coverage"]["updates"], 1)
        self.assertEqual(final_cache["source"]["draw_schedule"]["sha256"], result["schedule"]["sha256"])
        self.assertEqual(final_cache["source"]["draw_schedule"]["path"], str(compatible.resolve()))
        self.assertEqual(final_cache["source"]["schedule_metadata_rebind"]["previous"]["sha256"], schedule_sha)
        self.assertEqual(final_cache["source"]["schedule_metadata_rebind"]["input_schedule"]["sha256"], schedule_sha)

    def test_runtime_checkpoint_rebind_keeps_draw_vector(self):
        built = MODULE.make_extension_schedule(
            self.base_schedule, self.base_cache, self.base_rows,
            self.recovery_result, self.recovery_rows, self.combined_rows,
            self.checkpoint, self.output,
        )
        runtime = self.root / "runtime-bound.json"
        result = MODULE.rebind_runtime_checkpoint(self.output, self.checkpoint, runtime)
        rebound = json.loads(runtime.read_text(encoding="utf-8"))
        self.assertEqual(rebound["row_ids"], built["schedule"]["row_ids"])
        self.assertEqual(rebound["source_checkpoint"]["source_cursor"], 16)
        self.assertEqual(rebound["stage_transition"]["checkpoint_step"], 2)
        self.assertEqual(result["counts"]["updates"], 2)


class ConcatTests(unittest.TestCase):
    def test_concat_is_hash_bound_and_fresh(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            recovery = root / "recovery"
            output = Path("/mnt/e/sepalith/campaign-20260915/data-work") / f".test-cpt-extension-{uuid.uuid4().hex}"
            base.write_bytes(b"base\n")
            recovery.write_bytes(b"recovery\n")
            try:
                result = MODULE.concat_rows(
                    base, recovery, output,
                    base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                    recovery_sha256=hashlib.sha256(recovery.read_bytes()).hexdigest(),
                )
                self.assertEqual(output.read_bytes(), b"base\nrecovery\n")
                self.assertEqual(result["rows"], 2)
                self.assertEqual(result["bytes"], 14)
            finally:
                if output.exists():
                    output.unlink()


if __name__ == "__main__":
    unittest.main()
