"""Card C04: schedule v2 coverage and exposure caps, the v2 binder, and the bound artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sepalith.ops import night_queue
from sepalith.training.sft import c04_jobs
from sepalith.training.sft import full_weight_edit_sft as trainer
from sepalith.training.sft import recipe_v2
from sepalith.training.sft import schedule_v2
from sepalith.training.sft.campaign_edit_data import inspect
from sepalith.training.supervisors import request_graceful_stop

REPO = Path(__file__).resolve().parents[4]
DEPLOY = Path("/home/m0hawk/sandbox/Sepalith")
C04 = REPO / "docs/training/plan-20260927/c04"
ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/train-token-rows.jsonl")
ROWS_SHA = "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e"
ARMS = {"lr3e-6": 3e-6, "lr1e-5": 1e-5, "lr3e-5": 3e-5, "smoke-lr1e-5": 1e-5}
OLD_COVERAGE_FIELDS = {"eligible_rows", "eligible_edit_rows", "eligible_noop_rows", "distinct_rows_drawn", "first_all_rows_step",
                       "min_row_exposures", "max_row_exposures", "last_unique_draw_by_pool", "first_replay_draw_by_pool",
                       "replay_draws_by_reason", "edit_chunk_batches", "exposure_histogram"}
PATHS = recipe_v2.PathMap(DEPLOY, REPO)


def synthetic(edits, noops, seed=7):
    rows = []
    for index in range(edits + noops):
        total = 100 + (index * 37) % 900
        rows.append({"row_id": f"r{index:04d}", "family": "no_op" if index >= edits else "finish_block", "package_id": "p",
                     "semantic_noop": index >= edits, "prompt_tokens": total - 10, "target_body_tokens": 5,
                     "target_protocol_tokens": 4, "supervised_target_tokens": 10, "total_tokens": total,
                     "sequence_bucket": 512 if total <= 512 else 1024})
    return rows


class ScheduleTests(unittest.TestCase):
    def check(self, result, edits, noops, updates, edit_replays, noop_hist):
        coverage = result["coverage"]
        self.assertTrue(OLD_COVERAGE_FIELDS <= set(coverage))
        self.assertEqual(result["max_steps"], updates)
        self.assertEqual(coverage["exposure_histogram_by_pool"]["noop"], plain(noop_hist))
        self.assertEqual(coverage["exposure_histogram_by_pool"]["edit"],
                         plain({1: edits - edit_replays, **({2: edit_replays} if edit_replays else {})}))
        self.assertEqual(coverage["edit_replays"], edit_replays)
        self.assertLessEqual(edit_replays, 12)
        flags = {k: v for k, v in result["checks"].items() if k != "targets_over_1024_retained"}
        self.assertTrue(all(v is True for v in flags.values()), result["checks"])
        draws = result["draws"]
        for update in range(1, updates + 1):
            batch = draws[(update - 1) * 16: update * 16]
            self.assertEqual(sum(d["semantic_noop"] for d in batch), 3)
            if update < updates:
                self.assertTrue(all(d["presentation"] == 1 for d in batch if not d["semantic_noop"]))
        # No-op passes: pass k+1 starts only after every row was drawn k times.
        seen = {}
        for draw in (d for d in draws if d["semantic_noop"]):
            count = seen.get(draw["row_id"], 0) + 1
            if count > 1:
                self.assertEqual(len(seen), noops)
                self.assertGreaterEqual(min(seen.values()), count - 1)
            seen[draw["row_id"]] = count

    def test_small_cohort_composition_and_caps(self):
        result = plain(schedule_v2.build(synthetic(30, 7), [], seed=7))
        # 30 edits / 13 per update -> 3 updates, 9 edit replays in the last; 9 no-op draws over 7 rows.
        self.check(result, 30, 7, 3, 9, {1: 5, 2: 2})
        self.assertEqual(result["policy"]["max_noop_exposures"], 2)
        self.assertEqual(max(int(k) for k in result["coverage"]["exposure_histogram_by_pool"]["noop"]), 2)

    @unittest.skipUnless(ROWS.is_file(), "campaign TRAIN rows are not mounted")
    def test_real_cohort_matches_the_card_and_the_bound_schedule(self):
        rows, excluded = schedule_v2.load_rows(ROWS, ROWS_SHA, [512, 1024, 2048, 4096, 8192, 16384])
        result = plain(schedule_v2.build(rows, excluded))
        self.check(result, 13912, 1094, 1071, 11, {2: 69, 3: 1025})
        coverage = result["coverage"]
        self.assertEqual(coverage["exposure_histogram"], plain({1: 13901, 2: 80, 3: 1025}))
        self.assertEqual((coverage["noop_draws"], coverage["noop_share"]), (3213, 0.1875))
        self.assertEqual(result["policy"]["gate_updates"], [268, 536, 803, 1071])
        self.assertLess(coverage["last_unique_draw_by_pool"]["noop"], coverage["first_replay_draw_by_pool"]["noop"])
        self.assertLess(coverage["last_unique_draw_by_pool"]["edit"], coverage["first_replay_draw_by_pool"]["edit"])
        # The bound schedule file is exactly what this code produces.
        template = json.loads((C04 / "sft-v2-recipe.template.json").read_text())
        record = template["cohort"]["draw_schedule"]
        result["input"] = {"path": str(ROWS), "sha256": ROWS_SHA, "rows_seen": 15006,
                           "eligible_row_ids_sha256": hashlib.sha256(("\n".join(result["row_ids"]) + "\n").encode()).hexdigest()}
        produced = hashlib.sha256((json.dumps(result, indent=2) + "\n").encode()).hexdigest()
        self.assertEqual(produced, record["sha256"])
        # ... and the trainer's reader accepts it.
        loaded, draws, schedule, totals = inspect({"path": str(ROWS), "sha256": ROWS_SHA}, record, 16384, 16)
        self.assertEqual((len(loaded), len(draws), schedule["max_steps"]), (15006, 17136, 1071))
        self.assertEqual(totals, {k: template["cohort"][k] for k in ("input_tokens", "loss_tokens", "body_tokens")})

    def test_gate_updates_are_quarters(self):
        self.assertEqual(schedule_v2.gate_updates(1071), [268, 536, 803, 1071])
        self.assertEqual(schedule_v2.gate_updates(1033), [258, 516, 775, 1033])


def plain(value):
    """JSON round trip, so the tests index records without static type noise."""
    return json.loads(json.dumps(value))


def localized(path):
    return PATHS.localize(json.loads(Path(path).read_text()))


class BoundArtifactTests(unittest.TestCase):
    def test_template_pre_registers_card_values(self):
        template = json.loads((C04 / "sft-v2-recipe.template.json").read_text())
        registered = template["pre_registered"]
        self.assertEqual((registered["warmup_steps"], registered["min_lr_rate"], registered["weight_decay"], registered["side_lr_ratio"]),
                         (32, 0.1, 0.0, 0.1))
        self.assertEqual((registered["learning_rate_upper_bound"], registered["cpt_hidden_learning_rate"]), (3e-5, 3e-6))
        self.assertIn("ten times the CPT hidden LR", registered["learning_rate_upper_bound_rationale"])
        self.assertEqual(registered["pilot_learning_rates"], [3e-6, 1e-5, 3e-5])
        self.assertEqual(template["cohort"]["updates"], 1071)
        self.assertEqual(template["dev_panels"]["dev250"]["manifest"]["sha256"],
                         "07e370a8675f0c2578f822950df99a59ae704b48e17786013dccf638faba230a")
        self.assertEqual(template["dev_panels"]["dev250"]["sha256"], "75f3a5ee216400fdfe1e2fb1fb63224374f418d1c248a1c278ae0d09be487f22")

    def test_admission_template_is_example_only(self):
        skeleton = json.loads((C04 / "admissions/arm-admission.template.json").read_text())
        self.assertTrue(recipe_v2._markers(skeleton))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps(dict(skeleton, status="admitted", launch_authorized=True)))
            with self.assertRaisesRegex(ValueError, "EXAMPLE_ONLY or FILL"):
                recipe_v2.bind(C04 / "sft-v2-recipe.template.json", path, Path(tmp) / "out.json",
                               source_manifest=C04 / "bound/source-manifest.json", paths=PATHS)

    def test_bound_recipes_match_code_template_and_admissions(self):
        outputs = set()
        for arm, learning_rate in ARMS.items():
            recipe = localized(C04 / f"bound/sft-v2-{arm}.json")
            recipe_v2.validate_pre_registered(recipe)
            for key in ("root_admission", "template"):
                self.assertEqual(recipe_v2.sha256(recipe[key]["path"]), recipe[key]["sha256"], key)
            recipe_v2.verify_source(recipe["source"])  # the bound source manifest equals this tree
            runtime = recipe["runtime"]
            self.assertEqual((runtime["learning_rate"], runtime["optimizer"]["hidden_lr"]), (learning_rate, learning_rate))
            self.assertAlmostEqual(runtime["optimizer"]["side_lr"], learning_rate / 10, delta=1e-15)
            self.assertEqual(recipe["dev_gate"]["mandatory_steps"], [268, 536, 803, 1071])
            self.assertEqual(recipe["parent"]["files"]["model.safetensors"],
                             "4bf617f9ea66a31e547397b93f0998372686e076d03ba22961780abd285a3c51")
            self.assertEqual(recipe["arm"]["name"], arm)
            outputs.add(recipe["outputs"]["archive"])
            kwargs = trainer.training_arguments_kwargs(recipe, Path("/tmp/x"))
            self.assertEqual((kwargs["warmup_steps"], kwargs["lr_scheduler_type"], kwargs["lr_scheduler_kwargs"], kwargs["max_steps"]),
                             (32, "cosine_with_min_lr", {"min_lr_rate": 0.1}, 1071))
        self.assertEqual(len(outputs), len(ARMS))
        identities = {json.dumps(trainer.identity(localized(C04 / f"bound/sft-v2-{arm}.json")), sort_keys=True) for arm in ARMS}
        self.assertEqual(len(identities), len(ARMS))

    @unittest.skipUnless(ROWS.is_file(), "campaign data is not mounted")
    def test_load_bound_v2_with_a_stand_in_parent(self):
        # The real parent is 5 GB; everything else is verified for real.
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent"
            parent.mkdir()
            recipe = localized(C04 / "bound/sft-v2-lr1e-5.json")
            for name in recipe["parent"]["files"]:
                (parent / name).write_text(name)
                recipe["parent"]["files"][name] = recipe_v2.sha256(parent / name)
            recipe["parent"]["path"] = str(parent)
            path = Path(tmp) / "recipe.json"
            path.write_text(json.dumps(recipe))
            loaded = trainer.load_bound(path)
            self.assertEqual(loaded["schema"], recipe_v2.BOUND_SCHEMA)
            self.assertEqual(trainer.preflight(path)["updates"], 1071)
            for mutate, message in ((lambda r: r["runtime"].update(learning_rate=4e-5), "LR"),
                                    (lambda r: r["runtime"].update(warmup_steps=30), "LR schedule"),
                                    (lambda r: r["runtime"]["optimizer"].update(side_lr=1e-5), "side LR"),
                                    (lambda r: r["runtime"].update(evaluation_steps=[290, 580, 870, 1071]), "gate updates"),
                                    (lambda r: r["dev_gate"]["panels"]["dev250"].update(sha256="0" * 64), "input differs")):
                bad = copy.deepcopy(recipe)
                mutate(bad)
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, message):
                    trainer.load_bound(path)

    def test_binder_bounds_and_lr_rationale(self):
        template = C04 / "sft-v2-recipe.template.json"
        base = json.loads((C04 / "admissions/lr1e-5.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for learning_rate, ok in ((3e-5, True), (3.0000001e-5, False), (0.0, False), (1, False)):
                admission = copy.deepcopy(base)
                admission["arm"]["learning_rate"] = learning_rate
                path = Path(tmp) / f"a-{learning_rate}.json"
                path.write_text(json.dumps(admission))
                output = Path(tmp) / f"bound-{learning_rate}.json"
                if ok:
                    bound = plain(recipe_v2.bind(template, path, output, source_manifest=C04 / "bound/source-manifest.json", paths=PATHS))
                    self.assertEqual(bound["runtime"]["optimizer"]["side_lr"], 3e-6)
                else:
                    with self.assertRaisesRegex(ValueError, "editing LR"):
                        recipe_v2.bind(template, path, output, source_manifest=C04 / "bound/source-manifest.json", paths=PATHS)
            admission = copy.deepcopy(base)
            admission["template_sha256"] = "0" * 64
            path = Path(tmp) / "wrong-template.json"
            path.write_text(json.dumps(admission))
            with self.assertRaisesRegex(ValueError, "template hash"):
                recipe_v2.bind(template, path, Path(tmp) / "x.json", source_manifest=C04 / "bound/source-manifest.json", paths=PATHS)
        self.assertIn("ten times the CPT hidden LR", " ".join((recipe_v2.__doc__ or "").split()))

    def test_a_corrected_row_file_needs_only_a_new_data_queue(self):
        cohort = {"rows": {"sha256": "c" * 64}, "unique_rows": 15006}
        with tempfile.TemporaryDirectory() as tmp:
            queue = json.loads((REPO / "docs/campaign/work/lead/r2-full-weight-edit-sft-eval-gate-v1/data-queue.json").read_text())
            path = Path(tmp) / "queue.json"
            path.write_text(json.dumps(queue))
            record = {"path": str(path), "sha256": recipe_v2.sha256(path), "additional_roxy_rows": 10017,
                      "additional_roxy_status": "pending_root_admission"}
            with self.assertRaisesRegex(ValueError, "does not accept this row file"):
                recipe_v2.verify_data_queue_v2(record, cohort)
            queue["active_cohort"]["rows_sha256"] = "c" * 64
            path.write_text(json.dumps(queue))
            record["sha256"] = recipe_v2.sha256(path)
            recipe_v2.verify_data_queue_v2(record, cohort)

    def test_graceful_stop_request_knows_the_v2_schema(self):
        self.assertEqual(request_graceful_stop.TRAINERS[recipe_v2.BOUND_SCHEMA], "sepalith.training.sft.full_weight_edit_sft")


class JobFileTests(unittest.TestCase):
    def test_job_files_are_valid_and_point_at_bound_recipes(self):
        expected = plain(c04_jobs.jobs(DEPLOY))
        files = sorted((C04 / "jobs").glob("*.json"))
        self.assertEqual(sorted(path.stem for path in files), sorted(expected))
        for path in files:
            job = json.loads(path.read_text())
            self.assertEqual(job, expected[path.stem])
            night_queue.validate_job(copy.deepcopy(job))
            recipe = Path(job["command"][job["command"].index("--recipe") + 1])
            self.assertTrue(PATHS.to_local(str(recipe)).is_file(), recipe)
            self.assertIn("sepalith.training.sft.night_arm", job["command"])
        pilots = [expected[f"C08-arm-{arm}"] for arm in ("lr1e-5", "lr3e-6", "lr3e-5")]
        self.assertEqual([job["priority"] for job in pilots], [20, 21, 22])
        self.assertTrue(all(job["resumable"] and job["requires"] == ["C07-baseline"] and "--stop-after-gate" in job["command"]
                            for job in pilots))
        self.assertFalse(expected["C07-baseline"]["resumable"])


if __name__ == "__main__":
    unittest.main()
