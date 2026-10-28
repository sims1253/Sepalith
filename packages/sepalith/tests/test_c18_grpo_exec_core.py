"""Card C18 core (stdlib): prompt mixing, the replay schedule, R6x, selection, plateau, the final verdict,
the pool split, the bound recipes and the job files."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sepalith.training.grpo.algorithm import GRPOError
from sepalith.training.grpo_exec import mix, pool, recipe as recipe_module, rules

REPO = Path(__file__).resolve().parents[3]
BOUND = REPO / "docs/training/plan-20260927/c18/bound"
PILOT = BOUND / "grpo-exec-pilot-1-arm-a-803.json"
SMOKE = BOUND / "grpo-exec-smoke-arm-a-803.json"
JOBS = REPO / "docs/training/plan-20260927/c18/jobs"


class MixTests(unittest.TestCase):
    def setUp(self):
        self.exec_ids = [f"e{i:02d}" for i in range(7)]
        self.legacy_ids = [f"l{i}" for i in range(3)]

    def test_every_fifth_position_is_legacy_and_both_pools_cycle(self):
        sampler = mix.MixedSampler(self.exec_ids, self.legacy_ids, 3407, 5)
        drawn = sampler.draw(40)
        self.assertEqual([p for p, _ in drawn], list(range(40)))
        legacy = [p for p, i in drawn if i.startswith("l")]
        self.assertEqual(legacy, [4, 9, 14, 19, 24, 29, 34, 39])
        exec_draws = [i for _, i in drawn if i.startswith("e")]
        self.assertEqual(sorted(exec_draws[:7]), self.exec_ids)  # one full epoch before any repeat
        self.assertEqual(sampler.state()["legacy_cursor"], 8)
        self.assertEqual(sampler.state()["exec_cursor"], 32)

    def test_resume_from_the_cursor_continues_the_same_stream(self):
        whole = mix.MixedSampler(self.exec_ids, self.legacy_ids, 1, 5).draw(37)
        first = mix.MixedSampler(self.exec_ids, self.legacy_ids, 1, 5)
        head = first.draw(13)
        tail = mix.MixedSampler(self.exec_ids, self.legacy_ids, 1, 5, cursor=first.cursor).draw(24)
        self.assertEqual(head + tail, whole)

    def test_overlapping_pools_and_bad_settings_are_refused(self):
        with self.assertRaises(GRPOError):
            mix.MixedSampler(["a", "b"], ["b"], 1, 5)
        with self.assertRaises(GRPOError):
            mix.MixedSampler(["a"], ["b"], 1, 1)
        with self.assertRaises(GRPOError):
            mix.MixedSampler(["a"], ["b"], 1, 5, cursor=-1)

    def test_replay_schedule_is_stateless_and_walks_epochs(self):
        ids = [f"r{i}" for i in range(10)]
        first = mix.replay_ids(ids, 7, 1, 4) + mix.replay_ids(ids, 7, 2, 4)
        self.assertEqual(len(set(first)), 8)
        self.assertEqual(mix.replay_ids(ids, 7, 2, 4), mix.replay_ids(ids, 7, 2, 4))
        epoch = mix.replay_ids(ids, 7, 1, 5) + mix.replay_ids(ids, 7, 2, 5)
        self.assertEqual(sorted(epoch), sorted(ids))
        with self.assertRaises(GRPOError):
            mix.replay_ids(ids, 7, 0, 4)


def start(**override):
    value = {"valid": 75, "cap": 0, "edit": 28, "nofp": 9, "score250": 109, "val_all_pass": 0.30, "val_mean_reward": 0.38,
             "cpt_nll": 1.0}
    value.update(override)
    return value


def current(**override):
    value = {"valid": 75, "cap": 0, "edit": 28, "nofp": 9, "val_all_pass": 0.31, "val_mean_reward": 0.39, "cpt_nll": 1.005,
             "loss": 0.01}
    value.update(override)
    return value


def window(**override):
    value = {"zero_variance_share_after": 0.0, "completion_tokens_mean_drawn": 100.0}
    value.update(override)
    return value


def apply(cur, prior=(), win=None, previous=None):
    return rules.apply(cur, previous75=previous or start(), start=start(), window=win or window(), prior_windows=list(prior),
                       start_window=window())


def rows(result):
    return {row["name"]: row["outcome"] for row in result["rows"]}


class RuleTests(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual(rules.evaluation_steps()[:3], [50, 100, 150])
        self.assertEqual(rules.evaluation_steps()[-1], 1000)
        self.assertEqual(rules.evaluation_panels(100), ("dev75", "val", "cpt"))
        self.assertEqual(rules.evaluation_panels(150), ("dev75", "val", "cpt", "dev250"))
        self.assertEqual(rules.evaluation_panels(149), ())

    def test_healthy_evaluation_continues(self):
        result = apply(current())
        self.assertEqual(result["outcome"], rules.CONTINUE, result)
        self.assertEqual(set(rows(result)), {"edit_drop", "nofp_rise", "valid_drop", "cap", "val_all_pass_drop",
                                             "val_mean_reward_drop", "cpt_nll_rise", "zero_variance_streak", "length_factor"})

    def test_validation_all_pass_bands(self):
        self.assertEqual(rows(apply(current(val_all_pass=0.2801)))["val_all_pass_drop"], rules.CONTINUE)
        self.assertEqual(rows(apply(current(val_all_pass=0.28)))["val_all_pass_drop"], rules.STOP_FOR_USER)
        self.assertEqual(rows(apply(current(val_all_pass=0.26)))["val_all_pass_drop"], rules.ROLLBACK)
        self.assertEqual(apply(current(val_all_pass=0.26))["outcome"], rules.ROLLBACK)

    def test_validation_reward_bands(self):
        self.assertEqual(rows(apply(current(val_mean_reward=0.36)))["val_mean_reward_drop"], rules.STOP_FOR_USER)
        self.assertEqual(rows(apply(current(val_mean_reward=0.34)))["val_mean_reward_drop"], rules.ROLLBACK)

    def test_cpt_forgetting_bands(self):
        self.assertEqual(rows(apply(current(cpt_nll=1.015)))["cpt_nll_rise"], rules.CONTINUE)
        self.assertEqual(rows(apply(current(cpt_nll=1.02)))["cpt_nll_rise"], rules.STOP_FOR_USER)
        self.assertEqual(rows(apply(current(cpt_nll=1.0301)))["cpt_nll_rise"], rules.ROLLBACK)

    def test_score250_against_the_start_at_dev250_steps(self):
        self.assertEqual(rows(apply(current(score250=105)))["score250_drop_from_start"], rules.CONTINUE)
        self.assertEqual(rows(apply(current(score250=104)))["score250_drop_from_start"], rules.STOP_FOR_USER)
        self.assertEqual(rows(apply(current(score250=99)))["score250_drop_from_start"], rules.ROLLBACK)
        self.assertNotIn("score250_drop_from_start", rows(apply(current())))

    def test_r1_rows_compare_with_the_previous_evaluation(self):
        result = apply(current(nofp=11), previous=start(nofp=10))
        self.assertEqual(rows(result)["nofp_rise"], rules.CONTINUE)
        self.assertEqual(rows(apply(current(valid=67)))["valid_drop"], rules.ROLLBACK)

    def test_zero_variance_streak_and_length(self):
        high = window(zero_variance_share_after=0.6)
        self.assertEqual(rows(apply(current(), prior=[high, high], win=high))["zero_variance_streak"], rules.STOP_FOR_USER)
        self.assertEqual(rows(apply(current(), prior=[high], win=high))["zero_variance_streak"], rules.CONTINUE)
        self.assertEqual(rows(apply(current(), win=window(completion_tokens_mean_drawn=201.0)))["length_factor"], rules.STOP_FOR_USER)

    def test_missing_inputs_stop_for_the_user(self):
        bad = current()
        del bad["cpt_nll"]
        result = apply(bad)
        self.assertEqual(result["outcome"], rules.STOP_FOR_USER)
        self.assertTrue(result["problems"])
        self.assertEqual(rules.failure("x")["outcome"], rules.STOP_FOR_USER)

    def test_selection_prefers_validation_all_pass_then_reward_then_earliest(self):
        accepted = [{"step": 50, "val_all_pass": 0.32, "val_mean_reward": 0.40}, {"step": 100, "val_all_pass": 0.33, "val_mean_reward": 0.39},
                    {"step": 150, "val_all_pass": 0.33, "val_mean_reward": 0.41}, {"step": 200, "val_all_pass": 0.33, "val_mean_reward": 0.41}]
        self.assertEqual(dict(rules.select(accepted) or {})["step"], 150)
        self.assertIsNone(rules.select([]))

    def test_plateau(self):
        flat = [{"step": 50 * (i + 1), "val_all_pass": 0.305} for i in range(6)]
        self.assertTrue(rules.plateau(0.30, flat))
        self.assertFalse(rules.plateau(0.30, flat[:5]))
        rising = [*flat[:5], {"step": 300, "val_all_pass": 0.311}]
        self.assertFalse(rules.plateau(0.30, rising))
        late = [{"step": 50, "val_all_pass": 0.35}, *({"step": 50 * (i + 2), "val_all_pass": 0.355} for i in range(6))]
        self.assertTrue(rules.plateau(0.30, late))

    def test_final_verdict(self):
        holdout = {"all_pass": {"mean": 0.03, "ci95": [0.01, 0.05]}, "reward": {"mean": 0.02, "ci95": [0.0, 0.04]},
                   "exact_match": {"mean": 0.01, "ci95": [-0.01, 0.03]}}
        verdict = json.loads(json.dumps(rules.final_verdict(holdout, dev250_candidate=105, dev250_start=109, cpt_rise=0.02)))
        self.assertTrue(verdict["primary"]["met"] and verdict["adopt"])
        self.assertFalse(rules.final_verdict(holdout, dev250_candidate=104, dev250_start=109, cpt_rise=0.02)["adopt"])
        self.assertFalse(rules.final_verdict(holdout, dev250_candidate=110, dev250_start=109, cpt_rise=0.031)["adopt"])
        holdout["all_pass"]["ci95"] = [0.0, 0.05]
        verdict = json.loads(json.dumps(rules.final_verdict(holdout, dev250_candidate=110, dev250_start=109, cpt_rise=0.0)))
        self.assertFalse(verdict["primary"]["met"])


def task(function, package, family="finish_block", bucket="short", index=0):
    return {"id": f"{function}-{family}-{bucket}-{index}", "function_id": function, "package": package, "family": family,
            "curriculum": {"bucket": bucket, "prompt_tokens": 300, "completion_tokens": 50}}


class PoolTests(unittest.TestCase):
    def tasks(self):
        out = []
        for p in range(30):
            for f in range(3):
                function = f"f{p:02d}{f}"
                for bucket in ("short", "medium", "long"):
                    out += [task(function, f"pkg{p:02d}", bucket=bucket, index=i) for i in range(4)]
                out.append(task(function, f"pkg{p:02d}", family="rename_propagation"))
        return out

    def test_validation_takes_whole_packages_and_caps_tasks_per_function(self):
        train, val, info = pool.split_tasks(self.tasks(), val_share=0.10, train_per_function=8, val_per_function=3,
                                            held_functions=set(), held_packages=set())
        val_packages = {t["package"] for t in val}
        self.assertEqual(val_packages, set(json.loads(json.dumps(info["validation_packages"]))))
        self.assertFalse(val_packages & {t["package"] for t in train})
        self.assertGreaterEqual(len({t["function_id"] for t in val}), 9)
        per_function = {}
        for t in train:
            per_function.setdefault(t["function_id"], []).append(t)
        self.assertTrue(all(len(v) == 8 for v in per_function.values()))
        strata = {(t["family"], t["curriculum"]["bucket"]) for t in per_function["f000"]} if "f000" in per_function else None
        if strata is not None:
            self.assertEqual(len(strata), 4)  # round-robin over family x cut length
        self.assertTrue(all(sum(1 for t in val if t["function_id"] == f) == 3 for f in {t["function_id"] for t in val}))
        again = pool.split_tasks(self.tasks(), val_share=0.10, train_per_function=8, val_per_function=3, held_functions=set(),
                                 held_packages=set())
        self.assertEqual(again[0], train)

    def test_holdout_functions_or_packages_in_the_task_file_fail(self):
        with self.assertRaises(pool.PoolError):
            pool.split_tasks(self.tasks(), val_share=0.1, train_per_function=8, val_per_function=3, held_functions={"f000"},
                             held_packages=set())
        with self.assertRaises(pool.PoolError):
            pool.split_tasks(self.tasks(), val_share=0.1, train_per_function=8, val_per_function=3, held_functions=set(),
                             held_packages={"pkg03"})

    def test_untested_branches_are_scoped_and_bugfix_declarations_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "equivalent.txt").write_text(
                "# comment\n"
                "aaaa SCOPED[x > 0]: differs only for positive x\n"
                "bbbb BUGFIX[is.null(y)]: original returns NULL\n"
                "cccc QUIRK[z == 1]: incidental\n"
                "dddd plain equivalent rewrite\n"
                "eeee NUMTOL[1e-8]: rounding\n")
            self.assertEqual(pool.untested_mutants(directory), {"aaaa", "bbbb"})
            self.assertEqual(pool.untested_mutants(directory / "missing"), set())

    def test_sealed_paths_are_refused(self):
        with self.assertRaises(pool.PoolError):
            pool.no_sealed(Path("/mnt/e/x/DAT-07-final-a/rows.jsonl"))


@unittest.skipUnless(PILOT.is_file() and SMOKE.is_file(), "C18 bound recipes not written yet")
class BoundRecipeTests(unittest.TestCase):
    def test_bound_recipes_validate_and_carry_the_pre_registration(self):
        for path, kind in ((PILOT, "pilot"), (SMOKE, "smoke")):
            recipe = json.loads(path.read_text())
            recipe_module.validate(recipe)
            self.assertEqual(recipe["kind"], kind)
            self.assertEqual(recipe["pre_registered"], recipe_module.PRE_REGISTERED)
            self.assertEqual(recipe["parent"]["files"]["model.safetensors"], recipe_module.START_FILES["model.safetensors"])
            self.assertEqual(recipe["parent_reference"]["metrics"]["score250"], 109)
            self.assertEqual(recipe["runtime"]["optimizer"]["hidden_lr"], 1e-06)
            self.assertFalse(recipe["final_set_access"] or recipe["holdout_in_training"])
            self.assertFalse(recipe["memory_fit"]["kl_reference_fits"])
            self.assertNotEqual(recipe["pool"]["train"]["path"], recipe["pool"]["val"]["path"])
        self.assertEqual(json.loads(SMOKE.read_text())["dev_gate"]["val"]["limit"], recipe_module.SMOKE_VAL_LIMIT)
        self.assertIsNone(json.loads(PILOT.read_text())["dev_gate"]["val"]["limit"])

    def test_tampering_is_refused(self):
        recipe = json.loads(PILOT.read_text())
        for mutate in (lambda r: r["pre_registered"]["anti_forgetting"].__setitem__("coefficient", 0.2),
                       lambda r: r["runtime"].__setitem__("replay_coefficient", 0.2),
                       lambda r: r["runtime"].__setitem__("legacy_every", 4),
                       lambda r: r["dev_gate"]["val"].__setitem__("tasks", r["pool"]["train"] | {}),
                       lambda r: r.__setitem__("holdout_in_training", True),
                       lambda r: r["runtime"]["optimizer"].__setitem__("hidden_lr", 1e-5),
                       lambda r: r["runtime"].__setitem__("device", "cuda")):
            changed = copy.deepcopy(recipe)
            mutate(changed)
            with self.assertRaises(recipe_module.RecipeError):
                recipe_module.validate(changed)

    def test_pool_manifest_matches_the_recipe_and_excludes_the_holdout(self):
        recipe = json.loads(PILOT.read_text())
        manifest = json.loads((REPO / recipe["pool"]["manifest"]["path"]).read_text())
        self.assertEqual(manifest["train"]["sha256"], recipe["pool"]["train"]["sha256"])
        self.assertEqual(manifest["validation"]["sha256"], recipe["pool"]["val"]["sha256"])
        self.assertFalse(manifest["final_set_access"])
        self.assertEqual(manifest["built"]["source"]["holdout_packages_excluded"], 206)
        if not (pool.FREEZE_MANIFEST.is_file() and Path(recipe["pool"]["train"]["path"]).is_file()):
            self.skipTest("campaign data not present")
        functions, packages = pool.holdout_identity()
        seen = {}
        for split in ("train", "val"):
            tasks = pool.read_jsonl(Path(recipe["pool"][split]["path"]))
            seen[split] = {t["function_id"] for t in tasks}
            self.assertFalse(seen[split] & functions)
            self.assertFalse({t["package"] for t in tasks} & packages)
        self.assertFalse(seen["train"] & seen["val"])


@unittest.skipUnless((JOBS / "C18-grpo-exec-smoke.json").is_file(), "C18 job files not written yet")
class JobFileTests(unittest.TestCase):
    def test_jobs_validate_and_point_at_the_bound_recipes(self):
        from sepalith.ops import night_queue

        for name, kind, requires in (("C18-grpo-exec-smoke", "smoke", []), ("C18-grpo-exec-pilot-1", "run", ["C18-grpo-exec-smoke"])):
            job = json.loads((JOBS / f"{name}.json").read_text())
            night_queue.validate_job(job)
            argv = job["command"]
            self.assertIn("CUDA_VISIBLE_DEVICES=0", argv)
            self.assertTrue(argv[argv.index("-m") + 1] == "sepalith.training.grpo_exec.night" and argv[argv.index("-m") + 2] == kind)
            self.assertEqual(job["requires"], requires)
            recipe = Path(argv[argv.index("--recipe") + 1])
            local = REPO / recipe.relative_to(recipe_module.DEPLOY_ROOT)
            self.assertTrue(local.is_file())


if __name__ == "__main__":
    unittest.main()
