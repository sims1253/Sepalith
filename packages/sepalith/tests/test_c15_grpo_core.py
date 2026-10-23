"""Cards C14 step 1 and C15: the torch-free GRPO pieces.

Group advantages without std scaling, zero-variance detection, dynamic
sampling with refill and its limits, update assignment, the seeded pool
sampler and its resume cursor, R6 decision mapping, the pool selection rule,
the committed pool against its manifest, and the bound recipes against the
pre-registered C15 values and the committed source manifest.
"""
import json
import math
from pathlib import Path
import unittest

from sepalith.training.grpo import algorithm, pool, recipe as recipe_module, rules

REPO = Path(__file__).resolve().parents[3]
C14 = REPO / "docs/training/plan-20260927/c14"
C15 = REPO / "docs/training/plan-20260927/c15"


def plain(value):
    """JSON round trip: typed records become plain test data."""
    return json.loads(json.dumps(value))


def group(name, rewards, position=0):
    return algorithm.Group(name, position, tuple(rewards))


class AdvantageTests(unittest.TestCase):
    def test_mean_centred_without_std_scaling(self):
        self.assertEqual(algorithm.group_advantages([1.0, 0.0, 0.0, 1.0]), [0.5, -0.5, -0.5, 0.5])
        values = algorithm.group_advantages([1.0, 0.5, -0.5, 0.0])
        self.assertAlmostEqual(sum(values), 0.0)
        self.assertAlmostEqual(values[0], 1.0 - 0.25)  # not divided by the std (0.6455)

    def test_zero_variance_groups_get_exact_zeros(self):
        self.assertTrue(algorithm.zero_variance([0.1] * 8))
        self.assertEqual(algorithm.group_advantages([0.1] * 8), [0.0] * 8)
        self.assertFalse(algorithm.zero_variance([1.0] * 7 + [0.0]))

    def test_non_finite_reward_fails(self):
        with self.assertRaises(algorithm.GRPOError):
            algorithm.zero_variance([1.0, math.nan])


class DynamicSamplingTests(unittest.TestCase):
    def drawer(self, pattern):
        """``pattern`` gives, per drawn prompt in order, whether its group has signal."""
        calls, counter = [], [0]
        state = {"calls": calls}

        def draw(count):
            calls.append(count)
            out = []
            for _ in range(count):
                index = counter[0]
                counter[0] += 1
                signal = pattern(index)
                out.append(group(f"p{index}", [1.0, 0.0] * 4 if signal else [1.0] * 8, index))
            return out
        return draw, state

    def test_draws_one_and_a_half_times_and_drops_zero_variance(self):
        draw, state = self.drawer(lambda i: True)
        plan = algorithm.dynamic_sample(8, draw)
        self.assertEqual(state["calls"], [12])
        self.assertEqual(len(plan.selected), 8)
        self.assertEqual(plan.surplus_discarded, 4)
        self.assertFalse(plan.exhausted)
        self.assertEqual(plan.zero_variance_share_before, 0.0)

    def test_refills_until_full(self):
        draw, state = self.drawer(lambda i: i % 3 == 0)  # one prompt in three has signal
        plan = algorithm.dynamic_sample(8, draw)
        self.assertEqual(state["calls"][0], 12)
        self.assertEqual(state["calls"][1], math.ceil(1.5 * (8 - 4)))
        self.assertGreater(len(plan.round_sizes), 1)
        self.assertTrue(all(not g.zero_variance for g in plan.selected))
        self.assertEqual([g.prompt_id for g in plan.selected], sorted((g.prompt_id for g in plan.selected), key=lambda p: int(p[1:])))

    def test_refill_stops_at_the_draw_limit_and_reports_exhaustion(self):
        draw, state = self.drawer(lambda i: False)
        plan = algorithm.dynamic_sample(8, draw, max_draw_factor=4.0, max_rounds=10)
        self.assertEqual(sum(state["calls"]), 32)
        self.assertTrue(plan.exhausted)
        self.assertEqual(plan.selected, [])
        self.assertEqual(plan.zero_variance_share_before, 1.0)
        updates = algorithm.assign_updates(plan.selected, 2, 4)
        self.assertEqual([algorithm.zero_variance_share_after(u, 4) for u in updates], [1.0, 1.0])

    def test_round_limit(self):
        draw, state = self.drawer(lambda i: i == 0)
        plan = algorithm.dynamic_sample(8, draw, max_rounds=2)
        self.assertEqual(len(state["calls"]), 2)
        self.assertEqual(len(plan.selected), 1)
        updates = algorithm.assign_updates(plan.selected, 2, 4)
        self.assertEqual([len(u) for u in updates], [1, 0])
        self.assertEqual(algorithm.zero_variance_share_after(updates[0], 4), 0.75)

    def test_zero_variance_groups_never_enter_an_update(self):
        with self.assertRaises(algorithm.GRPOError):
            algorithm.zero_variance_share_after([group("a", [1.0] * 8)], 4)


class PoolSamplerTests(unittest.TestCase):
    def test_epochs_are_permutations_and_resume_matches(self):
        ids = [f"id{i}" for i in range(7)]
        sampler = algorithm.PoolSampler(ids, 3407)
        first = sampler.draw(7)
        self.assertEqual(sorted(i for _, i in first), sorted(ids))
        later = sampler.draw(10)
        resumed = algorithm.PoolSampler(ids, 3407, cursor=7)
        self.assertEqual(resumed.draw(10), later)
        self.assertEqual(sampler.state(), {"cursor": 17, "seed": 3407, "pool_size": 7})
        self.assertNotEqual([i for _, i in first], [i for _, i in algorithm.PoolSampler(ids, 1).draw(7)])


def stats(after=0.0, length=40.0, reward=0.3, loss=0.01, first=1, last=50):
    return {"first": first, "last": last, "zero_variance_share_after": after, "completion_tokens_mean_drawn": length,
            "reward_mean_drawn": reward, "loss": loss}


START = {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 117}


class R6Tests(unittest.TestCase):
    def decide(self, current, *, previous75=START, previous250=None, prior=(), window=None, reward_window=None):
        return rules.apply_r6({**current, "loss": 0.01}, previous75=previous75, previous250=previous250, start=START,
                              window=window or stats(), prior_windows=list(prior), start_window=stats(first=1, last=10),
                              reward_window=reward_window)

    def test_panels_by_step(self):
        self.assertEqual(rules.evaluation_panels(50), ("dev75",))
        self.assertEqual(rules.evaluation_panels(150), ("dev75", "dev250"))
        self.assertEqual(rules.evaluation_panels(49), ())
        self.assertEqual(rules.evaluation_steps()[:3], [50, 100, 150])
        self.assertEqual(rules.evaluation_steps()[-1], 1500)

    def test_unchanged_dev75_evaluation_continues_without_a_score_row(self):
        rule = self.decide({"valid": 74, "cap": 1, "edit": 27, "nofp": 8})
        self.assertEqual(rule["outcome"], rules.CONTINUE)
        self.assertEqual(rule["kind"], "grpo_dev75")
        self.assertNotIn("score250_drop", [row["name"] for row in rule["rows"]])

    def test_r1_bands(self):
        self.assertEqual(self.decide({"valid": 74, "cap": 1, "edit": 24, "nofp": 8})["outcome"], rules.STOP_FOR_USER)
        self.assertEqual(self.decide({"valid": 74, "cap": 1, "edit": 27, "nofp": 12})["outcome"], rules.ROLLBACK)
        self.assertEqual(self.decide({"valid": 67, "cap": 1, "edit": 27, "nofp": 8})["outcome"], rules.ROLLBACK)
        self.assertEqual(self.decide({"valid": 74, "cap": 2, "edit": 27, "nofp": 8})["outcome"], rules.STOP_FOR_USER)

    def test_dev250_evaluation_compares_score250_with_the_previous_dev250(self):
        current = {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 110}
        rule = self.decide(current, previous250=START, reward_window=stats(reward=0.2))
        self.assertEqual(rule["kind"], "grpo_dev75_dev250")
        row = next(r for r in rule["rows"] if r["name"] == "score250_drop")
        self.assertEqual((row["value"], row["outcome"]), (7, rules.STOP_FOR_USER))
        self.assertEqual(rule["outcome"], rules.STOP_FOR_USER)

    def test_reward_hacking_rolls_back(self):
        current = {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 107}
        previous = {**START, "score250": 111}  # the per-evaluation drop (4) passes R1
        rule = self.decide(current, previous250=previous, reward_window=stats(reward=0.5))
        self.assertEqual(next(r for r in rule["rows"] if r["name"] == "score250_drop")["outcome"], rules.CONTINUE)
        self.assertEqual(next(r for r in rule["rows"] if r["name"] == "reward_hacking")["outcome"], rules.ROLLBACK)
        self.assertEqual(rule["outcome"], rules.ROLLBACK)
        same = self.decide(current, previous250=previous, reward_window=stats(reward=0.3))
        self.assertEqual(same["outcome"], rules.CONTINUE)

    def test_zero_variance_streak_needs_three_evaluations(self):
        unchanged = {"valid": 74, "cap": 1, "edit": 27, "nofp": 8}
        self.assertEqual(self.decide(unchanged, prior=[stats(after=0.6)], window=stats(after=0.7))["outcome"], rules.CONTINUE)
        self.assertEqual(self.decide(unchanged, prior=[stats(after=0.4), stats(after=0.6)], window=stats(after=0.7))["outcome"],
                         rules.CONTINUE)
        rule = self.decide(unchanged, prior=[stats(after=0.51), stats(after=0.6)], window=stats(after=0.7))
        self.assertEqual(rule["outcome"], rules.STOP_FOR_USER)

    def test_length_factor(self):
        unchanged = {"valid": 74, "cap": 1, "edit": 27, "nofp": 8}
        self.assertEqual(self.decide(unchanged, window=stats(length=81.0))["outcome"], rules.STOP_FOR_USER)
        self.assertEqual(self.decide(unchanged, window=stats(length=19.0))["outcome"], rules.STOP_FOR_USER)
        self.assertEqual(self.decide(unchanged, window=stats(length=79.0))["outcome"], rules.CONTINUE)

    def test_missing_or_non_finite_inputs_stop(self):
        rule = rules.apply_r6({"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "loss": math.nan}, previous75=START, previous250=None,
                              start=START, window=stats(), prior_windows=[], start_window=stats(), reward_window=None)
        self.assertEqual(rule["outcome"], rules.STOP_FOR_USER)
        self.assertTrue(rule["problems"])
        self.assertEqual(self.decide({"valid": 74, "cap": 1, "edit": 27})["outcome"], rules.STOP_FOR_USER)

    def test_window_stats_need_every_update_once(self):
        events = [{"event": "grpo_update", "update": u, "zero_variance_share_after": 0.0, "loss": 0.1,
                   "completion_tokens_mean_drawn": 40.0, "reward_mean_drawn": 0.5, "groups_trained": 4} for u in range(1, 11)]
        window = plain(rules.window_stats(events, 1, 10))
        self.assertEqual(window["updates"], 10)
        self.assertAlmostEqual(window["reward_mean_drawn"], 0.5)
        with self.assertRaises(ValueError):
            rules.window_stats(events[:-1], 1, 10)
        with self.assertRaises(ValueError):
            rules.window_stats(events + events[:1], 1, 10)

    def test_r1_versus_start(self):
        self.assertEqual(rules.r1_versus_start({**START, "score250": 117}, START)["outcome"], rules.CONTINUE)
        self.assertTrue(rules.r1_versus_start({**START, "score250": 117}, START)["with_score250"])
        self.assertEqual(rules.r1_versus_start({"valid": 74, "cap": 1, "edit": 27, "nofp": 11}, START)["outcome"], rules.STOP_FOR_USER)


def census_row(identifier, family, count):
    return {"id": identifier, "family": family, "exact_count": count, "pass_rate": count / 8, "k": 8,
            "mean_completion_tokens": 40.0, "mean_reward": count / 8}


class PoolSelectionTests(unittest.TestCase):
    def test_largest_remainder(self):
        self.assertEqual(pool.largest_remainder({"a": 5, "b": 3, "c": 2}, 5), {"a": 3, "b": 1, "c": 1})
        self.assertEqual(sum(pool.largest_remainder({"a": 1086, "b": 1957, "c": 94}, 209).values()), 209)

    def test_band_plus_ten_percent_all_correct(self):
        rows = [census_row(f"b{i:02d}", "finish_block", 1 + i % 7) for i in range(18)]
        rows += [census_row(f"z{i:02d}", "finish_block", 0) for i in range(30)]
        rows += [census_row(f"c{i:02d}", "pipe_rewrite" if i < 6 else "no_op", 8) for i in range(10)]
        rows += [census_row("n00", "no_op", 3)]
        selection = plain(pool.select_pool(rows))
        counts = selection["counts"]
        self.assertEqual(counts["band"], 19)
        self.assertEqual(counts["all_correct_sampled"], 2)  # round(19 / 9)
        self.assertEqual(counts["pool"], 21)
        self.assertEqual(counts["by_family"]["no_op"]["band"], 1)
        ids = [row["id"] for row in selection["pool"]]
        self.assertEqual(ids, sorted(ids))
        self.assertFalse(any(i.startswith("z") for i in ids))
        self.assertEqual(selection, pool.select_pool(list(reversed(rows))))

    def test_bad_rows_fail(self):
        with self.assertRaises(pool.PoolError):
            pool.select_pool([census_row("a", "x", 9)])
        with self.assertRaises(pool.PoolError):
            pool.select_pool([census_row("a", "x", 3), census_row("a", "x", 3)])


class CommittedPoolTests(unittest.TestCase):
    def test_pool_matches_its_manifest(self):
        manifest = json.loads((C14 / "C14-grpo-pool-arm-a-536.json").read_text())
        rows = plain(pool.read_pool(REPO / manifest["pool"]["path"], manifest["pool"]["sha256"]))
        counts = manifest["counts"]
        self.assertEqual(len(rows), counts["pool"])
        self.assertEqual(sum(row["stratum"] == "band" for row in rows), counts["band"])
        self.assertEqual(sum(row["stratum"] == "all_correct" for row in rows), counts["all_correct_sampled"])
        self.assertTrue(all(1 <= row["exact_count"] <= 7 for row in rows if row["stratum"] == "band"))
        self.assertTrue(all(row["exact_count"] == 8 for row in rows if row["stratum"] == "all_correct"))
        self.assertEqual((counts["band"], counts["all_correct_sampled"], counts["pool"]), (1877, 209, 2086))
        for family, value in counts["by_family"].items():
            self.assertEqual(sum(row["family"] == family for row in rows), value["pool"])
        self.assertIn("TRAIN only", manifest["scope"])
        self.assertFalse(manifest["final_set_access"])


class RecipeTests(unittest.TestCase):
    def recipes(self):
        return [C15 / "bound/grpo-pilot-1-arm-a-536.json", C15 / "bound/grpo-smoke-arm-a-536.json"]

    def test_bound_recipes_carry_the_pre_registered_values_and_the_committed_source(self):
        for path in self.recipes():
            recipe = plain(recipe_module.load(path, root_override=REPO, check_data=False))
            runtime = recipe["runtime"]
            self.assertEqual((runtime["group_size"], runtime["rows_per_update"], runtime["rollout_rows_per_call"]), (8, 32, 16))
            self.assertEqual(runtime["optimizer"]["hidden_lr"], 1e-06)
            self.assertEqual(runtime["optimizer"]["side_lr"], 1e-07)
            self.assertEqual(runtime["device"], "cuda:0")
            self.assertEqual(recipe["parent"]["files"]["model.safetensors"],
                             "befe587a0d4969e33b0daaa8fe66247371ec32f331616db74724616878ad6030")
            self.assertEqual(recipe["parent_reference"]["metrics"]["score250"], 117)
            self.assertEqual(recipe["pre_registered"]["epsilon_high"], 0.28)
            self.assertIn("2026-10-23", recipe["r4_override"]["decision"])

    def test_pilot_and_smoke_write_to_different_places(self):
        pilot, smoke = (json.loads(path.read_text()) for path in self.recipes())
        self.assertEqual(pilot["kind"], "pilot")
        self.assertEqual(smoke["smoke"]["stops"], [4, 6])
        for key in ("archive", "graceful_stop"):
            self.assertNotEqual(pilot["outputs"][key], smoke["outputs"][key])
        self.assertNotEqual(pilot["dev_gate"]["evidence_directory"], smoke["dev_gate"]["evidence_directory"])

    def test_changed_pre_registered_value_is_refused(self):
        recipe = json.loads(self.recipes()[0].read_text())
        recipe["pre_registered"]["epsilon_high"] = 0.2
        with self.assertRaises(recipe_module.RecipeError):
            recipe_module.validate(recipe)
        recipe = json.loads(self.recipes()[0].read_text())
        recipe["runtime"]["optimizer"]["hidden_lr"] = 1e-05
        with self.assertRaises(recipe_module.RecipeError):
            recipe_module.validate(recipe)
        recipe = json.loads(self.recipes()[0].read_text())
        recipe["runtime"]["device"] = "cuda"
        with self.assertRaises(recipe_module.RecipeError):
            recipe_module.validate(recipe)


class JobTests(unittest.TestCase):
    def test_jobs_validate_and_the_pilot_requires_the_smoke(self):
        from sepalith.ops import night_queue

        smoke = night_queue.validate_job(json.loads((C15 / "jobs/C15-grpo-smoke.json").read_text()))
        pilot = night_queue.validate_job(json.loads((C15 / "jobs/C15-grpo-pilot-1.json").read_text()))
        self.assertEqual(pilot["requires"], ["C15-grpo-smoke"])
        self.assertTrue(pilot["resumable"])
        self.assertFalse(smoke["resumable"])
        for job, command in ((smoke, "smoke"), (pilot, "run")):
            argv = job["command"]
            module = argv.index("-m")
            self.assertEqual(argv[module + 1:module + 3], ["sepalith.training.grpo.night", command])
            self.assertIn("CUDA_VISIBLE_DEVICES=0", argv)
            self.assertTrue(argv[argv.index("--recipe") + 1].startswith("/home/m0hawk/sandbox/Sepalith/docs/training/plan-20260927/c15/bound/"))


if __name__ == "__main__":
    unittest.main()
