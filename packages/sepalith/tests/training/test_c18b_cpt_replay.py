"""Card C18b on CPU: the CPT replay term (gradient, telemetry, bit-exact resume), the CPT row shape, the install
hooks that point the C18 driver at the C18b modules, the recipe checks and the job files."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import LlamaForCausalLM

from sepalith.training.grpo import state
from sepalith.training.grpo import trainer as c15_trainer
from sepalith.training.grpo_exec import c18b, c18b_trainer, mix, night, recipe as c18_recipe, trainer

from test_c18_grpo_exec import EXEC, LEGACY, events, fake_generate, plain, prompts, replay_rows, routed_reward, tiny_model, trainer_recipe

REPO = Path(__file__).resolve().parents[4]
BOUND = REPO / "docs/training/plan-20260927/c18b/bound"
JOBS = REPO / "docs/training/plan-20260927/c18b/jobs"


def cpt_rows():
    generator = torch.Generator().manual_seed(11)
    return {f"c{i}": ((0,), tuple(torch.randint(22, 129000, (5 + i % 4,), generator=generator).tolist())) for i in range(9)}


def recipe_b(root):
    recipe = trainer_recipe(root)
    recipe["runtime"].update(cpt_replay_rows_per_update=2, cpt_replay_coefficient=0.2)
    return recipe


def build(recipe, *, model, update=0, cursor=0, resumed_from=None, stop_after=None):
    optimizer = c15_trainer.build_optimizer(model, recipe["runtime"], check_dispatch=False)
    return c18b_trainer.CptReplayTrainer(recipe, "r" * 64, model=model, optimizer=optimizer, ops=c15_trainer.DeviceOps(torch.device("cpu")),
                                         reward=routed_reward, exec_ids=EXEC, legacy_ids=LEGACY, prompts=prompts(), replay=replay_rows(),
                                         cpt_replay=cpt_rows(), settings=c15_trainer.Settings(stop_after=stop_after), update=update,
                                         cursor=cursor, resumed_from=resumed_from, generate=fake_generate)


class CptReplayTrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cpt_term_gradient_and_statistics(self):
        recipe = recipe_b(self.root)
        run = build(recipe, model=tiny_model())
        run.model.zero_grad(set_to_none=True)
        stats = run.replay_term(2)
        ours = {n: p.grad.detach().clone() for n, p in run.model.named_parameters() if p.grad is not None}
        model = run.model
        model.zero_grad(set_to_none=True)
        seed = recipe["runtime"]["seed"]
        total = torch.zeros(())
        for rows, coefficient, offset in ((replay_rows(), 0.1, 0), (cpt_rows(), 0.2, 101)):
            ids = mix.replay_ids(sorted(rows), seed + offset, 2, 3 if offset == 0 else 2)
            loss, tokens = torch.zeros(()), 0
            for identifier in ids:
                prompt, target = rows[identifier]
                logits = model(input_ids=torch.tensor([list(prompt + target)])).logits[0].float()
                logp = torch.log_softmax(logits[len(prompt) - 1:-1], dim=-1).gather(1, torch.tensor(target)[:, None]).squeeze(1)
                loss = loss - logp.sum()
                tokens += len(target)
            if offset:
                self.assertEqual(stats["cpt_replay_tokens"], tokens)
                self.assertAlmostEqual(stats["cpt_replay_nll"], float(loss.detach()) / tokens, places=4)
            total = total + coefficient * loss / tokens
        total.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(ours[name], parameter.grad, rtol=1e-4, atol=1e-6)
        self.assertGreater(stats["grad_norm_after_sft"], 0)

    def test_segment_logs_both_terms_and_resumes_bit_exact(self):
        shared: dict = {}

        def segment(root, stop_after, resume=None):
            recipe = recipe_b(root) if resume is None else shared["recipe"]
            if resume is None:
                shared["recipe"] = recipe
                torch.manual_seed(recipe["runtime"]["seed"])
                return build(recipe, model=tiny_model(), stop_after=stop_after), None
            saved = plain(state.verify_checkpoint(resume, "r" * 64))
            run = build(recipe, model=LlamaForCausalLM.from_pretrained(str(resume)), update=saved["update"],
                        cursor=saved["sampler"]["cursor"], resumed_from=saved["update"], stop_after=stop_after)
            c15_trainer.load_optimizer_state(run.optimizer, resume / "optimizer.pt")
            torch.set_rng_state(torch.load(resume / "rng_state.pt", weights_only=True)["cpu"])
            return run, None

        whole, _ = segment(self.root / "whole", 6)
        whole.run()
        first, _ = segment(self.root / "split", 4)
        first.run()
        resumed, _ = segment(self.root / "split", 6, self.root / "split" / "archive" / "full" / "checkpoint-4")
        resumed.run()
        for (name, a), (_, b) in zip(whole.model.state_dict().items(), resumed.model.state_dict().items(), strict=True):
            self.assertTrue(torch.equal(a, b), name)
        stepped = [e for e in events(self.root / "whole" / "archive") if e["optimizer_step"]]
        self.assertTrue(stepped)
        for event in stepped:
            self.assertEqual(event["cpt_replay_rows"], 2)
            self.assertGreater(event["cpt_replay_nll"], 0)
            self.assertGreaterEqual(event["grad_norm"], 0)


class ShapeAndHookTests(unittest.TestCase):
    def test_cpt_row_splits_head_and_supervised_span(self):
        row = {"input_ids": [0, 5, *range(100, 120), 1], "labels": [-100, -100, *range(100, 120), -100]}
        self.assertEqual(c18b.cpt_row(row), {"prompt_ids": [0, 5], "target_ids": list(range(100, 120))})
        self.assertIsNone(c18b.cpt_row({"input_ids": [0, *range(100, 120)], "labels": [-100, 7, *range(101, 120)]}))
        self.assertIsNone(c18b.cpt_row({"input_ids": [0, 1, 2], "labels": [-100, 1, 2]}))  # too short
        self.assertIsNone(c18b.cpt_row({"input_ids": [5, *range(100, 120)], "labels": [-100, *range(100, 120)]}))  # no BOS

    def test_install_points_the_driver_and_trainer_at_the_c18b_modules(self):
        saved = {name: getattr(night, name) for name in ("TRAINER", "DEV_EVALUATOR", "VAL_EVALUATOR", "CPT_PROBE", "UPDATE_FIELDS")}
        saved_trainer, saved_recipe = trainer.ExecGRPOTrainer, night.recipe_module
        try:
            c18b.install()
            self.assertEqual(night.TRAINER, "sepalith.training.grpo_exec.c18b_trainer")
            self.assertEqual(night.CPT_PROBE, "sepalith.training.grpo_exec.c18b_cpt_probe")
            self.assertIs(night.recipe_module, c18b)
            self.assertIs(trainer.ExecGRPOTrainer, c18b_trainer.CptReplayTrainer)
            self.assertIn("cpt_replay_nll", night.UPDATE_FIELDS)
            for name in c18b.ENTRY.values():
                __import__(name)
        finally:
            for name, value in saved.items():
                setattr(night, name, value)
            trainer.ExecGRPOTrainer = saved_trainer
            for module in (night, trainer):
                module.recipe_module = saved_recipe

    def test_pre_registration_differs_from_c18_only_where_intended(self):
        changed = {k for k in c18b.PRE_REGISTERED if c18b.PRE_REGISTERED[k] != c18_recipe.PRE_REGISTERED.get(k)}
        self.assertEqual(changed, {"card", "start", "anti_forgetting", "comparison_with_c18"})
        b_anti, c18_anti = plain(c18b.PRE_REGISTERED["anti_forgetting"]), plain(c18_recipe.PRE_REGISTERED["anti_forgetting"])
        sft = {k: v for k, v in b_anti.items() if k not in ("kind", "cpt", "why_cpt")}
        self.assertEqual(sft, {k: v for k, v in c18_anti.items() if k != "kind"})
        self.assertEqual((c18b.CPT_REPLAY_SETTINGS["rows_per_update"], c18b.CPT_REPLAY_SETTINGS["coefficient"]), (4, 0.2))


@unittest.skipUnless((BOUND / "grpo-exec-pilot-1-arm-a-803.json").is_file(), "C18b recipes not bound yet")
class BoundTests(unittest.TestCase):
    def test_bound_recipes_validate_and_tampering_fails(self):
        for kind in ("pilot", "smoke"):
            name = "grpo-exec-pilot-1-arm-a-803.json" if kind == "pilot" else "grpo-exec-smoke-arm-a-803.json"
            recipe = json.loads((BOUND / name).read_text())
            c18b.validate(recipe)
            self.assertEqual((recipe["card"], recipe["kind"]), ("C18b", kind))
            self.assertTrue(recipe["outputs"]["archive"].endswith(f"C18b-grpo-exec-{'pilot-1' if kind == 'pilot' else 'smoke'}"))
            self.assertEqual(recipe["parent"]["files"]["model.safetensors"], c18_recipe.START_FILES["model.safetensors"])
            for mutate in (lambda r: r["runtime"].__setitem__("cpt_replay_coefficient", 0.1),
                           lambda r: r["runtime"].__setitem__("replay_coefficient", 0.3),
                           lambda r: r["pool"].pop("cpt_replay"),
                           lambda r: r.__setitem__("card", "C18")):
                changed = copy.deepcopy(recipe)
                mutate(changed)
                with self.assertRaises(c18b.RecipeError):
                    c18b.validate(changed)

    @unittest.skipUnless((JOBS / "C18b-grpo-exec-smoke.json").is_file(), "C18b jobs not written yet")
    def test_jobs_validate_and_parse(self):
        from sepalith.ops import night_queue

        for name, kind in (("C18b-grpo-exec-smoke", "smoke"), ("C18b-grpo-exec-pilot-1", "run")):
            job = json.loads((JOBS / f"{name}.json").read_text())
            night_queue.validate_job(job)
            argv = job["command"]
            args = night.build_parser().parse_args(argv[argv.index("sepalith.training.grpo_exec.c18b_night") + 1:])
            self.assertEqual(args.command, kind)
            local = REPO / Path(str(args.recipe)).relative_to(c18_recipe.DEPLOY_ROOT)
            self.assertEqual(json.loads(local.read_text())["card"], "C18b")
        self.assertEqual(json.loads((JOBS / "C18b-grpo-exec-pilot-1.json").read_text())["requires"], ["C18b-grpo-exec-smoke"])


if __name__ == "__main__":
    unittest.main()
