"""Card C18 on CPU (no GPU): the exec GRPO trainer, the SFT replay term, CUDA discipline and the night driver.

- A tiny Llama with the real 130,560-token vocabulary: trainer segments with mixed exec and legacy
  prompts, routed rewards and the replay term (telemetry, save, bit-exact resume at a buffer
  boundary), and the replay term's gradient against autograd of the written loss.
- ``open_cuda`` against a fake ``torch``: explicit ``cuda:0``, the context before any memory statistic;
  a scan of the package for bare CUDA devices and memory calls without a device.
- The night driver with a fake launcher: the 803 start measurements, R6x decisions with DEV75,
  validation, CPT and DEV250 results, rollback, plateau, selection, the final hold-out job, the smoke.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sepalith.training.eval import campaign_eval
from sepalith.training.grpo import state
from sepalith.training.grpo import trainer as c15_trainer
from sepalith.training.grpo_exec import mix, night, recipe as recipe_module, rules, trainer, val_eval
from sepalith.training.offline import rules as offline_rules
from sepalith.training.sft import night_arm

REPO = Path(__file__).resolve().parents[4]
C15_PILOT = REPO / "docs/training/plan-20260927/c15/bound/grpo-pilot-1-arm-a-536.json"
JOBS = REPO / "docs/training/plan-20260927/c18/jobs"
VOCAB = 130560
BATCH = campaign_eval.generation_batching(32, "dynamic")


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=256, tie_word_embeddings=False, bos_token_id=0,
                         eos_token_id=1, pad_token_id=1)
    return LlamaForCausalLM(config)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    return json.loads(json.dumps(value))


def synthetic_recipe(root, kind="pilot"):
    """A structurally valid C18 recipe over temporary files (validate() passes; data files are fakes)."""
    base = json.loads(C15_PILOT.read_text())
    root.mkdir(parents=True, exist_ok=True)
    parent = root / "start" / "checkpoint-803"
    parent.mkdir(parents=True)
    for name in (*c15_trainer.TOKENIZER_FILES, "campaign-manifest.json"):
        (parent / name).write_text(name)
    files = {}
    for name in ("train", "val", "replay", "completions"):
        path = root / "data" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n")
        files[name] = {"path": str(path), "sha256": sha(path)}
    runtime = copy.deepcopy(base["runtime"])
    runtime.update({"legacy_every": 5, "replay_rows_per_update": 8, "replay_coefficient": 0.1, "buffer_seconds_prior": 240.0,
                    "save_seconds_prior": 300.0})
    runtime["reward"] = {**runtime["reward"], "exec_workers": 8, "exec_nice": 10}
    steps = rules.evaluation_steps()
    out = {"archive": str(root / "archive"), "graceful_stop": str(root / "control" / "stop.json"), "trainer": str(root / "archive" / "runtime")}
    gate = copy.deepcopy(base["dev_gate"])
    gate.update({"rule": "R6x", "evaluation_steps": steps, "mandatory_steps": steps, "evidence_directory": str(root / "gates"),
                 "val": {"tasks": files["val"], "n_tasks": 40, "batch_size": 32, "cap": 1024, "decoding": "greedy",
                         "limit": 8 if kind == "smoke" else None},
                 "cpt": {"panel": files["replay"], "rows": 499, "max_sequence_tokens": 2048, "load": "plain bf16"}})
    recipe = {
        "schema": recipe_module.BOUND_SCHEMA, "card": "C18", "kind": kind, "status": "bound_not_launched", "decided_by": "test",
        "arm": {"name": f"C18-test-{kind}", "purpose": "test"},
        "parent": {"path": str(parent), "step": 803, "tag": "arm-a-803",
                   "files": {"campaign-manifest.json": sha(parent / "campaign-manifest.json"),
                             "tokenizer.json": sha(parent / "tokenizer.json"), "model.safetensors": "0" * 64}},
        "parent_reference": {"results": {}, "metrics": {}, "cpt_recorded_nll_fp32_norms": 1.003412},
        "sft_recipe": base["sft_recipe"],
        "pool": {"manifest": files["train"], "train": {**files["train"], "tasks": 1, "functions": 1},
                 "val": {**files["val"], "tasks": 40, "functions": 10}, "legacy": {}, "replay": {**files["replay"], "rows": 1}},
        "memory_fit": {"kl_reference_fits": False}, "pre_registered": copy.deepcopy(recipe_module.PRE_REGISTERED),
        "runtime": runtime, "dev_gate": gate,
        "final_evaluation": {"holdout_tasks": files["train"], "freeze_manifest_sha256": "f" * 64,
                             "start_completions": files["completions"]},
        "outputs": out, "smoke": {"stops": [4, 6], "eval_step": 4, "eval_panels": ["dev75", "val", "cpt"]} if kind == "smoke" else None,
        "source": {"repository_root": str(REPO), "manifest_path": "x", "manifest_sha256": "0" * 64},
        "expected_model": base["expected_model"], "seed": 3407, "final_set_access": False, "holdout_in_training": False,
    }
    recipe_module.validate(recipe)
    return recipe


# Trainer -----------------------------------------------------------------------------------------

LEGACY = [f"l{i}" for i in range(4)]
EXEC = [f"e{i:02d}" for i in range(10)]


def prompts():
    generator = torch.Generator().manual_seed(5)
    return {identifier: (0, *torch.randint(22, 129000, (3 + index % 4,), generator=generator).tolist())
            for index, identifier in enumerate(EXEC + LEGACY)}


def replay_rows():
    generator = torch.Generator().manual_seed(9)
    return {f"r{i}": ((0, *torch.randint(22, 129000, (4,), generator=generator).tolist()),
                      (*torch.randint(22, 129000, (2 + i % 3,), generator=generator).tolist(), 1)) for i in range(12)}


def fake_generate(input_ids, attention_mask, **kwargs):
    cap = kwargs["max_new_tokens"]
    rows = []
    for _ in range(input_ids.shape[0]):
        length = int(torch.randint(1, cap + 1, (1,)))
        body = torch.randint(22, 129000, (cap,))
        if length < cap or bool(torch.rand(()) < 0.5):
            body[length - 1] = 1
            body[length:] = 1
        rows.append(body)
    return torch.cat([input_ids, torch.stack(rows)], dim=1)


def routed_reward(ids, completions):
    out = []
    for identifier, completion in zip(ids, completions, strict=True):
        if completion[-1] != 1:
            out.append({"reward": -0.5, "exact": False, "outcome": "cap_hit"})
        elif identifier.startswith("l"):  # legacy: v2.1-like
            exact = sum(completion) % 2 == 0
            out.append({"reward": 1.0 if exact else 0.25, "exact": exact, "outcome": "exact" if exact else "near_miss"})
        else:  # exec: graded
            behaviour = (sum(completion) % 5) / 4
            out.append({"reward": 0.5 * behaviour + 0.5 * (behaviour == 1.0), "exact": behaviour == 1.0, "outcome": "partial"})
    return out


def trainer_recipe(root):
    recipe = synthetic_recipe(root)
    recipe["runtime"].update({"completion_max_tokens": 6, "seq_chunk": 3, "vocab_chunk": 50000, "min_free_disk_bytes": 0,
                              "checkpoint_bytes_estimate": 0, "replay_rows_per_update": 3})
    recipe["runtime"]["optimizer"] = {**recipe["runtime"]["optimizer"], "hidden_lr": 0.02, "side_lr": 0.002}
    return recipe


def build(recipe, *, model, update=0, cursor=0, resumed_from=None, stop_after=None, coefficient=None):
    if coefficient is not None:
        recipe["runtime"]["replay_coefficient"] = coefficient
    optimizer = c15_trainer.build_optimizer(model, recipe["runtime"], check_dispatch=False)
    return trainer.ExecGRPOTrainer(recipe, "r" * 64, model=model, optimizer=optimizer, ops=c15_trainer.DeviceOps(torch.device("cpu")),
                                   reward=routed_reward, exec_ids=EXEC, legacy_ids=LEGACY, prompts=prompts(), replay=replay_rows(),
                                   settings=c15_trainer.Settings(stop_after=stop_after), update=update, cursor=cursor,
                                   resumed_from=resumed_from, generate=fake_generate)


def events(archive, kind="grpo_update"):
    return [json.loads(line) for line in (archive / "telemetry.jsonl").read_text().splitlines() if json.loads(line)["event"] == kind]


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def segment(self, root, *, stop_after, resume=None):
        recipe = trainer_recipe(root) if resume is None else self.recipe
        if resume is None:
            self.recipe = recipe
            model = tiny_model()
            torch.manual_seed(recipe["runtime"]["seed"])
            run = build(recipe, model=model, stop_after=stop_after)
        else:
            saved = plain(state.verify_checkpoint(resume, "r" * 64))
            model = LlamaForCausalLM.from_pretrained(str(resume))
            run = build(recipe, model=model, update=saved["update"], cursor=saved["sampler"]["cursor"], resumed_from=saved["update"],
                        stop_after=stop_after)
            c15_trainer.load_optimizer_state(run.optimizer, resume / "optimizer.pt")
            rng = torch.load(resume / "rng_state.pt", weights_only=True)
            torch.set_rng_state(rng["cpu"])
        return run, run.run()

    def test_segment_mixes_sources_and_logs_the_replay_term(self):
        run, report = self.segment(self.root / "a", stop_after=4)
        self.assertEqual((report["status"], report["update"]), ("saved", 4))
        updates = events(self.root / "a" / "archive")
        self.assertEqual([e["update"] for e in updates], [1, 2, 3, 4])
        for event in updates:
            for name in ("exec_groups_drawn", "legacy_groups_drawn", "exec_zero_variance_share_before", "exec_reward_mean_drawn",
                         "exec_groups_trained", "legacy_groups_trained", "replay_rows", "replay_nll", "grad_norm_grpo"):
                self.assertIn(name, event)
            if event["optimizer_step"]:
                self.assertEqual((event["replay_rows"], event["replay_tokens"] > 0), (3, True))
                self.assertGreater(event["replay_nll"], 0)
                self.assertGreater(event["grad_norm"], 0)
        first = updates[0]
        self.assertEqual(first["exec_groups_drawn"] + first["legacy_groups_drawn"], first["groups_drawn"])
        self.assertGreater(first["legacy_groups_drawn"], 0)
        saved = plain(state.verify_checkpoint(self.root / "a" / "archive" / "full" / "checkpoint-4", "r" * 64))
        self.assertEqual(saved["sampler"]["cursor"], run.sampler.cursor)
        self.assertEqual(saved["sampler"]["legacy_every"], 5)

    def test_resume_at_a_buffer_boundary_reproduces_the_uninterrupted_run(self):
        whole, _ = self.segment(self.root / "whole", stop_after=6)
        self.segment(self.root / "split", stop_after=4)
        resumed, report = self.segment(self.root / "split", stop_after=6,
                                       resume=self.root / "split" / "archive" / "full" / "checkpoint-4")
        self.assertEqual(report["from_update"], 4)
        for (name, a), (_, b) in zip(whole.model.state_dict().items(), resumed.model.state_dict().items(), strict=True):
            self.assertTrue(torch.equal(a, b), name)
        self.assertEqual(resumed.sampler.state(), whole.sampler.state())

    def test_replay_gradient_is_coefficient_times_token_mean_nll(self):
        recipe = trainer_recipe(self.root)
        run = build(recipe, model=tiny_model(), coefficient=0.1)
        run.model.zero_grad(set_to_none=True)
        stats = run.replay_term(3)
        ours = {n: p.grad.detach().clone() for n, p in run.model.named_parameters() if p.grad is not None}
        model = run.model
        model.zero_grad(set_to_none=True)
        ids = mix.replay_ids(sorted(replay_rows()), recipe["runtime"]["seed"], 3, 3)
        total, tokens = torch.zeros(()), 0
        for identifier in ids:
            prompt, target = replay_rows()[identifier]
            logits = model(input_ids=torch.tensor([list(prompt + target)])).logits[0].float()
            logp = torch.log_softmax(logits[len(prompt) - 1:-1], dim=-1).gather(1, torch.tensor(target)[:, None]).squeeze(1)
            total = total - logp.sum()
            tokens += len(target)
        (0.1 * total / tokens).backward()
        self.assertEqual(stats["replay_tokens"], tokens)
        self.assertAlmostEqual(stats["replay_nll"], float(total.detach()) / tokens, places=4)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(ours[name], parameter.grad, rtol=1e-4, atol=1e-6)

    def test_zero_coefficient_leaves_the_grpo_gradient_alone(self):
        run = build(trainer_recipe(self.root), model=tiny_model(), coefficient=0.0)
        run.model.zero_grad(set_to_none=True)
        stats = run.replay_term(1)
        self.assertEqual(stats["replay_loss"], 0.0)
        self.assertTrue(all(p.grad is None or not bool(torch.any(p.grad)) for p in run.model.parameters()))

    def test_prompts_must_cover_both_pools(self):
        recipe = trainer_recipe(self.root)
        model = tiny_model()
        optimizer = c15_trainer.build_optimizer(model, recipe["runtime"], check_dispatch=False)
        partial = {k: v for k, v in prompts().items() if k != "l0"}
        with self.assertRaises(c15_trainer.TrainerError):
            trainer.ExecGRPOTrainer(recipe, "r" * 64, model=model, optimizer=optimizer, ops=c15_trainer.DeviceOps(torch.device("cpu")),
                                    reward=routed_reward, exec_ids=EXEC, legacy_ids=LEGACY, prompts=partial, replay=replay_rows(),
                                    settings=c15_trainer.Settings())


PILOT = REPO / "docs/training/plan-20260927/c18/bound/grpo-exec-pilot-1-arm-a-803.json"


@unittest.skipUnless(PILOT.is_file() and Path("/mnt/e/sepalith/campaign-20260915/data-work/C18-grpo-exec-v1").is_dir(),
                     "C18 bound recipe or pool not present")
class RealInputTests(unittest.TestCase):
    def test_real_inputs_load_and_the_routed_reward_scores_gold_completions(self):
        recipe = plain(recipe_module.load(PILOT, root_override=REPO, check_source=False))
        prompts_, exec_ids, legacy_ids, replay, train, bundle = trainer.load_inputs(recipe, REPO)
        self.assertEqual(len(exec_ids), recipe["pool"]["train"]["tasks"])
        self.assertEqual(len(legacy_ids), 2086)
        self.assertEqual(len(replay), recipe["pool"]["replay"]["rows"])
        val_ids = {json.loads(line)["id"] for line in Path(recipe["pool"]["val"]["path"]).read_text().splitlines()}
        self.assertFalse(val_ids & set(prompts_))
        tasks: dict[tuple[str, str], dict] = {}
        with train.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["family"], row["curriculum"].get("bucket"))
                tasks.setdefault(key, row)
        chosen = list(tasks.values())
        reward, harness = trainer.make_reward(recipe, train, bundle)
        try:
            from sepalith.training.rl.reward_v21 import DATA_ROOT, TRAIN_INPUTS

            wanted = set(legacy_ids[:3])
            gold = {}
            with (DATA_ROOT / TRAIN_INPUTS["rows"][0]).open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["id"] in wanted:
                        gold[row["id"]] = tuple(row["input_ids"][row["target_start"]:])
            ids = [t["id"] for t in chosen] + sorted(gold)
            completions = [tuple(t["gold_completion_ids"]) for t in chosen] + [gold[i] for i in sorted(gold)]
            records = reward(ids, completions)
        finally:
            harness.close()
        exec_records = records[:len(chosen)]
        self.assertTrue(all(r["exact"] and r["reward"] == 1.0 for r in exec_records), exec_records)
        self.assertTrue(all(r.get("reward_policy") == "sepalith.reward.exec.v1" for r in exec_records))
        self.assertTrue(all(r["exact"] for r in records[len(chosen):]), records[len(chosen):])


# CUDA discipline ----------------------------------------------------------------------------------

class FakeCuda:
    def __init__(self, log):
        self.log = log

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args):
            self.log.append((name, args))
            if name == "is_available":
                return True
            if name == "device_count":
                return 1
            return None
        return call


class CudaTests(unittest.TestCase):
    def test_open_cuda_creates_the_context_before_memory_statistics(self):
        log = []
        fake = SimpleNamespace(cuda=FakeCuda(log), device=torch.device, zeros=lambda *a, **k: log.append(("zeros", (k["device"],))))
        device = val_eval.open_cuda(fake)
        self.assertEqual(device, torch.device("cuda", 0))
        names = [name for name, _ in log]
        self.assertEqual(names, ["is_available", "device_count", "set_device", "zeros", "synchronize", "reset_peak_memory_stats"])
        for name, args in log[2:]:
            self.assertEqual(args[0], torch.device("cuda", 0), name)

    def test_package_has_no_bare_cuda_device_and_memory_calls_name_the_device(self):
        package = REPO / "packages/sepalith/src/sepalith/training/grpo_exec"
        allowed = {"is_available", "device_count", "max_memory_allocated", "empty_cache"}
        for path in package.glob("*.py"):
            text = path.read_text()
            self.assertIsNone(re.search(r"""set_device\(\s*["']cuda["']\s*\)""", text), path.name)
            self.assertIsNone(re.search(r"""device_map=\{"": "cuda"\}""", text), path.name)
            for match in re.finditer(r"torch\.cuda\.(\w+)\(([^)]*)\)", text):
                self.assertIn(match.group(1), allowed, f"{path.name}: torch.cuda.{match.group(1)}")
                if match.group(1) == "max_memory_allocated":
                    self.assertEqual(match.group(2), "device", path.name)
        for name in ("val_eval.py", "cpt_probe.py"):
            text = (package / name).read_text()
            self.assertIn('DEVICE = "cuda:0"', text)
            self.assertLess(text.index("open_cuda(torch)"), text.index("max_memory_allocated"))


# Night driver --------------------------------------------------------------------------------------

def dev_counts(valid=75, cap=0, edit=28, nofp=9, score=109, noedit=1):
    dev75 = {"protocol_valid": valid, "cap_hit": cap, "edit_exact": edit, "strict_noop_false_suggestions": nofp,
             "predicted_noop": 23, "strict_noop_correct": 23}
    dev250 = {"edit_exact": score - 33, "strict_noop_correct": 33, "predicted_noop": 33 + noedit}
    return dev75, dev250


def write_dev(path, step, recipe_path, manifest, panel, values):
    Path(path).write_text(json.dumps({
        "schema": night_arm.RESULT_SCHEMA, "status": "complete", "step": step,
        "bound_recipe": {"path": str(recipe_path), "sha256": sha(recipe_path) if Path(recipe_path).is_file() else "0" * 64},
        "checkpoint": {"path": str(Path(manifest).parent), "campaign_manifest_sha256": sha(manifest)},
        "panel": {"path": panel["path"], "sha256": panel["sha256"]},
        "summary": {"denominators": {"cases": panel["cases"]}, "counts": values, "panel_sha256": panel["sha256"],
                    "generation_max_new_tokens": 1024, "generation_runtime": BATCH},
        "generation_runtime": {"dtype": "bf16", "cuda_visible_devices": "0", **BATCH}, "final_set_access": False,
        "elapsed_seconds": 240.0}))


def write_checkpoint(archive, step, recipe_sha):
    directory = archive / "full" / f"checkpoint-{step}"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"weights %d" % step)
    (directory / "trainer-state.json").write_text(json.dumps({"schema": state.STATE_SCHEMA, "update": step, "sampler": {"cursor": 0}}))
    files = {path.name: state.file_entry(path) for path in directory.iterdir()}
    (directory / "campaign-manifest.json").write_text(json.dumps({"schema": state.CHECKPOINT_SCHEMA, "step": step,
                                                                  "recipe_sha256": recipe_sha, "files": files}))


def update_event(update, resumed_from=None, **override):
    event = {"event": "grpo_update", "update": update, "buffer_position": (update - 1) % 2, "resumed_from": resumed_from,
             "groups_trained": 4, "zero_variance_share_before": 0.3, "zero_variance_share_after": 0.0, "mean_abs_advantage": 0.3,
             "reward_mean_drawn": 0.4, "exact_rate_drawn": 0.3, "completion_tokens_mean_drawn": 100.0, "clip_fraction": 0.01,
             "loss": 0.02, "optimizer_step": True, "update_seconds": 20.0, "rollout_seconds": 150.0, "old_logprob_seconds": 3.0,
             "replay_nll": 0.3, "grad_norm": 1.1, "grad_norm_grpo": 1.0, "exec_groups_drawn": 10, "legacy_groups_drawn": 2,
             "exec_zero_variance_share_before": 0.3, "max_memory_reserved": 28 * 2**30}
    event.update(override)
    return event


class FakeLauncher(night_arm.Launcher):
    def __init__(self, test):
        self.test = test
        self.calls = []
        self.dev = {}
        self.val = {0: (0.30, 0.38)}
        self.cpt = {0: 1.0}
        self.event = lambda update: {}
        self.on_train = None
        self.trainer_codes = []

    def run(self, label, module_args, *, budget, on_stop):
        module = module_args[0]
        self.calls.append((module, module_args))
        args = {name: module_args[index + 1] for index, name in enumerate(module_args[:-1])
                if name.startswith("--") and not module_args[index + 1].startswith("--")}
        if module == night.TRAINER:
            return self.train(module_args)
        recipe_path = Path(args["--bound-recipe"])
        recipe = json.loads(recipe_path.read_text())
        if "--start" in module_args:
            checkpoint, step = Path(recipe["parent"]["path"]), 0
        else:
            checkpoint = Path(args["--checkpoint"])
            step = int(checkpoint.name.removeprefix("checkpoint-"))
        manifest = checkpoint / "campaign-manifest.json"
        common = {"status": "complete", "step": step, "bound_recipe": {"path": str(recipe_path), "sha256": sha(recipe_path)},
                  "checkpoint": {"path": str(checkpoint), "campaign_manifest_sha256": sha(manifest)}, "final_set_access": False}
        output = Path(args["--output"])
        if module == night.DEV_EVALUATOR:
            dev75, dev250 = self.dev.get(step, dev_counts())
            panel = recipe["dev_gate"]["panels"][args["--panel"]]
            write_dev(output, step, recipe_path, manifest, panel, dev75 if args["--panel"] == "dev75" else dev250)
        elif module == night.VAL_EVALUATOR:
            val = recipe["dev_gate"]["val"]
            n = val["limit"] or val["n_tasks"]
            all_pass, reward = self.val.get(step, (0.31, 0.39))
            output.write_text(json.dumps({**common, "schema": night.VAL_SCHEMA, "holdout_access": False,
                                          "tasks": {"sha256": val["tasks"]["sha256"], "n": n},
                                          "gate_version": {"fingerprint": "7f8e073fbedb4b9d"}, "elapsed_seconds": 60.0,
                                          "metrics": {"overall": {"n": n, "all_pass": all_pass, "mean_reward": reward,
                                                                  "exact_match": 0.2, "protocol_valid": 0.97}}}))
        else:
            self.test.assertEqual(module, night.CPT_PROBE)
            cpt = recipe["dev_gate"]["cpt"]
            output.write_text(json.dumps({**common, "schema": night.CPT_SCHEMA, "panel": {"sha256": cpt["panel"]["sha256"], "rows": 499},
                                          "mean_causal_nll": float(self.cpt.get(step, 1.005)), "load_seconds": 60.0, "eval_seconds": 40.0}))
        return 0

    def train(self, module_args):
        if self.trainer_codes:
            code = self.trainer_codes.pop(0)
            if code:
                return code
        recipe_path = Path(module_args[module_args.index("--recipe") + 1])
        recipe = json.loads(recipe_path.read_text())
        start = int(Path(module_args[module_args.index("--resume") + 1]).name.removeprefix("checkpoint-")) if "--resume" in module_args else 0
        target = min(step for step in rules.evaluation_steps() if step > start)
        if "--stop-after-update" in module_args:
            target = min(target, int(module_args[module_args.index("--stop-after-update") + 1]))
        archive = Path(recipe["outputs"]["archive"])
        archive.mkdir(parents=True, exist_ok=True)
        with (archive / "telemetry.jsonl").open("a") as stream:
            if start == 0:
                stream.write(json.dumps({"event": "run_start", "allocator": {"allocator_cap_mib": 30669.0}}) + "\n")
            for update in range(start + 1, target + 1):
                stream.write(json.dumps({**update_event(update, start or None), **self.event(update)}) + "\n")
            stream.write(json.dumps({"event": "checkpoint_durable", "step": target, "seconds": 220.0}) + "\n")
        write_checkpoint(archive, target, sha(recipe_path))
        if self.on_train is not None:
            self.on_train(target)
        return 0


def driver_recipe(root, kind="pilot"):
    recipe = synthetic_recipe(root, kind)
    parent = Path(recipe["parent"]["path"])
    start75, start250 = dev_counts()
    results = {}
    for name, values in (("dev75", start75), ("dev250", start250)):
        path = root / "start" / f"{name}.json"
        write_dev(path, 803, root / "sft.json", parent / "campaign-manifest.json", recipe["dev_gate"]["panels"][name], values)
        results[name] = {"path": str(path), "sha256": sha(path)}
    recipe["parent_reference"]["results"] = results
    recipe["parent_reference"]["metrics"] = offline_rules.offline_metrics({"counts": start75}, {"counts": start250})
    path = root / f"recipe-{kind}.json"
    path.write_text(json.dumps(recipe))
    return path


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recipe_path = driver_recipe(self.root)
        self.launcher = FakeLauncher(self)
        self.clock = {"now": 1000.0}
        self.deadline = 1000.0 + 10**6

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self):
        settings = night_arm.Settings(recipe_path=self.recipe_path, guard=False, eval_budget_seconds=100.0, min_train_seconds=10.0,
                                      job_result=self.root / "job-result.json", clock=lambda: self.clock["now"],
                                      deadline=self.deadline, stop_request_file=self.root / "stop-request")
        return night.ExecGRPOArm(settings, lambda s, w, stop: self.launcher, verify_files=False)

    def decision(self, step):
        return json.loads(self.arm().decisions(step)[-1].read_text())

    def modules(self, name):
        return [args for module, args in self.launcher.calls if module == name]

    def test_start_measurements_then_gates_then_rollback_selects_the_best_accepted(self):
        self.launcher.val.update({50: (0.31, 0.39), 100: (0.33, 0.40), 150: (0.32, 0.40), 200: (0.25, 0.30)})
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        starts = [args for args in self.modules(night.VAL_EVALUATOR) + self.modules(night.CPT_PROBE) if "--start" in args]
        self.assertEqual(len(starts), 2)
        self.assertEqual(self.decision(50)["panels"], ["cpt", "dev75", "val"])
        self.assertEqual(self.decision(150)["panels"], ["cpt", "dev250", "dev75", "val"])
        self.assertIn("score250_drop_from_start", [r["name"] for r in self.decision(150)["rule"]["rows"]])
        for step in (50, 100, 150):
            self.assertEqual(self.decision(step)["outcome"], rules.CONTINUE)
        self.assertEqual(self.decision(200)["outcome"], rules.ROLLBACK)
        summary = json.loads((self.root / "archive" / "pilot-summary.json").read_text())
        self.assertEqual(summary["status"], "rolled_back")
        self.assertEqual(summary["selected"]["step"], 100)
        self.assertTrue(summary["result_checkpoint"].endswith("checkpoint-100"))
        # checkpoint 100 had no DEV250: the driver runs it after the pilot
        selection = [a for a in self.modules(night.DEV_EVALUATOR) if a[a.index("--panel") + 1] == "dev250"]
        self.assertTrue(selection[-1][selection[-1].index("--checkpoint") + 1].endswith("checkpoint-100"))
        self.assertEqual(summary["selected"]["dev250"]["score250"], 109)
        self.assertAlmostEqual(summary["selected"]["cpt_rise_vs_start"], 0.005)
        job = json.loads((self.root / "gates" / "final-holdout-job.json").read_text())
        argv = job["command"]
        self.assertEqual(argv[argv.index("-m") + 1], "sepalith.exec_eval.jobs")
        models = json.loads(Path(argv[argv.index("--models") + 1]).read_text())["models"]
        self.assertEqual([m["label"] for m in models], ["arm-a-803", "c18-selected"])
        self.assertTrue(models[1]["path"].endswith("checkpoint-100"))
        self.assertTrue((self.root / "gates" / "final" / "completions-arm-a-803.jsonl").is_file())
        # running again does not repeat the start measurements or the evaluations
        calls = len(self.launcher.calls)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        self.assertEqual(len(self.launcher.calls), calls)

    def test_cpt_forgetting_rolls_back(self):
        self.launcher.cpt.update({50: 1.01, 100: 1.04})
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        rows = {r["name"]: r["outcome"] for r in self.decision(100)["rule"]["rows"]}
        self.assertEqual(rows["cpt_nll_rise"], rules.ROLLBACK)
        self.assertTrue(json.loads((self.root / "archive" / "pilot-summary.json").read_text())["result_checkpoint"].endswith("checkpoint-50"))

    def test_first_evaluation_rollback_returns_803(self):
        self.launcher.val[50] = (0.20, 0.30)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        summary = json.loads((self.root / "archive" / "pilot-summary.json").read_text())
        self.assertIsNone(summary["selected"])
        self.assertTrue(summary["result_checkpoint"].endswith("start/checkpoint-803"))

    def test_plateau_completes_the_pilot(self):
        self.launcher.val.update({step: (0.305, 0.385) for step in rules.evaluation_steps()})
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        summary = json.loads((self.root / "archive" / "pilot-summary.json").read_text())
        self.assertEqual((summary["status"], summary["last_evaluation"]), ("plateau", 300))
        self.assertEqual(summary["selected"]["step"], 50)  # equal val all-pass and reward: the earliest

    def test_stop_band_needs_the_user_and_an_override_continues(self):
        self.launcher.dev[100] = dev_counts(nofp=11)
        self.launcher.val.update({50: (0.32, 0.40), 100: (0.33, 0.41), 150: (0.34, 0.42)})
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        self.assertEqual(self.decision(100)["outcome"], rules.STOP_FOR_USER)
        arm = self.arm()
        (self.root / "gates" / "override-step-100.json").write_text(json.dumps({
            "schema": night_arm.OVERRIDE_SCHEMA, "step": 100, "bound_recipe_sha256": arm.recipe_sha, "outcome": "continue",
            "reason": "test"}))

        def hook(target):
            if target >= 150:
                self.clock["now"] = self.deadline + 1
        self.launcher.on_train = hook
        self.assertEqual(self.arm().run(), night_arm.EXIT_DEFERRED)
        self.assertEqual(self.decision(100)["decided_by"], "human_override")

    def test_evaluator_crash_stops_for_the_user(self):
        class Broken(FakeLauncher):
            def run(self, label, module_args, *, budget, on_stop):
                if module_args[0] == night.VAL_EVALUATOR and "--start" not in module_args:
                    return 1
                return super().run(label, module_args, budget=budget, on_stop=on_stop)
        self.launcher = Broken(self)
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        self.assertIn("evaluator crash", self.decision(50)["rule"]["problems"][0])

    def test_start_measurements_defer_near_the_deadline(self):
        self.deadline = 1000.0 + 100
        self.assertEqual(self.arm().run(), night_arm.EXIT_DEFERRED)
        self.assertFalse(self.modules(night.TRAINER))


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recipe_path = driver_recipe(self.root, "smoke")
        self.launcher = FakeLauncher(self)

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self):
        settings = night_arm.Settings(recipe_path=self.recipe_path, guard=False, min_train_seconds=10.0,
                                      job_result=self.root / "job-result.json", clock=lambda: 1000.0, deadline=10**9)
        return night.ExecGRPOArm(settings, lambda s, w, stop: self.launcher, verify_files=False)

    def test_smoke_trains_evaluates_and_resumes(self):
        self.assertEqual(night.run_smoke(self.arm()), night_arm.EXIT_DONE)
        receipt = json.loads((self.root / "archive" / "smoke-receipt.json").read_text())
        self.assertEqual((receipt["status"], receipt["updates"]), ("pass", [1, 2, 3, 4, 5, 6]), receipt["problems"])
        self.assertEqual(sorted(receipt["results"]), ["cpt", "dev75", "val"])
        self.assertAlmostEqual(receipt["cpt"]["rise"], 0.005)
        self.assertEqual(receipt["val_subset"]["n"], 8)
        self.assertGreater(receipt["projected_updates_per_night"], 0)
        self.assertEqual(receipt["memory_headroom_mib"], round(30669.0 - 28 * 1024, 1))
        trains = self.launcher.calls
        resumes = [a for m, a in trains if m == night.TRAINER and "--resume" in a]
        self.assertEqual(Path(resumes[0][resumes[0].index("--resume") + 1]).name, "checkpoint-4")
        self.assertEqual(len([a for m, a in trains if m == night.CPT_PROBE and "--start" in a]), 1)

    def test_smoke_fails_without_an_optimizer_step_or_legacy_draws(self):
        self.launcher.event = lambda update: {"optimizer_step": False, "legacy_groups_drawn": 0}
        self.assertEqual(night.run_smoke(self.arm()), night_arm.EXIT_FAILED)
        problems = " ".join(json.loads((self.root / "archive" / "smoke-receipt.json").read_text())["problems"])
        self.assertIn("no optimizer step ran", problems)
        self.assertIn("no legacy group", problems)


@unittest.skipUnless((JOBS / "C18-grpo-exec-smoke.json").is_file(), "C18 job files not written yet")
class JobCommandTests(unittest.TestCase):
    def test_job_commands_parse_with_the_driver(self):
        for name in ("C18-grpo-exec-smoke", "C18-grpo-exec-pilot-1"):
            argv = json.loads((JOBS / f"{name}.json").read_text())["command"]
            args = night.build_parser().parse_args(argv[argv.index("sepalith.training.grpo_exec.night") + 1:])
            self.assertEqual(args.command, "smoke" if "smoke" in name else "run")
            self.assertFalse(args.no_guard)
            local = REPO / Path(str(args.recipe)).relative_to(recipe_module.DEPLOY_ROOT)
            self.assertEqual(json.loads(local.read_text())["kind"], "smoke" if "smoke" in name else "pilot")


if __name__ == "__main__":
    unittest.main()
