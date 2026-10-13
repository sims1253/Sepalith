"""DEV generation dtype, device pinning and two-panel runs on a tiny CPU model.

``run_driver`` runs ``run_dev_generation.main`` end to end with only the
production-specific edges replaced: the bound recipe and checkpoint receipts,
the pinned tokenizer contract, and CUDA availability.  The model is a real
one-layer Llama, loaded through the driver's own ``from_pretrained`` call in
the requested dtype, on CPU.  ``docs/training/plan-20260927/c05/evaluator_equivalence.py``
reuses this harness to compare the default path with the pre-C05 driver.
"""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import transformers

from sepalith.campaign_protocol import Cursor, Position, PromptContext, ReplacementRange
from sepalith.training.contracts import campaign_tokenizer_contract, trainer_tokenizer_alignment
from sepalith.training.eval import campaign_eval, milestone_gate, prepare_dev_gate
from sepalith.training.eval import run_dev_generation as driver

CAP = 8
PARAMETER_TENSORS, PARAMETERS = 381, 2_516_756_480


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context(value="x <- 1"):
    return PromptContext(
        path="R/example.R", prefix=("f <- function() {",), selected_references=(), history=(), diagnostics=(),
        retrieval=(), scope_mode="off", scope_lines=(), suffix_lines=("}",), region_old=(value,),
        cursor=Cursor(0, len(value), len(value)), replacement_range=ReplacementRange(
            "file:///R/example.R", 1, "0" * 64, Position(1, 0), Position(1, len(value))),
    )


class TextTokenizer:
    """Lossless character tokenizer; not a claim about the production tokenizer."""

    def encode(self, text, **kwargs):
        return [ord(char) + 512 for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token - 512) if token >= 512 else "�" for token in ids)


def write_panel(path, count, prefix):
    rows = []
    for index in range(count):
        noop = index % 3 == 0
        rows.append({"id": f"{prefix}-{index:03d}", "split": "dev", "family": "no_op" if noop else "rename_propagation",
                     "package_id": f"pkg-{index % 7}", "operation": "no_op" if noop else "replace",
                     "region_new": ["x <- 1"] if noop else [f"x <- {index + 100}"], "context": context().to_dict()})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def tiny_checkpoint(directory):
    config = transformers.LlamaConfig(
        vocab_size=1024, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=4096, bos_token_id=0, eos_token_id=1, pad_token_id=1,
        tie_word_embeddings=False)
    torch.manual_seed(3407)
    transformers.LlamaForCausalLM(config).save_pretrained(directory)
    return Path(directory)


class DenseView:
    """A tiny model that reports the production parameter census.

    Forward, generate, eval/train and module traversal go to the real tiny
    model; calls are recorded so runs can be compared.
    """

    def __init__(self, inner, log):
        self.inner, self.log = inner, log

    @property
    def training(self):
        return self.inner.training

    def named_parameters(self):
        sizes = [1] * (PARAMETER_TENSORS - 1) + [PARAMETERS - (PARAMETER_TENSORS - 1)]
        return [(f"model.p{i}", SimpleNamespace(numel=lambda n=n: n)) for i, n in enumerate(sizes)]

    def parameters(self):
        return self.inner.parameters()

    def modules(self):
        return self.inner.modules()

    def eval(self):
        self.log.append(["eval"])
        self.inner.eval()
        return self

    def train(self, mode=True):
        self.inner.train(mode)
        return self

    def __call__(self, **kwargs):
        self.log.append(["forward", sorted(kwargs), kwargs.get("use_cache")])
        return self.inner(**kwargs)

    def generate(self, **kwargs):
        ids = kwargs["input_ids"]
        self.log.append(["generate", {k: v for k, v in kwargs.items() if k not in ("input_ids", "attention_mask")},
                         hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest(),
                         kwargs["attention_mask"].tolist() == torch.ones_like(ids).tolist()])
        return self.inner.generate(**kwargs)


def build_fixture(root, *, dev250_cases=250):
    """Write the shared inputs (recipe, checkpoint receipts, panels) for driver runs."""
    root = Path(root)
    checkpoint = root / "archive/full/checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "campaign-manifest.json").write_text("{}\n")
    (checkpoint / "tokenizer.json").write_text('{"fixture": true}\n')
    dev75 = write_panel(root / "dev75.jsonl", 75, "d75")
    dev250 = write_panel(root / "dev250.jsonl", dev250_cases, "d250")
    recipe = {"dev_gate": {"mandatory_steps": [2], "generation_max_new_tokens": CAP},
              "parent": {"path": str(root / "parent"), "files": {"tokenizer.json": sha(checkpoint / "tokenizer.json")}},
              "development": {"panel": {"path": str(dev75), "sha256": sha(dev75)}},
              "cohort": {"max_sequence_tokens": 4096}}
    recipe_path = root / "bound.json"
    recipe_path.write_text(json.dumps(recipe, sort_keys=True) + "\n")
    return SimpleNamespace(root=root, checkpoint=checkpoint, dev75=dev75, dev250=dev250, recipe=recipe,
                           recipe_path=recipe_path)


def new_run(fx, name):
    """Fresh output locations for one driver run over a shared fixture."""
    directory = fx.root / "runs" / name
    return SimpleNamespace(dir=directory, output=directory / "dev75.json", dev250_output=directory / "dev250.json")


def write_admission(fx, *, panel_sha, output, extra=None):
    admission = {"schema": "sepalith.sft11.full-weight-edit-dev-generation-admission.v1", "status": "admitted",
                 "purpose": "development_generation", "cuda_authorized": True, "final_set_access": False,
                 "bound_recipe_sha256": sha(fx.recipe_path),
                 "checkpoint_manifest_sha256": sha(fx.checkpoint / "campaign-manifest.json"),
                 "panel_sha256": panel_sha, "output": str(output), "deadline_seconds": 3600, **(extra or {})}
    path = fx.root / f"admission-{len(list(fx.root.glob('admission-*')))}.json"
    path.write_text(json.dumps(admission) + "\n")
    return path


def run_driver(module, fx, run, tiny, argv=(), *, cuda_visible="0", admission=None):
    """Run ``module.main`` on fixture ``fx`` into ``run``; return the recorded calls and files."""
    log, loads = [], []
    from_pretrained_original = transformers.AutoModelForCausalLM.from_pretrained

    def from_pretrained(path, **kwargs):
        loads.append({"path": path, **{k: (str(v) if k == "torch_dtype" else v) for k, v in kwargs.items()}})
        inner = from_pretrained_original(str(tiny), torch_dtype=kwargs["torch_dtype"], local_files_only=True)
        return DenseView(inner, log)

    stdout = io.StringIO()
    environ = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    if cuda_visible is not None:
        environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    with ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, environ, clear=True))
        stack.enter_context(mock.patch.object(module, "load_bound", lambda path: json.loads(Path(path).read_text())))
        stack.enter_context(mock.patch.object(module, "identity", lambda recipe: {"fixture": True}))
        stack.enter_context(mock.patch.object(module, "verify_checkpoint", lambda *args, **kwargs: None))
        stack.enter_context(mock.patch.object(module, "PANEL_SHA", sha(fx.dev75)))
        if hasattr(module, "DEV250_PANEL_SHA"):
            stack.enter_context(mock.patch.object(module, "DEV250_PANEL_SHA", sha(fx.dev250)))
            stack.enter_context(mock.patch.object(module, "DEV250_CASES", len(fx.dev250.read_text().splitlines())))
        stack.enter_context(mock.patch.object(torch.cuda, "is_available", lambda: True))
        stack.enter_context(mock.patch.object(torch.cuda, "device_count", lambda: 1))
        stack.enter_context(mock.patch.object(campaign_tokenizer_contract, "load_pinned_reference_tokenizer",
                                              lambda path: "reference"))
        stack.enter_context(mock.patch.object(transformers.AutoTokenizer, "from_pretrained",
                                              lambda *args, **kwargs: TextTokenizer()))
        stack.enter_context(mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained))
        stack.enter_context(mock.patch.object(trainer_tokenizer_alignment, "restore_trainer_eog_alignment",
                                              lambda model, tokenizer, reference: {"repaired": []}))
        stack.enter_context(mock.patch.object(trainer_tokenizer_alignment, "embedding_identity", lambda model: {}))
        stack.enter_context(mock.patch.object(trainer_tokenizer_alignment, "assert_runtime_tokenizer",
                                              lambda *args: {"stage": args[-1]}))
        stack.enter_context(redirect_stdout(stdout))
        if admission is None:
            admission = write_admission(fx, panel_sha=sha(fx.dev75), output=run.output)
        # Drive main() through sys.argv, as the command line does (the pre-C05 main takes no arguments).
        stack.enter_context(mock.patch.object(sys, "argv", [
            "run_dev_generation", "--bound-recipe", str(fx.recipe_path), "--checkpoint", str(fx.checkpoint),
            "--dev-admission", str(admission), "--output", str(run.output), *argv]))
        module.main()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    files = {}
    for path in (run.output, run.dev250_output):
        if path.exists():
            result = json.loads(path.read_text())
            files[path.name] = {"result": result, "cases": json.loads(Path(result["cases"]["path"]).read_text())}
    return {"loads": loads, "calls": log, "stdout": stdout.getvalue().splitlines(), "files": files,
            "cuda_visible_devices": visible}


def normalized(record, run_dir):
    """Drop wall-clock fields and the run directory so two runs can be compared."""
    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items() if k not in ("elapsed_seconds", "generation_seconds")}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value.replace(str(run_dir), "<run>") if isinstance(value, str) else value
    record = scrub(record)
    for item in record["files"].values():
        item["result"]["cases"].pop("sha256")  # cases.json holds generation_seconds
    record["stdout"] = [{k: v for k, v in json.loads(line).items() if k != "sha256"} for line in record["stdout"]]
    return record


class DevGenerationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(8, torch.get_num_threads()))
        cls.tmp = tempfile.TemporaryDirectory(prefix="sepalith-c05-")
        cls.tiny = tiny_checkpoint(Path(cls.tmp.name) / "tiny")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def fixture(self, **kwargs):
        return build_fixture(Path(tempfile.mkdtemp(dir=self.tmp.name)), **kwargs)

    def both_argv(self, fx, run):
        return ["--panel", "both", "--dev250-panel", str(fx.dev250), "--dev250-output", str(run.dev250_output)]

    def test_default_run_is_the_pc_bf16_dev75_call(self):
        fx = self.fixture()
        run = new_run(fx, "default")
        record = run_driver(driver, fx, run, self.tiny)
        self.assertEqual(record["loads"], [{"path": str(fx.checkpoint), "torch_dtype": "torch.bfloat16",
                                            "device_map": {"": "cuda"}, "local_files_only": True,
                                            "trust_remote_code": False}])
        generate = [call[1] for call in record["calls"] if call[0] == "generate"]
        self.assertEqual(len(generate), 75)
        self.assertEqual(generate[0], {"max_new_tokens": CAP, "do_sample": False, "num_beams": 1,
                                       "eos_token_id": [1, 130073], "pad_token_id": 1, "bos_token_id": 0,
                                       "repetition_penalty": 1.0, "return_dict_in_generate": False,
                                       "use_cache": True})
        result = record["files"]["dev75.json"]["result"]
        self.assertEqual(set(result), {"schema", "status", "step", "bound_recipe", "checkpoint", "panel", "summary",
                                       "cases", "tokenizer_contract", "tokenizer_repair", "elapsed_seconds",
                                       "final_set_access"})
        self.assertEqual(result["summary"]["denominators"]["cases"], 75)
        self.assertEqual(len(record["stdout"]), 1)
        self.assertNotIn("dev250.json", record["files"])
        explicit_run = new_run(fx, "explicit")
        explicit = run_driver(driver, fx, explicit_run, self.tiny, ["--dtype", "bf16", "--panel", "dev75"],
                              admission=write_admission(fx, panel_sha=sha(fx.dev75), output=explicit_run.output))
        self.assertEqual(normalized(explicit, explicit_run.dir), normalized(record, run.dir))

    def test_evaluator_runs_in_all_three_dtypes_on_a_tiny_cpu_model(self):
        fx = self.fixture()
        for dtype, torch_dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
            with self.subTest(dtype=dtype):
                model = campaign_eval.load_development_model(self.tiny, dtype=dtype, device="cpu")
                self.assertEqual({p.dtype for p in model.parameters()}, {torch_dtype})
                record = run_driver(driver, fx, new_run(fx, dtype), self.tiny, ["--dtype", dtype])
                self.assertEqual(record["loads"][0]["torch_dtype"], str(torch_dtype))
                self.assertEqual(record["loads"][0]["device_map"], {"": "cuda"})
                result = record["files"]["dev75.json"]["result"]
                summary, cases = result["summary"], record["files"]["dev75.json"]["cases"]["results"]
                self.assertEqual(summary["denominators"]["cases"], 75)
                self.assertEqual(len(cases), 75)
                self.assertTrue(all(0 < len(c["generated_ids"]) <= CAP for c in cases))
                self.assertTrue(0 < summary["target_nll"] < 100 and 0 < summary["prompt_nll"] < 100)
                if dtype == "bf16":
                    self.assertNotIn("generation_runtime", result)
                    prepare_dev_gate.require_bf16_result(result)
                else:
                    self.assertEqual(result["generation_runtime"], {"dtype": dtype, "cuda_visible_devices": "0"})
                    with self.assertRaisesRegex(ValueError, "only bf16"):
                        prepare_dev_gate.require_bf16_result(result)

    def test_both_panels_share_one_model_load_and_match_single_panel_runs(self):
        fx = self.fixture()
        run = new_run(fx, "both")
        admission = write_admission(fx, panel_sha=sha(fx.dev75), output=run.output,
                                    extra={"dev250_panel_sha256": sha(fx.dev250),
                                           "dev250_output": str(run.dev250_output)})
        both = run_driver(driver, fx, run, self.tiny, self.both_argv(fx, run), admission=admission)
        self.assertEqual(len(both["loads"]), 1)
        self.assertEqual([c[0] for c in both["calls"]].count("eval"), 1)
        self.assertEqual(len(both["stdout"]), 2)
        dev250_result = both["files"]["dev250.json"]["result"]
        self.assertEqual(dev250_result["summary"]["denominators"]["cases"], 250)
        self.assertEqual(dev250_result["panel"], {"path": str(fx.dev250), "sha256": sha(fx.dev250)})
        self.assertNotIn("generation_runtime", dev250_result)

        run75 = new_run(fx, "dev75")
        alone75 = run_driver(driver, fx, run75, self.tiny,
                             admission=write_admission(fx, panel_sha=sha(fx.dev75), output=run75.output))
        run250 = new_run(fx, "dev250")
        alone250 = run_driver(driver, fx, run250, self.tiny, ["--panel", "dev250", "--dev250-panel", str(fx.dev250)],
                              admission=write_admission(fx, panel_sha=sha(fx.dev250), output=run250.output))
        both_n = normalized(both, run.dir)
        self.assertEqual(both_n["files"]["dev75.json"], normalized(alone75, run75.dir)["files"]["dev75.json"])
        # With --panel dev250, --output (dev75.json in the run directory) holds the DEV250 result.
        single = normalized(alone250, run250.dir)["files"]["dev75.json"]
        self.assertEqual(single["result"]["panel"], dev250_result["panel"])
        self.assertEqual({k: v for k, v in both_n["files"]["dev250.json"]["result"]["summary"].items() if k != "cases_path"},
                         {k: v for k, v in single["result"]["summary"].items() if k != "cases_path"})
        self.assertEqual(both_n["files"]["dev250.json"]["cases"]["results"], single["cases"]["results"])

    def test_both_panels_require_the_extended_admission(self):
        fx = self.fixture()
        run = new_run(fx, "refused")
        with self.assertRaisesRegex(ValueError, "admission fields differ"):
            run_driver(driver, fx, run, self.tiny, self.both_argv(fx, run))
        wrong = write_admission(fx, panel_sha=sha(fx.dev75), output=run.output,
                                extra={"dev250_panel_sha256": sha(fx.dev75), "dev250_output": str(run.dev250_output)})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            run_driver(driver, fx, run, self.tiny, self.both_argv(fx, run), admission=wrong)
        with self.assertRaisesRegex(ValueError, "--dev250-output goes with --panel both"):
            run_driver(driver, fx, run, self.tiny, ["--dev250-output", str(run.dev250_output)])
        self.assertFalse(run.output.exists() or run.dev250_output.exists())

    def test_cuda_device_pinning(self):
        self.assertEqual(campaign_eval.pin_cuda_device(1, env := {}), "1")
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(campaign_eval.pin_cuda_device(1, env := {"CUDA_VISIBLE_DEVICES": "3, 5"}), "5")
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "5"})
        self.assertEqual(campaign_eval.pin_cuda_device(0, {"CUDA_VISIBLE_DEVICES": "2"}), "2")
        for index, environ, message in ((2, {"CUDA_VISIBLE_DEVICES": "0,1"}, "outside"),
                                        (0, {"CUDA_VISIBLE_DEVICES": ""}, "hides every"),
                                        (-1, {}, "non-negative"), (True, {}, "non-negative")):
            with self.assertRaisesRegex(ValueError, message):
                campaign_eval.pin_cuda_device(index, environ)
        # Kaggle: two T4s and no lease variable; pin the second one.
        fx = self.fixture()
        run = new_run(fx, "pinned")
        record = run_driver(driver, fx, run, self.tiny, ["--cuda-device", "1", "--dtype", "fp16"], cuda_visible=None)
        self.assertEqual(record["cuda_visible_devices"], "1")
        self.assertEqual(record["files"]["dev75.json"]["result"]["generation_runtime"],
                         {"dtype": "fp16", "cuda_visible_devices": "1"})
        # Without a pin or a lease the driver still refuses to pick a device.
        with self.assertRaisesRegex(ValueError, "root CUDA lease"):
            run_driver(driver, fx, new_run(fx, "unleased"), self.tiny, cuda_visible=None)

    def test_gate_evidence_rejects_non_bf16_results(self):
        milestone_gate.require_bf16_result({"summary": {}})
        milestone_gate.require_bf16_result({"generation_runtime": {"dtype": "bf16", "cuda_visible_devices": "1"}})
        for dtype in ("fp16", "fp32"):
            with self.assertRaisesRegex(ValueError, "only bf16"):
                milestone_gate.require_bf16_result({"generation_runtime": {"dtype": dtype}})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            campaign_eval.generation_dtype("int8")

    @unittest.skipUnless(Path(milestone_gate.DEV250_PANEL_PATH).is_file(), "DEV250 v1 panel is not on this host")
    def test_pinned_dev250_panel_loads_in_the_evaluator(self):
        path = Path(milestone_gate.DEV250_PANEL_PATH)
        self.assertEqual(sha(path), milestone_gate.DEV250_PANEL_SHA)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        self.assertEqual(len(rows), milestone_gate.DEV250_CASES)
        with tempfile.TemporaryDirectory() as tmp:
            campaign_eval.development_evaluator({
                "development_panel": {"path": str(path), "sha256": milestone_gate.DEV250_PANEL_SHA},
                "renderer_id": "zeta2-prm03-v1", "development_case_ids": sorted(r["id"] for r in rows),
                "development_max_new_tokens": 1024, "parameters": {"max_sequence_tokens": 16384},
                "evaluation_output_directory": tmp})


if __name__ == "__main__":
    unittest.main(verbosity=2)
