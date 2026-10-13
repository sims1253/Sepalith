#!/usr/bin/env python3
"""Show that the C05 DEV driver's default path behaves exactly like the old one.

Loads ``run_dev_generation.py`` and ``campaign_eval.py`` as they were at
``--base-ref`` (default ``origin/main``) and runs both drivers through the
harness in ``packages/sepalith/tests/training/test_dev_generation_runtime.py``:
same fixture, same real tiny Llama on CPU (loaded through the driver's own
``from_pretrained`` call), CUDA availability mocked, no GPU touched.

Compared after removing wall-clock fields and the per-run output directory:
the ``from_pretrained`` arguments, every forward and ``generate`` call (keyword
arguments and a hash of the input ids), the ``eval()`` call, the printed
line, the result JSON and the per-case JSON.  The new driver is run twice:
with no new flags and with ``--dtype bf16 --panel dev75`` spelled out.

Run with the training interpreter on CPU, e.g.
``CUDA_VISIBLE_DEVICES= PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python
docs/training/plan-20260927/c05/evaluator_equivalence.py --receipt docs/training/plan-20260927/receipts/C05-evaluator-equivalence.json``
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
DRIVER = "packages/sepalith/src/sepalith/training/eval/run_dev_generation.py"
EVALUATOR = "packages/sepalith/src/sepalith/training/eval/campaign_eval.py"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def module_from_source(name, source, filename):
    module = types.ModuleType(name)
    module.__file__ = filename
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "packages/sepalith/tests/training"))
    import torch
    torch.set_num_threads(min(8, torch.get_num_threads()))
    harness = importlib.import_module("test_dev_generation_runtime")
    new_driver = harness.driver

    base = git("rev-parse", args.base_ref).strip()
    old_driver_source = git("show", f"{base}:{DRIVER}")
    old_evaluator_source = git("show", f"{base}:{EVALUATOR}")
    old_evaluator = module_from_source("old_campaign_eval", old_evaluator_source, f"{base}:{EVALUATOR}")
    old_driver = module_from_source("old_run_dev_generation", old_driver_source, f"{base}:{DRIVER}")

    with tempfile.TemporaryDirectory(prefix="sepalith-c05-equivalence-") as tmp:
        tiny = harness.tiny_checkpoint(Path(tmp) / "tiny")
        fx = harness.build_fixture(Path(tmp) / "fixture")
        records = {}
        run = harness.new_run(fx, "old")
        # The old driver imports development_evaluator inside main(); serve it the old module.
        with mock.patch.dict(sys.modules, {"sepalith.training.eval.campaign_eval": old_evaluator}):
            records["old"] = harness.normalized(harness.run_driver(old_driver, fx, run, tiny), run.dir)
        for name, argv in (("new_default", []), ("new_explicit_bf16_dev75", ["--dtype", "bf16", "--panel", "dev75"])):
            run = harness.new_run(fx, name)
            admission = harness.write_admission(fx, panel_sha=harness.sha(fx.dev75), output=run.output)
            records[name] = harness.normalized(
                harness.run_driver(new_driver, fx, run, tiny, argv, admission=admission), run.dir)

    old = records["old"]
    calls = [call[0] for call in old["calls"]]
    comparison = {name: {part: records[name][part] == old[part] for part in ("loads", "calls", "stdout", "files")}
                  for name in ("new_default", "new_explicit_bf16_dev75")}
    identical = all(all(parts.values()) for parts in comparison.values())
    receipt = {
        "schema": "sepalith.c05.evaluator-equivalence.v1",
        "base_ref": args.base_ref, "base_commit": base,
        "sources": {
            "old_driver_sha256": hashlib.sha256(old_driver_source.encode()).hexdigest(),
            "old_evaluator_sha256": hashlib.sha256(old_evaluator_source.encode()).hexdigest(),
            "new_driver_sha256": hashlib.sha256((ROOT / DRIVER).read_bytes()).hexdigest(),
            "new_evaluator_sha256": hashlib.sha256((ROOT / EVALUATOR).read_bytes()).hexdigest(),
        },
        "fixture": {"panel_cases": 75, "generation_max_new_tokens": harness.CAP,
                    "model": "random one-layer LlamaForCausalLM (vocab 1024, hidden 16), CPU, loaded in the dtype the driver requests",
                    "mocked": ["load_bound", "identity", "verify_checkpoint", "PANEL_SHA", "torch.cuda.is_available",
                               "torch.cuda.device_count", "load_pinned_reference_tokenizer", "AutoTokenizer.from_pretrained",
                               "tokenizer alignment/audit hooks", "parameter census (381 tensors, 2,516,756,480)"]},
        "observed_old": {"from_pretrained": old["loads"], "forward_calls": calls.count("forward"),
                         "generate_calls": calls.count("generate"), "eval_calls": calls.count("eval"),
                         "generate_kwargs": next(call[1] for call in old["calls"] if call[0] == "generate")},
        "normalized_away": ["elapsed_seconds", "generation_seconds", "cases.sha256 and printed sha256 (they hash files that hold generation_seconds)", "the per-run output directory"],
        "identical": comparison,
        "result": "identical" if identical else "differs",
    }
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.write_text(text)
    print(text, end="")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
