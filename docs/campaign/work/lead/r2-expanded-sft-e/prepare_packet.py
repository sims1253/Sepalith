#!/usr/bin/env python3
"""Build the exact-source D/full500 to terminal1000 continuation packet."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
WORK = PLAN / "docs/campaign/work/lead/r2-expanded-sft-e"
D = PLAN / "docs/campaign/work/lead/r2-expanded-sft-d"
CHECKPOINT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-d/full/checkpoint-500")
OUTPUT = Path("/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-e")
ARCHIVE = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-e")
SUPERVISION = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-e-host-supervision")
RUNNER_ROOT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-sft-e-v1")
SOURCE_ID = "bb2f9fdc1d016679533beab7d9906c9b996c9ce59d36e54ff98101adbaeed384"
SCHEDULE_SHA = "2ad2074d194b99bdef6826eb74543669fd8a7cf3490aedd692844b6402bed498"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def record(path):
    return {"path": str(path), "sha256": sha(path)}


def build_recipe():
    recipe = json.loads((D / "recipe.json").read_text())
    checkpoint_manifest = json.loads((CHECKPOINT / "campaign-manifest.json").read_text())
    recipe.update({
        "id": "sft11-expanded-postsft500-e-resume500-20260914",
        "status": "prepared_exact_d_source_resume500_to_terminal1000_root_admission_required",
        "archive_dir": str(ARCHIVE),
        "output_dir": str(OUTPUT),
        "resume_from": str(CHECKPOINT),
        "resume_milestone": 500,
        "mandatory_stop_steps": [],
        "decision_steps": [],
        "milestones": [250, 500, 1000],
        "launch_authorized": False,
    })
    recipe["checkpoint"]["evaluation_steps"] = [250, 500, 750, 1000]
    recipe.pop("resume_identity_compatibility", None)
    recipe["recovery_binding"] = {
        "predecessor_attempt": "SFT11-expanded-postsft500-d",
        "resume_step": 500,
        "consumed_draws": 8000,
        "terminal_step": 1000,
        "terminal_consumed_draws": 16000,
        "schedule_sha256": SCHEDULE_SHA,
        "predecessor_source": SOURCE_ID,
        "current_source": SOURCE_ID,
        "checkpoint_manifest_sha256": sha(CHECKPOINT / "campaign-manifest.json"),
        "continuity": ["adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "campaign-state.json"],
        "resume_semantics": "ordinary exact-identity resume; no source compatibility exception or migration",
        "archive_rewrite_forbidden": True,
        "displaced_750_gate": "HF evaluation is scheduled at750, but exact frozen D source cannot declare750 as a decision or mandatory stop; terminal control remains1000",
    }
    recipe["source_binding"]["identity_binding"] = "identity.source and D/full500 source are byte-identical bb2f9f; ordinary exact-source resume, no migration"
    recipe["parent_binding"]["optimizer_resume"] = "required from exact full expanded D/checkpoint-500; base parent lineage unchanged"
    c_prefix = "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-c/full/checkpoint-250/"
    inputs = [row for row in recipe["inputs"] if not row["path"].startswith(c_prefix)]
    inputs.extend(record(CHECKPOINT / name) for name in ["campaign-manifest.json", *checkpoint_manifest["files"].keys()])
    inputs.extend((record(D / "recipe.json"), record(D / "source-manifest.json")))
    recipe["inputs"] = sorted({row["path"]: row for row in inputs}.values(), key=lambda row: row["path"])
    write(WORK / "recipe.json", recipe)
    return recipe


def build_runner(recipe):
    old = json.loads((D / "runner-recipe.json").read_text())
    runner = deepcopy(old)
    runner.update({
        "id": recipe["id"],
        "description": "Root-controlled exact-source SFT-11 E continuation: D/full500 optimizer/RNG/sampler to terminal1000 with scheduled HF diagnostics at750 and1000",
        "snapshot": SOURCE_ID,
    })
    runner["provenance"] = {
        "task": "SFT-11-expanded-E",
        "owner": "lead",
        "status": "prepared_root_review_snapshot_enqueue_and_launch_required",
        "acceptance": "Resume exact D/full500 through unchanged schedule to terminal full1000; D500 native gate remains prerequisite and native1000 acceptance remains separate",
        "prior_terminal_reason": "D stopped cleanly at mandatory500 with exact full checkpoint",
        "gate_semantics": "D frozen source schedules HF readout750 but cannot stop there; no750 mandatory decision is claimed",
        "budget_source": "Fresh per-job 6084-second trainer horizon under6144-second V4 guard; host safety may stop at any full50 cadence",
    }
    runner["inputs"] = deepcopy(recipe["inputs"])
    runner["inputs"].extend((record(WORK / "recipe.json"), record(D / "source-manifest.json")))
    for path in (
        PLAN / "docs/campaign/work/lead/host-memory-guard-v4/cuda_host_guard.py",
        PLAN / "docs/campaign/work/lead/host-memory-guard-v4/host_memory_policy.py",
        PLAN / "docs/campaign/work/lead/host-memory-guard-v4/post_load_cache.py",
        PLAN / "docs/campaign/work/lead/host-memory-guard-v4/root-policy-tests.json",
        PLAN / "docs/campaign/work/lead/r2-cpt-broad-f/prepare_guard_command.py",
    ):
        runner["inputs"].append(record(path))
    runner["inputs"] = sorted({row["path"]: row for row in runner["inputs"]}.values(), key=lambda row: row["path"])
    runner["steps"][0]["argv"][4] = str(WORK / "recipe.json")
    runner["steps"][1]["argv"][5] = str(SUPERVISION)
    runner["steps"][1]["id"] = "guarded-expanded-e-500-to1000"
    write(WORK / "runner-recipe.json", runner)
    return runner


def build_launcher():
    pins = {
        str(WORK / "recipe.json"): sha(WORK / "recipe.json"),
        str(WORK / "runner-recipe.json"): sha(WORK / "runner-recipe.json"),
        str(D / "source-manifest.json"): sha(D / "source-manifest.json"),
        str(CHECKPOINT / "campaign-manifest.json"): sha(CHECKPOINT / "campaign-manifest.json"),
    }
    source = '''"""Run one root-reviewed SFT-11 E queue item and retain supervisor identity."""
import datetime,hashlib,json,os
from pathlib import Path
import sys,time
sys.path.insert(0,"/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src")
from sepalith.runner import Runner
RUNNER_ROOT={runner_root!r}
PINS={pins!r}
for name,expected in PINS.items():
 actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
 if actual!=expected:raise RuntimeError(f"{{name}} differs from reviewed E packet")
for path in map(Path,{fresh!r}):
 if path.exists():raise RuntimeError(f"E output is not fresh: {{path}}")
work=Path(__file__).resolve().parent;started=time.monotonic();launch={{"at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pid":os.getpid(),"start_stat":Path('/proc/self/stat').read_text(),"owner":"root","task":"SFT-11-expanded-E-resume500-to1000","pins":PINS}}
with (work/'supervisor-launch.json').open('x') as stream:json.dump(launch,stream,indent=2)
terminal={{}}
try:
 runner=Runner(RUNNER_ROOT);runner.resume();result=runner.run_next();terminal={{"result":result,"seconds":time.monotonic()-started}}
except BaseException as error:
 terminal={{"failure":repr(error),"seconds":time.monotonic()-started}};raise
finally:
 terminal['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (work/'supervisor-terminal.json').open('x') as stream:json.dump(terminal,stream,indent=2)
'''.format(runner_root=str(RUNNER_ROOT), pins=pins, fresh=[str(OUTPUT), str(ARCHIVE), str(SUPERVISION)])
    (WORK / "run_owned.py").write_text(source)


def build_commands():
    python = "/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python"
    runner = "/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src/sepalith/runner.py"
    commands = {
        "schema": "sepalith.sft11.expanded-e-resume500.commands.v1",
        "status": "prepared_not_executed_root_owned",
        "pins": {
            "runner_snapshot": SOURCE_ID,
            "recipe_sha256": sha(WORK / "recipe.json"),
            "runner_recipe_sha256": sha(WORK / "runner-recipe.json"),
            "d_source_manifest_sha256": sha(D / "source-manifest.json"),
            "resume_manifest_sha256": sha(CHECKPOINT / "campaign-manifest.json"),
            "run_owned_sha256": sha(WORK / "run_owned.py"),
        },
        "preflight": [
            "/usr/bin/env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "PYTHONPATH=" + str(D / "source/packages/sepalith/src") + ":" + str(D / "source/experiments/training"),
            python, str(D / "source/experiments/training/campaign_expanded_sft.py"), str(WORK / "recipe.json"), "--preflight-only",
        ],
        "snapshot": [python, runner, "--state", str(RUNNER_ROOT), "snapshot", "--repo", str(D / "source"), "--include", "experiments", "--include", "packages"],
        "enqueue_after_snapshot_matches_pin": [python, runner, "--state", str(RUNNER_ROOT), "enqueue", str(WORK / "runner-recipe.json")],
        "launch_after_root_review_and_native_d500_gate": [python, str(WORK / "run_owned.py")],
    }
    write(WORK / "commands.json", commands)


def main():
    recipe = build_recipe()
    build_runner(recipe)
    build_launcher()
    build_commands()
    print(json.dumps({"recipe": sha(WORK / "recipe.json"), "runner_recipe": sha(WORK / "runner-recipe.json"), "run_owned": sha(WORK / "run_owned.py"), "commands": sha(WORK / "commands.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
