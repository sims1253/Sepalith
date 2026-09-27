"""Fail-closed front door for the prepared DDP adapter."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from a100_execution_transition import validate
from a100_distributed import ddp_training_arguments

def preflight_template(path: Path):
    value=json.loads(Path(path).read_text()); result=validate(value, verify_large=False)
    result["training_arguments"]=ddp_training_arguments(
        output_dir=value["outputs"]["trainer"], max_steps=value["runtime"]["max_steps"],
        learning_rate=value["runtime"]["learning_rate"], warmup_steps=value["runtime"]["warmup_steps"],
        seed=value["seed"])
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=("preflight-template","run")); p.add_argument("--recipe",type=Path,required=True)
    a=p.parse_args()
    if a.command=="preflight-template": print(json.dumps(preflight_template(a.recipe),sort_keys=True)); return 0
    raise RuntimeError("DDP CUDA run remains fail-closed pending root admission and actual-model parity wiring")
if __name__=="__main__": raise SystemExit(main())
