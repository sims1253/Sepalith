"""Permit the planned final checkpoint in addition to ordinary cadence boundaries.

The frozen trainer already saves and evaluates max_steps. Only its execution-stop
validator incorrectly requires max_steps to be divisible by checkpoint cadence.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

TRAINER=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt450-cadence64-continuation-preparation-v2/source/experiments/training/full_weight_cpt_trainer.py')
EXPECTED='5184354d974cc9bedfbc055f1d6b4c7091582e50a6d2b68c677c6a43ddfc79ee'
OLD='(stop - offset) % recipe["runtime"]["checkpoint_every"] == 0'
NEW='stop == recipe["runtime"]["max_steps"] or (stop - offset) % recipe["runtime"]["checkpoint_every"] == 0'

def load():
    source=TRAINER.read_text()
    assert hashlib.sha256(TRAINER.read_bytes()).hexdigest()==EXPECTED
    sys.path.insert(0,str(TRAINER.parent))
    spec=importlib.util.spec_from_file_location('cpt_terminal_trainer',TRAINER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='validate_execution_stop')
    code=ast.get_source_segment(source,function)
    assert code.count(OLD)==1
    exec(compile(code.replace(OLD,NEW),str(Path(__file__).resolve()),'exec'),module.__dict__)
    return module

def main():
    return load().main()

if __name__=='__main__':
    raise SystemExit(main())
