#!/usr/bin/env python3
"""LR-sweep arm close-out: pull adapter -> merge (CPU, gated) -> export Q8 -> battery.

Per-arm flow (established pieces, composed):
  1. hf download scholzmx/sepalith-lora/<run>/final_lora -> work/<arm>/final_lora
  2. merge onto the GDN base via the pft1_merge_b4.py gates (same 96-module profile)
  3. export Q8 GGUF (export path per b4 line)
  4. the frozen 513 battery (same paired pattern as s2_quality, ARMS=[banked-b4, arm])
Usage: lr_eval.py pull <arm-run-name> | merge <arm-dir> | all <arm-run-name>
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path('/home/m0hawk/Documents/Sepalith')
WT = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-e6aed5ff')
WORK = Path('/mnt/h/sepalith/runs/lr-sweep-20260910')
BASE_GGUF = REPO / 'experiments/models/b4_qwen35_2b-Q8_0.gguf'
VENV = REPO / '.venv-sft/bin/python'


def sh(cmd, **kw):
    print('$', ' '.join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def pull(run_name):
    dst = WORK / run_name / 'final_lora'
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh(['/usr/bin/env', 'hf', 'download', 'scholzmx/sepalith-lora',
        '--include', f'{run_name}/*', '--local-dir', str(dst) + '-repo'])
    import shutil as _sh
    _sh.move(str(dst) + f'-repo/{run_name}/final_lora', dst)
    print('pulled:', dst)


def merge(run_name):
    """Gated merge: reuse pft1_merge_b4.py flow via env-parameterized copy."""
    arm = WORK / run_name
    script = arm / 'merge_arm.py'
    src = (REPO / 'scripts/pft1_merge_b4.py').read_text()
    src = src.replace('LORA = Path("/mnt/h/sepalith/runs/b4_qwen35_2b/final_lora")',
                      f'LORA = Path("{arm}/final_lora")')
    src = src.replace('OUT = Path("/mnt/h/sepalith/runs/pft1_b4_merged")',
                      f'OUT = Path("{arm}/merged")')
    src = src.replace('print(f"base: {bp}", flush=True)',
                      'if not bp.exists():\n    bp = REPO / "experiments/models/qwen3.5-2b-base-text-hf"\nprint(f"base: {bp}", flush=True)')
    script.write_text(src)
    sh([VENV, script])
    print('merged:', arm / 'merged')


def all_(run_name):
    pull(run_name)
    merge(run_name)
    # export + battery steps land with the arms; battery runner reuses s2_quality pattern
    print('TODO export+battery for', run_name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['pull', 'merge', 'all'])
    ap.add_argument('run_name')
    a = ap.parse_args()
    dict(pull=pull, merge=lambda r: merge(r), all=all_)[a.action](a.run_name)
