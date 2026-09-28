#!/usr/bin/env python3
import json
from pathlib import Path
p=Path(__file__).resolve().parents[1]
c=json.loads((p/'root-commands.template.json').read_text())['commands']
cpu=c['cpu_frontdoor'];gpu=c['guard_child_after_cpu_preflight']
assert 'CUDA_VISIBLE_DEVICES=' in cpu
assert 'CUDA_VISIBLE_DEVICES=0' in gpu
assert 'CUDA_VISIBLE_DEVICES=' not in gpu
assert '--ordinary-canary-transition-admission' in cpu and '--ordinary-canary-transition-admission' in gpu
assert 'selected-packed330-transition-admission' not in ' '.join(gpu) or 'ROOT_FRESH_SELECTED_PACKED330_TRANSITION_ADMISSION' in gpu
trainer=(p/'source/experiments/training/full_weight_cpt_trainer.py').read_text()
assert 'xformers' not in trainer and 'varlen_attention' not in trainer
assert 'cu130-overlay' not in ' '.join(gpu)
print('PASS CPU masks CUDA; guarded child selects CUDA0; transition gate present; production runner remains ordinary')
