#!/usr/bin/env python3
"""CPU-only contract tests for the proposed saved-FP32 restoration seam."""
import importlib.util
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

HELPER = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-long-eval-root-v2/saved_precision.py')
spec = importlib.util.spec_from_file_location('saved_precision_under_review', HELPER)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Linear(3, 1, bias=False, dtype=torch.bfloat16)
        self.proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)


with tempfile.TemporaryDirectory(prefix='sepalith-dtype-audit-') as directory:
    root = Path(directory)
    exact = torch.tensor([[1.003497337, -0.3333377, 0.100019]], dtype=torch.float32)
    bf16 = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    weights = root/'model.safetensors'
    save_file({'norm.weight': exact, 'proj.weight': bf16}, str(weights))
    model = Tiny()
    model.norm.weight.data.copy_(exact.to(torch.bfloat16))
    audit = module.restore_saved_fp32(model, weights)
    assert audit['fp32_tensors_restored'] == 1 and audit['exact_saved_values_verified']
    assert model.norm.weight.dtype == torch.float32
    assert torch.equal(model.norm.weight.detach(), exact)
    assert model.proj.weight.dtype == torch.bfloat16
    assert audit['tensors'][0]['changed_elements'] > 0

    bad = root/'bad.safetensors'
    save_file({'norm.weight': torch.tensor([[float('nan'), 0, 0]], dtype=torch.float32)}, str(bad))
    try:
        module.restore_saved_fp32(Tiny(), bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('nonfinite saved FP32 tensor was accepted')

    wrong = root/'wrong.safetensors'
    save_file({'missing.weight': exact}, str(wrong))
    try:
        module.restore_saved_fp32(Tiny(), wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing parameter name was accepted')

print('PASS: exact restore, BF16 preservation, nonfinite rejection, missing-name rejection')
