#!/usr/bin/env python3
"""CPU-only contract tests for the frozen saved-FP32 restoration seam."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file

HELPER = Path(__file__).resolve().parents[1] / 'source/experiments/training/saved_precision.py'
spec = importlib.util.spec_from_file_location('saved_precision_under_review', HELPER)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.norm=torch.nn.Linear(3,1,bias=False,dtype=torch.bfloat16);self.proj=torch.nn.Linear(2,2,bias=False,dtype=torch.bfloat16)

class SavedPrecisionTests(unittest.TestCase):
 def test_exact_restore_preserves_bf16_and_rejects_bad_fp32(self):
  with tempfile.TemporaryDirectory(prefix='sepalith-dtype-gate-') as directory:
   root=Path(directory);exact=torch.tensor([[1.003497337,-0.3333377,0.100019]],dtype=torch.float32);bf16=torch.tensor([[1,2],[3,4]],dtype=torch.bfloat16);weights=root/'model.safetensors';save_file({'norm.weight':exact,'proj.weight':bf16},str(weights));model=Tiny();model.norm.weight.data.copy_(exact.to(torch.bfloat16));before=model.proj.weight.detach().clone();audit=module.restore_saved_fp32(model,weights);self.assertEqual(audit['fp32_tensors_restored'],1);self.assertTrue(audit['exact_saved_values_verified']);self.assertEqual(model.norm.weight.dtype,torch.float32);self.assertTrue(torch.equal(model.norm.weight.detach(),exact));self.assertEqual(model.proj.weight.dtype,torch.bfloat16);self.assertTrue(torch.equal(model.proj.weight.detach(),before));self.assertGreater(audit['tensors'][0]['changed_elements'],0)
   bad=root/'bad.safetensors';save_file({'norm.weight':torch.tensor([[float('nan'),0,0]],dtype=torch.float32)},str(bad))
   with self.assertRaises(AssertionError):module.restore_saved_fp32(Tiny(),bad)
   wrong=root/'wrong.safetensors';save_file({'missing.weight':exact},str(wrong))
   with self.assertRaises(AssertionError):module.restore_saved_fp32(Tiny(),wrong)

if __name__=='__main__':unittest.main(verbosity=2)
