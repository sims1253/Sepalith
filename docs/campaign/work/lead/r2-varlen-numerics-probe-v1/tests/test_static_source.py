import ast,json,sys,unittest
from pathlib import Path
PKT=Path(__file__).parents[1]
class Tests(unittest.TestCase):
 def test_gpu_source_has_no_optimizer_or_mutation(self):
  text=(PKT/'source/varlen_numerics_probe.py').read_text();tree=ast.parse(text)
  self.assertNotIn('torch.optim',text);self.assertNotIn('.step()',text);self.assertIn('no_optimizer_created',text)
  self.assertIn('file_identity(weights) != weight_identity_before',text)
 def test_command_and_guard_are_exact(self):
  cmd=json.loads((PKT/'command.json').read_text());guard=json.loads((PKT/'guard-command.json').read_text())
  self.assertEqual(cmd[0],'/usr/bin/env');self.assertIn('PYTHONNOUSERSITE=1',cmd);self.assertEqual(cmd[-2:],["--seed","20260914"])
  self.assertEqual(guard[guard.index('--seconds')+1],'2400');self.assertEqual(guard[guard.index('--admission-free-mib')+1],'14336')
 def test_sources_compile(self):
  for p in (PKT/'source').glob('*.py'):compile(p.read_text(),str(p),'exec')
if __name__=='__main__':unittest.main()
