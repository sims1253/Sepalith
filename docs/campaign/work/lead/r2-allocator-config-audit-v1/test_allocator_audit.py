import importlib.util,os,sys,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).parent
SRC=HERE/'source/experiments/training/allocator_runtime_audit.py'
spec=importlib.util.spec_from_file_location('allocator_runtime_audit',SRC);m=importlib.util.module_from_spec(spec);sys.modules['allocator_runtime_audit']=m;spec.loader.exec_module(m)
class Memory:
 @staticmethod
 def get_allocator_backend():return 'native'
class Cuda:
 memory=Memory()
 @staticmethod
 def is_initialized():return False
class Torch:cuda=Cuda()
class Tests(unittest.TestCase):
 def test_wsl_expandable_removed_and_roundup_reported(self):
  env={'WSL_DISTRO_NAME':'Ubuntu-22.04','PYTORCH_ALLOC_CONF':'expandable_segments:True'}
  with patch.dict(os.environ,env,clear=True):
   before=m.capture_before_unsloth();os.environ['PYTORCH_ALLOC_CONF']='roundup_power2_divisions:[32:256,64:128,256:64,>:32]';out=m.observe_after_unsloth(Torch,before)
  self.assertEqual(out['interpretation'],'expandable_segments_requested_but_removed_by_unsloth_wsl_policy');self.assertFalse(out['expandable_segments_effective_by_environment']);self.assertTrue(out['roundup_fallback_effective_by_environment']);self.assertEqual(out['allocator_backend'],'native')
 def test_preserved_unified_config(self):
  with patch.dict(os.environ,{'PYTORCH_ALLOC_CONF':'max_split_size_mb:128'},clear=True):
   before=m.capture_before_unsloth();out=m.observe_after_unsloth(Torch,before)
  self.assertEqual(out['interpretation'],'requested_unified_allocator_config_preserved')
 def test_backend_does_not_claim_expandable(self):
  with patch.dict(os.environ,{},clear=True):out=m.observe_after_unsloth(Torch,{'environment':{},'wsl_distro_name_present':False,'wsl_interop_present':False})
  self.assertFalse(out['expandable_segments_effective_by_environment']);self.assertIn('does not itself distinguish',out['note'])
if __name__=='__main__':unittest.main()
