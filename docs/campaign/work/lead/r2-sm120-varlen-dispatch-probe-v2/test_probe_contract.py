#!/usr/bin/env python3
import importlib.util,unittest
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('probe',P/'probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class Event:
 def __init__(self,key):self.key=key
class Profile:
 def __init__(self,keys=None,error=None):self.keys=keys;self.error=error
 def __enter__(self):
  if self.error:raise self.error
  return self
 def __exit__(self,*args):return False
 def key_averages(self):return [Event(x)for x in self.keys]
class Profiler:
 class ProfilerActivity:CPU='cpu'
 def __init__(self,profile):self.value=profile
 def profile(self,**kwargs):return self.value
class Torch:
 def __init__(self,profile):self.profiler=Profiler(profile)
class ProbeContract(unittest.TestCase):
 def test_capture_labels_only_real_profiler_keys_actual(self):
  value=probe.profiler_capture(Torch(Profile(['aten::empty','xformers_flash::flash_fwd','aten::_flash_attention_forward'])),lambda *args:None,'torch_varlen',['inferred::name'])
  self.assertEqual(value['status'],'captured');self.assertEqual(value['inferred_operator_names'],[]);self.assertIn('aten::_flash_attention_forward',value['actual_aten_operator_keys']);self.assertIn('xformers_flash::flash_fwd',value['actual_operator_keys'])
 def test_unavailable_capture_keeps_inference_separate(self):
  value=probe.profiler_capture(Torch(Profile(error=RuntimeError('no profiler'))),lambda *args:None,'torch_varlen',['aten::inferred'])
  self.assertEqual(value['status'],'capture_unavailable');self.assertEqual(value['actual_operator_keys'],[]);self.assertEqual(value['inferred_operator_names'],['aten::inferred']);self.assertEqual(value['error_type'],'RuntimeError')
 def test_probe_source_has_seeded_cotangent_isolation_and_unchanged_timing_design(self):
  source=(P/'probe.py').read_text();self.assertIn("manual_seed(20260915+total)",source);self.assertIn("backward_mode=='cotangent'",source);self.assertIn("outside_delta==0.0 and inside_delta>0.0",source);self.assertIn('for repetition in range(7)',source);self.assertIn('if repetition>=2',source);self.assertEqual(len(probe.PROFILES),3);self.assertEqual(probe.PROFILES['single_16384'],[16384])
if __name__=='__main__':unittest.main(verbosity=2)
