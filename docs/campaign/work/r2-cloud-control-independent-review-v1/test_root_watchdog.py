import importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
H=Path(__file__).resolve().parent;W=H.parent/'r2-cloud-control-entry-v1';sys.dont_write_bytecode=True;sys.path.insert(0,str(W));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import cloud_entry as c
import root_arm_watchdog as a
class Tests(unittest.TestCase):
 def binding(self):
  return {'schema':1,'admitted':True,'run_id':'a'*32,'image_uri':c.IMAGE,'instance_type':'g5.2xlarge','provider_timeout_seconds':6900,'watchdog_timeout_seconds':7200,'watchdog_armed_receipt_sha256':None,'watchdog_armed_at_utc':'2026-09-13T20:00:00Z','absolute_deadline_utc':'2026-09-13T22:00:00Z','artifact_repo':'scholzmx/sepalith-lora','artifact_prefix':'r2-control/'+'a'*32,'setup_seconds':1800,'training_seconds':4200,'upload_seconds':600,'cleanup_seconds':60}
 def invoke(self,b):
  with tempfile.TemporaryDirectory(dir=H) as d:
   p=Path(d)/'binding.json';p.write_text(json.dumps(b))
   with patch.object(sys,'argv',['root_arm_watchdog.py',str(p),'--output',str(Path(d)/'guard')]),patch.object(c.time,'time',return_value=c.utc('2026-09-13T20:00:01Z')),patch.object(a.os,'execv') as execute:
    a.main();return execute.call_args.args
 def test_exact_two_hour_binding_constructs_pinned_watchdog_argv(self):
  exe,argv=self.invoke(self.binding());self.assertEqual(exe,sys.executable);self.assertEqual(argv[1],'-B');self.assertEqual(argv[2],str(a.WATCHDOG));self.assertEqual(argv[argv.index('--deadline-utc')+1],'2026-09-13T22:00:00Z');self.assertEqual(argv[argv.index('--name')+1],'sepalith-r2-control-'+'a'*32)
 def test_extra_second_rejected_before_exec(self):
  b=self.binding();b['absolute_deadline_utc']='2026-09-13T22:00:01Z'
  with self.assertRaises(ValueError):self.invoke(b)
 def test_unadmitted_binding_rejected(self):
  b=self.binding();b['admitted']=False
  with self.assertRaises(ValueError):self.invoke(b)
 def test_direct_entry_rejects_missing_receipt_hash(self):
  with self.assertRaises(TypeError):c.validate_binding(self.binding(),c.utc('2026-09-13T20:00:01Z'))
if __name__=='__main__':unittest.main(verbosity=2)
