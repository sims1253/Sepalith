import copy,datetime,hashlib,json,os,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import cloud_entry as c
from artifact_upload import validate_remote,hashes
def binding():
 return {'schema':1,'admitted':True,'run_id':'a'*32,'image_uri':c.IMAGE,'instance_type':'g5.2xlarge','provider_timeout_seconds':6900,'watchdog_timeout_seconds':7200,
 'watchdog_armed_receipt_sha256':'b'*64,'watchdog_armed_at_utc':'2026-09-13T20:00:00Z','absolute_deadline_utc':'2026-09-13T22:00:00Z','artifact_repo':'scholzmx/sepalith-lora','artifact_prefix':'r2-control/'+'a'*32,'setup_seconds':1800,'training_seconds':4200,'upload_seconds':600,'cleanup_seconds':60}
class CloudTests(unittest.TestCase):
 def test_deadline_counts_elapsed_setup_and_upload_reserve(self):
  b=binding();now=c.utc('2026-09-13T20:30:00Z');hard=c.validate_binding(b,now)
  self.assertEqual(hard,c.utc('2026-09-13T21:55:00Z'))
  self.assertEqual(c.remaining_phase(hard,4200,660,now),min(now+4200,hard-660))
  self.assertEqual(c.remaining_phase(hard,4200,660,hard-3000),hard-660)
  with self.assertRaises(ValueError):c.validate_binding(b,hard)
  with self.assertRaises(ValueError):c.remaining_phase(hard,4200,660,hard-600)
  b['admitted']=False
  with self.assertRaises(ValueError):c.validate_binding(b,now)
 def test_prelaunch_low_memory_does_not_create_process(self):
  with tempfile.TemporaryDirectory(dir=c.HERE) as temp,patch.object(c.subprocess,'Popen') as popen:
   with self.assertRaises(ValueError):c.guarded_run([sys.executable,'-c','pass'],env={},cwd=temp,deadline=time.time()+3,log=Path(temp)/'log',mem_probe=lambda:7*1024**3)
   popen.assert_not_called()
 def test_live_memory_failure_cleans_owned_identity(self):
  values=iter([9*1024**3,7*1024**3])
  with tempfile.TemporaryDirectory(dir=c.HERE) as temp:
   log=Path(temp)/'log'
   with self.assertRaises(ValueError):c.guarded_run([sys.executable,'-c','import time;time.sleep(30)'],env={'PATH':os.environ['PATH']},cwd=temp,deadline=time.time()+3,log=log,mem_probe=lambda:next(values))
   result=json.loads(Path(str(log)+'.guard.json').read_text())
   self.assertEqual(result['reason'],'Linux MemAvailable below8GiB');self.assertEqual(result['release']['remaining'],[])
   identity=result['pid_identity'];current=c.proc(identity['pid'])
   self.assertTrue(current is None or current['start_tick']!=identity['start_tick'])
 def test_guard_deadline_and_child_failure_are_not_success(self):
  with tempfile.TemporaryDirectory(dir=c.HERE) as temp:
   for script,seconds in [('import time;time.sleep(30)',.3),('raise SystemExit(7)',3)]:
    log=Path(temp)/str(seconds)
    with self.assertRaises(ValueError):c.guarded_run([sys.executable,'-c',script],env={'PATH':os.environ['PATH']},cwd=temp,deadline=time.time()+seconds,log=log,mem_probe=lambda:9*1024**3)
    self.assertEqual(json.loads(Path(str(log)+'.guard.json').read_text())['release']['remaining'],[])
 def test_exact_relocation_preserves_identity_and_rejects_unknown_inputs(self):
  with tempfile.TemporaryDirectory(dir=c.HERE) as temp:
   root=Path(temp);b=binding();b['payload_files']=[];records={}
   for name in ['token_rows','draw_schedule','development_panel']:
    q=root/(name+'.json');q.write_text('{}\n');digest=c.sha(q);original='/local/'+name+'.json'
    b['payload_files'].append({'original_path':original,'relative_path':q.name,'sha256':digest});records[name]={'path':original,'sha256':digest}
   params={'max_steps':1000,'per_device_batch':2,'gradient_accumulation':8}
   r={'stage':'task_sft_prm03_v1','parameters':params,'model_path':'/local/model','inputs':list(records.values()),**records,
    'identity':{'schedule':copy.deepcopy(params),'parent':{'kind':'midtrain_control','revision':c.REVISION,'weights_sha256':c.MODEL_PINS['model.safetensors']},'policy':{'initialization':'new_lora_on_midtrain_control'}},'mandatory_stop_steps':[250],'decision_steps':[250],'launch_authorized':True,'resume_from':None}
   with patch.object(c,'HERE',root):
    relocated=c.relocate_recipe(r,b,root/'fresh');self.assertEqual(relocated['identity'],r['identity']);self.assertEqual(relocated['token_rows']['sha256'],r['token_rows']['sha256'])
    bad=copy.deepcopy(r);bad['inputs'][0]['path']='/unbound/path'
    with self.assertRaisesRegex(ValueError,'unbound input'):c.relocate_recipe(bad,b,root/'fresh')
    bad=copy.deepcopy(r);bad['inputs'][0]['sha256']='9'*64
    with self.assertRaisesRegex(ValueError,'changed content identity'):c.relocate_recipe(bad,b,root/'fresh')
    with self.assertRaises(ValueError):c.bounded_path(root,'../escape')
    (root/'link').symlink_to(root/'token_rows.json')
    with self.assertRaises(ValueError):c.bounded_path(root,'link')
 def test_remote_persistence_hash_failure_rejects_success(self):
  with tempfile.TemporaryDirectory(dir=c.HERE) as temp:
   q=Path(temp)/'artifact';q.write_bytes(b'private synthetic artifact');local={'artifact':hashes(q)}
   validate_remote(local,{'artifact':{'bytes':q.stat().st_size,'sha256':local['artifact']['sha256']}})
   validate_remote(local,{'artifact':{'bytes':q.stat().st_size,'git_blob_sha1':local['artifact']['git_blob_sha1']}})
   for remote in [{},{'artifact':{'bytes':q.stat().st_size,'sha256':'0'*64}}]:
    with self.assertRaises(ValueError):validate_remote(local,remote)
 def test_training_environment_drops_private_upload_token(self):
  with patch.dict(os.environ,{'HF_TOKEN':'synthetic-unit-marker','HUGGING_FACE_HUB_TOKEN':'synthetic-unit-marker'}):
   env=c.clean_env(Path('/tmp/unit-only'));self.assertNotIn('HF_TOKEN',env);self.assertNotIn('HUGGING_FACE_HUB_TOKEN',env)
 def test_real_torch_backward_profile_does_not_update_or_leave_gradients(self):
  sys.path[:0]=[str(c.HERE/'source/experiments/training'),str(c.HERE/'source/packages/sepalith/src')]
  import torch
  torch.set_num_threads(1)
  from campaign_sft_data import target_only_collator
  from cloud_train import backward_profile
  class Toy(torch.nn.Module):
   def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(.25));self.config=SimpleNamespace(use_cache=True)
  model=Toy();model.eval();before=model.p.detach().clone();rng=torch.get_rng_state().clone()
  rows=[{'input_ids':[0,20]+[21]*n+[31,41,1],'target_start':2+n,'target_body_tokens':[31],'target_terminal_tokens':[41]} for n in [0,5,2]]
  trainer=SimpleNamespace(args=SimpleNamespace(per_device_train_batch_size=2),train_dataset=rows,data_collator=target_only_collator,_prepare_inputs=lambda x:x,
   compute_loss_context_manager=lambda:__import__('contextlib').nullcontext(),compute_loss=lambda m,b,**kw:m.p.square())
  result=backward_profile(trainer,model,device_stats=lambda:{'synthetic_cpu':True})
  self.assertEqual(result['real_input_lengths'],[10,7]);self.assertEqual(result['optimizer_updates'],0)
  self.assertTrue(torch.equal(model.p,before));self.assertIsNone(model.p.grad);self.assertFalse(model.training);self.assertTrue(model.config.use_cache);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  trainer.compute_loss=lambda m,b,**kw:m.p*float('nan')
  with self.assertRaisesRegex(ValueError,'not finite'):backward_profile(trainer,model,device_stats=lambda:{})
  self.assertIsNone(model.p.grad);self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main(verbosity=2)
