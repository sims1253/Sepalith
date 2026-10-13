"""Synthetic orchestration and CPU backward checks; no provider/model/network calls."""
import contextlib,copy,hashlib,io,json,os,random,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
H=Path(__file__).resolve().parent;W=H.parent/'r2-cloud-control-entry-v1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});sys.dont_write_bytecode=True
sys.path[:0]=[str(W),str(W/'source/experiments/training'),str(W/'source/packages/sepalith/src')]
import cloud_entry as c

class OrchestrationTests(unittest.TestCase):
 def exercise(self,upload_fail=False,train_fail=False):
  with tempfile.TemporaryDirectory(dir=H) as directory:
   d=Path(directory);payload=d/'payload';payload.mkdir();run=d/'run';run.mkdir();events=[]
   (payload/'trainer-source-manifest.json').write_text('{"files":[]}')
   recipe={'checkpoint_reserve_seconds':300};(payload/'recipe.json').write_text(json.dumps(recipe))
   now=time.time();iso=lambda x:__import__('datetime').datetime.fromtimestamp(x,__import__('datetime').timezone.utc).isoformat()
   b={'schema':1,'admitted':True,'run_id':'a'*32,'image_uri':c.IMAGE,'instance_type':'g5.2xlarge','provider_timeout_seconds':6900,'watchdog_timeout_seconds':7200,'watchdog_armed_receipt_sha256':'b'*64,'watchdog_armed_at_utc':iso(now-1),'absolute_deadline_utc':iso(now+7199),'artifact_repo':'scholzmx/sepalith-lora','artifact_prefix':'r2-control/'+'a'*32,'setup_seconds':1800,'training_seconds':4200,'upload_seconds':600,'cleanup_seconds':60,'trainer_source_manifest_sha256':c.sha(payload/'trainer-source-manifest.json'),'recipe':{'relative_path':'recipe.json','sha256':c.sha(payload/'recipe.json')}}
   binding=d/'binding.json';binding.write_text(json.dumps(b))
   def guarded(argv,**kw):
    events.append({'argv':list(map(str,argv)),'deadline':kw['deadline'],'has_token':'HF_TOKEN' in kw['env']})
    name=Path(str(argv[0])).name;args=list(map(str,argv));art=run/'artifacts'
    if 'uv-bootstrap.log' in str(kw['log']):
     q=run/'bootstrap-tools/bin/uv';q.parent.mkdir(parents=True);q.write_text('synthetic executable marker')
    elif args[1:3]==['python','install']:
     q=run/'python/managed/bin/python3.10';q.parent.mkdir(parents=True);q.write_text('synthetic interpreter marker')
    elif 'venv-bootstrap.log' in str(kw['log']):
     q=run/'venv/bin/python';q.parent.mkdir(parents=True);q.write_text('synthetic interpreter marker')
    elif any(x.endswith('cloud_launch.py') for x in args):
     if train_fail:raise ValueError('synthetic training failure')
     q=art/'training';q.mkdir();(q/'terminal.json').write_text(json.dumps({'step':250,'status':'lead_decision'}));(q/'cloud-backward-profile.json').write_text(json.dumps({'status':'pass','optimizer_updates':0}))
    elif any(x.endswith('artifact_upload.py') for x in args) and 'final' in args:
     if upload_fail:raise ValueError('synthetic upload failure')
     (run/'persistence-receipt.json').write_text('{"status":"synthetic_uploaded"}')
    return {'synthetic':True}
   with patch.object(c,'HERE',payload),patch.object(c.tempfile,'mkdtemp',return_value=str(run)),patch.object(c,'guarded_run',side_effect=guarded),patch.object(c,'relocate_recipe',side_effect=lambda r,b,run:copy.deepcopy(r)),patch.object(sys,'argv',['cloud_entry.py',str(binding)]),patch.dict(os.environ,{'HF_TOKEN':'synthetic-secret-marker'}),contextlib.redirect_stdout(io.StringIO()) as output:
    if upload_fail or train_fail:
     with self.assertRaises(SystemExit) as error:c.main()
     self.assertEqual(error.exception.code,1)
    else:c.main()
   terminal=json.loads((run/'artifacts/local-terminal.json').read_text())
   self.assertEqual(terminal['training_success'],not train_fail)
   self.assertNotIn('synthetic-secret-marker',output.getvalue())
   final=[e for e in events if 'final' in e['argv']];self.assertEqual(len(final),1)
   self.assertLessEqual(final[0]['deadline'],c.utc(b['watchdog_armed_at_utc'])+6900-60)
   for e in events:
    self.assertEqual(e['has_token'],any(x.endswith('artifact_upload.py') for x in e['argv']))
   training=[e for e in events if any(x.endswith('cloud_launch.py') for x in e['argv'])];self.assertEqual(len(training),1)
   bound=json.loads((run/'artifacts/recipe.json').read_text());self.assertLessEqual(c.utc(bound['deadline']),c.utc(b['watchdog_armed_at_utc'])+6900-660)
   return events
 def test_success_reaches_training_and_final_upload(self):self.exercise()
 def test_final_upload_failure_is_exit_one(self):self.exercise(upload_fail=True)
 def test_training_failure_still_uploads_but_never_succeeds(self):self.exercise(train_fail=True)

class BackwardTests(unittest.TestCase):
 def test_rng_optimizer_gradients_and_trainable_values_restored(self):
  import numpy as np,torch
  from cloud_train import backward_profile
  from campaign_sft_data import target_only_collator
  torch.set_num_threads(1)
  class Toy(torch.nn.Module):
   def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(.25));self.q=torch.nn.Parameter(torch.tensor(.5),requires_grad=False);self.config=SimpleNamespace(use_cache=True)
  model=Toy();model.eval();optimizer=torch.optim.AdamW([model.p],lr=.1)
  rows=[{'input_ids':[0,20]+[21]*n+[31,41,1],'target_start':2+n,'target_body_tokens':[31],'target_terminal_tokens':[41]} for n in [1,20,6]]
  def loss(m,b,**kw):return m.p.square()*(1+random.random()+float(np.random.random())+torch.rand(()))
  trainer=SimpleNamespace(args=SimpleNamespace(per_device_train_batch_size=2),train_dataset=rows,data_collator=target_only_collator,_prepare_inputs=lambda x:x,compute_loss_context_manager=contextlib.nullcontext,compute_loss=loss,optimizer=optimizer)
  py=random.getstate();ns=np.random.get_state();ts=torch.get_rng_state().clone();before=[p.detach().clone() for p in model.parameters()];optim=copy.deepcopy(optimizer.state_dict())
  result=backward_profile(trainer,model,device_stats=lambda:{'scope':'CPU synthetic'})
  self.assertEqual(result['real_input_lengths'],[25,11]);self.assertEqual(optimizer.state_dict(),optim)
  self.assertEqual(random.getstate(),py);self.assertEqual(np.random.get_state()[0],ns[0]);self.assertTrue(np.array_equal(np.random.get_state()[1],ns[1]));self.assertEqual(np.random.get_state()[2:],ns[2:]);self.assertTrue(torch.equal(torch.get_rng_state(),ts))
  for p,old in zip(model.parameters(),before):self.assertTrue(torch.equal(p,old));self.assertIsNone(p.grad)
  self.assertFalse(model.training);self.assertTrue(model.config.use_cache);self.assertFalse(torch.cuda.is_initialized())
  trainer.compute_loss=lambda m,b,**kw:m.p*torch.tensor(float('nan'))
  with self.assertRaises(ValueError):backward_profile(trainer,model,device_stats=lambda:{})
  self.assertIsNone(model.p.grad);self.assertFalse(model.training);self.assertTrue(model.config.use_cache)

if __name__=='__main__':unittest.main(verbosity=2)
