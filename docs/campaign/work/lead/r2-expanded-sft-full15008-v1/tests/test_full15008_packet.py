#!/usr/bin/env python3
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
WORK=Path(__file__).resolve().parents[1]
SOURCE=WORK/'source/experiments/training'
sys.path.insert(0,str(SOURCE))
import campaign_expanded_sft as adapter
import campaign_sft_data as data
import campaign_launch

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
class PacketTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.recipe=json.loads((WORK/'recipe.json').read_text()); cls.schedule=json.loads(Path(cls.recipe['draw_schedule']['path']).read_text())
 def test_fresh_parent_and_optimizer_reset(self):
  r=self.recipe
  self.assertIsNone(r['resume_from']);self.assertIsNone(r['resume_milestone'])
  self.assertEqual(r['identity']['policy']['initialization'],'new_lora_on_sft_merged_parent')
  self.assertEqual(r['identity']['parent']['weights_sha256'],adapter.SELECTED_PARENT_WEIGHTS_SHA256)
  self.assertEqual((adapter.SELECTED_PARENT_STEP,adapter.SELECTED_PARENT_CURSOR),(750,12000))
  self.assertEqual(r['identity']['parent']['previous_checkpoint_manifest_sha256'],'37d48d9118922bcb55ddf7ce0af1ef46456eafe7a01497a193e56744a7f3e0ab')
  adapter.expanded_target_only_policy(r);adapter.validate_sft_merged_parent(r)
 def test_schedule_and_all_finish_rows_bound(self):
  r=self.recipe;s=self.schedule
  self.assertEqual((r['parameters']['max_steps'],r['parameters']['per_device_batch']*r['parameters']['gradient_accumulation']),(1160,16))
  self.assertEqual((r['identity']['data']['candidate_rows'],r['identity']['data']['new_finish_rows']),(15006,3503))
  self.assertEqual((s['draw_count'],s['coverage']['distinct_rows_drawn'],s['coverage']['first_all_rows_step']),(18560,15006,1160))
  self.assertTrue(all(s['checks'][k] for k in ('all_eligible_ids_drawn','no_target_truncation','every_unique_noop_before_noop_replay','every_unique_edit_before_edit_replay')))
  self.assertEqual(s['excluded'],[])
 def test_target_above_dev_cap_is_valid_and_malformed_rejected(self):
  long=None
  with open(self.recipe['token_rows']['path']) as f:
   for line in f:
    row=json.loads(line)
    if len(row['input_ids'])-row['target_start']>192:long=row;break
  self.assertIsNotNone(long);summary=adapter.validate_expanded_rows(self.recipe,[long])
  self.assertGreater(summary['max_target_tokens_including_eos'],192)
  bad=copy.deepcopy(long);bad['input_ids']=bad['input_ids'][:-1]
  with self.assertRaises(ValueError):adapter.validate_expanded_rows(self.recipe,[bad])
 def test_structured_draw_adapter_uses_draws_not_inventory(self):
  row={'id':'z','split':'train','renderer_id':'zeta2-prm03-v1','family':'finish_block','package_id':'p','input_ids':[0,4,5,6,7,1],'target_start':3,'target_body_tokens':[6],'target_terminal_tokens':[7]}
  with tempfile.TemporaryDirectory() as td:
   td=Path(td); rp=td/'r.jsonl';rp.write_text(json.dumps(row)+'\n')
   sched={'schema':'sepalith.sft.full-coverage-batchmix-draws.v2','max_steps':1,'draw_count':1,'row_ids':['z'],'excluded':[],'policy':{'effective_batch':1},'input':{'sha256':sha(rp),'rows_seen':1},'coverage':{'distinct_rows_drawn':1},'checks':{'all_eligible_ids_drawn':True,'no_target_truncation':True},'draws':[{'draw_index':0,'update':1,'row_id':'z','total_tokens':6,'supervised_target_tokens':3,'semantic_noop':False}]}
   sp=td/'s.json';sp.write_text(json.dumps(sched))
   rows,draws,normalized,exposure=data.inspect_training_data({'path':str(rp),'sha256':sha(rp)},{'path':str(sp),'sha256':sha(sp)},renderer_id='zeta2-prm03-v1',max_sequence_tokens=4096,max_steps=1,effective_batch=1,split_id='split')
   self.assertEqual((draws,normalized['row_ids'],normalized['split_id']),([0],['z'],'split'))
   sched['draws'][0]['update']=2;sp.write_text(json.dumps(sched))
   with self.assertRaisesRegex(ValueError,'Invalid full-coverage v2 draw'):
    data.inspect_training_data({'path':str(rp),'sha256':sha(rp)},{'path':str(sp),'sha256':sha(sp)},renderer_id='zeta2-prm03-v1',max_sequence_tokens=4096,max_steps=1,effective_batch=1,split_id='split')
 def test_resume_and_evaluation_lifecycle(self):
  r=self.recipe
  self.assertEqual(r['target_gate_start_steps'],[0,240,480,720]);self.assertEqual(r['target_resume_steps'],[240,480,720])
  self.assertEqual(r['checkpoint'],{'light_every':40,'full_every':40,'evaluation_steps':[240,480,720,1160]})
  self.assertEqual(r['mandatory_stop_steps'],[]);self.assertIn('requires separate evidence',r['runtime_horizon']['note'])
  self.assertIn('control_callback(', (SOURCE/'campaign_control.py').read_text())
  self.assertIn('checkpoint_callback(', (SOURCE/'campaign_checkpoint.py').read_text())
  self.assertIn('target_only_startup_gate(', (SOURCE/'target_only_gate.py').read_text())
  self.assertIn('development_evaluator(', (SOURCE/'campaign_eval.py').read_text())
  self.assertEqual(adapter.reviewed_sft.runtime_options(r), {'loader_kwargs': {'use_gradient_checkpointing': True}, 'gradient_checkpointing': True, 'logging_steps': 1})
 def test_runner_guard_argv_and_source_pin(self):
  runner=json.loads((WORK/'runner-recipe.json').read_text());argv=runner['steps'][1]['argv']
  self.assertEqual(runner['snapshot'],'28e224b22497ea9881aba29fdea1c9dd5725681da19099a54bfb800ae2f4bbde')
  self.assertEqual(argv[argv.index('--output')+1],'/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-expanded-full15006-a-host-supervision')
  self.assertEqual(argv[argv.index('--seconds')+1],'15000')
  self.assertEqual(campaign_launch.training_entrypoint(self.recipe).name, 'campaign_expanded_sft.py')
 def test_source_manifest_exact(self):
  m=json.loads((WORK/'source-manifest.json').read_text());self.assertEqual(m['id'],self.recipe['identity']['source'])
  for item in m['files']:self.assertEqual(sha(WORK/'source'/item['path']),item['sha256'])
 def test_reject_wrong_parent_or_horizon(self):
  for mutate in ('parent','horizon'):
   r=copy.deepcopy(self.recipe)
   if mutate=='parent':r['identity']['parent']['weights_sha256']='0'*64
   else:r['parameters']['max_steps']=1000;r['identity']['schedule']=copy.deepcopy(r['parameters'])
   with self.assertRaises(ValueError):adapter.expanded_target_only_policy(r)
if __name__=='__main__':unittest.main(verbosity=2)
