import copy, hashlib, importlib.util, json, os, tempfile, unittest
from pathlib import Path

PACKET=Path(__file__).parent
SPEC=importlib.util.spec_from_file_location('assembler',PACKET/'assemble_root_admission.py');M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
def dump(path,value): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n');return {'path':str(path.resolve()),'sha256':M.sha256(path)}

class TestAssembler(unittest.TestCase):
 def setUp(self):
  self.td=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR','/mnt/e/sepalith/campaign-20260915/tmp'));self.r=Path(self.td.name)
  self.template={'schema':'sepalith.sft11.full-weight-cpt-stage-transition-template.v1','launch_authorized':False};self.template_pin=dump(self.r/'template.json',self.template)
  self.rows={'bytes':99,'sha256':'a'*64};self.cache_rows={'path':str((self.r/'rows.jsonl').resolve()),'sha256':'a'*64};self.schedule={'schema':'sepalith.sft11.cpt-draw-schedule.v1','status':'complete_candidate_pending_root_admission','training_admission':False,'method':'one_pass_plus_named_replay_v1','seed':3407,'split_id':'cpt_train_final_union_ctx16384_v1','effective_batch':16,'max_steps':11443,'replay_count':4,'replay_row_ids':['r0','r1','r2','r3'],'token_rows_sha256':'a'*64,'coverage':{'all_final_union_rows_required':True,'all_unique_rows_before_replay':True,'draws':183088,'first_unique_draw_position_exclusive':183084,'minimal_named_tail_replay':True,'unique_rows':183084,'updates':11443},'stage_transition':{'destination_sampler':'fresh_cursor_zero','global_max_step':11509,'global_optimizer_step_offset':66,'initial_stage_cursor':0}}
  self.schedule_pin=dump(self.r/'schedule.json',self.schedule)
  self.result={'schema':'sepalith.cpt.lossless-rechunk-result.v1','status':'complete','artifacts':{'cpt_train_ctx16384.jsonl':self.rows},'outputs':{'16384':{'rows':183084,'input_tokens':100,'payload_tokens':60,'supervised_tokens':70,'terminal_eos':20}},'totals':{'documents':20,'payload_tokens':60},'guarantees':{'all_source_tokens_once_per_context':True,'prior_token_overlap_masked':True,'raw_source_read':False,'retokenized':False,'single_terminal_eos_per_document':True}}
  self.result_pin=dump(self.r/'result.json',self.result)
  self.source_recipe={'runtime':{'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_ratio':2/66,'max_steps':66,'optimizer':{'arm':'aurora_mix','hidden_lr':3e-6,'side_lr':3e-7}}};self.source_recipe_pin=dump(self.r/'source-recipe.json',self.source_recipe)
  self.decision_pin=dump(self.r/'decision.json',{});self.source_manifest_pin=dump(self.r/'source-manifest.json',{})
  self.cache=self.r/'cache';self.cache.mkdir()
  files={}
  for name in ('input_ids.i32le','labels.i32le','draw_ordinals.i64le','index.sqlite3'):
   q=self.cache/name;q.write_bytes(name.encode());files[name]={'bytes':q.stat().st_size,'sha256':M.sha256(q)}
  self.cache_manifest={'schema':'sepalith.sft11.cpt-streaming-cache.v1','status':'complete','source':{'rows':self.cache_rows,'draw_schedule':self.schedule_pin},'max_sequence_tokens':16384,'counts':{'rows':183084,'documents':20,'input_tokens':100,'payload_tokens':60,'loss_tokens':70,'packages':7,'draws':183088,'named_replays':4,'updates':11443},'contract':{'dtype':'signed little-endian int32','attention_mask':'derived all ones after source validation','complete_documents':True,'target_truncation':False,'unique_rows_before_replay':True},'files':files}
  (self.cache/'manifest.json').write_text(json.dumps(self.cache_manifest)+'\n');self.cache_sha=M.sha256(self.cache/'manifest.json')
  self.frontier={'documents':1999,'path':'/x/frontier','sha256':'f'*64,'content_dedup_status':'pending_rehash_and_content_dedup'}
  self.cfg={'schema':'sepalith.sft11.cpt-expanded-stage-launch-inputs.v1','launch_authorized':False,'cohort_id':'test','scope':'current_terminal_union_snapshot_pending_1999_cap_frontier','stage_transition_template':self.template_pin,'lossless_result':self.result_pin,'draw_schedule':self.schedule_pin,'streaming_cache_path':str(self.cache),'expected':{'rows':{**self.rows,'path':self.cache_rows['path']},'lossless_counts':self.result['outputs']['16384'],'unique_rows':183084,'documents':20,'input_tokens':100,'payload_tokens':60,'loss_tokens':70,'named_replays':4,'stage_updates':11443,'cache_counts_without_packages':{k:v for k,v in self.cache_manifest['counts'].items() if k!='packages'},'coverage':self.schedule['coverage'],'schedule':{k:self.schedule[k] for k in ('effective_batch','max_steps','method','replay_count','schema','seed','split_id','status','training_admission')},'schedule_transition':self.schedule['stage_transition']},'required_lossless_guarantees':self.result['guarantees'],'required_cache_contract':self.cache_manifest['contract'],'pending_frontier':self.frontier,'dtype_audit':dump(self.r/'dtype.json',{}),'source_checkpoint66':{'source_checkpoint':{'candidate_id':'c','path':'/checkpoint','manifest_sha256':'c'*64},'source_decision':self.decision_pin,'source_manifest':self.source_manifest_pin,'source_recipe':self.source_recipe_pin,'saved_precision':{'fp32_elements':2,'fp32_tensors':1,'identity_source':'test'}},'selected':{'checkpoint_every':128,'evaluation_stage_steps':[128,11443],'gradient_accumulation':16,'learning_rate':3e-6,'mandatory_stop_stage_step':128,'micro_batch':1,'optimizer':self.source_recipe['runtime']['optimizer'],'scheduler':'constant_with_warmup','selected_stage_milestones':[128,11443],'telemetry_every':1,'warmup_steps':2}}
  self.cfgpin=dump(self.r/'inputs.json',self.cfg)
  snap={'scope':self.cfg['scope'],'rows_sha256':'a'*64,'cache_manifest_sha256':self.cache_sha,'draw_schedule_sha256':self.schedule_pin['sha256'],'unique_rows':183084,'documents':20,'packages':7,'input_tokens':100,'payload_tokens':60,'loss_tokens':70,'named_replays':4,'stage_updates':11443,'max_sequence_tokens':16384}
  self.adm={'schema':'sepalith.sft11.cpt-expanded-snapshot-data-admission.v1','status':'admitted','training_admission':True,'stage_transition_binding_authorized':True,'all_source_pool_closed':False,'snapshot':snap,'pending_frontier':self.frontier}
 def tearDown(self): self.td.cleanup()
 def run_ok(self):
  a=dump(self.r/'admission.json',self.adm);out=self.r/'out.json';result=M.assemble(self.r/'inputs.json',a['path'],out);return json.loads(out.read_text()),result
 def test_exact_terminal_admission(self):
  out,res=self.run_ok();self.assertTrue(out['launch_authorized']);self.assertEqual(out['cohort']['packages'],7);self.assertEqual(out['selected']['evaluation_stage_steps'],[128,11443]);self.assertEqual(res['global_first_gate_step'],194);self.assertEqual(res['global_terminal_step'],11509)
 def test_absent_or_building_cache_fails(self):
  (self.cache/'manifest.json').unlink()
  with self.assertRaisesRegex(ValueError,'absent'):self.run_ok()
  (self.cache/'manifest.json').write_text(json.dumps({**self.cache_manifest,'status':'building'}))
  with self.assertRaisesRegex(ValueError,'not terminal'):self.run_ok()
 def test_data_admission_required_and_must_authorize_binding(self):
  with self.assertRaisesRegex(ValueError,'required'):M.assemble(self.r/'inputs.json',self.r/'missing.json',self.r/'o')
  self.adm['stage_transition_binding_authorized']=False
  with self.assertRaisesRegex(ValueError,'did not authorize'):self.run_ok()
 def test_snapshot_count_or_cache_identity_mismatch_fails(self):
  self.adm['snapshot']['packages']=8
  with self.assertRaisesRegex(ValueError,'snapshot differs'):self.run_ok()
 def test_open_frontier_cannot_be_hidden(self):
  self.adm['all_source_pool_closed']=True
  with self.assertRaisesRegex(ValueError,'obscures'):self.run_ok()
 def test_optimizer_or_milestone_drift_fails(self):
  self.cfg['selected']['optimizer']['side_lr']=1e-6;dump(self.r/'inputs.json',self.cfg)
  with self.assertRaisesRegex(ValueError,'not byte-semantic'):self.run_ok()
 def test_missing_or_wrong_source_warmup_fails(self):
  self.source_recipe['runtime']['warmup_ratio']=1/66;dump(self.r/'source-recipe.json',self.source_recipe);self.cfg['source_checkpoint66']['source_recipe']['sha256']=M.sha256(self.r/'source-recipe.json');dump(self.r/'inputs.json',self.cfg)
  with self.assertRaisesRegex(ValueError,'source warmup differs'):self.run_ok()
 def test_config_is_preparation_only(self):
  cfg=json.loads((PACKET/'launch-inputs.json').read_text());guard=json.loads((PACKET/'guard-proposal.json').read_text());ad=json.loads((PACKET/'root-data-admission.template.json').read_text());self.assertFalse(cfg['launch_authorized']);self.assertFalse(guard['execution_authorized']);self.assertFalse(ad['training_admission']);self.assertFalse(ad['stage_transition_binding_authorized']);self.assertIsNone(ad['snapshot']['cache_manifest_sha256']);self.assertEqual(guard['host_memory_gib'],{'startup_available':14,'soft_floor':6,'hard_floor':4})

if __name__=='__main__':unittest.main()
