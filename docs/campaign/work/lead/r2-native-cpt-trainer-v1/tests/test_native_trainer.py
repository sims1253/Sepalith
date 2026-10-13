import hashlib,importlib.util,json,os,struct,sys,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source'/'experiments'/'training';sys.path.insert(0,str(SRC))
import native_stage,native_launch_attestation,native_runtime_overlay,native_checkpoint_publish
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,sort_keys=True)+'\n')
def tensor(path):
 header=json.dumps({'w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},separators=(',',':')).encode();header+=b' '*((8-len(header)%8)%8);Path(path).write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<f',1.0))
def make_checkpoint(root,identity=None,step=1):
 root.mkdir();tensor(root/'model.safetensors');values={'config.json':b'{}\n','generation_config.json':b'{}\n','tokenizer.json':b'{"v":1}\n','tokenizer_config.json':b'{}\n','optimizer.pt':b'optimizer','scheduler.pt':b'scheduler','rng_state.pth':b'rng','trainer_state.json':json.dumps({'global_step':step}).encode()+b'\n','campaign-state.json':json.dumps({'identity':identity or {'x':1},'step':step,'full':True,'checkpoint_kind':'full_weights','sampler':{'x':1}}).encode()+b'\n'}
 for n,b in values.items():(root/n).write_bytes(b)
 files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in root.iterdir()};manifest={'schema_version':1,'step':step,'full':True,'checkpoint_kind':'full_weights','identity':identity or {'x':1},'files':files};dump(root/'campaign-manifest.json',manifest);return sha(root/'campaign-manifest.json')

class NativeTrainerTests(unittest.TestCase):
 def setUp(self):
  bulk=Path(os.environ['TMPDIR']);self.t=tempfile.TemporaryDirectory(dir=bulk);self.root=Path(self.t.name)
  nt=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/native-cpt-trainer-test-scratch');nt.mkdir(parents=True,exist_ok=True);self.nt=tempfile.TemporaryDirectory(dir=nt);self.native=Path(self.nt.name)
 def tearDown(self):self.t.cleanup();self.nt.cleanup()
 def files(self):
  records={}
  for name in ('root-admission','data-admission','corpus','dtype','cohort','validation','dispatch'):
   p=self.root/f'{name}.json';dump(p,{'name':name});records[name]={'path':str(p),'sha256':sha(p)}
  return records
 def prepare(self):
  parent=self.root/'parent';parent_manifest=make_checkpoint(parent)
  cache=self.root/'cache';cache.mkdir()
  for name,data in {'input_ids.i32le':b'1234','labels.i32le':b'5678','draw_ordinals.i64le':b'12345678','index.sqlite3':b'sqlite'}.items():(cache/name).write_bytes(data)
  rows=self.root/'rows.jsonl';rows.write_bytes(b'{"row_id":"r"}\n');schedule=self.root/'schedule.json';schedule.write_bytes(b'{"row_ids":["r"]}\n')
  cf={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in cache.iterdir()};dump(cache/'manifest.json',{'schema':'sepalith.sft11.cpt-streaming-cache.v1','status':'complete','source':{'rows':{'sha256':sha(rows)},'draw_schedule':{'sha256':sha(schedule)}},'files':cf})
  spec={'schema':'sepalith.sft11.native-stage-spec.v1','native_root':str(self.native/'base'),'max_native_bytes':1024**3,'min_free_after_bytes':70*1024**3,'objects':[{'name':'cache','kind':'manifest_tree','source':str(cache),'manifest':{'path':str(cache/'manifest.json'),'sha256':sha(cache/'manifest.json')}},{'name':'rows','kind':'file','source':str(rows),'sha256':sha(rows)},{'name':'schedule','kind':'file','source':str(schedule),'sha256':sha(schedule)}]};sp=self.root/'spec';dump(sp,spec);rp=self.root/'receipt';receipt=native_stage.build(sp,rp)
  rec=self.files();parent_files={n:sha(parent/n) for n in ('config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json')}
  base={'schema':'sepalith.sft11.full-weight-cpt-stage-transition-bound.v1','status':'root_admitted_not_launched','seed':1,'launch_authorized':True,'source':{'manifest_path':'old','manifest_sha256':'a'*64},'root_admission':rec['root-admission'],'data_admission':rec['data-admission'],'corpus_manifest':rec['corpus'],'dtype_audit':rec['dtype'],'optimizer_dispatch':{**rec['dispatch'],'ordered_rows_sha256':'x'},'validation':{**rec['validation'],'max_sequence_tokens':2048,'rows':499},'parent':{'candidate_id':'p','kind':'full_weight_stage_checkpoint','path':str(parent),'files':parent_files,'saved_precision':{'fp32_tensors':0,'fp32_elements':0}},'cohort':{'id':'c','manifest':rec['cohort'],'rows':{'path':str(rows),'sha256':sha(rows)},'draw_schedule':{'path':str(schedule),'sha256':sha(schedule)},'streaming_cache':{'path':str(cache),'manifest_sha256':sha(cache/'manifest.json')},'max_sequence_tokens':16384,'unique_rows':128,'documents':1,'packages':1,'input_tokens':1,'payload_tokens':1,'loss_tokens':1,'named_replays':0,'updates':8,'scope':'fixture'},'transition':{'global_optimizer_step_offset':66,'source_checkpoint':{'path':str(parent),'manifest_sha256':parent_manifest}},'runtime':{'max_steps':74,'effective_batch':16,'micro_batch':1,'gradient_accumulation':16,'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_steps':2,'checkpoint_every':4,'mandatory_stop_step':70,'evaluation_steps':[70,74],'selected_milestones':[70,74],'optimizer':{},'telemetry_every':1},'outputs':{'trainer':'/mnt/e/old-trainer','archive':'/mnt/e/old-archive','graceful_stop':str(self.root/'stop')},'checkpoint_storage':{'mode':'e_same_filesystem_atomic','c_hot_stage':'not_admitted','required_same_filesystem':True,'trainer_root':'/mnt/e/old-trainer','archive_root':'/mnt/e/old-archive'},'retention':{'durable_archive_latest':2}}
  recipe=self.root/'canonical.json';dump(recipe,base);rsha=sha(rp);ad=self.root/'relocation.json';dump(ad,{'schema':'sepalith.sft11.native-relocation-admission.v1','status':'admitted','launch_authorized':True,'bound_recipe_sha256':sha(recipe),'stage_receipt_sha256':rsha,'object_roles':{'streaming_cache':'cache','rows':'rows','draw_schedule':'schedule'}})
  ms=self.root/'migration.json';runtime_sha=sha(ROOT/'source-manifest.json');dump(ms,{'schema':'sepalith.sft11.native-runtime-source-migration-admission.v1','status':'admitted','launch_authorized':True,'canonical_bound_recipe_sha256':sha(recipe),'scientific_source_manifest_sha256':'a'*64,'runtime_source_manifest_sha256':runtime_sha,'stage_receipt_sha256':rsha})
  out=self.root/'runtime.json';native_runtime_overlay.make_overlay(recipe,rp,rsha,ad,out,runtime_source_manifest=ROOT/'source-manifest.json',migration_admission=ms,trainer_root=self.native/'training',archive_root='/mnt/e/sepalith/campaign-20260915/test-native-cpt-archive');return out,rp,receipt
 def test_actual_frontdoor_and_full_resume_through_locked_attestation(self):
  recipe,rp,receipt=self.prepare();sys.path.insert(0,str(SRC));import full_weight_cpt_trainer as trainer
  runtime=json.loads(recipe.read_text());identity=trainer.identity(runtime);resume=self.native/'resume';manifest_sha=make_checkpoint(resume,identity,70);state=json.loads((resume/'campaign-state.json').read_text());state['sampler']={'method':'sequential_frozen_draw_schedule_stage_local','global_optimizer_step_offset':66,'draw_schedule_sha256':runtime['cohort']['draw_schedule']['sha256'],'stage_cursor':64,'cursor':64,'global_step':70,'ignore_data_skip':True};dump(resume/'campaign-state.json',state);manifest=json.loads((resume/'campaign-manifest.json').read_text());manifest['files']['campaign-state.json']={'bytes':(resume/'campaign-state.json').stat().st_size,'sha256':sha(resume/'campaign-state.json')};dump(resume/'campaign-manifest.json',manifest);manifest_sha=sha(resume/'campaign-manifest.json')
  admission=self.root/'continuation';dump(admission,{'schema':'sepalith.sft11.cpt-stage-transition-continuation-admission.v1','status':'admitted','decision':'continue','launch_authorized':True,'bound_recipe_sha256':sha(recipe),'checkpoint':str(resume.resolve()),'checkpoint_manifest_sha256':manifest_sha,'step':70})
  code='import json,sys;sys.path.insert(0,sys.argv[1]);import full_weight_cpt_trainer as t;from campaign_checkpoint import verify_checkpoint;from stage_transition_contract import stage_cursor_from_checkpoint;r=t.load_bound(sys.argv[2]);t.validate_continuation(sys.argv[2],r,sys.argv[3],sys.argv[4]);m=verify_checkpoint(sys.argv[3],t.identity(r),require_full=True,expected_checkpoint_kind="full_weights",native_bundle_id=r["_native_validation"]["bundle_id"]);c=stage_cursor_from_checkpoint(json.load(open(sys.argv[3]+"/campaign-state.json")),66,r["cohort"]["draw_schedule"]["sha256"]);assert m["step"]==70 and c==64'
  rc=native_launch_attestation.run(rp,sha(rp),resume,manifest_sha,[sys.executable,'-B','-c',code,str(SRC),str(recipe),str(resume),str(admission)]);self.assertEqual(rc,0)
 def test_real_packet_source_manifest_frontdoor_and_tamper(self):
  import full_weight_cpt_trainer as trainer
  record={'runtime_source':{'manifest_path':str(ROOT/'source-manifest.json'),'manifest_sha256':sha(ROOT/'source-manifest.json')}};trainer.verify_source(record)
  bad=dict(record);bad['runtime_source']=dict(record['runtime_source']);bad['runtime_source']['manifest_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'manifest differs'):trainer.verify_source(bad)
 def test_interrupted_publish_never_exposes_E_name(self):
  source=self.root/'checkpoint';msha=make_checkpoint(source);destination=self.root/'E'/'checkpoint-1';original=native_checkpoint_publish.copy_hash;calls=[]
  def fail(a,b,c):calls.append(1);return (_ for _ in ()).throw(RuntimeError('interrupted')) if len(calls)==2 else original(a,b,c)
  native_checkpoint_publish.copy_hash=fail
  try:
   with self.assertRaisesRegex(RuntimeError,'interrupted'):native_checkpoint_publish.publish(source,destination,msha,require_cross_filesystem=False)
  finally:native_checkpoint_publish.copy_hash=original
  self.assertFalse(destination.exists())
 def test_runtime_overlay_rejects_source_migration_identity_change(self):
  recipe,rp,_=self.prepare();value=json.loads(recipe.read_text());value['source']['manifest_sha256']='b'*64;dump(recipe,value)
  code='import sys;sys.path.insert(0,sys.argv[1]);import full_weight_cpt_trainer as t;t.load_bound(sys.argv[2])'
  resume=self.native/'resume';msha=make_checkpoint(resume)
  rc=native_launch_attestation.run(rp,sha(rp),resume,msha,[sys.executable,'-B','-c',code,str(SRC),str(recipe)]);self.assertNotEqual(rc,0)
 def test_native_hot_retention_and_cursor_contract(self):
  import full_weight_cpt_trainer as trainer
  recipe,_,_=self.prepare();runtime=json.loads(recipe.read_text());args=trainer.training_arguments_kwargs(runtime,self.native/'training');self.assertEqual(args['save_total_limit'],1);self.assertTrue(args['ignore_data_skip'])
  from stage_transition_contract import stage_cursor_from_checkpoint
  state={'sampler':{'method':'sequential_frozen_draw_schedule_stage_local','global_optimizer_step_offset':66,'draw_schedule_sha256':'d','stage_cursor':64,'cursor':64,'global_step':70,'ignore_data_skip':True}}
  self.assertEqual(stage_cursor_from_checkpoint(state,66,'d'),64)
  state['sampler']['cursor']=48
  with self.assertRaisesRegex(ValueError,'cursor differs'):stage_cursor_from_checkpoint(state,66,'d')

if __name__=='__main__':unittest.main()
