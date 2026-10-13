import hashlib, importlib.util, json, os
from pathlib import Path
import subprocess, sys, tempfile, unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'source'/f'{name}.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
stage=load('native_stage');publish_mod=load('native_checkpoint_publish');overlay=load('native_runtime_overlay');capacity_mod=load('native_capacity');attest=load('native_attestation')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,sort_keys=True)+'\n')

class NativeStorageTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
  self.root=Path(self.t.name)
  native_tests=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/native-staging-test-scratch');native_tests.mkdir(parents=True,exist_ok=True)
  self.nt=tempfile.TemporaryDirectory(dir=native_tests);self.native=Path(self.nt.name)
 def tearDown(self):self.t.cleanup();self.nt.cleanup()
 def fixture(self):
  src=self.root/'source';src.mkdir();(src/'a.bin').write_bytes(b'a'*101);(src/'state.json').write_bytes(b'{"ok":true}\n')
  files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (src/'a.bin',src/'state.json')}
  manifest=src/'manifest.json';dump(manifest,{'files':files})
  row=self.root/'rows.jsonl';row.write_bytes(b'{"id":"r"}\n')
  schedule=self.root/'schedule.json';schedule.write_bytes(b'{"draws":["r"]}\n')
  spec={'schema':'sepalith.sft11.native-stage-spec.v1','native_root':str(self.native/'bundle'),'max_native_bytes':1024**3,'min_free_after_bytes':70*1024**3,'objects':[
   {'name':'cache','kind':'manifest_tree','source':str(src),'manifest':{'path':str(manifest),'sha256':sha(manifest)}},
   {'name':'rows','kind':'file','source':str(row),'sha256':sha(row)},
   {'name':'schedule','kind':'file','source':str(schedule),'sha256':sha(schedule)}]}
  sp=self.root/'spec.json';dump(sp,spec);rp=self.root/'receipt.json';receipt=stage.build(sp,rp);return src,row,schedule,sp,rp,receipt
 def test_stage_and_verified_child(self):
  *_,rp,receipt=self.fixture(); receipt_sha=sha(rp)
  held_receipt,held=stage.verify_and_lock(rp,receipt_sha)
  self.assertEqual(held_receipt['bundle_id'],receipt['bundle_id']);self.assertGreaterEqual(len(held),5)
  for fd in held:os.close(fd)
  code=stage.verify_run(rp,receipt_sha,[sys.executable,'-c','import os;assert os.environ["SEPALITH_NATIVE_STAGE_BUNDLE_ID"]'])
  self.assertEqual(code,0)
 def test_verified_child_can_consume_fd_attestation(self):
  *_,rp,receipt=self.fixture();receipt_sha=sha(rp);obj=next(x for x in receipt['objects'] if x['name']=='rows');path=str(Path(obj['staged_path'])/'rows.jsonl');expected=obj['files']['rows.jsonl']
  helper=ROOT/'source'/'native_attestation.py';code='import importlib.util,os; s=importlib.util.spec_from_file_location("a",os.environ["HELPER"]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.require_attested(os.environ["PATH_TO_CHECK"],os.environ["HASH"],int(os.environ["BYTES"]),os.environ["SEPALITH_NATIVE_STAGE_BUNDLE_ID"])'
  old=dict(os.environ);os.environ.update({'HELPER':str(helper),'PATH_TO_CHECK':path,'HASH':expected['sha256'],'BYTES':str(expected['bytes'])})
  try:self.assertEqual(stage.verify_run(rp,receipt_sha,[sys.executable,'-c',code]),0)
  finally:os.environ.clear();os.environ.update(old)
 def test_same_size_native_mutation_rejected(self):
  *_,rp,receipt=self.fixture();obj=next(x for x in receipt['objects'] if x['name']=='rows');path=Path(obj['staged_path'])/'rows.jsonl'
  os.chmod(path,0o600);data=path.read_bytes();path.write_bytes(b'X'+data[1:]);os.chmod(path,0o400)
  with self.assertRaisesRegex(ValueError,'native input bytes differ'):stage.verify_and_lock(rp,sha(rp))
 def test_symlink_and_wrong_hash_rejected(self):
  src=self.root/'real';src.write_bytes(b'x');link=self.root/'link';link.symlink_to(src)
  spec={'schema':'sepalith.sft11.native-stage-spec.v1','native_root':str(self.native/'bundle'),'max_native_bytes':100,'min_free_after_bytes':70*1024**3,'objects':[{'name':'x','kind':'file','source':str(link),'sha256':sha(src)}]};sp=self.root/'s';dump(sp,spec)
  with self.assertRaises(OSError):stage.build(sp,self.root/'r')
 def test_overlay_preserves_identity_and_requires_admission(self):
  src,row,schedule,sp,rp,receipt=self.fixture()
  # Add the checkpoint role by treating the small manifest tree as a checkpoint fixture.
  value=json.loads(sp.read_text());value['objects'].append({'name':'checkpoint','kind':'manifest_tree','source':str(src),'manifest':{'path':str(src/'manifest.json'),'sha256':sha(src/'manifest.json')}});dump(sp,value)
  # Build into a fresh root because content-addressed destinations never overwrite.
  value['native_root']=str(self.native/'bundle2');dump(sp,value);receipt=stage.build(sp,rp);receipt_sha=sha(rp)
  base={'parent':{'candidate_id':'p','path':str(src),'files':{'model.safetensors':'m','tokenizer.json':'t'},'saved_precision':{}},'transition':{'source_checkpoint':{'path':str(src),'manifest_sha256':'c'},'global_optimizer_step_offset':1},'cohort':{'id':'d','max_sequence_tokens':16384,'rows':{'path':str(row),'sha256':sha(row)},'draw_schedule':{'path':str(schedule),'sha256':sha(schedule)},'streaming_cache':{'path':str(src),'manifest_sha256':sha(src/'manifest.json')},'updates':2},'source':{'manifest_sha256':'s'},'runtime':{'max_steps':3,'effective_batch':16,'micro_batch':1,'gradient_accumulation':16,'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_steps':2,'checkpoint_every':1,'mandatory_stop_step':2,'optimizer':{}}}
  recipe=self.root/'recipe';dump(recipe,base);ad=self.root/'admission';dump(ad,{'schema':'sepalith.sft11.native-relocation-admission.v1','status':'admitted','bound_recipe_sha256':sha(recipe),'stage_receipt_sha256':receipt_sha,'object_roles':{'source_checkpoint':'checkpoint','streaming_cache':'cache','rows':'rows','draw_schedule':'schedule'}})
  out=self.root/'runtime';result=overlay.make_overlay(recipe,rp,receipt_sha,ad,out);runtime=json.loads(out.read_text())
  self.assertTrue(result['identity_unchanged']);self.assertEqual(overlay.identity_view(base),overlay.identity_view(runtime));self.assertNotEqual(runtime['parent']['path'],base['parent']['path'])
  bad=json.loads(ad.read_text());bad['bound_recipe_sha256']='0'*64;dump(ad,bad)
  with self.assertRaisesRegex(ValueError,'another recipe'):overlay.make_overlay(recipe,rp,receipt_sha,ad,self.root/'bad')
 def checkpoint(self):
  src=self.root/'native-checkpoint';src.mkdir();names={'model.safetensors':b'm','optimizer.pt':b'o','scheduler.pt':b's','rng_state.pth':b'r','trainer_state.json':b'{"global_step":1}','campaign-state.json':b'{}','tokenizer.json':b't'}
  for name,data in names.items():(src/name).write_bytes(data)
  files={name:{'bytes':len(data),'sha256':sha(src/name)} for name,data in names.items()};manifest={'checkpoint_kind':'full_weights','full':True,'step':1,'files':files};dump(src/'campaign-manifest.json',manifest);return src,sha(src/'campaign-manifest.json')
 def test_checkpoint_publish_exact_and_atomic(self):
  src,msha=self.checkpoint();dest=self.root/'durable'/'checkpoint-1';receipt=publish_mod.publish(src,dest,msha,require_cross_filesystem=False)
  self.assertEqual(receipt['status'],'durable_atomic_complete');self.assertEqual(sha(dest/'campaign-manifest.json'),msha)
  for name,value in json.loads((dest/'campaign-manifest.json').read_text())['files'].items():self.assertEqual(sha(dest/name),value['sha256'])
 def test_checkpoint_failure_never_exposes_destination(self):
  src,msha=self.checkpoint();dest=self.root/'durable'/'checkpoint-1';original=publish_mod.copy_hash;calls=[]
  def fail(source,destination,expected):
   calls.append(str(source))
   if len(calls)==2:raise RuntimeError('injected')
   return original(source,destination,expected)
  publish_mod.copy_hash=fail
  try:
   with self.assertRaisesRegex(RuntimeError,'injected'):publish_mod.publish(src,dest,msha,require_cross_filesystem=False)
  finally:publish_mod.copy_hash=original
  self.assertFalse(dest.exists())
 def test_current_template_binds_small_manifests_and_budget(self):
  spec=json.loads((ROOT/'stage-spec.template.json').read_text());declared=0
  by_name={o['name']:o for o in spec['objects']}
  for name in ('checkpoint','cache'):
   o=by_name[name];mp=Path(o['manifest']['path']);self.assertEqual(sha(mp),o['manifest']['sha256'])
   manifest=json.loads(mp.read_text());declared+=mp.stat().st_size+sum(v['bytes'] for v in manifest['files'].values())
  for name in ('rows','schedule'):
   o=by_name[name];self.assertEqual(Path(o['source']).stat().st_size,o['bytes']);declared+=o['bytes']
  self.assertEqual(declared,26581176959);self.assertLessEqual(declared,spec['max_native_bytes']);self.assertLessEqual(spec['max_native_bytes'],70*1024**3)
  recipe=json.loads((ROOT.parent/'r2-cpt90-prefix-extension-root-v2'/'bound-recipe.json').read_text())
  self.assertEqual(recipe['parent']['path'],by_name['checkpoint']['source']);self.assertEqual(recipe['cohort']['streaming_cache']['path'],by_name['cache']['source']);self.assertEqual(recipe['cohort']['rows']['sha256'],by_name['rows']['sha256'])
 def test_capacity_guard_counts_bundle_and_hot_checkpoints(self):
  report=capacity_mod.capacity('/home/m0hawk/.local/state/sepalith/campaign-20260915',26581176959,17250934153,17250934153)
  self.assertEqual(report['projected_campaign_native_bytes'],61083045265);self.assertLessEqual(report['projected_campaign_native_bytes'],70*1024**3)
  with self.assertRaisesRegex(ValueError,'exceeds cap'):capacity_mod.capacity('/home/m0hawk/.local/state/sepalith/campaign-20260915',60*1024**3,0,11*1024**3)

if __name__=='__main__':unittest.main()
