import fcntl, hashlib, importlib.util, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PACKET=PLAN/'docs/campaign/work/lead/r2-native-cpt194-continuation-preparation-v1'
RECIPE=PLAN/'docs/campaign/work/lead/r2-cpt90-prefix-extension-root-v2/bound-recipe.json'
STAGE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/native-staging/cpt-prefix-extension-reusable-data-v1.receipt.json')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class TestFrontdoor(unittest.TestCase):
 def setUp(self):
  self.tmp=Path(tempfile.mkdtemp(prefix='native194-',dir='/mnt/e/sepalith/campaign-20260915/data-work'))
  self.overlay=load('overlay',PACKET/'source/experiments/training/native_runtime_overlay.py')
 def admissions(self, source):
  relocation={'schema':'sepalith.sft11.native-relocation-admission.v1','status':'admitted','launch_authorized':True,'bound_recipe_sha256':sha(RECIPE),'stage_receipt_sha256':sha(STAGE),'object_roles':{'streaming_cache':'cache','rows':'rows','draw_schedule':'schedule'}}
  migration={'schema':'sepalith.sft11.native-runtime-source-migration-admission.v1','status':'admitted','launch_authorized':True,'scientific_source_manifest_sha256':json.loads(RECIPE.read_text())['source']['manifest_sha256'],'runtime_source_manifest_sha256':sha(source),'canonical_bound_recipe_sha256':sha(RECIPE),'stage_receipt_sha256':sha(STAGE)}
  rp=self.tmp/'relocation.json';mp=self.tmp/'migration.json';rp.write_text(json.dumps(relocation));mp.write_text(json.dumps(migration));return rp,mp
 def test_real_overlay_and_runtime_source_frontdoor(self):
  source=PACKET/'source-manifest.json';rp,mp=self.admissions(source);out=self.tmp/'runtime.json'
  self.overlay.make_overlay(RECIPE,STAGE,sha(STAGE),rp,out,runtime_source_manifest=source,migration_admission=mp,trainer_root='/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-test-only',archive_root='/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-native-test-only')
  value=json.loads(out.read_text());storage=value['checkpoint_storage']
  self.assertEqual((storage['trainer_transient_save_total_limit'],storage['native_retained_after_durable_publication']),(2,1))
  sys.path.insert(0,str(PACKET/'source/experiments/training'))
  trainer=load('trainer',PACKET/'source/experiments/training/full_weight_cpt_trainer.py');trainer.verify_source(value)
 def test_old_v1_schema_rejected_by_real_overlay(self):
  source=self.tmp/'source-v1.json';value=json.loads((PACKET/'source-manifest.json').read_text());value['schema']='sepalith.sft11.native-cpt-trainer-source.v1';source.write_text(json.dumps(value))
  rp,mp=self.admissions(source)
  with self.assertRaisesRegex(ValueError,'schema differs'):
   self.overlay.make_overlay(RECIPE,STAGE,sha(STAGE),rp,self.tmp/'bad.json',runtime_source_manifest=source,migration_admission=mp,trainer_root='/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-test-only',archive_root='/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-native-test-only')
 def test_actual_checkpoint_and_capacity_preparation(self):
  prep=load('prep',PACKET/'prepare_continuation.py');out=self.tmp/'prepared';result=prep.prepare(PACKET/'source-manifest.json',out)
  self.assertEqual((result['checkpoint_step'],result['resume_cursor'],result['stop_cursor']),(194,2048,4096))
  self.assertTrue(json.loads((out/'capacity-review.json').read_text())['capacity_pass'])
  self.assertFalse(json.loads((out/'continuation-admission-194.prepared.json').read_text())['launch_authorized'])
 def test_native_attestation_propagates_validated_cuda_lease(self):
  sys.path.insert(0,str(PACKET/'source/experiments/training'))
  attestation=load('attestation',PACKET/'source/experiments/training/native_launch_attestation.py')
  lock=self.tmp/'cuda.lock';lock.touch();fd=os.open(lock,os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  previous=os.environ.get('SEPALITH_CUDA_LOCK_FD');os.environ['SEPALITH_CUDA_LOCK_FD']=str(fd);seen=self.tmp/'seen'
  try:
   code='import os,sys;os.fstat(int(os.environ["SEPALITH_CUDA_LOCK_FD"]));open(sys.argv[1],"w").write("ok")'
   rc=attestation.run_child([sys.executable,'-c',code,str(seen)],dict(os.environ),[],lock)
   self.assertEqual(rc,0);self.assertEqual(seen.read_text(),'ok')
  finally:
   if previous is None:os.environ.pop('SEPALITH_CUDA_LOCK_FD',None)
   else:os.environ['SEPALITH_CUDA_LOCK_FD']=previous
   os.close(fd)

if __name__=='__main__': unittest.main()
