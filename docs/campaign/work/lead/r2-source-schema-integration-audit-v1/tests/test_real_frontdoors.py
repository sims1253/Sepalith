import importlib.util,json,subprocess,sys,unittest
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
A=PLAN/'docs/campaign/work/lead/r2-a100-fullweight-trainer-v2';V=PLAN/'docs/campaign/work/lead/r2-varlen-numerics-probe-v2'
class Tests(unittest.TestCase):
 def test_unchanged_trainer_rejects_custom_schema(self):
  path=PLAN/'docs/campaign/work/lead/r2-full-weight-cpt-stage-transition-v3/source/experiments/training/full_weight_cpt_trainer.py';spec=importlib.util.spec_from_file_location('oldtrainer',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  recipe={'source':{'manifest_path':str(A/'source-manifest.json'),'manifest_sha256':'812d3b42e9e03308ca97b2437ff27d5aca5b39ab3503ff154cdbb95171feae7d'}}
  with self.assertRaisesRegex(ValueError,'source schema differs'):m.verify_source(recipe)
 def test_actual_a100_v2_frontdoor_real_manifest(self):
  cmd=['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(A/'source/experiments/training/a100_fullweight_trainer.py'),'preflight-template','--recipe',str(A/'recipe.template.json')]
  run=subprocess.run(cmd,text=True,capture_output=True,timeout=30);self.assertEqual(run.returncode,0,run.stderr);out=json.loads(run.stdout);self.assertEqual(out['source_closure']['sha256'],'812d3b42e9e03308ca97b2437ff27d5aca5b39ab3503ff154cdbb95171feae7d')
 def test_actual_varlen_v2_runtime_verifier_real_manifest(self):
  sys.path.insert(0,str(V/'source'));from runtime_source_closure import verify_runtime_source
  out=verify_runtime_source(V/'source-manifest.json','sepalith.sft11.varlen-numerics-probe-source.v2');self.assertEqual(out['sha256'],'77ffdf45c21823394726dac76f2e6d38228a7753edff38943e48af671b57c9ba')
 def test_tampered_real_manifest_fails(self):
  sys.path.insert(0,str(A/'source/experiments/training'));from runtime_source_closure import verify_runtime_source
  value=json.loads((A/'source-manifest.json').read_text());value['schema']='sepalith.full-weight-cpt-stage-transition-source.v1';tmp=Path('/mnt/e/sepalith/campaign-20260915/tmp/source-schema-audit-tampered.json');tmp.write_text(json.dumps(value))
  try:
   with self.assertRaises(ValueError):verify_runtime_source(tmp,'sepalith.sft11.a100-fullweight-source.v2')
  finally:tmp.unlink()
if __name__=='__main__':unittest.main()
