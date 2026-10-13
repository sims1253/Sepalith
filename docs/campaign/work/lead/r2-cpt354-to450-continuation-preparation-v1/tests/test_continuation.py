import hashlib,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];SRC=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-selected330-recovery-preparation-v2/source/experiments/training');sys.path.insert(0,str(SRC))
import full_weight_cpt_trainer as T
import ordinary_canary_resume as O

class TestContinuation(unittest.TestCase):
 def test_recipe_keeps_exact_identity_and_fresh_outputs(self):
  base=json.loads(Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-selected330-recovery-root-v2/runtime-recipe.json').read_text());new=json.loads((P/'runtime-recipe.review.json').read_text())
  self.assertEqual(T.identity(base),T.identity(new));self.assertEqual(O.canonical_sha(T.identity(new)),'46a5a1aced9f533e8ba4cd25aebe795dedc64d390a0d563cfc7e406c03324739');self.assertNotEqual(base['outputs'],new['outputs'])
 def test_same_identity_resume_reuses_stored_lineage_without_packed_path(self):
  recipe=json.loads((P/'runtime-recipe.review.json').read_text());ident=T.identity(recipe)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cp=root/'checkpoint-354';cp.mkdir();ad=root/'old-transition.json';ad.write_text('{}\n');lineage={'schema':'sepalith.sft11.selected-packed330-resume-lineage.v2','transition_admission':str(ad),'transition_admission_sha256':hashlib.sha256(ad.read_bytes()).hexdigest()}
   files={n:{'bytes':1,'sha256':'x'} for n in O.FULL_FILES};(cp/'campaign-manifest.json').write_text(json.dumps({'step':354,'full':True,'checkpoint_kind':'full_weights','identity':ident,'files':files})+'\n');(cp/'campaign-state.json').write_text(json.dumps({'step':354,'full':True,'checkpoint_kind':'full_weights','identity':ident,'sampler':{'resume_lineage':lineage}})+'\n')
   observed,stored=O.resolve_resume_identity(P/'runtime-recipe.review.json',ident,cp,None);self.assertEqual(observed,ident);self.assertEqual(stored,lineage)
 def test_exact_manifest_and_aligned_stop_templates(self):
  c=json.loads((P/'continuation-354.template.json').read_text());s=json.loads((P/'execution-stop-450.template.json').read_text());self.assertEqual(c['checkpoint_manifest_sha256'],'c95770f2ba069795ab559222b6fd0488c4cd65100ff3fa44a61b38786969f9f3');self.assertEqual((c['step'],s['resume_global_step'],s['stop_at_global_step']),(354,354,450));self.assertEqual((450-66)%24,0)
 def test_wrapper_uses_354_and_dynamic_exact_manifest(self):
  s=(P/'run_with_allocator.py').read_text();self.assertIn('checkpoint-354',s);self.assertIn("os.environ['SEPALITH_RESUME_MANIFEST_SHA256']",s);self.assertNotIn('checkpoint-330',s);self.assertIn("EXPECTED = '5184354d",s)
 def test_commands_omit_packed_transition_and_bind_manifest(self):
  v=json.loads((P/'root-commands.template.json').read_text());joined=json.dumps(v);self.assertNotIn('ordinary-canary-transition-admission',joined);self.assertNotIn('ROOT_BINDS_VERIFIED_CHECKPOINT354_MANIFEST',joined);self.assertIn('c95770f2ba069795ab559222b6fd0488c4cd65100ff3fa44a61b38786969f9f3',joined)

if __name__=='__main__':unittest.main()
