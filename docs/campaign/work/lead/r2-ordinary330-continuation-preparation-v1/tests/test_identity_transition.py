#!/usr/bin/env python3
import copy,hashlib,importlib.util,json,tempfile,unittest,sys
from pathlib import Path
PACKET=Path(__file__).resolve().parents[1];SRC=PACKET/'source/experiments/training';sys.path.insert(0,str(SRC))
import ordinary_canary_resume as O
FILES=O.FULL_FILES

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True)+'\n')
def production():return {'parent':{'x':1},'tokenizer':{'x':1},'renderer':{'x':1},'data':{'x':1},'source':{'x':1},'policy':{'stage':'full_weight_cpt_stage_transition_v1','optimizer':{'arm':'aurora_mix'}},'schedule':{'x':1}}
class TestTransition(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(prefix='ordinary330-');self.root=Path(self.t.name);self.recipe=self.root/'recipe.json';write(self.recipe,{'recipe':'x'});self.prod=production();self.resume=self.root/'checkpoint-330';self.ad=self.root/'admission.json';self.make_checkpoint(O.canary_identity(self.prod));self.make_admission()
 def tearDown(self):self.t.cleanup()
 def make_checkpoint(self,ident,**sampler):
  s={'cursor':4224,'stage_cursor':4224,'global_step':330,'global_optimizer_step_offset':66,'effective_batch':16,**sampler};m={'step':330,'full':True,'checkpoint_kind':'full_weights','identity':ident,'files':{x:{'bytes':1,'sha256':'x'} for x in FILES}};st={'step':330,'full':True,'checkpoint_kind':'full_weights','identity':ident,'sampler':s};write(self.resume/'campaign-manifest.json',m);write(self.resume/'campaign-state.json',st)
 def make_admission(self,**changes):
  m=self.resume/'campaign-manifest.json';v={'schema':O.SCHEMA,'status':'admitted','decision':'ordinary_canary_to_production','launch_authorized':True,'bound_recipe_sha256':O.sha(self.recipe),'checkpoint':str(self.resume.resolve()),'checkpoint_manifest_sha256':O.sha(m),'checkpoint_step':330,'source_identity_sha256':O.canonical_sha(O.canary_identity(self.prod)),'destination_identity_sha256':O.canonical_sha(self.prod),'allowed_policy_delta':O.ORDINARY_POLICY,'payload_action':'load_original_full_state_without_rewrite_or_copy',**changes};write(self.ad,v)
 def reject(self,pattern=''):
  with self.assertRaisesRegex(ValueError,pattern):O.resolve_resume_identity(self.recipe,self.prod,self.resume,self.ad)
 def test_exact_transition_and_immutable_bytes(self):
  before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.resume.iterdir()};ident,lineage=O.resolve_resume_identity(self.recipe,self.prod,self.resume,self.ad);after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.resume.iterdir()};self.assertEqual(ident,O.canary_identity(self.prod));self.assertEqual(before,after);self.assertEqual(lineage['source_checkpoint_manifest_sha256'],O.sha(self.resume/'campaign-manifest.json'))
 def test_missing_admission_rejected(self):
  with self.assertRaisesRegex(ValueError,'requires root admission'):O.resolve_resume_identity(self.recipe,self.prod,self.resume,None)
 def test_packed_or_mutated_policy_rejected(self):
  x=O.canary_identity(self.prod);x['policy']['execution_arm']='varlen_candidate';self.make_checkpoint(x);self.make_admission();self.reject('neither production nor exact ordinary')
 def test_cursor_mutation_rejected(self):self.make_checkpoint(O.canary_identity(self.prod),cursor=4223);self.make_admission();self.reject('cursor/schedule')
 def test_weights_only_inventory_rejected(self):
  x=json.loads((self.resume/'campaign-manifest.json').read_text());x['files'].pop('optimizer.pt');write(self.resume/'campaign-manifest.json',x);self.make_admission();self.reject('full-state inventory')
 def test_admission_recipe_and_identity_mutations_rejected(self):
  self.make_admission(bound_recipe_sha256='bad');self.reject('another recipe');self.make_admission(destination_identity_sha256='bad');self.reject('identity transition hashes')
 def test_production_resume_needs_no_transition_and_carries_lineage(self):
  _,lineage=O.resolve_resume_identity(self.recipe,self.prod,self.resume,self.ad);self.make_checkpoint(self.prod,resume_lineage=lineage);ident,again=O.resolve_resume_identity(self.recipe,self.prod,self.resume,None);self.assertEqual(ident,self.prod);self.assertEqual(again,lineage)
if __name__=='__main__':unittest.main()
