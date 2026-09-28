#!/usr/bin/env python3
import copy,hashlib,json,tempfile,unittest,sys
from pathlib import Path
PACKET=Path(__file__).resolve().parents[1];SRC=PACKET/'source/experiments/training';sys.path.insert(0,str(SRC))
import ordinary_canary_resume as O
FILES=O.FULL_FILES

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True)+'\n')
def production():return {'parent':{'x':1},'tokenizer':{'x':1},'renderer':{'x':1},'data':{'x':1},'source':{'x':1},'policy':{'stage':'full_weight_cpt_stage_transition_v1','optimizer':{'arm':'aurora_mix'}},'schedule':{'x':1}}
class TestTransition(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(prefix='selected-packed330-');self.root=Path(self.t.name);self.packet=self.root/'packet';(self.packet/'selected-source').mkdir(parents=True);self.recipe=self.root/'recipe.json';write(self.recipe,{'recipe':'x'});self.prod=production();self.resume=self.root/'checkpoint-330';self.ad=self.root/'admission.json';self.canary_recipe=self.root/'canary-recipe.json';write(self.canary_recipe,{'arm':'varlen_candidate'});self.metric=self.root/'metric.json';write(self.metric,{'winner':'packed'});self.payload=self.root/'payload.json';write(self.payload,{'verified':True})
  O.PACKET=self.packet;O.SOURCE_CHECKPOINT=self.resume;O.CANARY_RECIPE=self.canary_recipe;O.CANARY_RECIPE_SHA=O.sha(self.canary_recipe);O.SELECTION_EVIDENCE={'metric_selection':{'path':str(self.metric),'sha256':O.sha(self.metric)},'payload_review':{'path':str(self.payload),'sha256':O.sha(self.payload)}}
  self.make_checkpoint(O.canary_identity(self.prod));O.SOURCE_MANIFEST_SHA=O.sha(self.resume/'campaign-manifest.json');O.SOURCE_STATE_SHA=O.sha(self.resume/'campaign-state.json');self.freeze();self.make_admission()
 def tearDown(self):self.t.cleanup()
 def make_checkpoint(self,ident,**sampler):
  s={'cursor':4224,'stage_cursor':4224,'global_step':330,'global_optimizer_step_offset':66,'effective_batch':16,'draw_schedule_sha256':'5ea4a04f4082f93ad8e7211a139e19e7ffffd33f4ef2661d93e42134933a3cac',**sampler};m={'step':330,'full':True,'checkpoint_kind':'full_weights','identity':ident,'files':{x:{'bytes':1,'sha256':'x'} for x in FILES}};st={'step':330,'full':True,'checkpoint_kind':'full_weights','identity':ident,'sampler':s};write(self.resume/'campaign-manifest.json',m);write(self.resume/'campaign-state.json',st)
 def freeze(self):
  for src,name in [(self.resume/'campaign-manifest.json','campaign-manifest.json'),(self.resume/'campaign-state.json','campaign-state.json'),(self.canary_recipe,'canary-runtime-recipe.json'),(self.metric,'metric-selection-result.json'),(self.payload,'payload-review.json')]: (self.packet/'selected-source'/name).write_bytes(src.read_bytes())
 def make_admission(self,**changes):
  v={'schema':O.SCHEMA,'status':'admitted','decision':'selected_packed330_weights_to_ordinary_production_execution','launch_authorized':True,'bound_recipe_sha256':O.sha(self.recipe),'source_arm':'varlen_candidate','destination_execution':'ordinary_sdpa_unpacked','checkpoint':str(self.resume.resolve()),'checkpoint_manifest_sha256':O.sha(self.resume/'campaign-manifest.json'),'campaign_state_sha256':O.sha(self.resume/'campaign-state.json'),'checkpoint_step':330,'source_canary_recipe_sha256':O.CANARY_RECIPE_SHA,'source_identity_sha256':O.canonical_sha(O.canary_identity(self.prod)),'destination_identity_sha256':O.canonical_sha(self.prod),'selection_evidence':O.SELECTION_EVIDENCE,'allowed_source_policy':O.PACKED_POLICY,'payload_action':'load_original_full_state_without_rewrite_or_copy',**changes};write(self.ad,v)
 def reject(self,pattern=''):
  with self.assertRaisesRegex(ValueError,pattern):O.resolve_resume_identity(self.recipe,self.prod,self.resume,self.ad)
 def test_selected_packed_transition_and_immutable_bytes(self):
  before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.resume.iterdir()};ident,lineage=O.resolve_resume_identity(self.recipe,self.prod,self.resume,self.ad);after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.resume.iterdir()};self.assertEqual(ident,O.canary_identity(self.prod));self.assertEqual(before,after);self.assertEqual(lineage['source_arm'],'varlen_candidate');self.assertEqual(lineage['destination_execution'],'ordinary_sdpa_unpacked')
 def test_missing_admission_rejected(self):
  with self.assertRaisesRegex(ValueError,'requires root admission'):O.resolve_resume_identity(self.recipe,self.prod,self.resume,None)
 def test_ordinary_or_wrong_arm_rejected(self):
  ordinary=copy.deepcopy(self.prod);ordinary['policy'].update({'execution_experiment':'native_full_optimizer_varlen_canary_v1','execution_arm':'ordinary_reference','logical_effective_batch':16,'loss_reduction':'global_supervised_token_mean','physical_gradient_accumulation':16,'attention_packing':'ordinary_padded_rows_v1','attention_backend':'sdpa_ordinary'});self.make_checkpoint(ordinary);self.reject('neither production nor exact selected packed')
 def test_wrong_checkpoint_path_rejected(self):
  other=self.root/'other-checkpoint-330';other.mkdir();[(other/x.name).write_bytes(x.read_bytes()) for x in self.resume.iterdir()]
  with self.assertRaisesRegex(ValueError,'selected checkpoint330'):O.resolve_resume_identity(self.recipe,self.prod,other,self.ad)
 def test_draw_or_optimizer_inventory_mutation_rejected(self):
  self.make_checkpoint(O.canary_identity(self.prod),cursor=4223);O.SOURCE_MANIFEST_SHA=O.sha(self.resume/'campaign-manifest.json');O.SOURCE_STATE_SHA=O.sha(self.resume/'campaign-state.json');self.freeze();self.make_admission();self.reject('cursor/schedule')
  self.make_checkpoint(O.canary_identity(self.prod));m=json.loads((self.resume/'campaign-manifest.json').read_text());m['files'].pop('optimizer.pt');write(self.resume/'campaign-manifest.json',m);self.reject('full-state inventory')
 def test_recipe_selection_and_destination_mutations_rejected(self):
  self.make_admission(bound_recipe_sha256='bad');self.reject('another recipe');self.make_admission(selection_evidence={});self.reject('selection evidence');self.make_admission(destination_identity_sha256='bad');self.reject('identity transition hashes')
 def test_future_production_resume_carries_lineage(self):
  _,lineage=O.resolve_resume_identity(self.recipe,self.prod,self.resume,self.ad);self.make_checkpoint(self.prod,resume_lineage=lineage);ident,again=O.resolve_resume_identity(self.recipe,self.prod,self.resume,None);self.assertEqual(ident,self.prod);self.assertEqual(again,lineage)
if __name__=='__main__':unittest.main()
