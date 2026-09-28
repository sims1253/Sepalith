"""Root-gated immutable packed-canary checkpoint to ordinary production transition."""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
from sepalith.training.paths import campaign_path

SCHEMA='sepalith.sft11.selected-packed330-to-ordinary-transition-admission.v2'
CADENCE64_SCHEMA='sepalith.sft11.cpt450-cadence64-identity-transition-admission.v1'
CADENCE24_IDENTITY_SHA='46a5a1aced9f533e8ba4cd25aebe795dedc64d390a0d563cfc7e406c03324739'
CADENCE64_IDENTITY_SHA='6ac45eec834ee675bfecd880b06972ec028b7314497c47ca0130d6d32c43ce48'
CADENCE64_ALLOWED_CHANGES=['checkpoint cadence 24 to 64 after accepted checkpoint450','mandatory external review boundary 354 to 706','fresh continuation runtime output roots']
PACKET=campaign_path('work/lead/r2-cpt450-cadence64-continuation-preparation-v2')
SOURCE_CHECKPOINT=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-varlen-canary-322-varlen_candidate-v1/runtime/checkpoint-330')
SOURCE_MANIFEST_SHA='2b35854537588499fb703af2127029ff649e410cbab65374948723273b520951'
SOURCE_STATE_SHA='fb6d4d456e2318c70301a8b39f21e54069570c48c9c90de80e3f09155361a8e8'
CANARY_RECIPE=campaign_path('work/lead/r2-native-varlen322-root-v1/varlen_candidate/recipe.json')
CANARY_RECIPE_SHA='12d09ed0158070b256b0976e6f80fc0e46e1f552c1556b44dbc7a6cf360d5997'
SELECTION_EVIDENCE={
 'metric_selection':{'path':str(campaign_path('work/lead/r2-varlen330-metric-root-review-v1/result.json')),'sha256':'b9368a6d6ec1ebf30cd1e59669ce3b66f065fb3c04dc77f6f97a4d4f27b93a21'},
 'payload_review':{'path':str(campaign_path('work/lead/r2-varlen330-evaluation-preparation-v2/payload-review.json')),'sha256':'fc42ebb45d9cddbded49c9c356cdc371b630b5e7c42f35e7bf804c5b0c6f287b'},
}
PACKED_POLICY={
 'execution_experiment':'native_full_optimizer_varlen_canary_v1',
 'execution_arm':'varlen_candidate',
 'logical_effective_batch':16,
 'loss_reduction':'global_supervised_token_mean',
 'physical_gradient_accumulation':1,
 'attention_packing':'varlen_block_diagonal_v1',
 'attention_backend':'xformers',
}
FULL_FILES={'campaign-state.json','chat_template.jinja','config.json','generation_config.json','model.safetensors','optimizer.pt','rng_state.pth','scheduler.pt','tokenizer.json','tokenizer_config.json','trainer_state.json','training_args.bin'}
def require(v,m):
 if not v:raise ValueError(m)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def canary_identity(production_identity):
 value=copy.deepcopy(production_identity);value['policy'].update(PACKED_POLICY);return value

def recovery_destination_identity(observed):
 """Derive the one admitted recovery identity from the frozen packed source.

 The checkpoint's learned state and optimizer schedule remain authoritative.
 Only checkpoint cadence and the first mandatory review boundary change.
 """
 value=copy.deepcopy(observed)
 for key in PACKED_POLICY:value['policy'].pop(key,None)
 require(value['schedule'].get('checkpoint_every')==128 and value['schedule'].get('mandatory_stop_step')==194,'selected source schedule differs')
 value['schedule']['checkpoint_every']=24
 value['schedule']['mandatory_stop_step']=354
 return value

def _verify_frozen_selection():
 pairs=(
  (SOURCE_CHECKPOINT/'campaign-manifest.json',PACKET/'selected-source/campaign-manifest.json',SOURCE_MANIFEST_SHA),
  (SOURCE_CHECKPOINT/'campaign-state.json',PACKET/'selected-source/campaign-state.json',SOURCE_STATE_SHA),
  (CANARY_RECIPE,PACKET/'selected-source/canary-runtime-recipe.json',CANARY_RECIPE_SHA),
  (Path(SELECTION_EVIDENCE['metric_selection']['path']),PACKET/'selected-source/metric-selection-result.json',SELECTION_EVIDENCE['metric_selection']['sha256']),
  (Path(SELECTION_EVIDENCE['payload_review']['path']),PACKET/'selected-source/payload-review.json',SELECTION_EVIDENCE['payload_review']['sha256']),
 )
 for original,frozen,expected in pairs:
  require(original.is_file() and frozen.is_file(),'selected source evidence missing')
  require(sha(original)==sha(frozen)==expected,'selected source evidence differs')

def _validate_lineage(value):
 require(value.get('schema')=='sepalith.sft11.selected-packed330-resume-lineage.v2','resume lineage schema differs')
 admission=Path(value.get('transition_admission',''));require(admission.is_file() and sha(admission)==value.get('transition_admission_sha256'),'resume lineage admission differs')
 return value

def _accepted_evidence(record,name):
 path=Path(record.get('path',''));require(path.is_file() and sha(path)==record.get('sha256'),f'{name} differs')
 value=json.loads(path.read_text());require(value.get('status')=='accepted' and value.get('checkpoint_step')==450,f'{name} not accepted')

def _resolve_cadence64(recipe_path,production_identity,resume,manifest,state,admission_path):
 observed=manifest['identity'];require(canonical_sha(observed)==CADENCE24_IDENTITY_SHA and canonical_sha(production_identity)==CADENCE64_IDENTITY_SHA,'cadence64 identity pair differs')
 expected=copy.deepcopy(observed);expected['schedule']['checkpoint_every']=64;expected['schedule']['mandatory_stop_step']=706
 require(production_identity==expected,'cadence64 destination changes more than cadence/review boundary')
 require(admission_path is not None,'cadence64 identity transition requires root admission');path=Path(admission_path);require(path.is_file(),'cadence64 identity admission missing');ad=json.loads(path.read_text())
 require(ad.get('schema')==CADENCE64_SCHEMA and ad.get('status')=='admitted' and ad.get('decision')=='continue_checkpoint450_with_cadence64' and ad.get('launch_authorized')is True,'cadence64 identity transition not admitted')
 require(ad.get('bound_recipe_sha256')==sha(recipe_path),'cadence64 transition refers to another recipe')
 mp=resume/'campaign-manifest.json';sp=resume/'campaign-state.json'
 require(Path(ad.get('checkpoint','')).resolve()==resume.resolve() and ad.get('checkpoint_manifest_sha256')==sha(mp) and ad.get('campaign_state_sha256')==sha(sp),'cadence64 checkpoint differs')
 require(ad.get('checkpoint_step')==manifest.get('step')==state.get('step')==450,'cadence64 checkpoint step differs')
 sampler=state.get('sampler',{});require((sampler.get('global_step'),sampler.get('cursor'),sampler.get('stage_cursor'),sampler.get('global_optimizer_step_offset'),sampler.get('effective_batch'),sampler.get('draw_schedule_sha256'))==(450,6144,6144,66,16,'5ea4a04f4082f93ad8e7211a139e19e7ffffd33f4ef2661d93e42134933a3cac'),'cadence64 checkpoint sampler differs')
 require(ad.get('source_identity_sha256')==CADENCE24_IDENTITY_SHA and ad.get('destination_identity_sha256')==CADENCE64_IDENTITY_SHA and ad.get('allowed_changes')==CADENCE64_ALLOWED_CHANGES,'cadence64 identity admission differs')
 source_recipe=Path(ad.get('source_runtime_recipe',''));require(source_recipe.is_file() and sha(source_recipe)==ad.get('source_runtime_recipe_sha256'),'cadence64 source recipe differs')
 _accepted_evidence(ad.get('checkpoint450_review',{}),'checkpoint450 review');_accepted_evidence(ad.get('matched450_review',{}),'matched450 review')
 require(ad.get('payload_action')=='load_checkpoint450_full_state_without_rewrite_or_optimizer_reset','cadence64 payload action differs')
 prior=state.get('sampler',{}).get('resume_lineage');require(prior is not None,'cadence64 checkpoint lost prior lineage')
 return observed,_validate_lineage(prior)

def resolve_resume_identity(recipe_path,production_identity,resume,transition_admission=None):
 """Return the real source identity; future production checkpoints retain lineage."""
 resume=Path(resume);mp=resume/'campaign-manifest.json';state_path=resume/'campaign-state.json'
 require(mp.is_file() and state_path.is_file(),'resume metadata missing')
 manifest=json.loads(mp.read_text());state=json.loads(state_path.read_text())
 require(manifest.get('full')is True and manifest.get('checkpoint_kind')=='full_weights' and set(manifest.get('files',{}))==FULL_FILES,'resume is not the exact full-state inventory')
 require(state.get('full')is True and state.get('checkpoint_kind')=='full_weights' and state.get('step')==manifest.get('step'),'resume full-state metadata differs')
 require(manifest.get('identity')==state.get('identity'),'resume identity metadata disagrees')
 observed=manifest.get('identity')
 if observed==production_identity:
  require(transition_admission is None,'same-identity resume supplied a transition admission')
  prior=state.get('sampler',{}).get('resume_lineage')
  return observed,(_validate_lineage(prior) if prior is not None else None)
 if canonical_sha(observed)==CADENCE24_IDENTITY_SHA and canonical_sha(production_identity)==CADENCE64_IDENTITY_SHA:
  return _resolve_cadence64(recipe_path,production_identity,resume,manifest,state,transition_admission)
 _verify_frozen_selection()
 frozen=json.loads((PACKET/'selected-source/campaign-manifest.json').read_text()).get('identity')
 require(observed==frozen,'resume identity is neither production nor exact frozen selected packed canary')
 require(production_identity==recovery_destination_identity(observed),'destination identity changes more than recovery cadence/review boundary')
 require(resume.resolve()==SOURCE_CHECKPOINT.resolve(),'packed resume is not the selected checkpoint330')
 require(transition_admission is not None,'selected packed canary identity transition requires root admission')
 admission_path=Path(transition_admission);require(admission_path.is_file(),'identity transition admission missing');ad=json.loads(admission_path.read_text())
 require(ad.get('schema')==SCHEMA and ad.get('status')=='admitted' and ad.get('decision')=='selected_packed330_weights_to_ordinary_production_execution' and ad.get('launch_authorized')is True,'identity transition is not admitted')
 require(ad.get('bound_recipe_sha256')==sha(recipe_path),'identity transition refers to another recipe')
 require(ad.get('source_arm')=='varlen_candidate' and ad.get('destination_execution')=='ordinary_sdpa_unpacked','identity transition arm/execution differs')
 require(Path(ad.get('checkpoint','')).resolve()==resume.resolve() and ad.get('checkpoint_manifest_sha256')==sha(mp)==SOURCE_MANIFEST_SHA and ad.get('campaign_state_sha256')==sha(state_path)==SOURCE_STATE_SHA,'identity transition checkpoint differs')
 require(ad.get('checkpoint_step')==manifest.get('step')==330,'identity transition step differs')
 sampler=state.get('sampler',{});require((state.get('step'),sampler.get('global_step'),sampler.get('cursor'),sampler.get('stage_cursor'),sampler.get('global_optimizer_step_offset'),sampler.get('effective_batch'),sampler.get('draw_schedule_sha256'))==(330,330,4224,4224,66,16,'5ea4a04f4082f93ad8e7211a139e19e7ffffd33f4ef2661d93e42134933a3cac'),'identity transition cursor/schedule differs')
 require(ad.get('source_identity_sha256')==canonical_sha(observed) and ad.get('destination_identity_sha256')==canonical_sha(production_identity),'identity transition hashes differ')
 require(ad.get('source_canary_recipe_sha256')==CANARY_RECIPE_SHA and ad.get('selection_evidence')==SELECTION_EVIDENCE,'canary recipe or selection evidence differs')
 require(ad.get('allowed_source_policy')==PACKED_POLICY,'identity transition source policy differs')
 require(ad.get('payload_action')=='load_original_full_state_without_rewrite_or_copy','identity transition payload action differs')
 lineage={'schema':'sepalith.sft11.selected-packed330-resume-lineage.v2','source_checkpoint':str(resume.resolve()),'source_checkpoint_manifest_sha256':sha(mp),'source_campaign_state_sha256':sha(state_path),'source_arm':'varlen_candidate','source_canary_recipe_sha256':CANARY_RECIPE_SHA,'source_identity_sha256':canonical_sha(observed),'destination_execution':'ordinary_sdpa_unpacked','destination_identity_sha256':canonical_sha(production_identity),'selection_evidence':SELECTION_EVIDENCE,'transition_admission':str(admission_path.resolve()),'transition_admission_sha256':sha(admission_path)}
 return observed,lineage
