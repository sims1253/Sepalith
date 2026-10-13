"""Root-gated immutable ordinary-canary checkpoint to production identity transition."""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path

SCHEMA='sepalith.sft11.ordinary-canary-resume-transition-admission.v1'
ORDINARY_POLICY={
 'execution_experiment':'native_full_optimizer_varlen_canary_v1',
 'execution_arm':'ordinary_reference',
 'logical_effective_batch':16,
 'loss_reduction':'global_supervised_token_mean',
 'physical_gradient_accumulation':16,
 'attention_packing':'ordinary_padded_rows_v1',
 'attention_backend':'sdpa_ordinary',
}
FULL_FILES={'campaign-state.json','chat_template.jinja','config.json','generation_config.json','model.safetensors','optimizer.pt','rng_state.pth','scheduler.pt','tokenizer.json','tokenizer_config.json','trainer_state.json','training_args.bin'}
def require(v,m):
 if not v:raise ValueError(m)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def canary_identity(production_identity):
 value=copy.deepcopy(production_identity);value['policy'].update(ORDINARY_POLICY);return value

def _validate_lineage(value):
 require(value.get('schema')=='sepalith.sft11.ordinary-canary-resume-lineage.v1','resume lineage schema differs')
 admission=Path(value.get('transition_admission',''));require(admission.is_file() and sha(admission)==value.get('transition_admission_sha256'),'resume lineage admission differs')
 return value

def resolve_resume_identity(recipe_path,production_identity,resume,transition_admission=None):
 """Return the immutable checkpoint identity and lineage to carry into future checkpoints."""
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
 expected=canary_identity(production_identity)
 require(observed==expected,'resume identity is neither production nor exact ordinary canary')
 require(transition_admission is not None,'ordinary canary identity transition requires root admission')
 admission_path=Path(transition_admission);require(admission_path.is_file(),'identity transition admission missing');ad=json.loads(admission_path.read_text())
 require(ad.get('schema')==SCHEMA and ad.get('status')=='admitted' and ad.get('decision')=='ordinary_canary_to_production' and ad.get('launch_authorized')is True,'identity transition is not admitted')
 require(ad.get('bound_recipe_sha256')==sha(recipe_path),'identity transition refers to another recipe')
 require(Path(ad.get('checkpoint','')).resolve()==resume.resolve() and ad.get('checkpoint_manifest_sha256')==sha(mp),'identity transition checkpoint differs')
 require(ad.get('checkpoint_step')==manifest.get('step')==330,'identity transition step differs')
 sampler=state.get('sampler',{});require((state.get('step'),sampler.get('global_step'),sampler.get('cursor'),sampler.get('stage_cursor'),sampler.get('global_optimizer_step_offset'),sampler.get('effective_batch'))==(330,330,4224,4224,66,16),'identity transition cursor/schedule differs')
 require(ad.get('source_identity_sha256')==canonical_sha(observed) and ad.get('destination_identity_sha256')==canonical_sha(production_identity),'identity transition hashes differ')
 require(ad.get('allowed_policy_delta')==ORDINARY_POLICY,'identity transition policy delta differs')
 require(ad.get('payload_action')=='load_original_full_state_without_rewrite_or_copy','identity transition payload action differs')
 lineage={'schema':'sepalith.sft11.ordinary-canary-resume-lineage.v1','source_checkpoint':str(resume.resolve()),'source_checkpoint_manifest_sha256':sha(mp),'source_identity_sha256':canonical_sha(observed),'destination_identity_sha256':canonical_sha(production_identity),'transition_admission':str(admission_path.resolve()),'transition_admission_sha256':sha(admission_path)}
 return observed,lineage
