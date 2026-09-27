"""Fail-closed validation for a root-admitted native physical-path overlay."""
import hashlib,json
from pathlib import Path
from native_attestation import require_attested,require_attested_identity

def require(v,m):
 if not v:raise ValueError(m)
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def canonical_sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

RECOVERY_ALLOWED_CHANGES=[
 'selected packed330 identity-transition validation',
 'immutable packed canary and selection lineage propagation',
 'fresh ordinary-production runtime output roots',
 'recovery checkpoint cadence 24 with first mandatory stop 354',
]
CANONICAL_IDENTITY_SHA='730c2aba916ae9ab8a97a1584207e7c1d7095e23ce1988cdd968df7fd17844b6'
RECOVERY_IDENTITY_SHA='46a5a1aced9f533e8ba4cd25aebe795dedc64d390a0d563cfc7e406c03324739'
CADENCE64_ALLOWED_CHANGES=[
 'checkpoint cadence 24 to 64 after accepted checkpoint450',
 'mandatory external review boundary 354 to 706',
 'fresh continuation runtime output roots',
]

def validate_cadence64_identity(source,recipe,identity_function,migration,check_evidence=True):
 old=identity_function(source);new=identity_function(recipe)
 require(migration.get('allowed_changes')==CADENCE64_ALLOWED_CHANGES,'cadence64 allowed changes differ')
 require(migration.get('source_identity_sha256')==canonical_sha(old)==RECOVERY_IDENTITY_SHA,'cadence64 source identity differs')
 require(migration.get('destination_identity_sha256')==canonical_sha(new),'cadence64 destination identity differs')
 for key in old:
  if key!='schedule':require(old[key]==new[key],f'cadence64 changed scientific identity:{key}')
 stable=lambda x:{k:v for k,v in x.items() if k not in {'checkpoint_every','mandatory_stop_step'}}
 require(stable(old['schedule'])==stable(new['schedule']),'cadence64 changed optimizer/data/horizon schedule')
 require((old['schedule'].get('checkpoint_every'),old['schedule'].get('mandatory_stop_step'))==(24,354),'cadence64 source schedule differs')
 require((new['schedule'].get('checkpoint_every'),new['schedule'].get('mandatory_stop_step'))==(64,706),'cadence64 destination schedule differs')
 if check_evidence:
  source_path=Path(migration.get('source_runtime_recipe',''))
  require(source_path.is_file() and sha256(source_path)==migration.get('source_runtime_recipe_sha256'),'cadence64 source recipe differs')
  require(identity_function(json.loads(source_path.read_text()))==old,'cadence64 source recipe identity differs')
  for name in ('checkpoint450_review','matched450_review'):
   record=migration.get(name,{});path=Path(record.get('path',''))
   require(path.is_file() and sha256(path)==record.get('sha256'),f'{name} differs')
   evidence=json.loads(path.read_text());require(evidence.get('status')=='accepted' and evidence.get('checkpoint_step')==450,f'{name} not accepted')
 return canonical_sha(new)

def validate_recovery_identity(canonical,recipe,identity_function,migration):
 old=identity_function(canonical);new=identity_function(recipe)
 if old==new:return
 if migration.get('allowed_changes')==CADENCE64_ALLOWED_CHANGES:
  source_path=Path(migration.get('source_runtime_recipe',''))
  require(source_path.is_file() and sha256(source_path)==migration.get('source_runtime_recipe_sha256'),'cadence64 source recipe differs')
  validate_cadence64_identity(json.loads(source_path.read_text()),recipe,identity_function,migration)
  return
 require(migration.get('allowed_changes')==RECOVERY_ALLOWED_CHANGES,'runtime recovery allowed changes differ')
 require(migration.get('canonical_destination_identity_sha256')==canonical_sha(old)==CANONICAL_IDENTITY_SHA,'canonical destination identity differs')
 require(migration.get('recovery_destination_identity_sha256')==canonical_sha(new)==RECOVERY_IDENTITY_SHA,'recovery destination identity differs')
 for key in old:
  if key!='schedule':require(old[key]==new[key],f'recovery changed scientific identity:{key}')
 stable=lambda x:{k:v for k,v in x.items() if k not in {'checkpoint_every','mandatory_stop_step'}}
 require(stable(old['schedule'])==stable(new['schedule']),'recovery changed optimizer/data schedule')
 require((old['schedule'].get('checkpoint_every'),old['schedule'].get('mandatory_stop_step'))==(128,194),'canonical recovery schedule differs')
 require((new['schedule'].get('checkpoint_every'),new['schedule'].get('mandatory_stop_step'))==(24,354),'admitted recovery schedule differs')

def validate(recipe,identity_function):
 relocation=recipe.get('storage_relocation',{});bundle=relocation.get('bundle_id')
 require(relocation.get('schema')=='sepalith.sft11.native-runtime-overlay.v1' and isinstance(bundle,str),'native relocation missing')
 canonical_path=Path(relocation['canonical_bound_recipe']);require(canonical_path.is_file() and sha256(canonical_path)==relocation['canonical_bound_recipe_sha256'],'canonical bound recipe differs')
 canonical=json.loads(canonical_path.read_text())
 receipt_path=Path(relocation['stage_receipt']);require(receipt_path.is_file() and sha256(receipt_path)==relocation['stage_receipt_sha256'],'stage receipt differs')
 receipt=json.loads(receipt_path.read_text());require(receipt.get('bundle_id')==bundle and receipt.get('status')=='staged_immutable_verified','stage bundle differs')
 admission_path=Path(relocation['admission']);require(admission_path.is_file() and sha256(admission_path)==relocation['admission_sha256'],'relocation admission differs')
 admission=json.loads(admission_path.read_text());require(admission.get('schema')=='sepalith.sft11.native-relocation-admission.v1' and admission.get('status')=='admitted' and admission.get('launch_authorized')is True,'relocation not admitted')
 require(admission.get('bound_recipe_sha256')==relocation['canonical_bound_recipe_sha256'] and admission.get('stage_receipt_sha256')==relocation['stage_receipt_sha256'],'relocation admission identity differs')
 migration=recipe.get('runtime_source_migration',{});mp=Path(migration.get('admission',''))
 require(mp.is_file() and sha256(mp)==migration.get('admission_sha256'),'runtime source migration admission differs')
 mv=json.loads(mp.read_text());require(mv.get('schema')=='sepalith.sft11.native-runtime-source-migration-admission.v1' and mv.get('status')=='admitted' and mv.get('launch_authorized')is True,'runtime source migration not admitted')
 validate_recovery_identity(canonical,recipe,identity_function,mv)
 require(mv.get('scientific_source_manifest_sha256')==recipe['source']['manifest_sha256'],'scientific source identity changed')
 require(mv.get('runtime_source_manifest_sha256')==recipe['runtime_source']['manifest_sha256'],'runtime source admission differs')
 require(mv.get('canonical_bound_recipe_sha256')==relocation['canonical_bound_recipe_sha256'],'runtime source admission recipe differs')
 require(mv.get('stage_receipt_sha256')==relocation['stage_receipt_sha256'],'runtime source admission stage differs')
 cache=Path(recipe['cohort']['streaming_cache']['path']);require(require_attested_identity(cache/'manifest.json',bundle)['sha256']==recipe['cohort']['streaming_cache']['manifest_sha256'],'staged cache manifest identity differs');cm=json.loads((cache/'manifest.json').read_text())
 for name,item in cm['files'].items():require_attested(cache/name,item['sha256'],item['bytes'],bundle)
 require(require_attested_identity(recipe['cohort']['rows']['path'],bundle)['sha256']==recipe['cohort']['rows']['sha256'],'staged rows identity differs')
 require(require_attested_identity(recipe['cohort']['draw_schedule']['path'],bundle)['sha256']==recipe['cohort']['draw_schedule']['sha256'],'staged schedule identity differs')
 if 'source_checkpoint' in relocation.get('canonical_paths',{}):
  parent=Path(recipe['parent']['path']);require(require_attested_identity(parent/'campaign-manifest.json',bundle)['sha256']==recipe['transition']['source_checkpoint']['manifest_sha256'],'staged parent manifest identity differs');pm=json.loads((parent/'campaign-manifest.json').read_text())
  for name,expected_sha in recipe['parent']['files'].items():
   require(name in pm['files'],'parent file absent from checkpoint manifest');require_attested(parent/name,expected_sha,pm['files'][name]['bytes'],bundle)
 os=__import__('os');resume_bytes=int(os.environ.get('SEPALITH_NATIVE_RESUME_BYTES','0'));resume_path=Path(os.environ.get('SEPALITH_NATIVE_RESUME_PATH','/nonexistent'))
 if str(resume_path.resolve()).startswith(str(Path(recipe['outputs']['trainer']).resolve())+'/'):resume_bytes=0
 require(resume_bytes>=0,'native resume bytes differ')
 return {'bundle_id':bundle,'stage_receipt_sha256':relocation['stage_receipt_sha256'],'total_bytes':receipt['total_bytes']+resume_bytes,'canonical_bound_recipe_sha256':relocation['canonical_bound_recipe_sha256']}

def canonical_source_checkpoint(recipe,physical):
 relocation=recipe.get('storage_relocation',{});canonical=relocation.get('canonical_paths',{}).get('source_checkpoint')
 if canonical and Path(physical).resolve()==Path(recipe['transition']['source_checkpoint']['path']).resolve():return str(Path(canonical).resolve())
 return str(Path(physical).resolve())
