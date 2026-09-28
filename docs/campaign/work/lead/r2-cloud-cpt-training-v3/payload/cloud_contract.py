"""Frozen constants and fail-closed checks for the cloud CPT packet."""
import datetime, hashlib, json, math, os, re, time
from pathlib import Path

IMAGE='docker.io/anyscale/ray@sha256:3b904e7cbb1736a8a17359708197d62333daa0bc8285d7de66f1d22aa78754f9'
INSTANCE='g5.2xlarge';REPO='scholzmx/sepalith-lora'
RECIPE_SHA='254de1150b0fcc8e8566178905542bdadc1255610a2dc3cb8a9168dc7017d538'
TRANSPORT_SHA='a9afe5d58034b56e36abf83a0c1d169ec578b79cc215daa32603ccd210e7045b'

def require(ok,reason):
 if not ok:raise ValueError(reason)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def utc(value):
 d=datetime.datetime.fromisoformat(value.replace('Z','+00:00'));require(d.tzinfo is not None,'UTC offset missing');return d.timestamp()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def validate_binding(b,now=None):
 now=time.time() if now is None else now
 require(b.get('schema')=='sepalith.cloud-cpt.binding.v1' and b.get('admitted') is True,'root cloud admission pending')
 require(re.fullmatch('[0-9a-f]{32}',b.get('run_id','')),'fresh UUID32 required')
 require(b.get('image_uri')==IMAGE and b.get('instance_type')==INSTANCE,'image or instance differs')
 require(b.get('input_repo')==REPO and b.get('artifact_repo')==REPO,'private repository differs')
 require(b.get('input_manifest_sha256')==TRANSPORT_SHA and b.get('recipe_sha256')==RECIPE_SHA,'frozen input identity differs')
 require(re.fullmatch('[0-9a-f]{40,64}',b.get('input_revision','')),'immutable input revision required')
 require(re.fullmatch('[0-9a-f]{64}',b.get('root_recipe_admission_sha256','')) and b['root_recipe_admission_sha256']!='0'*64,'root recipe admission missing')
 require(b.get('root_recipe_admission_relative_path')=='root-recipe-admission.json','root recipe admission path differs')
 require(re.fullmatch('[0-9a-f]{64}',b.get('watchdog_armed_receipt_sha256','')),'watchdog receipt binding missing')
 require(b.get('watchdog_armed_receipt_relative_path')=='watchdog-armed.json','watchdog receipt path differs')
 require(re.fullmatch('[0-9a-f]{64}',b.get('payload_manifest_sha256','')),'payload manifest binding missing')
 require(b.get('artifact_prefix')=='r2-cpt/'+b['run_id'],'private artifact prefix differs')
 require(b.get('provider_timeout_seconds')==28800 and b.get('watchdog_timeout_seconds')==29100,'deadline layers differ')
 require(b.get('max_incremental_charge_usd')==28.0,'charge reservation differs')
 armed=utc(b['watchdog_armed_at_utc']);absolute=utc(b['absolute_deadline_utc'])
 require(armed<=now<absolute<=armed+29100,'expired or expanded cloud window')
 phases=b.get('phase_seconds')
 require(phases=={'bootstrap':1800,'staging':3600,'training':19800,'upload':3000,'cleanup':300},'phase budgets differ')
 require(not any('TOKEN' in str(k).upper() or 'SECRET' in str(k).upper() for k in b),'credential fields forbidden')
 resume=b.get('resume')
 if resume is not None:
  required={'revision','inventory_path','inventory_sha256','step'}
  require(isinstance(resume,dict) and set(resume)==required,'resume binding fields differ')
  require(re.fullmatch('[0-9a-f]{40,64}',resume['revision']) and re.fullmatch('[0-9a-f]{64}',resume['inventory_sha256']),'resume hashes invalid')
  require(type(resume['step']) is int and resume['step'] in (317,634,951,1268,1585),'resume step not full cadence')
  require(re.fullmatch(r'r2-cpt/[0-9a-f]{32}/checkpoint-receipts/checkpoint-'+str(resume['step'])+r'\.json',resume['inventory_path']),'resume inventory path differs')
 return absolute

def remaining(hard,cap,reserve,now=None):
 now=time.time() if now is None else now;value=min(now+cap,hard-reserve)
 require(math.isfinite(value) and value>now,'no phase budget remains');return value
