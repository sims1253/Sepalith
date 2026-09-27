"""Fail-closed contract for one bootstrap-only diagnostic allocation."""
import datetime,hashlib,json,re,time
from pathlib import Path

IMAGE='docker.io/anyscale/ray@sha256:3b904e7cbb1736a8a17359708197d62333daa0bc8285d7de66f1d22aa78754f9'
INSTANCE='g5.2xlarge';REPO='scholzmx/sepalith-lora'
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def utc(value):return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def validate(b,now=None):
 now=time.time() if now is None else now
 require(b.get('schema')=='sepalith.cloud-cpt.bootstrap-diagnostic.v2' and b.get('admitted') is True,'diagnostic root admission pending')
 require(re.fullmatch('[0-9a-f]{32}',b.get('run_id','')),'fresh diagnostic UUID32 required')
 require(b.get('image_uri')==IMAGE and b.get('instance_type')==INSTANCE,'image or instance differs')
 require(b.get('artifact_repo')==REPO and b.get('artifact_prefix')=='r2-cpt-bootstrap-diagnostic/'+b['run_id'],'private diagnostic prefix differs')
 require(b.get('provider_timeout_seconds')==600 and b.get('watchdog_timeout_seconds')==900 and b.get('max_retries')==0,'diagnostic lifecycle bound differs')
 require(b.get('training_authorized') is False and b.get('input_staging_authorized') is False,'diagnostic scope expanded')
 require(re.fullmatch('[0-9a-f]{64}',b.get('payload_manifest_sha256','')),'payload binding missing')
 require(re.fullmatch('[0-9a-f]{64}',b.get('root_admission_sha256','')),'root admission binding missing')
 require(b.get('root_admission_relative_path')=='root-diagnostic-admission.json','root admission path differs')
 require(re.fullmatch('[0-9a-f]{64}',b.get('watchdog_armed_receipt_sha256','')),'watchdog binding missing')
 require(b.get('watchdog_armed_receipt_relative_path')=='watchdog-armed.json','watchdog path differs')
 deadline=utc(b['absolute_deadline_utc']);armed=utc(b['watchdog_armed_at_utc'])
 require(armed<=now<deadline<=armed+900,'diagnostic window expired or expanded')
 require(not any('TOKEN' in str(k).upper() or 'SECRET' in str(k).upper() for k in b),'credential fields forbidden')
 return deadline
def verify_file(path,digest,label):require(Path(path).is_file() and not Path(path).is_symlink() and sha(path)==digest,label+' differs')
