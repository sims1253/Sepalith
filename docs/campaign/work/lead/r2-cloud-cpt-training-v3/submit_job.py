"""Submit exactly one root-admitted CPT job; root executes this file."""
import contextlib,datetime,hashlib,io,json,logging,os,re,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(ok,why):
 if not ok:raise ValueError(why)
def main():
 import anyscale
 from anyscale.job.models import JobConfig
 binding_path=ROOT/'binding.root.json';b=json.loads(binding_path.read_text());run=b['run_id'];name='sepalith-cpt-'+run
 require(b.get('admitted') is True and re.fullmatch('[0-9a-f]{32}',run),'root binding is not admitted')
 require(sha(ROOT/'payload-manifest.json')==b['payload_manifest_sha256'],'payload manifest differs')
 require(sha(ROOT/b['root_recipe_admission_relative_path'])==b['root_recipe_admission_sha256'],'root recipe admission differs')
 require(sha(ROOT/b['watchdog_armed_receipt_relative_path'])==b['watchdog_armed_receipt_sha256'],'watchdog armed receipt differs')
 armed=json.loads((ROOT/b['watchdog_armed_receipt_relative_path']).read_text());require(armed['name']==name and Path('/proc/'+str(armed['pid'])).exists(),'live watchdog differs')
 require(not (ROOT/'submission-started.json').exists(),'submission was already attempted')
 deadline=datetime.datetime.fromisoformat(b['absolute_deadline_utc']).timestamp();require(time.time()+27000<deadline,'insufficient provider window remains')
 api=anyscale.Anyscale()._anyscale_client._internal_api_client;credits=api.get_credits_v2_api_v2_organization_billing_credits_v2_get(_request_timeout=20).to_dict();credits=credits.get('result') or credits
 require(float(credits['current_balance_usd'])>=28 and float(credits['amount_spent_usd'])+28<=60,'USD28 campaign reservation unavailable')
 with (ROOT/'submission-started.json').open('x') as f:json.dump({'at':stamp(),'run_id':run,'pid':os.getpid()},f)
 token=os.environ['HF_TOKEN'];config=JobConfig(name=name,cloud='Anyscale Cloud',working_dir=str(ROOT),image_uri=b['image_uri'],ray_version='2.57.0',entrypoint='bash payload/root-bound-entry.sh',compute_config={'head_node':{'instance_type':'g5.2xlarge'},'worker_nodes':[]},max_retries=0,timeout_s=28800,env_vars={'HF_TOKEN':token,'PYTHONDONTWRITEBYTECODE':'1','SEPALITH_CPT_BINDING':'binding.root.json'},excludes=['__pycache__','*.pyc','.env','.git','tests.log','input-verification.json'])
 logging.disable(logging.CRITICAL);capture=io.StringIO()
 try:
  with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):job_id=anyscale.job.submit(config)
  require(isinstance(job_id,str) and job_id.startswith('prodjob_'),'provider job ID differs')
  receipt={'at':stamp(),'status':'submitted_not_runtime_accepted','job_id':job_id,'name':name,'run_id':run,'provider_timeout_seconds':28800,'max_incremental_charge_usd':28.0,'absolute_deadline_utc':b['absolute_deadline_utc']};(ROOT/'submission-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
 except Exception as e:
  message=str(e).replace(token,'[REDACTED]')[:1000] if isinstance(e,ValueError) else 'Provider exception; raw request suppressed'
  receipt={'at':stamp(),'status':'submission_failed_no_retry_authorized','error_type':type(e).__name__,'message':message,'run_id':run,'stack':[{'file':x.filename,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)]};(ROOT/'submission-error.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(1)
if __name__=='__main__':
 try:main()
 except Exception as e:print(json.dumps({'status':'pre_submission_rejected','error_type':type(e).__name__}));raise SystemExit(1)
