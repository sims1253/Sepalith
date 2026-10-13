#!/usr/bin/env python3
"""Run only the exact CPT bootstrap commands with durable sanitized phase evidence."""
import argparse,base64,datetime,hashlib,json,os,re,shutil,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
from diagnostic_contract import require,sha,validate,verify_file
from managed_python import PROBE,discover,validate as validate_python

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
UV_VERSION='0.11.23';PYTHON_VERSION='3.10.19';MAX_TAIL_BYTES=4096
SECRET=re.compile(rb'hf_[A-Za-z0-9]+')
PHASE_COMMANDS=[
 ('pip_uv',[sys.executable,'-m','pip','--disable-pip-version-check','--no-input','install','--target','{run}/bootstrap','uv==0.11.23'],120),
 ('uv_version',['{run}/bootstrap/bin/uv','--version'],15),
 ('python_install',['{run}/bootstrap/bin/uv','python','install','3.10.19'],180),
 ('python_discovery',['{run}/bootstrap/bin/uv','python','find','--managed-python','--no-python-downloads','--no-project','--resolve-links','3.10.19'],30),
 ('python_probe',['{interpreter}','-I','-c','<same managed_python PROBE>'],30),
 ('venv',['{run}/bootstrap/bin/uv','venv','--python','{interpreter}','{run}/venv'],45),
 ('requirements',['{run}/bootstrap/bin/uv','pip','install','--python','{run}/venv/bin/python','-r','payload/requirements.txt'],120),
]
def event(phase,status,**fields):print(json.dumps({'schema':'sepalith.cloud-cpt.bootstrap-event.v2','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'status':status,**fields},sort_keys=True),flush=True)
def sanitized(data,limit):
 data=SECRET.sub(b'[REDACTED]',data)
 data=re.sub(rb'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+',rb'\1[REDACTED]',data)
 data=re.sub(rb'(://)[^/@\s:]+:[^/@\s]+@',rb'\1[REDACTED]@',data)
 text=data[-limit:].decode('utf-8','replace')
 while len(text.encode('utf-8'))>limit:text=text[1:]
 return text
def safe_tail(path):
 try:data=Path(path).read_bytes()[-MAX_TAIL_BYTES:]
 except OSError:return ''
 return sanitized(data,MAX_TAIL_BYTES)
def safe_message(error):return sanitized(str(error).encode('utf-8','replace'),1024)
def env(run):
 credential_markers=('TOKEN','SECRET','CREDENTIAL','PASSWORD','PASSWD','API_KEY','ACCESS_KEY','PRIVATE_KEY','AUTH')
 clean={k:v for k,v in os.environ.items() if not any(marker in k.upper() for marker in credential_markers)}
 clean.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_PROGRESS_BARS='1',UV_CACHE_DIR=str(run/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(run/'python'))
 return clean
def phase(name,argv,deadline,run,clean,log=None):
 log=run/(name+'.log') if log is None else Path(log);event(name,'START',argv=argv)
 remaining=min(deadline-time.time(),PHASE_CAPS[name]);require(remaining>0,'diagnostic deadline exhausted')
 with log.open('xb') as f:proc=subprocess.run(argv,cwd=HERE,env=clean,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
 tail=safe_tail(log);event(name,'END' if proc.returncode==0 else 'EXCEPTION',exit_code=proc.returncode,log_tail=tail)
 require(proc.returncode==0,name+' failed');return tail
def persist(binding,record):
 token=os.environ.get('HF_TOKEN');require(bool(token),'private persistence token absent')
 path=binding['artifact_prefix']+'/diagnostic-terminal.json';body=json.dumps(record,sort_keys=True).encode();payload=[{'key':'header','value':{'summary':'CPT bootstrap diagnostic','description':''}},{'key':'file','value':{'content':base64.b64encode(body).decode(),'path':path,'encoding':'base64'}}]
 data=b''.join(json.dumps(row,separators=(',',':')).encode()+b'\n' for row in payload);request=urllib.request.Request('https://huggingface.co/api/models/'+binding['artifact_repo']+'/commit/main',data=data,headers={'Authorization':'Bearer '+token,'Content-Type':'application/x-ndjson'},method='POST')
 with urllib.request.urlopen(request,timeout=30) as response:result=json.loads(response.read())
 require(re.fullmatch('[0-9a-f]{40,64}',result.get('commitOid','')),'diagnostic persistence revision missing');return result['commitOid']
def main():
 parser=argparse.ArgumentParser();parser.add_argument('binding',type=Path);args=parser.parse_args();current='binding';binding=None;trusted=False;run=None;result=None
 try:
  event(current,'START');binding=json.loads(args.binding.read_text());deadline=validate(binding);trusted=True;event(current,'END')
  verify_file(ROOT/'payload-manifest.json',binding['payload_manifest_sha256'],'payload manifest');manifest=json.loads((ROOT/'payload-manifest.json').read_text())
  for row in manifest['files']:verify_file(ROOT/row['path'],row['sha256'],'payload file')
  admission=ROOT/binding['root_admission_relative_path'];verify_file(admission,binding['root_admission_sha256'],'root admission');a=json.loads(admission.read_text());require(a.get('admitted') is True and str(a.get('status','')).lower().find('diagnostic')>=0,'diagnostic admission is not affirmative');require(binding['payload_manifest_sha256'] in json.dumps(a,sort_keys=True),'diagnostic admission does not bind payload')
  armed=ROOT/binding['watchdog_armed_receipt_relative_path'];verify_file(armed,binding['watchdog_armed_receipt_sha256'],'watchdog receipt');w=json.loads(armed.read_text());require(w.get('name')=='sepalith-cpt-'+binding['run_id'] and w.get('deadline')==binding['absolute_deadline_utc'],'watchdog identity differs')
  require(Path('/mnt/local_storage').is_dir() and shutil.disk_usage('/mnt/local_storage').free>=24*1024**3,'local storage below24GiB');require(shutil.disk_usage('/').free>=16*1024**3,'root filesystem below16GiB')
  run=Path(tempfile.mkdtemp(prefix='sepalith-cpt-bootstrap-'+binding['run_id']+'-',dir='/mnt/local_storage'));clean=env(run);global PHASE_CAPS;PHASE_CAPS={name:cap for name,_,cap in PHASE_COMMANDS}
  commands=[];current='pip_uv';uv_target=run/'bootstrap';phase(current,[sys.executable,'-m','pip','--disable-pip-version-check','--no-input','install','--target',str(uv_target),'uv==0.11.23'],deadline,run,clean);commands.append(current)
  uv=uv_target/'bin/uv';current='uv_version';phase(current,[str(uv),'--version'],deadline,run,clean);require((run/'uv_version.log').read_text().strip().split()[:2]==['uv',UV_VERSION],'uv version differs');commands.append(current)
  current='python_install';phase(current,[str(uv),'python','install',PYTHON_VERSION],deadline,run,clean);commands.append(current)
  current='python_discovery';interpreter=discover(uv,run/'python',clean,min(deadline,time.time()+30),lambda argv,env,cwd,cap,log: phase(current,argv,cap,run,env,log),HERE,run/'python_discovery.log');commands.append(current)
  current='python_probe';phase(current,[str(interpreter),'-I','-c',PROBE],deadline,run,clean);validate_python(json.loads((run/'python_probe.log').read_text()),interpreter,run/'python');commands.append(current)
  current='venv';phase(current,[str(uv),'venv','--python',str(interpreter),str(run/'venv')],deadline,run,clean);commands.append(current)
  current='requirements';phase(current,[str(uv),'pip','install','--python',str(run/'venv/bin/python'),'-r',str(HERE/'requirements.txt')],deadline,run,clean);commands.append(current)
  result={'schema':'sepalith.cloud-cpt.bootstrap-diagnostic-terminal.v2','status':'PASS','phase':'complete','commands_completed':commands,'training_started':False,'input_staging_started':False}
 except BaseException as error:
  result={'schema':'sepalith.cloud-cpt.bootstrap-diagnostic-terminal.v2','status':'FAIL','phase':current,'error_type':type(error).__name__,'error_message':safe_message(error),'log_tail':safe_tail(run/(current+'.log')) if run else '','training_started':False,'input_staging_started':False};event(current,'EXCEPTION',error_type=type(error).__name__,error_message=result['error_message'],log_tail=result['log_tail'])
 if binding and trusted:
  try:result['persistence_revision']=persist(binding,result);event('persistence','END',revision=result['persistence_revision'])
  except BaseException as error:event('persistence','EXCEPTION',error_type=type(error).__name__,error_message=safe_message(error))
 print(json.dumps(result,sort_keys=True),flush=True);raise SystemExit(0 if result['status']=='PASS' else 1)
if __name__=='__main__':main()
