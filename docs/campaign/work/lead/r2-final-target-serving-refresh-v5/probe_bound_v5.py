#!/usr/bin/env python3
import argparse,ctypes,hashlib,json,os,signal,subprocess,tempfile
from pathlib import Path
from prepare_refresh_v5 import validate_binding,require,write_new
from record_prelaunch_verification import validate as validate_prelaunch
from cuda_lock_contract import require_lock_fd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);ap.add_argument('--verification',required=True);ap.add_argument('--verification-sha256',required=True);ap.add_argument('--cuda-lock-fd',required=True,type=int);ap.add_argument('--profile',choices=('ngram','draft'),required=True);ap.add_argument('--mode',choices=('ordinary','ngram_mod','released_dspark','existing_trained_dspark'),required=True);ap.add_argument('--url',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);require(out.is_absolute() and not out.exists(),'fresh absolute probe output required');bp=Path(a.binding).resolve();b=json.loads(bp.read_text());validate_binding(b,require_admission=True,check_large_payloads=False,serving_light=True);validate_prelaunch(b,bp,a.verification,a.verification_sha256);require_lock_fd(a.cuda_lock_fd);require(os.get_inheritable(a.cuda_lock_fd),'CUDA lock fd not inherited');profile=b['spec_profiles'][a.profile]
 if a.profile=='ngram':require(a.mode in {'ordinary','ngram_mod'},'mode/profile mismatch')
 else:require(a.mode in {'ordinary','released_dspark','existing_trained_dspark'},'mode/profile mismatch')
 key=b['selected_quant_artifact_key'];model=Path(b['outputs'][key]['path']);require(model.is_file(),'selected quant missing');draft=None
 if a.mode in {'released_dspark','existing_trained_dspark'}:
  d=b['drafts'][a.mode];require(Path(d['path']).is_file(),'draft missing');draft={'key':a.mode,'sha256':d['sha256'],'target_relation':d['target_relation']}
 bound={'binding_sha256':sha(bp),'target_identity':b['target']['identity'],'tokenizer_json_sha256':b['tokenizer_contract']['tokenizer_json_sha256'],'artifact_key':key,'artifact_sha256':b['outputs'][key]['sha256'],'prelaunch_verification_sha256':a.verification_sha256,'mode':a.mode,'profile_id':profile['id'],'draft':draft}
 probe=Path(__file__).with_name('paired_panel_probe_v5.py')
 with tempfile.TemporaryDirectory(prefix='sepalith-spec-v3-') as td:
  raw=Path(td)/'raw.json';cmd=['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(probe),'--url',a.url,'--panel',profile['path'],'--manifest',profile['manifest_path'],'--context','10240','--cap','64','--reps',str(profile['repetitions']),'--out',str(raw)]
  def pdeath():
   if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError('PR_SET_PDEATHSIG failed')
  proc=subprocess.run(cmd,check=False,close_fds=True,pass_fds=(a.cuda_lock_fd,),preexec_fn=pdeath);require(raw.is_file(),'probe result missing');result=json.loads(raw.read_text());result['artifact_binding']=bound;result['prelaunch_verification_sha256']=a.verification_sha256;write_new(a.out,result)
 return 0 if proc.returncode in (0,1) else proc.returncode
if __name__=='__main__':raise SystemExit(main())
