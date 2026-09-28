#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess,tempfile
from pathlib import Path
from prepare_refresh_v2 import validate_binding,artifact,require,write_new
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);ap.add_argument('--profile',choices=('ngram','draft'),required=True);ap.add_argument('--mode',choices=('ordinary','ngram_mod','released_dspark','existing_trained_dspark'),required=True);ap.add_argument('--url',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);require(out.is_absolute() and not out.exists(),'fresh absolute probe output required');bp=Path(a.binding);b=json.loads(bp.read_text());validate_binding(b,require_admission=True,check_large_payloads=True);profile=b['spec_profiles'][a.profile]
 if a.profile=='ngram':require(a.mode in {'ordinary','ngram_mod'},'mode/profile mismatch')
 else:require(a.mode in {'ordinary','released_dspark','existing_trained_dspark'},'mode/profile mismatch')
 model=artifact(b['outputs']['q8'],'q8');draft=None
 if a.mode in {'released_dspark','existing_trained_dspark'}:
  d=b['drafts'][a.mode];artifact(d,a.mode);artifact({'path':d['header_receipt_path'],'sha256':d['header_receipt_sha256']},a.mode+' header');draft={'key':a.mode,'sha256':d['sha256'],'target_relation':d['target_relation']}
 bound={'binding_sha256':sha(bp),'target_identity':b['target']['identity'],'tokenizer_json_sha256':b['tokenizer_contract']['tokenizer_json_sha256'],'artifact_key':'q8','artifact_sha256':b['outputs']['q8']['sha256'],'mode':a.mode,'profile_id':profile['id'],'draft':draft}
 probe=Path(__file__).with_name('paired_panel_probe_v2.py')
 with tempfile.TemporaryDirectory(prefix='sepalith-spec-v2-') as td:
  raw=Path(td)/'raw.json';cmd=['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(probe),'--url',a.url,'--panel',profile['path'],'--manifest',profile['manifest_path'],'--context','10240','--cap','64','--reps',str(profile['repetitions']),'--out',str(raw)];proc=subprocess.run(cmd,check=False);require(raw.is_file(),'probe result missing');result=json.loads(raw.read_text());result['artifact_binding']=bound;write_new(a.out,result)
 return 0 if proc.returncode in (0,1) else proc.returncode
if __name__=='__main__':raise SystemExit(main())
