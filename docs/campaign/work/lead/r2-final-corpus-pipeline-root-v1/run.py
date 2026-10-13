import json,os,time,hashlib,subprocess,datetime
from pathlib import Path
BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work')
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead')
OUT=BASE/'CPT-final-union-v1-ctx16384';SCHEDULE=BASE/'CPT-final-union-v1-draw-schedule.json';CACHE=BASE/'CPT-streaming-input-v1/final-union-cache'
PY='/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def emit(v):print(json.dumps(v),flush=True)
def live(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
try:
 start=time.monotonic()
 while live(1588884):
  if time.monotonic()-start>7200:raise RuntimeError('conversion observation deadline reached; do not restart conversion')
  time.sleep(5)
 result=json.loads((OUT/'result.json').read_text());union=json.loads((BASE/'CPT-final-union-v1/manifest.json').read_text())
 assert result['status']=='complete'
 assert result['manifest_sha256']==sha(BASE/'CPT-final-union-v1/input-manifest.json')
 assert result['totals']=={'input_rows':union['counts']['rows'],'documents':union['counts']['documents'],'payload_tokens':union['counts']['payload_tokens']}
 assert result['outputs']['16384']['payload_tokens']==460833265 and result['outputs']['16384']['terminal_eos']==177190
 schedule_source=PLAN/'r2-final-corpus-launch-inputs-v1/make_one_pass_schedule.py'
 cache_source=PLAN/'r2-full-weight-cpt-full-corpus-trainer-v3/source/experiments/training/cpt_streaming_cache.py'
 assert sha(schedule_source)=='a34a06ea7e56191eeb68c2c99432b855fb6189810db98aa1eab621797adec8fd'
 assert sha(cache_source)=='0729f5a3f8a5363bc6ac5c2fc59c7ad2e22c125f0e39e5b43756e1eb5b940976'
 rows=OUT/'cpt_train_ctx16384.jsonl'
 emit({'event':'conversion_verified','counts':result['totals'],'outputs':result['outputs']})
 subprocess.run([PY,'-B',str(schedule_source),'--rows',str(rows),'--rechunk-result',str(OUT/'result.json'),'--output',str(SCHEDULE),'--seed','3407','--split-id','cpt_train_final_union_ctx16384_v1'],check=True,timeout=7200)
 schedule=json.loads(SCHEDULE.read_text());assert schedule['token_rows_sha256']==result['artifacts'][rows.name]['sha256']
 emit({'event':'schedule_verified','coverage':schedule['coverage']})
 CACHE.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run([PY,'-B',str(cache_source),'--rows',str(rows),'--schedule',str(SCHEDULE),'--output',str(CACHE),'--max-sequence-tokens','16384','--rows-sha256',schedule['token_rows_sha256'],'--schedule-sha256',sha(SCHEDULE)],check=True,timeout=7200)
 terminal={'status':'pipeline_complete_pending_root_data_admission','cache_manifest_sha256':sha(CACHE/'manifest.json'),'schedule_sha256':sha(SCHEDULE),'conversion_result_sha256':sha(OUT/'result.json'),'training_admission':False}
except Exception as e:
 terminal={'status':'failed','error':repr(e),'training_admission':False}
 terminal['observed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (BASE/'CPT-final-corpus-pipeline-root-v1.terminal.json').write_text(json.dumps(terminal,indent=2)+'\n');emit(terminal);raise
terminal['observed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(BASE/'CPT-final-corpus-pipeline-root-v1.terminal.json').write_text(json.dumps(terminal,indent=2)+'\n');emit(terminal)
