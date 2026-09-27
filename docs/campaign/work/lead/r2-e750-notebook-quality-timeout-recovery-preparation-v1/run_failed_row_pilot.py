#!/usr/bin/env python3
"""Root-admitted offline CPU thread pilot for the one censored DEV row."""
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
BASE=Path.home()/'.local/share/sepalith-e750-notebook-cpu-quality-v1'
FROZEN=BASE/'packet-final';sys.path.insert(0,str(FROZEN))
import run_suite as common
import campaign_protocol as protocol
import native_transport as transport
ARMS=((4,'0,2,4,6'),(6,'0,2,4,6,8,10'),(8,'0,2,4,6,8,10,1,3'))
CAP=192;CASE_DEADLINE=300;PORT=18528;MAX_SECONDS=1800;STOP=False

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def validate(packet_path,row_path):
 if sha(FROZEN/'packet.json')!='2b6dd8210ede1db19b00fa1aa8e0815eb4034618acfbc920a5facd31ae27cfb3':raise ValueError('frozen packet changed')
 if sha(FROZEN/'source-manifest.json')!='fa18fefe0d771f8493b59ad08b0968ab7cc62e6532691c2e9ec73f331221db9c':raise ValueError('frozen source changed')
 base=json.loads((FROZEN/'packet.json').read_text());audit=common.validate_packet(base,FROZEN/'packet.json')
 packet=json.loads(packet_path.read_text())
 if sha(Path(__file__).resolve())!=packet['runner_sha256']:raise ValueError('pilot runner changed')
 if packet.get('schema')!='sepalith.run06.e750-notebook-timeout-recovery-pilot.v1' or packet.get('status')!='prepared_no_launch':raise ValueError('pilot packet schema/status')
 if packet['arms']!=[{'threads':n,'cpu_list':c} for n,c in ARMS] or packet['cap']!=CAP or packet['case_deadline_seconds']!=CASE_DEADLINE or packet['maximum_seconds']!=MAX_SECONDS:raise ValueError('pilot design drift')
 if sha(row_path)!=packet['failed_row_sha256']:raise ValueError('failed row hash')
 row=json.loads(row_path.read_text());rows,_=common.load_rows(base);original=next(x for x in rows if x['id']==packet['failed_row_id'])
 if canonical(row)!=canonical(original) or len(row['prompt_ids'])!=2619:raise ValueError('failed row binding')
 return packet,base,audit,row

def run(a):
 packet,base,audit,row=validate(a.packet.resolve(),a.row.resolve())
 if a.preflight_only:return {'status':'preflight_pass_no_model_load','failed_row_id':row['id'],'prompt_tokens':len(row['prompt_ids']),'arms':packet['arms']}
 if a.admission is None:raise ValueError('root admission required')
 adm=json.loads(a.admission.read_text());required={'schema':'sepalith.run06.e750-notebook-timeout-recovery-root-admission.v1','status':'admitted','pilot_packet_sha256':sha(a.packet.resolve()),'base_packet_sha256':audit['packet_sha256'],'failed_row_sha256':sha(a.row.resolve()),'arms':packet['arms'],'cap':CAP,'case_deadline_seconds':CASE_DEADLINE,'maximum_threads':8,'maximum_seconds':MAX_SECONDS,'host':'m0hawk@192.168.178.40','purpose':'offline_failed_row_thread_deadline_pilot_not_production_latency','source_reviewed':True}
 if any(adm.get(k)!=v for k,v in required.items()):raise ValueError('admission mismatch')
 if adm.get('run_id')!=a.run_root.name:raise ValueError('run ID mismatch')
 now=datetime.datetime.now(datetime.timezone.utc);created=datetime.datetime.fromisoformat(adm['created_at']);expires=datetime.datetime.fromisoformat(adm['expires_at'])
 if created.tzinfo is None or expires.tzinfo is None or not created<=now<expires or (expires-created).total_seconds()>1800:raise ValueError('admission time invalid')
 root=a.run_root.resolve();prefix=Path(packet['run_root_prefix'])
 if root.exists() or root.parent!=prefix:raise ValueError('fresh direct run root required')
 root.mkdir(mode=0o700);tokenizer,tok_audit=transport._load_tokenizer(Path(base['tokenizer_dir']));common.write(root/'tokenizer-audit.json',tok_audit)
 started=time.monotonic();terminal={'schema':'sepalith.run06.e750-notebook-timeout-recovery-terminal.v1','status':'failed','arms':[]}
 try:
  for threads,cpus in ARMS:
   if STOP or time.monotonic()-started>=MAX_SECONDS:raise RuntimeError('pilot stop/global deadline')
   arm=root/f'threads-{threads}';arm.mkdir();common.free(PORT);proc=None;tick=None;log=(arm/'server.log').open('xb')
   argv=['/usr/bin/taskset','--cpu-list',cpus,str(base['runtime_server']),'-m',base['model']['path'],'--host','127.0.0.1','--port',str(PORT),'-t',str(threads),'-tb',str(threads),'--threads-http','1','--parallel','1','-c','4096','-b','256','-ub','256','-ngl','0','-lv','4','--metrics']
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
   try:
    proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);tick=common.proc_tick(proc.pid);common.write(arm/'launch.json',{'pid':proc.pid,'start_tick':tick,'argv':argv,'threads':threads,'cpu_list':cpus})
    until=time.monotonic()+180
    while time.monotonic()<until and proc.poll() is None:
     h=common.health(PORT)
     if h and h[0]==200:break
     time.sleep(.25)
    else:raise RuntimeError('server failed health/load')
    def before(_):
     if STOP or proc.poll() is not None or common.proc_tick(proc.pid)!=tick:raise RuntimeError('owned server identity changed')
     if common.statid(base['model']['path'])!=audit['model_stat']:raise RuntimeError('model stat changed')
    transport.COMPLETION_CAP=CAP
    case={'id':row['id'],'family':row['family'],'package_id':row['package_id'],'operation':row['operation'],'expected_noop':row['operation']=='no_op','expected_region':row['region_new'],'context':protocol.PromptContext.from_mapping(row['context']),'prompt':row['prompt_text'],'prompt_sha256':row['prompt_sha256'],'target_sha256':row['target_sha256'],'hf_prompt_ids':row['prompt_ids'],'hf_prompt_tokens_with_bos':len(row['prompt_ids']),'hf_prompt_ids_sha256':row['prompt_ids_sha256']}
    result=transport._run_case(f'http://127.0.0.1:{PORT}',case,protocol,tokenizer,CASE_DEADLINE,before_request=before)
    common.write(arm/'result.json',result)
   finally:
    clean=common.cleanup(proc,tick);log.close();clean['port_free']=True
    try:common.free(PORT)
    except OSError:clean['port_free']=False
    common.write(arm/'terminal.json',clean);terminal['arms'].append({'threads':threads,'cpu_list':cpus,**clean})
  terminal['status']='completed_pilot';return terminal
 finally:
  terminal['elapsed_seconds']=time.monotonic()-started;common.write(root/'pilot-terminal.json',terminal)
def main():
 global STOP
 def stop(*_):
  global STOP;STOP=True
 for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,stop)
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--row',type=Path,required=True);p.add_argument('--admission',type=Path);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 try:print(json.dumps(run(a),indent=2));return 0
 except Exception as e:print(json.dumps({'status':'failed','error':f'{type(e).__name__}: {e}'}));return 1
if __name__=='__main__':raise SystemExit(main())
