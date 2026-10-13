#!/usr/bin/env python3
"""Root-admitted serial notebook CPU DEV75 cap replay. Never executes generated R."""
import argparse,copy,datetime,hashlib,json,os,signal,socket,subprocess,sys,threading,time
from pathlib import Path
from urllib.request import urlopen
import campaign_protocol as protocol
import native_transport as transport
CAPS=(192,384,768); STOP=False

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def csha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(path,value):
 tmp=path.with_suffix(path.suffix+'.new')
 with tmp.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def statid(path):
 s=Path(path).stat();return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def free(port):
 with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
def health(port):
 try:
  with urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as r:return r.status,json.loads(r.read())
 except Exception:return None
def proc_tick(pid):
 t=Path(f'/proc/{pid}/stat').read_text();return t[t.rfind(')')+2:].split()[19]
def cleanup(proc,tick):
 if proc is None:return {'pid':None,'absent':True,'action':'not_started'}
 p=Path(f'/proc/{proc.pid}');action='already_absent'
 try:
  if p.exists() and proc_tick(proc.pid)==tick:
   os.killpg(proc.pid,signal.SIGTERM);action='TERM'
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);action='TERM_then_KILL';proc.wait(timeout=5)
 except ProcessLookupError:pass
 return {'pid':proc.pid,'start_tick':tick,'absent':not p.exists() or proc.poll() is not None,'action':action}
def validate_packet(packet,path):
 if packet.get('schema')!='sepalith.run06.e750-notebook-cpu-quality.v1' or packet.get('status')!='prepared_no_launch':raise ValueError('packet status/schema')
 if packet['caps']!=list(CAPS) or packet['context']!=4096 or packet['offline_case_deadline_seconds']!=120:raise ValueError('cap/context/deadline drift')
 if packet['production_reference_case_deadline_seconds']!=5:raise ValueError('production reference deadline changed')
 for key in ('runner','protocol','transport','profile','cap_contract','panel','panel_manifest','model_integrity'):
  x=packet['files'][key];p=Path(x['path']);
  if sha(p)!=x['sha256']:raise ValueError(key+' hash mismatch')
 for x in packet['runtime_files']:
  if sha(x['path'])!=x['sha256'] or Path(x['path']).stat().st_size!=x['bytes']:raise ValueError('runtime mismatch:'+x['path'])
 for name,x in packet['tokenizer'].items():
  if sha(x['path'])!=x['sha256'] or Path(x['path']).stat().st_size!=x['bytes']:raise ValueError('tokenizer mismatch:'+name)
 mi=json.loads(Path(packet['files']['model_integrity']['path']).read_text());m=packet['model']
 if mi['remote']['sha256']!=m['sha256'] or mi['remote']['bytes']!=m['bytes'] or mi['remote']['path']!=m['path']:raise ValueError('model integrity binding mismatch')
 st=Path(m['path']).stat();ri=mi['remote']
 if st.st_size!=m['bytes'] or (st.st_mode&0o777)!=0o400 or st.st_dev!=ri['device'] or st.st_ino!=ri['inode'] or int(st.st_mtime)!=ri['mtime_epoch'] or int(st.st_ctime)!=ri['ctime_epoch']:raise ValueError('model stat/mode mismatch')
 for name,x in packet['preserve'].items():
  if sha(x['path'])!=x['sha256']:raise ValueError('preserved profile mismatch:'+name)
 return {'packet_sha256':sha(path),'model_stat':statid(m['path']),'model_integrity_sha256':sha(packet['files']['model_integrity']['path'])}
def load_rows(packet):
 m=json.loads(Path(packet['files']['panel_manifest']['path']).read_text());rows=[json.loads(x) for x in Path(packet['files']['panel']['path']).read_text().splitlines()]
 if len(rows)!=75 or [r['id'] for r in rows]!=m['case_ids'] or sum(r['operation']=='no_op' for r in rows)!=32:raise ValueError('DEV75 closure mismatch')
 for r in rows:
  if hashlib.sha256(r['prompt_text'].encode()).hexdigest()!=r['prompt_sha256'] or csha(r['prompt_ids'])!=r['prompt_ids_sha256']:raise ValueError('prompt binding mismatch:'+r['id'])
 return rows,m
def summary(cap,rows,records,status):
 q=lambda r:r.get('quality',{});complete=[r for r in records if r.get('response_complete')]
 return {'schema':'sepalith.run06.e750-notebook-cap-quality-result.v1','status':status,'cap':cap,'offline_case_deadline_seconds':120,'production_latency_evidence':False,'expected_ids':[r['id'] for r in rows],'completed_ids':[r['id'] for r in records],'unattempted_ids':[r['id'] for r in rows[len(records):]],'denominators':{'expected':75,'attempted':len(records),'response_complete':len(complete),'edit':43,'noop':32},'counts':{'protocol_valid':sum(bool(q(r).get('protocol_valid')) for r in records),'edit_exact':sum(bool(q(r).get('edit_exact')) for r in records),'strict_noop_correct':sum(bool(q(r).get('strict_noop_correct')) for r in records),'false_suggestions':sum(bool(q(r).get('noop_false_positive')) for r in records),'cap_hit':sum(bool(r.get('cap',{}).get('hit')) for r in records),'canonical_eos':sum(r.get('eos',{}).get('status')=='canonical_eos' for r in records),'noncanonical_eog':sum(r.get('eos',{}).get('status')=='noncanonical_native_eog' for r in records),'transport_failed':sum(r.get('failure_class')=='transport' for r in records),'mechanical_failed':sum(r.get('failure_class')=='mechanical' for r in records)},'records':records}
def evaluate(endpoint,cap,rows,tokenizer,before,output):
 transport.COMPLETION_CAP=cap;records=[];write(output,summary(cap,rows,records,'partial'))
 for row in rows:
  if len(row['prompt_ids'])+cap>4096:raise ValueError('context budget mismatch:'+row['id'])
  case={'id':row['id'],'family':row['family'],'package_id':row['package_id'],'operation':row['operation'],'expected_noop':row['operation']=='no_op','expected_region':row['region_new'],'context':protocol.PromptContext.from_mapping(row['context']),'prompt':row['prompt_text'],'prompt_sha256':row['prompt_sha256'],'target_sha256':row['target_sha256'],'hf_prompt_ids':row['prompt_ids'],'hf_prompt_tokens_with_bos':len(row['prompt_ids']),'hf_prompt_ids_sha256':row['prompt_ids_sha256']}
  rec=transport._run_case(endpoint,case,protocol,tokenizer,120,before_request=before);records.append(rec);write(output,summary(cap,rows,records,'partial'))
  if not rec.get('response_complete'):break
 result=summary(cap,rows,records,'complete' if len(records)==75 else 'incomplete_preserved');write(output,result);return result
def run(args):
 packet_path=args.packet.resolve();packet=json.loads(packet_path.read_text());audit=validate_packet(packet,packet_path);rows,manifest=load_rows(packet)
 if args.preflight_only:return {'status':'preflight_pass_no_model_load','audit':audit,'cap_fit':manifest['cap_fit']}
 if args.admission is None:raise ValueError('root admission required')
 adm=json.loads(args.admission.read_text());required={'schema':'sepalith.run06.e750-notebook-cpu-root-admission.v1','status':'admitted','packet_sha256':audit['packet_sha256'],'model_sha256':packet['model']['sha256'],'model_integrity_sha256':audit['model_integrity_sha256'],'caps':list(CAPS),'maximum_threads':2,'offline_case_deadline_seconds':120,'maximum_seconds':28800,'host':'m0hawk@192.168.178.40','purpose':'offline_DEV75_CPU_quality_not_production_latency','source_reviewed':True}
 if any(adm.get(k)!=v for k,v in required.items()):raise ValueError('root admission mismatch:'+next(k for k,v in required.items() if adm.get(k)!=v))
 if not isinstance(adm.get('run_id'),str) or not adm['run_id'] or adm['run_id']!=args.run_root.name:raise ValueError('root admission run ID mismatch')
 now=datetime.datetime.now(datetime.timezone.utc);created=datetime.datetime.fromisoformat(adm['created_at']);expires=datetime.datetime.fromisoformat(adm['expires_at'])
 if created.tzinfo is None or expires.tzinfo is None:raise ValueError('root admission timestamps must include timezone')
 if not created<=now<expires or (expires-created).total_seconds()>28800:raise ValueError('root admission expired/future')
 root=args.run_root.resolve();prefix=Path(packet['run_root_prefix'])
 if root.exists() or root.parent!=prefix:raise ValueError('fresh direct run root required')
 root.mkdir(mode=0o700);tokenizer,tok_audit=transport._load_tokenizer(Path(packet['tokenizer_dir']));write(root/'tokenizer-audit.json',tok_audit)
 model_stat=audit['model_stat'];server=Path(packet['runtime_server']);started=time.monotonic();term={'status':'failed','arms':[]}
 try:
  for cap in CAPS:
   if STOP:raise RuntimeError('stop requested')
   arm=root/f'cap-{cap}';arm.mkdir();free(packet['port']);proc=None;tick=None;log=(arm/'server.log').open('xb')
   argv=['/usr/bin/taskset','--cpu-list','0,2',str(server),'-m',packet['model']['path'],'--host','127.0.0.1','--port',str(packet['port']),'-t','2','-tb','2','--threads-http','1','--parallel','1','-c','4096','-b','256','-ub','256','-ngl','0','-lv','4','--metrics']
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
   try:
    proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);tick=proc_tick(proc.pid);write(arm/'launch.json',{'pid':proc.pid,'start_tick':tick,'argv':argv,'cap':cap,'packet_sha256':audit['packet_sha256']})
    until=time.monotonic()+180
    while time.monotonic()<until and proc.poll() is None:
     h=health(packet['port'])
     if h and h[0]==200:break
     time.sleep(.25)
    else:raise RuntimeError('CPU server failed health/load')
    def before(_deadline):
     if STOP or proc.poll() is not None or proc_tick(proc.pid)!=tick:raise RuntimeError('owned server identity changed')
     if statid(packet['model']['path'])!=model_stat:raise RuntimeError('model stat changed')
    result=evaluate(f"http://127.0.0.1:{packet['port']}",cap,rows,tokenizer,before,arm/'results.json')
    if result['status']!='complete':raise RuntimeError('arm incomplete; preserve evidence and stop suite')
   finally:
    clean=cleanup(proc,tick);log.close();clean['port_free']=True
    try:free(packet['port'])
    except OSError:clean['port_free']=False
    write(arm/'terminal.json',clean);term['arms'].append({'cap':cap,**clean})
  for name,x in packet['preserve'].items():
   if sha(x['path'])!=x['sha256']:raise RuntimeError('preserved profile changed:'+name)
  term['status']='completed'
  return term
 finally:
  term['elapsed_seconds']=time.monotonic()-started;write(root/'suite-terminal.json',term)
def main():
 global STOP
 def stop(*_):STOP=True
 for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,stop)
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--admission',type=Path);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 try:print(json.dumps(run(a),indent=2));return 0
 except Exception as e:print(json.dumps({'status':'failed','error':f'{type(e).__name__}: {e}'}));return 1
if __name__=='__main__':raise SystemExit(main())
