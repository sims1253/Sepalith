"""Root-only fresh CUDA DEV rehearsal controller. Preparation never calls main."""
import argparse,datetime,hashlib,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CANDIDATE=HERE.parent/'native_evaluator'
sys.path.insert(0,str(CANDIDATE))
import native_identity as identity
import native_transport as transport
BUNDLE=Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453')
MODEL=Path(identity.PROFILE['selection']['q8_path'])
PYTHON='/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'
STOP=False
CLIENT_RSS_BYTES=1536*1024**2
SERVER_RSS_BYTES=8*1024**3
CONTROLLER_RSS_BYTES=1024**3
HOST_SECONDS=780
CHILD_SECONDS=720
def rss_ceiling(role):
    return {'client':CLIENT_RSS_BYTES,'server':SERVER_RSS_BYTES,'controller':CONTROLLER_RSS_BYTES}[role]
def interrupted(*args):
    global STOP
    STOP=True
def now():return datetime.datetime.now(datetime.timezone.utc)
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.new')
    with temporary.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
def check(start,limit):
    if STOP:raise RuntimeError('root stop')
    if time.monotonic()-start>limit:raise TimeoutError('bounded runtime expired')
    mem=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if int(mem.split()[1])<8*1024*1024:raise RuntimeError('MemAvailable below 8GiB')
def sha(path):return identity.sha(path)
def record(path,expected=None):
    # Admission names are profile aliases (.so.0), while /proc/maps names are
    # canonical targets (.so.0.20.0). Preserve both instead of conflating them.
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:raise ValueError('absolute artifact alias required')
    resolved=path.resolve(strict=True);before=identity.stat_identity(path);digest=sha(resolved)
    if identity.stat_identity(path)!=before or path.resolve(strict=True)!=resolved:raise ValueError('artifact changed during hash:'+str(path))
    if expected is not None and digest!=expected:raise ValueError('artifact hash mismatch:'+str(path))
    return {'path':str(path),'resolved_path':str(resolved),'sha256':digest,'stat':before}
def free_port(port):
    with socket.socket() as s:
        # A released listener can leave TCP TIME_WAIT sockets. SO_REUSEADDR
        # permits that state but still rejects an existing listening server.
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(('127.0.0.1',port))
def native_argv(port):
    return [str(BUNDLE/'llama-server'),'-m',str(MODEL),'--host','127.0.0.1','--port',str(port),'-t','6','-tb','6','--threads-http','2','--parallel','1','-c','4096','-b','256','-ub','256','-ngl','99','-lv','4']
def load_evidence(text):
    offloads=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',text)
    if not offloads or offloads[-1]!=('43','43'):raise ValueError('43/43 GPU offload not observed')
    if not re.search(r'CUDA0 model buffer size\s*=\s*[1-9][0-9.]* MiB',text):raise ValueError('CUDA0 model buffer not observed')
    return {'offloaded':43,'total_layers':43,'cuda_model_buffer_observed':True,'log_sha256':hashlib.sha256(text.encode()).hexdigest()}
def process_record(proc,role):
    _,tick=identity.proc_stat(Path(f'/proc/{proc.pid}/stat').read_text())
    return {'pid':proc.pid,'start_tick':tick,'role':role}
def _stop_owned(row):
    p=Path('/proc')/str(row['pid']);action='already absent'
    if p.exists():
        _,tick=identity.proc_stat((p/'stat').read_text())
        if tick!=row['start_tick']:return dict(row,action='PID reused; no signal',absent=False)
        if os.getpgid(row['pid'])!=row['pid']:return dict(row,action='not owned session leader; no signal',absent=False)
        try:os.killpg(row['pid'],signal.SIGTERM);action='TERM owned group'
        except ProcessLookupError:pass
        deadline=time.monotonic()+10
        while p.exists() and time.monotonic()<deadline:
            state,_=identity.proc_stat((p/'stat').read_text())
            if state=='Z':break
            time.sleep(.1)
        if p.exists():
            state,tick=identity.proc_stat((p/'stat').read_text())
            if state!='Z' and tick==row['start_tick']:
                try:os.killpg(row['pid'],signal.SIGKILL);action+=' then KILL'
                except ProcessLookupError:pass
    return dict(row,action=action,absent=not p.exists())
def stop_owned(row):
    try:return _stop_owned(row)
    except (FileNotFoundError,ProcessLookupError):return dict(row,action='owned process disappeared during cleanup',absent=True)
def validate_approval(a):
    if a.get('schema')!='dat08.native-dev-root-launch.v1' or a.get('status')!='root_admitted':raise ValueError('explicit root launch admission required')
    for flag in ('source_policy_reviewed','exclusive_gpu_owner','no_other_clients','native_loading_trust_accepted','dev_only'):
        if a.get(flag) is not True:raise ValueError('root authorization missing:'+flag)
    if not a.get('lease_id') or not a.get('run_id'):raise ValueError('root run/lease identity required')
    if a['profile_sha256']!=identity.digest(identity.PROFILE):raise ValueError('selected profile differs')
    if a.get('output_cap')!=identity.PROFILE['output']:raise ValueError('root output cap/profile mismatch')
    if not 1024<=a['port']<=65535:raise ValueError('loopback port invalid')
    if not 1<=a['client_seconds']<=600 or a['host_seconds']!=HOST_SECONDS:raise ValueError('reviewed guard limits differ')
    if a.get('client_rss_bytes')!=CLIENT_RSS_BYTES or a.get('server_rss_bytes')!=SERVER_RSS_BYTES:raise ValueError('reviewed RSS limits differ')
    created=datetime.datetime.fromisoformat(a['created_at']);expires=datetime.datetime.fromisoformat(a['expires_at'])
    if not created<=now()<expires or (expires-created).total_seconds()>7200:raise ValueError('root approval expired/future')
    for name,digest in a['input_files'].items():record(name,digest)
    if a['input_files'].get(str(HERE/'root_controller.py'))!=sha(HERE/'root_controller.py'):raise ValueError('controller not root pinned')
    for required in ('source-closure.prepared.json','native-mapped-code-allowlist.json','guarded_dev_client.py','client_source_policy.py','origin_snapshot.py','elf-loader-metadata.json'):
        if str(HERE/required) not in a['input_files']:raise ValueError('required input pin absent:'+required)
    return a
def child(a,run):
    start=time.monotonic();server=client=None;rows=[];failure=None
    def launch(argv,env,name):
        with (run/(name+'.log')).open('xb') as log:proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:row=process_record(proc,name)
        except BaseException:
            # The unreaped Popen child cannot have its PID reused. Clean it up
            # even if /proc observation fails before the ownership ledger write.
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
            raise
        rows.append(row);save(run/'owned-processes.json',rows)
        save(run/(name+'-launch.json'),dict(row,argv=argv,utc=now().isoformat()))
        return proc
    try:
        graph=json.loads((HERE/'source-closure.prepared.json').read_text())
        if graph['unresolved']:raise ValueError('concrete source closure unresolved')
        # Explicit root review plus all graph hashes and runtime guard, not status-only admission.
        for item in graph['files'].values():check(start,CHILD_SECONDS);record(item['path'],item['sha256'])
        graph['status']='root_admitted';graph['root_launch_approval_sha256']=identity.digest(a)
        save(run/'source-closure.json',graph);graph_sha=identity.digest(graph)
        for name,digest in a['data_files'].items():check(start,CHILD_SECONDS);record(name,digest)
        bundle=[record(BUNDLE/f['name'],f['sha256']) for f in identity.PROFILE['bundle']['files']]
        model=record(MODEL,identity.Q8);check(start,CHILD_SECONDS);free_port(a['port'])
        allow=json.loads((HERE/'native-mapped-code-allowlist.json').read_text())
        for item in allow['files']:record(item['path'],item['sha256'])
        loader=json.loads((HERE/'elf-loader-metadata.json').read_text())
        if loader['missing_system_cache_names']:raise ValueError('ELF loader dependency unresolved')
        for name,item in loader['loader_metadata_pins'].items():
            if item['exists']:record(name,item['sha256'])
            elif Path(name).exists():raise ValueError('previously absent loader override appeared:'+name)
        save(run/'prelaunch-artifacts.json',{'model':model,'bundle':bundle,'mapped_code_allowlist':allow,'source_sha256':graph_sha,'root_approval_sha256':identity.digest(a)})
        env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LD_','PYTHON'))}
        env.update(GGML_CUDA_GRAPH_OPT='0',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        server=launch(native_argv(a['port']),env,'server');endpoint='http://127.0.0.1:'+str(a['port'])
        ready_deadline=time.monotonic()+60
        while True:
            check(start,CHILD_SECONDS)
            if server.poll() is not None:raise RuntimeError('fresh server exited during load')
            try:_,props,_=transport._request_json(endpoint,'/props',None,1)
            except Exception:
                if time.monotonic()>ready_deadline:raise TimeoutError('60 second native load guard')
                time.sleep(.2);continue
            if props.get('model_path')!=str(MODEL) or props.get('total_slots')!=1 or props.get('default_generation_settings',{}).get('n_ctx')!=4096:raise ValueError('loaded native props differ')
            break
        save(run/'props-after-load.json',props)
        # Synthetic text only. One warmup settles lazy runtime mappings before admission.
        _,tokens,_=transport._request_json(endpoint,'/tokenize',transport.make_tokenize_payload('x <- 1\n'),3)
        payload=transport.make_completion_payload([0,*tokens['tokens']]);payload['n_predict']=8
        warmup=transport._stream_completion(endpoint,payload,time.monotonic()+10)
        if warmup.get('stream_complete') is not True:raise ValueError('synthetic warmup incomplete')
        save(run/'synthetic-warmup.json',{'request':payload,'response':warmup,'scope':'synthetic only; no quality evidence'})
        raw_maps=Path(f'/proc/{server.pid}/maps').read_text();(run/'native-maps-after-warmup.txt').write_text(raw_maps)
        paths=set()
        for line in raw_maps.splitlines():
            fields=line.split(None,5)
            if 'x' not in fields[1]:continue
            name=fields[5] if len(fields)==6 else ''
            if name in ('[vdso]','[vsyscall]'):continue
            if not name.startswith('/') or name.endswith(' (deleted)'):raise ValueError('unreviewed native executable mapping')
            paths.add(str(Path(name).resolve()))
        if paths!={f['path'] for f in allow['files']}:raise ValueError('fresh native mapped library set differs from root-pinned allowlist')
        mapped=[record(f['path'],f['sha256']) for f in allow['files']]
        evidence=load_evidence((run/'server.log').read_text());save(run/'load-evidence.json',evidence)
        lock=run/'exclusive-client.lock';lock.touch(exist_ok=False)
        admission={'schema':'dat08.native-process-admission.v1','status':'root_admitted','purpose':'dev-rehearsal','profile_sha256':identity.digest(identity.PROFILE),'output_cap':identity.PROFILE['output'],'run_id':a['run_id'],'lease_id':a['lease_id'],'pid':server.pid,'start_tick':rows[0]['start_tick'],'uid':os.getuid(),'created_at':now().isoformat(),'expires_at':(now()+datetime.timedelta(seconds=660)).isoformat(),'endpoint':endpoint,'fresh_process':True,'exclusive_owner':a['exclusive_gpu_owner'],'no_other_clients':a['no_other_clients'],'root_verified_model_before_launch':True,'cuda_backend_observed':evidence['cuda_model_buffer_observed'],'all_layers_offloaded':evidence['offloaded']==evidence['total_layers'],'load_evidence_sha256':sha(run/'load-evidence.json'),'exclusive_client_lock':str(lock),'model':model,'bundle_files':bundle,'mapped_code_files':mapped,'argv':native_argv(a['port']),'root_launch_approval_sha256':identity.digest(a)}
        owner=identity.NativeOwner(admission);owner.acquire()
        try:save(run/'native-controller-before.json',owner.verify(full_hash=True))
        finally:owner.close()
        save(run/'native-admission.json',admission)
        client_env=dict(env);client_env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        argv=[PYTHON,'-I','-S','-B',str(HERE/'guarded_dev_client.py'),'--native-admission',str(run/'native-admission.json'),'--source-manifest',str(run/'source-closure.json'),'--source-sha256',graph_sha,'--output',str(run/'dev-results'),'--deadline-seconds',str(a['client_seconds'])]
        client=launch(argv,client_env,'client');deadline=time.monotonic()+a['client_seconds']+30
        while client.poll() is None:
            check(start,CHILD_SECONDS)
            if time.monotonic()>deadline:raise TimeoutError('client deadline with 30 second exit allowance')
            time.sleep(.1)
        if client.returncode:raise RuntimeError('DEV client failed; retain partial evidence')
        owner.acquire()
        try:save(run/'native-controller-after.json',owner.verify(full_hash=True))
        finally:owner.close()
        result=json.loads((run/'dev-results/results.json').read_text())
        if result['status']!='complete' or result['denominators']['attempted_cases']!=75 or result['denominators']['edit_cases']!=43 or result['denominators']['strict_noop_cases']!=32:raise ValueError('DEV denominator incomplete')
    except BaseException as error:failure={'type':type(error).__name__,'message':str(error)}
    finally:
        cleanup=[stop_owned(row) for row in reversed(rows)]
        for proc in (client,server):
            if proc is not None:
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        for item in cleanup:item['absent']=not Path('/proc',str(item['pid'])).exists()
        try:free_port(a['port']);port_free=True
        except OSError:port_free=False
        save(run/'controller-terminal.json',{'failure':failure,'seconds':time.monotonic()-start,'cleanup':cleanup,'port_free':port_free,'final_access':False,'quality_promotion':False})
    return int(bool(failure) or not port_free or any(not x['absent'] for x in cleanup))
def main():
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--child',action='store_true');args=p.parse_args()
    a=validate_approval(json.loads(args.approval.read_text()));run=args.run_root
    if not run.is_absolute() or run.is_symlink() or not run.parent.is_dir():raise ValueError('absolute root-owned fresh output parent required')
    if args.child:return child(a,run)
    run.mkdir(mode=0o700,exist_ok=False);save(run/'root-approval.json',a)
    start=time.monotonic();failure=None
    with (run/'controller.log').open('xb') as log:proc=subprocess.Popen([sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'--approval',str(args.approval),'--run-root',str(run),'--child'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    controller=process_record(proc,'controller');save(run/'host-launch.json',controller)
    try:
        while proc.poll() is None:
            check(start,a['host_seconds'])
            children=json.loads((run/'owned-processes.json').read_text()) if (run/'owned-processes.json').exists() else []
            for row in [controller,*children]:
                try:
                    rss=int(next(x for x in Path(f"/proc/{row['pid']}/status").read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
                    ceiling=rss_ceiling(row['role'])
                    if rss>ceiling:raise RuntimeError(row['role']+' RSS guard')
                except (FileNotFoundError,StopIteration):pass
            time.sleep(.2)
    except BaseException as error:failure=repr(error)
    finally:
        stop_owned(controller)
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:pass
        children=json.loads((run/'owned-processes.json').read_text()) if (run/'owned-processes.json').exists() else []
        cleanup=[stop_owned(row) for row in reversed(children)]
        save(run/'host-terminal.json',{'failure':failure,'controller_returncode':proc.returncode,'seconds':time.monotonic()-start,'cleanup':cleanup,'final_access':False})
    return int(bool(failure) or proc.returncode!=0)
if __name__=='__main__':raise SystemExit(main())
