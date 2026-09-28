"""Root-only fresh CUDA FINAL-purpose controller. Preparation never calls main."""
import argparse,datetime,fcntl,hashlib,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CANDIDATE=HERE.parent/'r2-final-evaluator-v1'
sys.path.insert(0,str(CANDIDATE))
import native_identity as identity
import native_transport as transport
import final_binding as binding
import final_row_gate as row_gate
ENTRY=HERE.parent/'r2-final-entrypoint-v1'
CONSTRUCTOR_PYTHON='/home/m0hawk/Documents/Sepalith/.venv/bin/python'
BUNDLE=Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453')
MODEL=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf')
CUDA_LOCK=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock')
PYTHON='/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'
STOP=False
CONTROLLER_GRAPH=None
CONTROLLER_EPOCH=None
CLIENT_RSS_BYTES=1536*1024**2
SERVER_RSS_BYTES=8*1024**3
CONTROLLER_RSS_BYTES=1024**3
HOST_SECONDS=4860
CHILD_SECONDS=4800
def rss_ceiling(role):
    return {'client':CLIENT_RSS_BYTES,'constructor':CLIENT_RSS_BYTES,'server':SERVER_RSS_BYTES,'controller':CONTROLLER_RSS_BYTES}[role]
def interrupted(*args):
    global STOP
    STOP=True
def now():return datetime.datetime.now(datetime.timezone.utc)
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.new')
    with temporary.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
def check(start,limit):
    if CONTROLLER_EPOCH is not None and {k:identity.stat_identity(v['path']) for k,v in CONTROLLER_GRAPH['files'].items()}!=CONTROLLER_EPOCH:raise ValueError('controller source changed during job')
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
def acquire_cuda_lock():
    if CUDA_LOCK.is_symlink() or not CUDA_LOCK.is_file():raise ValueError('root must initialize the CUDA lock file')
    fd=os.open(CUDA_LOCK,os.O_RDWR|os.O_NOFOLLOW)
    try:
        validate_cuda_lock(fd)
        return fd
    except BaseException:os.close(fd);raise

def validate_cuda_lock(fd):
    actual=os.fstat(fd);expected=CUDA_LOCK.stat()
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino) or CUDA_LOCK.is_symlink():raise ValueError('inherited CUDA lock identity differs')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)

def controller_source_guard(graph,full_hash=False):
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:raise ValueError('fresh isolated no-site no-bytecode controller required')
    allowed={str(Path(v['path']).resolve()) for v in graph['files'].values()}
    for path in graph['must_remain_absent']:
        if Path(path).exists():raise ValueError('unexpected controller bytecode appeared:'+path)
    for name,module in tuple(sys.modules.items()):
        origin=getattr(module,'__file__',None)
        if origin and Path(origin).is_absolute() and Path(origin).is_file() and str(Path(origin).resolve()) not in allowed:raise ValueError('unexpected controller module origin:'+name)
    mapped=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        row=line.split(None,5)
        if 'x' not in row[1]:continue
        name=row[5] if len(row)==6 else ''
        if name in ('[vdso]','[vsyscall]'):continue
        if not name.startswith('/') or name.endswith(' (deleted)'):raise ValueError('unexpected controller executable mapping')
        mapped.add(str(Path(name).resolve()))
    if mapped!=set(graph['executable_maps']):raise ValueError('controller executable origins changed')
    if full_hash:
        for item in graph['files'].values():record(item['path'],item['sha256'])

def verify_process_bundle_files(graph,expected):
    # This is prelaunch byte verification of ANOTHER process's admitted bundle.
    # Its unchanged entrypoint verifies full graph structure and its own runtime
    # origins. Calling that origin verifier here would compare child __main__
    # against this controller and fail for the wrong process.
    if identity.digest(graph)!=expected or graph.get('schema')!='dat08.source-closure.v1' or graph.get('status')!='root_admitted' or graph.get('unresolved')!=[]:raise ValueError('separate process graph not root admitted')
    if not graph.get('files') or not graph.get('roots'):raise ValueError('separate process graph missing')
    for value in graph['files'].values():record(value['path'],value['sha256'])
    return graph

def release(a):
    # Root launch approval and freeze receipt are control metadata. Read no other
    # path until BOTH actual Monday UTC and explicit root weight/harness freeze.
    freeze=json.loads(Path(a['freeze_path']).read_text())
    row_gate._validate_freeze(freeze,expected_weights_sha256=identity.Q8,
        expected_harness_sha256=a['harness_sha256'],now=now())
    if identity.digest(freeze)!=a['prelaunch_freeze_sha256']:raise ValueError('root prelaunch freeze digest changed')
    if freeze.get('model_binding_kind')!='fresh-pinned-native-process.v1' or freeze.get('native_profile_sha256')!=identity.digest(identity.PROFILE):raise ValueError('root native loading/profile freeze missing')
    for field in ('source_closure_sha256','constructor_source_closure_sha256'):
        if freeze.get(field)!=a[field]:raise ValueError('root source graph freeze differs:'+field)
    return freeze

def validate_approval(a):
    if a.get('schema')!='dat08.native-final-root-launch.v1' or a.get('status')!='root_admitted':raise ValueError('explicit FINAL root launch admission required')
    for flag in ('source_policy_reviewed','exclusive_gpu_owner','no_other_clients','native_loading_trust_accepted','final_only','authorize_derived_runtime_freeze'):
        if a.get(flag) is not True:raise ValueError('root authorization missing:'+flag)
    if not a.get('lease_id') or not a.get('run_id'):raise ValueError('root run/lease identity required')
    if a.get('cuda_lock_path')!=str(CUDA_LOCK):raise ValueError('global CUDA lock path differs')
    if a['profile_sha256']!=identity.digest(identity.PROFILE):raise ValueError('selected profile differs')
    if not 1024<=a['port']<=65535:raise ValueError('loopback port invalid')
    if not 1<=a['client_seconds']<=3600 or a['host_seconds']!=HOST_SECONDS or a['constructor_seconds']!=600 or a['root_review_seconds']!=300:raise ValueError('reviewed guard limits differ')
    if a.get('client_rss_bytes')!=CLIENT_RSS_BYTES or a.get('server_rss_bytes')!=SERVER_RSS_BYTES:raise ValueError('reviewed RSS limits differ')
    created=datetime.datetime.fromisoformat(a['created_at']);expires=datetime.datetime.fromisoformat(a['expires_at'])
    if not created<=now()<expires or (expires-created).total_seconds()>7200:raise ValueError('root approval expired/future')
    release(a) # BEFORE input hashing, final path stat/read, output creation, or launch.
    for name,digest in a['input_files'].items():record(name,digest)
    if a['input_files'].get(str(HERE/'root_controller.py'))!=sha(HERE/'root_controller.py'):raise ValueError('controller not root pinned')
    for required in (HERE/'native-mapped-code-allowlist.json',HERE/'elf-loader-metadata.json',HERE/'controller-source-closure.prepared.json',ENTRY/'construct_final.py',ENTRY/'evaluate_final.py',ENTRY/'entry_gate.py',ENTRY/'client_source_policy.py',ENTRY/'origin_snapshot.py',Path(a['source_manifest']),Path(a['constructor_source_manifest'])):
        if str(required) not in a['input_files']:raise ValueError('required input pin absent:'+str(required))
    for field,path in (('source_closure_sha256',a['source_manifest']),('constructor_source_closure_sha256',a['constructor_source_manifest'])):
        graph=json.loads(Path(path).read_text());verify_process_bundle_files(graph,a[field])
    closure=json.loads((HERE/'controller-source-closure.prepared.json').read_text())
    if closure['unresolved']:raise ValueError('controller source unresolved')
    controller_source_guard(closure,full_hash=True)
    global CONTROLLER_GRAPH,CONTROLLER_EPOCH
    CONTROLLER_GRAPH=closure;CONTROLLER_EPOCH={k:identity.stat_identity(v['path']) for k,v in closure['files'].items()}
    return a

def derive_run_freeze(a,admission):
    freeze=release(a)
    if a.get('authorize_derived_runtime_freeze') is not True:raise ValueError('derived runtime freeze not authorized')
    if admission.get('purpose')!='final-evaluation' or admission.get('run_id')!=a['run_id'] or admission.get('lease_id')!=a['lease_id']:raise ValueError('derived freeze run identity differs')
    result=dict(freeze)
    result['native_process_admission_sha256']=identity.digest(admission)
    result['prelaunch_freeze_sha256']=identity.digest(freeze)
    result['root_launch_approval_sha256']=identity.digest(a)
    return result

def review_manifest(a,run,freeze,admission):
    release(a)
    completed=json.loads(Path(a['constructor_output'],'constructor-complete.json').read_text())
    if completed.get('status')!='constructed_and_validated':raise ValueError('constructor incomplete')
    manifest=json.loads(Path(a['rows_manifest']).read_text())
    if completed['rows_manifest_sha256']!=manifest['manifest_sha256']:raise ValueError('constructor manifest digest differs')
    row_gate._validate_manifest(manifest,expected_manifest_sha256=manifest['manifest_sha256'],freeze_sha256=identity.digest(freeze))
    return manifest

def validate_review(review,a,freeze,admission,manifest):
    release(a)
    if review.get('schema')!='dat08.native-final-case-review.v1' or review.get('status')!='root_admitted' or review.get('coverage_reviewed') is not True:raise ValueError('explicit completed-case root review required')
    expected={'run_id':a['run_id'],'lease_id':a['lease_id'],'freeze_sha256':identity.digest(freeze),'native_admission_sha256':identity.digest(admission),'manifest_sha256':manifest['manifest_sha256'],'rows_sha256':manifest['rows_artifact_sha256'],'expected_cases':manifest['row_count']}
    for key,value in expected.items():
        if review.get(key)!=value:raise ValueError('root case review differs:'+key)
    created=datetime.datetime.fromisoformat(review['created_at']);expires=datetime.datetime.fromisoformat(review['expires_at'])
    if not created<=now()<expires or expires>datetime.datetime.fromisoformat(admission['expires_at']):raise ValueError('root case review expiry differs')
    return expected

def child(a,run):
    start=time.monotonic();server=client=constructor=None;rows=[];failure=None;final_access_attempted=False
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
        release(a)
        graph=json.loads(Path(a['source_manifest']).read_text())
        verify_process_bundle_files(graph,a['source_closure_sha256'])
        constructor_graph=json.loads(Path(a['constructor_source_manifest']).read_text())
        verify_process_bundle_files(constructor_graph,a['constructor_source_closure_sha256'])
        graph_sha=identity.digest(graph)
        for name,digest in a['constructor_input_files'].items():check(start,CHILD_SECONDS);record(name,digest)
        for key in ('requests','case_specs','source_authorization','train_identities','dev_identities'):
            if a[key] not in a['constructor_input_files']:raise ValueError('constructor input not root pinned:'+key)
        output=Path(a['constructor_output'])
        if not output.is_absolute() or output.exists() or not output.parent.is_dir():raise ValueError('fresh absolute constructor output required')
        if Path(a['rows_manifest'])!=output/'rows-manifest.json' or Path(a['rows'])!=output/'canonical-rows.json':raise ValueError('root final paths differ from constructor outputs')
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
        admission={'schema':'dat08.native-process-admission.v1','status':'root_admitted','purpose':'final-evaluation','profile_sha256':identity.digest(identity.PROFILE),'run_id':a['run_id'],'lease_id':a['lease_id'],'pid':server.pid,'start_tick':rows[0]['start_tick'],'uid':os.getuid(),'created_at':now().isoformat(),'expires_at':min(now()+datetime.timedelta(seconds=4740),datetime.datetime.fromisoformat(a['expires_at'])).isoformat(),'endpoint':endpoint,'fresh_process':True,'exclusive_owner':a['exclusive_gpu_owner'],'no_other_clients':a['no_other_clients'],'root_verified_model_before_launch':True,'cuda_backend_observed':evidence['cuda_model_buffer_observed'],'all_layers_offloaded':evidence['offloaded']==evidence['total_layers'],'load_evidence_sha256':sha(run/'load-evidence.json'),'exclusive_client_lock':str(lock),'model':model,'bundle_files':bundle,'mapped_code_files':mapped,'argv':native_argv(a['port']),'root_launch_approval_sha256':identity.digest(a)}
        owner=identity.NativeOwner(admission);owner.acquire()
        try:save(run/'native-controller-before.json',owner.verify(full_hash=True))
        finally:owner.close()
        save(run/'native-admission.json',admission)
        client_env=dict(env);client_env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        freeze=derive_run_freeze(a,admission);save(run/'final-freeze.json',freeze)
        constructor_argv=[CONSTRUCTOR_PYTHON,'-I','-S','-B',str(ENTRY/'construct_final.py'),'--freeze',str(run/'final-freeze.json'),'--source-manifest',a['constructor_source_manifest'],'--source-sha256',a['constructor_source_closure_sha256'],'--harness-sha256',a['harness_sha256'],'--output',a['constructor_output']]
        for key in ('requests','case_specs','source_authorization','train_identities','dev_identities'):
            constructor_argv.extend(['--'+key.replace('_','-'),a[key]])
        for path in a['allowed_roots']:constructor_argv.extend(['--allowed-root',path])
        final_access_attempted=True
        constructor=launch(constructor_argv,client_env,'constructor');constructor_deadline=time.monotonic()+a['constructor_seconds']
        while constructor.poll() is None:
            check(start,CHILD_SECONDS);release(a)
            if time.monotonic()>constructor_deadline:raise TimeoutError('constructor deadline')
            time.sleep(.2)
        if constructor.returncode:raise RuntimeError('gated constructor failed; no final client')
        manifest=review_manifest(a,run,freeze,admission)
        save(run/'awaiting-root-case-review.json',{'schema':'dat08.native-final-case-review.v1','status':'ROOT_REVIEW_REQUIRED','coverage_reviewed':False,'run_id':a['run_id'],'lease_id':a['lease_id'],'freeze_sha256':identity.digest(freeze),'native_admission_sha256':identity.digest(admission),'manifest_sha256':manifest['manifest_sha256'],'rows_sha256':manifest['rows_artifact_sha256'],'expected_cases':manifest['row_count'],'created_at':'ROOT_UTC','expires_at':'ROOT_UTC'})
        review_path=run/'final-evaluation-approval.json';review_deadline=time.monotonic()+a['root_review_seconds']
        while not review_path.exists():
            check(start,CHILD_SECONDS);release(a)
            if time.monotonic()>review_deadline:raise TimeoutError('300 second bounded root case review wait')
            if server.poll() is not None:raise RuntimeError('fresh server exited during root review')
            time.sleep(.2)
        review=json.loads(row_gate._safe_metadata_path(review_path).read_text())
        validate_review(review,a,freeze,admission,manifest)
        record(a['rows'],manifest['rows_artifact_sha256'])
        argv=[PYTHON,'-I','-S','-B',str(ENTRY/'evaluate_final.py'),'--freeze',str(run/'final-freeze.json'),'--native-admission',str(run/'native-admission.json'),'--source-manifest',a['source_manifest'],'--source-sha256',graph_sha,'--rows-manifest',a['rows_manifest'],'--manifest-sha256',manifest['manifest_sha256'],'--rows',a['rows'],'--harness-sha256',a['harness_sha256'],'--output',str(run/'final-results'),'--deadline-seconds',str(a['client_seconds'])]
        client=launch(argv,client_env,'client');deadline=time.monotonic()+a['client_seconds']+30
        while client.poll() is None:
            check(start,CHILD_SECONDS)
            if time.monotonic()>deadline:raise TimeoutError('client deadline with 30 second exit allowance')
            time.sleep(.1)
        if client.returncode:raise RuntimeError('FINAL client failed; retain partial evidence')
        owner.acquire()
        try:save(run/'native-controller-after.json',owner.verify(full_hash=True))
        finally:owner.close()
        result=json.loads((run/'final-results/results.json').read_text())
        if result['status']!='complete' or result['denominators']['attempted_cases']!=manifest['row_count'] or result['completed_case_ids']!=manifest['case_ids']:raise ValueError('FINAL denominator incomplete')
        release(a);verify_process_bundle_files(graph,graph_sha);verify_process_bundle_files(constructor_graph,a['constructor_source_closure_sha256'])
        controller_source_guard(CONTROLLER_GRAPH,full_hash=True)
    except BaseException as error:failure={'type':type(error).__name__,'message':str(error)}
    finally:
        cleanup=[stop_owned(row) for row in reversed(rows)]
        for proc in (client,constructor,server):
            if proc is not None:
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        for item in cleanup:item['absent']=not Path('/proc',str(item['pid'])).exists()
        try:free_port(a['port']);port_free=True
        except OSError:port_free=False
        save(run/'controller-terminal.json',{'failure':failure,'seconds':time.monotonic()-start,'cleanup':cleanup,'port_free':port_free,'final_access_attempted':final_access_attempted,'quality_promotion':False})
    return int(bool(failure) or not port_free or any(not x['absent'] for x in cleanup))
def main():
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--child',action='store_true');p.add_argument('--cuda-lock-fd',type=int);args=p.parse_args()
    a=validate_approval(json.loads(args.approval.read_text()));run=args.run_root
    if not run.is_absolute() or run.is_symlink() or not run.parent.is_dir():raise ValueError('absolute root-owned fresh output parent required')
    if args.child:
        if args.cuda_lock_fd is None:raise ValueError('inherited global CUDA lock required')
        validate_cuda_lock(args.cuda_lock_fd)
        try:return child(a,run)
        finally:os.close(args.cuda_lock_fd)
    if args.cuda_lock_fd is not None:raise ValueError('lock FD is child-only')
    lock_fd=acquire_cuda_lock()
    try:return host(a,run,args,lock_fd)
    finally:os.close(lock_fd)

def host(a,run,args,lock_fd):
    run.mkdir(mode=0o700,exist_ok=False);save(run/'root-approval.json',a)
    start=time.monotonic();failure=None
    with (run/'controller.log').open('xb') as log:proc=subprocess.Popen([sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'--approval',str(args.approval),'--run-root',str(run),'--child','--cuda-lock-fd',str(lock_fd)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
    try:controller=process_record(proc,'controller')
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        raise
    save(run/'host-launch.json',controller)
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
        save(run/'host-terminal.json',{'failure':failure,'controller_returncode':proc.returncode,'seconds':time.monotonic()-start,'cleanup':cleanup,'final_access_scope':'root-gated constructor and evaluator'})
    return int(bool(failure) or proc.returncode!=0)
if __name__=='__main__':raise SystemExit(main())
