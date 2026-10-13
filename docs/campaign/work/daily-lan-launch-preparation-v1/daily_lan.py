#!/usr/bin/env python3
"""Root-reviewed LAN serving session. Linux only; no editor or credential management."""
from pathlib import Path
import argparse, contextlib, ctypes, datetime as dt, fcntl, hashlib, json, os, shlex, signal, socket, subprocess, resource, sys, time, urllib.request, uuid
HERE = Path(__file__).resolve().parent
CAMPAIGN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
NATIVE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
LOCK = NATIVE/'resource-locks/cuda0.lock'
LEDGER = CAMPAIGN/'RESOURCE-LEASES.md'
MODEL = NATIVE/'models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf'
BIN = Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server')
HOST = 'm0hawk@192.168.178.40'
REMOTE = '/home/m0hawk/.local/share/sepalith-daily-lan'
VSIX_SHA = 'b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1'
SESSION_SECONDS = 8*60*60

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write_json(path,value):
    with open(path,'x', opener=lambda p,f:os.open(p,f,0o600)) as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def identity(pid):
    base=Path('/proc')/str(pid)
    try:
        fields=(base/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X','x'): return None
        return {'pid':pid,'startTick':fields[19],'uid':base.stat().st_uid}
    except FileNotFoundError:return None

def owned_pidfd(expected):
    if identity(expected['pid']) != expected: raise RuntimeError('owned_process_changed_or_exited')
    fd=os.pidfd_open(expected['pid'])
    if identity(expected['pid']) != expected:
        os.close(fd);raise RuntimeError('owned_process_changed_or_exited')
    return fd

def signal_owned(expected,sig):
    try:fd=owned_pidfd(expected)
    except (ProcessLookupError,RuntimeError):return False
    try: signal.pidfd_send_signal(fd,sig);return True
    except ProcessLookupError:return False
    finally:os.close(fd)

@contextlib.contextmanager
def exclusive_lock(path=LOCK):
    # Root creates this stable inode. Never unlink, truncate, replace, or recreate it.
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        if os.fstat(fd).st_uid != os.getuid(): raise RuntimeError('CUDA_lock_owner_mismatch')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('CUDA_lease_busy')
        yield fd
    finally:os.close(fd)

def ledger_free(text):
    rows=[x for x in text.splitlines() if x.startswith('| RTX 5090 |')]
    if len(rows)!=1 or rows[0].split('|')[2].strip()!='None':raise RuntimeError('CUDA_lease_busy_or_unknown')

def check_manifest():
    pins=json.loads((HERE/'source-manifest.json').read_text())
    for name,digest in pins.items():
        if sha(HERE/name)!=digest:raise RuntimeError('daily_source_identity_mismatch:'+name)
    if sha(HERE/'candidate.vsix')!=VSIX_SHA:raise RuntimeError('selected_VSIX_mismatch')

def check_admission(admission,state):
    value=json.loads(Path(admission).read_text())
    expected={'schema':1,'status':'admitted','purpose':'daily-lan-deployment','sourceManifestSha256':sha(HERE/'source-manifest.json'),'cudaLockPath':str(LOCK)}
    if any(value.get(k)!=v for k,v in expected.items()):raise RuntimeError('root_deployment_admission_required')
    ledger_free(LEDGER.read_text())
    marker=state/'deployment-verified.json'
    if marker.exists():
        if json.loads(marker.read_text())['admissionSha256']!=sha(admission):raise RuntimeError('deployment_admission_changed')
    else:
        if value.get('initialLeaseLedgerSha256')!=sha(LEDGER):raise RuntimeError('initial_CUDA_lease_changed')
        write_json(marker,{'admissionSha256':sha(admission),'sourceManifestSha256':expected['sourceManifestSha256']})
    return value

def native_argv():
    return [str(BIN),'-m',str(MODEL),'--alias','sepalith','--temp','0','--host','127.0.0.1','--port','18403','-c','4096','-b','256','-ub','256','--parallel','1','-t','6','-tb','6','--threads-http','2','-ngl','99','-lv','4']

def ssh_argv(*tail):
    return ['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',*tail]

def parent_guard(parent):
    # Children retain CUDA flock and die if supervisor disappears. No process-group kill.
    resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024*1024,64*1024*1024))
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent:os._exit(125)

class Owned:
    def __init__(self,run,lockfd):self.run=run;self.lockfd=lockfd;self.children=[];self.completed=[];self.cancelled=lambda:False
    def launch(self,role,argv,env=None,stdin=None):
        parent=os.getpid()
        with (self.run/(role+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(self.lockfd,),preexec_fn=lambda:parent_guard(parent))
        ident=identity(child.pid)
        if ident is None:child.wait();raise RuntimeError(role+'_exited_during_start')
        self.children.append((role,child,ident));write_json(self.run/(role+'-identity.json'),ident)
        if stdin is not None:child.stdin.write(stdin);child.stdin.close()
        return child
    def check(self):
        if self.cancelled():raise RuntimeError('operator_stop')
        for role,child,ident in self.children:
            if child.poll() is not None or identity(child.pid)!=ident:raise RuntimeError(role+'_exited_or_changed')
    def complete(self,child):
        row=next(row for row in self.children if row[1] is child)
        if child.poll() is None:raise RuntimeError('child_not_complete')
        self.completed.append({'role':row[0],'identity':row[2],'exitCode':child.returncode,'survived':identity(child.pid)==row[2]})
        self.children.remove(row)
    def cleanup(self):
        result=list(self.completed)
        for role,child,ident in reversed(self.children):
            signal_owned(ident,signal.SIGTERM)
            try:child.wait(timeout=6)
            except subprocess.TimeoutExpired:
                signal_owned(ident,signal.SIGKILL)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:pass
            result.append({'role':role,'identity':ident,'exitCode':child.returncode,'survived':identity(child.pid)==ident})
        return result

def fetch_json(url,expected_instance=None):
    with urllib.request.urlopen(url,timeout=2) as r:
        if expected_instance and r.headers.get('X-Sepalith-Instance-Id')!=expected_instance:raise RuntimeError('instance_header_mismatch')
        raw=r.read(1024*1024+1)
        if len(raw)>1024*1024:raise RuntimeError('readiness_response_too_large')
        return json.loads(raw)

def ready(owner,url,predicate,instance=None):
    until=time.monotonic()+45
    while time.monotonic()<until:
        owner.check()
        try:value=fetch_json(url,instance)
        except (OSError,ValueError):time.sleep(.2);continue
        if not predicate(value):raise RuntimeError('runtime_readiness_identity_mismatch')
        return value
    raise RuntimeError('runtime_readiness_timeout')

def run_session(state,admission):
    state=Path(state).resolve();state.mkdir(mode=0o700,parents=True,exist_ok=True)
    check_manifest()
    with exclusive_lock() as lockfd:
        check_admission(admission,state)
        for port in (18403,18423):
            with socket.socket() as s:s.bind(('127.0.0.1',port))
        session=state/str(uuid.uuid4());session.mkdir(mode=0o700)
        instance=session.name;binding={'schema':1,'endpoint':'http://127.0.0.1:18403','instanceId':instance,'manifest':json.loads((HERE/'primary-manifest.json').read_text()),'backend':'cuda'}
        write_json(session/'binding.json',binding);write_json(session/'supervisor.json',identity(os.getpid()))
        profile=binding['manifest'];bundle=profile['bundles'][0]
        # Full artifact preflight is a root-run operation; preparation tests never call it.
        if MODEL.stat().st_size!=profile['model']['bytes'] or sha(MODEL)!=profile['model']['sha256']:raise RuntimeError('model_identity_mismatch')
        for item in bundle['files']:
            file=BIN.parent/item['name']
            if file.stat().st_size!=item['bytes'] or sha(file)!=item['sha256']:raise RuntimeError('native_artifact_mismatch:'+item['name'])
        stopping=False
        def stop(*_):
            nonlocal stopping
            stopping=True
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
        owner=Owned(session,lockfd);owner.cancelled=lambda:stopping;failure=None;start=time.monotonic()
        env={k:v for k,v in os.environ.items() if not k.startswith('GGML_')}
        env.update(CUDA_VISIBLE_DEVICES='0',GGML_CUDA_GRAPH_OPT='0',LD_LIBRARY_PATH=str(BIN.parent),OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        try:
            native=owner.launch('native',native_argv(),env)
            props=ready(owner,'http://127.0.0.1:18403/props',lambda x:x.get('model_path')==str(MODEL) and x.get('default_generation_settings',{}).get('n_ctx')==4096 and x.get('total_slots')==1)
            if 'offloaded 43/43 layers to GPU' not in (session/'native.log').read_text():raise RuntimeError('full_GPU_offload_unproven')
            ident={**identity(native.pid),'modelPath':str(MODEL)}
            write_json(session/'gateway-config.json',{'schema':1,'binding':binding,'native':ident,'admission':{'leaseId':'daily-'+instance,'preflightReceiptSha256':sha(admission)}})
            owner.launch('gateway',['node','--experimental-strip-types',str(HERE/'gateway/gateway.mjs'),'--config',str(session/'gateway-config.json'),'--receipt',str(session/'gateway-events.jsonl'),'--daily-max-run-ms',str(SESSION_SECONDS*1000)])
            expected={'schema':1,'instanceId':instance,'modelSha256':profile['model']['sha256'],'serverSha256':bundle['launchProfile']['serverSha256'],'backend':'cuda','contextSize':4096,'maxOutputTokens':192,'renderer':profile['modelProfile']['renderer'],'cudaGraphOptimization':0}
            ready(owner,'http://127.0.0.1:18423/sepalith/runtime',lambda x:x==expected,instance)
            # Receiver is separately root-installed/pinned; credentials stay in normal SSH handling.
            command=shlex.join(['python3',REMOTE+'/package/notebook_setup.py','bind','--self-sha256',sha(HERE/'notebook_setup.py')])
            upload=owner.launch('binding-upload',ssh_argv(HOST,command),stdin=json.dumps(binding).encode())
            upload.wait(timeout=15)
            if upload.returncode!=0:raise RuntimeError('notebook_binding_upload_failed')
            owner.complete(upload)
            tunnel=owner.launch('ssh',ssh_argv('-N','-R','127.0.0.1:18403:127.0.0.1:18423',HOST))
            time.sleep(1);owner.check()
            command=shlex.join(['python3',REMOTE+'/package/notebook_setup.py','probe','--self-sha256',sha(HERE/'notebook_setup.py')])
            probe=owner.launch('notebook-probe',ssh_argv(HOST,command))
            probe.wait(timeout=15)
            if probe.returncode!=0:raise RuntimeError('notebook_forward_identity_probe_failed')
            owner.complete(probe)
            write_json(session/'ready.json',{'instanceId':instance,'status':'desktop_and_notebook_forward_verified_extension_handshake_pending','sessionSeconds':SESSION_SECONDS,'gateway':expected})
            print(json.dumps({'run':str(session),'instanceId':instance,'next':'Notebook: Sepalith Stop server, then Start server; require verified Remote CUDA.'}),flush=True)
            while not stopping and not (session/'stop').exists() and time.monotonic()-start<SESSION_SECONDS:
                owner.check();ledger_free(LEDGER.read_text())
                mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                if int(mem['MemAvailable'].split()[0])<8*1024*1024:raise RuntimeError('host_memory_floor_8GiB')
                time.sleep(.5)
        except Exception as exc:failure=type(exc).__name__+': '+str(exc)
        finally:
            cleanup=owner.cleanup();write_json(session/'terminal.json',{'failure':failure,'seconds':time.monotonic()-start,'cleanup':cleanup,'resourceReleaseProven':not any(x['survived'] for x in cleanup)})
        return 1 if failure or any(x['survived'] for x in cleanup) else 0

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('start');p.add_argument('--state-dir',required=True);p.add_argument('--admission',required=True)
    for name in ('stop','status'):
        p=sub.add_parser(name);p.add_argument('run',type=Path)
    args=parser.parse_args()
    if args.command=='start':return run_session(args.state_dir,args.admission)
    ident=json.loads((args.run/'supervisor.json').read_text())
    if args.command=='stop':
        if not signal_owned(ident,signal.SIGTERM):
            if not (args.run/'terminal.json').exists():raise RuntimeError('supervisor_gone_release_requires_root_review')
    print(json.dumps({'supervisorAlive':identity(ident['pid'])==ident,'terminal':json.loads((args.run/'terminal.json').read_text()) if (args.run/'terminal.json').exists() else None}))
    return 0
if __name__=='__main__':raise SystemExit(main())
