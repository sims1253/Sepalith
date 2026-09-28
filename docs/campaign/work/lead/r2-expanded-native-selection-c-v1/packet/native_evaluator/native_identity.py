"""Root-frozen fresh native process binding. Does not start or stop a server."""
import copy, datetime, fcntl, hashlib, json, os, time
from pathlib import Path
from urllib.parse import urlsplit
import native_transport as transport
HERE=Path(__file__).resolve().parent
PROFILE=json.loads((HERE/'profile.json').read_text())
from selection_binding import validate_profile
validate_profile(PROFILE)
Q8=PROFILE['model']['sha256']

def digest(x):return transport.canonical_sha256(x)
def stat_identity(path):
    s=Path(path).stat()
    return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def sha(path):return transport.sha256_file(Path(path),4*1024*1024)
def proc_stat(text):
    values=text[text.rfind(')')+2:].split()
    return values[0],values[19]

def _mapping_entry(line):
    """Parse one procfs map while allowing only the observed CUDA shared zero map."""
    fields=line.split(None,5)
    if len(fields)<6:
        if len(fields)>=2 and 'x' in fields[1]:
            raise ValueError('unreviewed executable mapping')
        return None
    raw=fields[5]
    if raw in ('[vdso]','[vsyscall]'):
        return {'path':raw,'permissions':fields[1],'device':fields[3],
                'inode':fields[4],'raw':line,'deleted':False,'trusted_special':True}
    if raw.endswith(' (deleted)'):
        # CUDA allocators expose these non-executable shared /dev/zero rows after
        # warmup.  Keep the original line in the audit and fail closed for every
        # other deleted mapping shape, including code and model files.
        if raw!='/dev/zero (deleted)' or fields[1]!='rw-s' or fields[3].lower()!='00:01':
            raise ValueError('deleted native mapping')
        return {'path':raw,'permissions':fields[1],'device':fields[3],
                'inode':fields[4],'raw':line,'deleted':True,'trusted_special':False}
    if not raw.startswith('/'):
        if 'x' in fields[1]:
            raise ValueError('unreviewed executable mapping')
        return None
    return {'path':str(Path(raw).resolve()),'permissions':fields[1],
            'device':fields[3],'inode':fields[4],'raw':line,'deleted':False,
            'trusted_special':False}

def validate_admission(a):
    if a.get('schema')!='dat08.native-process-admission.v1' or a.get('status')!='root_admitted':raise ValueError('native admission missing')
    if a.get('profile_sha256')!=digest(PROFILE) or a.get('model',{}).get('sha256')!=Q8:raise ValueError('native profile/model mismatch')
    if a.get('fresh_process') is not True or a.get('exclusive_owner') is not True or a.get('no_other_clients') is not True:
        raise ValueError('fresh exclusive native owner required')
    if a.get('root_verified_model_before_launch') is not True:raise ValueError('independent prelaunch artifact verification required')
    if a.get('cuda_backend_observed') is not True or a.get('all_layers_offloaded') is not True or len(a.get('load_evidence_sha256',''))!=64:
        raise ValueError('independently reviewed CUDA load evidence required')
    if not isinstance(a.get('pid'),int) or a['pid']<2 or not str(a.get('start_tick','')).isdigit():raise ValueError('native pid identity missing')
    parsed=urlsplit(a['endpoint'])
    if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.path or parsed.query or parsed.fragment or parsed.username or not parsed.port:
        raise ValueError('exact loopback native endpoint required')
    wanted={f['name']:f['sha256'] for f in PROFILE['bundle']['files']}
    if len(a['bundle_files'])!=len(wanted):raise ValueError('native bundle file denominator differs')
    for item in a['bundle_files']+a.get('mapped_code_files',[])+[a['model']]:
        path=Path(item['path'])
        if not path.is_absolute() or '..' in path.parts:raise ValueError('absolute pinned native paths required')
    if {Path(f['path']).name:f['sha256'] for f in a['bundle_files']}!=wanted:raise ValueError('delivered CUDA bundle differs')
    argv=a['argv'];server=next(f['path'] for f in a['bundle_files'] if Path(f['path']).name=='llama-server')
    expected=[server,'-m',a['model']['path'],'--host','127.0.0.1','--port',str(parsed.port),
              '-t','6','-tb','6','--threads-http','2','--parallel','1','-c','4096','-b','256','-ub','256','-ngl','99','-lv','4']
    if argv!=expected:raise ValueError('native argv differs from reviewed profile')
    if not a.get('mapped_code_files') or not a.get('lease_id') or not a.get('run_id'):raise ValueError('root native source/owner admission incomplete')
    return a

class NativeOwner:
    def __init__(self,admission):
        self.a=copy.deepcopy(validate_admission(admission));self.lock=None
        self.files=self.a['bundle_files']+self.a['mapped_code_files']+[self.a['model']]
        self.before={}
    def acquire(self):
        path=Path(self.a['exclusive_client_lock'])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():raise ValueError('root-created client lock required')
        self.lock=path.open('r+')
        fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    def close(self):
        if self.lock:self.lock.close();self.lock=None
    def verify(self,*,full_hash=False,deadline=None):
        a=self.a;now=datetime.datetime.now(datetime.timezone.utc)
        def check_deadline():
            if deadline is not None and deadline-time.monotonic()<=0:
                raise transport.CaseTimeout('case deadline elapsed during native identity check')
        check_deadline()
        created=datetime.datetime.fromisoformat(a['created_at'].replace('Z','+00:00'))
        expires=datetime.datetime.fromisoformat(a['expires_at'].replace('Z','+00:00'))
        if not created<=now<expires or (expires-created).total_seconds()>7200:raise ValueError('native admission expired or future')
        check_deadline()
        base=Path('/proc')/str(a['pid']);state,tick=proc_stat((base/'stat').read_text())
        if tick!=str(a['start_tick']) or state in ('Z','X','x') or base.stat().st_uid!=a['uid']:raise ValueError('native process identity drift')
        check_deadline()
        argv=(base/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        if argv!=a['argv'] or (base/'exe').resolve()!=Path(argv[0]).resolve():raise ValueError('native executable/argv drift')
        # Retain only the relevant environment assertion, never raw environment values.
        env=(base/'environ').read_bytes().split(b'\0')
        if [e for e in env if e.startswith(b'GGML_CUDA_GRAPH_OPT=')]!=[b'GGML_CUDA_GRAPH_OPT=0']:raise ValueError('CUDA GraphOpt not zero')
        for f in self.files:
            check_deadline()
            path=Path(f['path']);identity=stat_identity(path)
            if identity!=f['stat']:raise ValueError('admitted artifact metadata changed:'+str(path))
            if full_hash and sha(path)!=f['sha256']:raise ValueError('admitted artifact bytes changed:'+str(path))
        check_deadline()
        maps=(base/'maps').read_text();mapped=[];model_seen=False;executables=set()
        for line in maps.splitlines():
            check_deadline()
            entry=_mapping_entry(line)
            if entry is None:continue
            mapped.append(entry);path=entry['path']
            if 'x' in entry['permissions'] and not entry.get('trusted_special'):executables.add(path)
            if path==str(Path(a['model']['path']).resolve()):
                st=a['model']['stat'];dev='%02x:%02x'%(os.major(st['device']),os.minor(st['device']))
                if int(entry['inode'])!=st['inode'] or entry['device'].lower()!=dev.lower():raise ValueError('mapped model file identity differs')
                model_seen=True
        if not model_seen:raise ValueError('selected Q8 file is not mapped by owned native process')
        expected={str(Path(f['path']).resolve()) for f in a['mapped_code_files']}
        if executables!=expected:raise ValueError('native executable map closure changed')
        for f in a['bundle_files']:
            if Path(f['path']).name not in ('libggml-cpu.so.0','libllama-cli-impl.so','libmtmd.so.0') and str(Path(f['path']).resolve()) not in {r['path'] for r in mapped}:
                raise ValueError('required native bundle implementation not mapped')
        check_deadline()
        port=urlsplit(a['endpoint']).port;address='0100007F:%04X'%port
        inodes={v[9] for line in (base/'net/tcp').read_text().splitlines()[1:] if (v:=line.split())[1]==address and v[3]=='0A'}
        owned=set()
        for fd in (base/'fd').iterdir():
            check_deadline()
            try:target=os.readlink(fd)
            except FileNotFoundError:continue
            if target.startswith('socket:['):owned.add(target[8:-1])
        if len(inodes)!=1 or not inodes.issubset(owned):raise ValueError('owned unique loopback listener missing')
        check_deadline()
        props_timeout=1.5 if deadline is None else min(1.5,max(0.001,deadline-time.monotonic()))
        _,props,_=transport._request_json(a['endpoint'],'/props',None,props_timeout)
        check_deadline()
        if props.get('model_path')!=a['model']['path'] or props.get('total_slots')!=1 or props.get('default_generation_settings',{}).get('n_ctx')!=4096:
            raise ValueError('native props profile drift')
        check_deadline()
        state,tick=proc_stat((base/'stat').read_text())
        if tick!=str(a['start_tick']) or state in ('Z','X','x'):raise ValueError('native process changed during observation')
        return {'pid':a['pid'],'start_tick':tick,'uid':a['uid'],'props':props,'argv':argv,
                'mapped_files':mapped,'full_hash_verified':full_hash,'cuda_graph_opt':0,
                'admission_sha256':digest(a),'binding_scope':'fresh pinned native loading trust; not direct in-memory tensor proof'}
