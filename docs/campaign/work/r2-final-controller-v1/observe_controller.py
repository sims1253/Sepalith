"""Bounded CPU import-only source inventory. No final paths, models or server."""
import hashlib,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
started=time.monotonic()
import root_controller as controller
# Exercise stdlib CLI parsing only; never call controller main or release.
controller.argparse.ArgumentParser().parse_args([])
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
files={};absent=set();modules={};unresolved=[]
def add(path):
    p=Path(path)
    if p.suffix in ('.gguf','.safetensors','.jsonl','.pt'):raise ValueError('source paths only')
    if not p.is_file():
        if p.suffix=='.pyc':absent.add(str(p))
        return None
    p=p.resolve();key=hashlib.sha256(str(p).encode()).hexdigest()[:20]
    if key not in files:files[key]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'dependencies':[],'modules':[],'external_imports':[]}
    return key
interpreter=add(Path(sys.executable).resolve())
for name,module in sorted(sys.modules.items()):
    if module is None:continue
    spec=getattr(module,'__spec__',None);origin=getattr(spec,'origin',None);file=getattr(module,'__file__',None);cached=getattr(module,'__cached__',None);bound=[]
    for path in (file,cached):
        if path and Path(path).is_absolute():
            key=add(path)
            if key:bound.append(key)
    if origin in ('built-in','frozen'):bound.append(interpreter)
    if name in ('typing.io','typing.re'):
        bound.append(add(sys.modules['typing'].__file__))
    if name=='__main__':bound.append(add(__file__))
    if not bound:unresolved.append(name)
    modules[name]={'file':file,'cached':cached,'origin':origin,'bindings':sorted(set(bound))}
    if file and Path(file).is_file():files[add(file)]['modules'].append(name)
# Native profile is parsed during the controller import, never model bytes.
add(controller.CANDIDATE/'profile.json')
executable=[]
for line in Path('/proc/self/maps').read_text().splitlines():
    row=line.split(None,5)
    if 'x' not in row[1]:continue
    path=row[5] if len(row)==6 else ''
    if path in ('[vdso]','[vsyscall]'):continue
    if not path.startswith('/') or path.endswith(' (deleted)'):raise ValueError('unexpected executable origin')
    executable.append(str(Path(path).resolve()));add(path)
files[interpreter]['dependencies']=[k for k in files if k!=interpreter]
graph={'schema':'dat08.source-closure.v1','status':'preparation_only','roots':[interpreter],'files':files,'unresolved':unresolved,'scope':'native root controller standard-library process; constructor/client have separate frozen origin policies','observed_modules':modules,'must_remain_absent':sorted(absent),'executable_maps':sorted(set(executable)),'trust':'Pinned fresh CPython -I -S -B and pinned native extension factories; OS loader/kernel remain trusted. Explicit process bundle, not static import analysis.'}
(HERE/'controller-source-closure.prepared.json').write_text(json.dumps(graph,indent=2)+'\n')
result={'modules':len(modules),'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'unresolved':unresolved,'seconds':time.monotonic()-started,'peak_rss_bytes':int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')).split()[1])*1024,'model_load':False,'native_launch':False,'final_access':False}
(HERE/'controller-observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
