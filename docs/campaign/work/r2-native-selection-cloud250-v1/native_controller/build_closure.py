"""Build an explicit observed process bundle. Never scan or hash model/data files."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent.parent/'final-source-closure-preparation-v3'
CANDIDATE=HERE.parent/'native_evaluator'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    old=json.loads((OLD/'candidate-source-graph.json').read_text())
    observed=json.loads((HERE/'client-origins.json').read_text())
    prior_text=(HERE.parent.parent/'final-native-rehearsal-preparation-v3/client-origins.json').read_text()
    prior_text=prior_text.replace(str(HERE.parent.parent/'final-native-evaluator-preparation-v3'),str(CANDIDATE)).replace(str(HERE.parent.parent/'final-native-rehearsal-preparation-v3'),str(HERE))
    prior=json.loads(prior_text)
    previous={m['name']:m for m in prior['modules']};current={m['name']:m for m in observed['modules']}
    comparison={'scope':'Independent fresh v3 observation versus retained v2 observations; only the two reviewed source directory names rebased','previous_modules':len(previous),'current_modules':len(current),'added':sorted(set(current)-set(previous)),'removed':sorted(set(previous)-set(current)),'changed':[name for name in sorted(set(current)&set(previous)) if current[name]!=previous[name]],'exact_matches':sum(current[name]==previous[name] for name in set(current)&set(previous))}
    (HERE/'independent-origin-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    files={};paths={};absent=set();unresolved=[]
    def add(path,why):
        p=Path(path)
        if p.suffix in ('.safetensors','.gguf','.pt','.jsonl'):raise ValueError('code-only path required:'+str(p))
        if not p.is_absolute():return None # Generated module is bound to its reviewed factory below.
        if not p.is_file():
            if p.suffix=='.pyc':absent.add(str(p))
            return None
        p=p.resolve();s=str(p);key='file_'+hashlib.sha256(s.encode()).hexdigest()[:20]
        if key not in files:files[key]={'path':s,'sha256':sha(p),'bytes':p.stat().st_size,'dependencies':[],'modules':[],'external_imports':[],'evidence':[why]};paths[s]=key
        return key
    interp=add(Path(observed['interpreter']).resolve(),'trusted CPython interpreter/frozen/builtin factory')
    for m in observed['modules']:
        for field in ('file','cached'):
            if m[field]:add(m[field],'actual client '+field)
    for path in observed['executable_maps']:add(path,'actual client executable mapping')
    for p in HERE.glob('*.py'):add(p,'root controller/policy source')
    for p in CANDIDATE.glob('*.py'):add(p,'unchanged native candidate source')
    add(CANDIDATE/'profile.json','exact selected native profile')
    policy={'schema':'dat08.runtime-origin-policy.v1','python_version':observed['version'],'interpreter_realpath':str(Path(observed['interpreter']).resolve()),'sys_path':observed['sys_path'],'modules':{},'must_remain_absent':sorted(absent),'executable_paths':observed['executable_maps']}
    groups={}
    for m in observed['modules']:
        name=m['name'];bindings=set();kind='file'
        for field in ('file','cached'):
            if m[field] and Path(m[field]).is_file():bindings.add(paths[str(Path(m[field]).resolve())])
        prior=old['runtime_policy']['modules'].get(name)
        if m['spec_origin'] in ('built-in','frozen'):bindings.add(interp);kind=m['spec_origin']
        elif not m['file'] or not Path(m['file']).is_file():
            if prior:
                kind=prior['kind']
                for k in prior['bindings']:
                    bound=add(old['files'][k]['path'],'source-reviewed '+kind+' factory')
                    if bound:bindings.add(bound)
            else:unresolved.append('unreviewed no-file factory:'+name)
        if not bindings:unresolved.append('module has no source binding:'+name)
        # Original helper checks __file__ only where that really exists. Other
        # origins are enforced by the additive interpreter/factory policy.
        if m['file'] and Path(m['file']).is_file():
            files[paths[str(Path(m['file']).resolve())]]['modules'].append(name)
        policy['modules'][name]={'kind':kind,'bindings':sorted(bindings),'observation':m}
        groups[kind]=groups.get(kind,0)+1
    policy['must_remain_absent']=sorted(absent)
    # Explicit allowed process bundle, not a fabricated static import graph.
    files[interp]['dependencies']=[k for k in files if k!=interp]
    graph={'schema':'dat08.source-closure.v1','status':'preparation_only','roots':[interp],'files':files,'runtime_policy':policy,'unresolved':unresolved,'graph_semantics':'Observed tokenizer/native-client process bundle plus source-reviewed interpreter and factory bindings. Edges express allowed process membership, not static import claims.','scope':'DEV rehearsal only; no final constructor closure or final admission','trust_boundaries':['Trusted fresh CPython -I -S with explicit roots; pinned pyc and source bytes, immutable source epoch.','Pinned native extension factories; dynamic aliases checked against exact observations and code fingerprints.','OS kernel/vdso/vsyscall, process loader and CUDA driver/device execution explicitly trusted; not arbitrary monkeypatch resistance.']}
    temporary=HERE/'source-closure.prepared.json.new'
    temporary.write_text(json.dumps(graph,indent=2)+'\n')
    os.replace(temporary,HERE/'source-closure.prepared.json')
    # This is a historical allowed library set; root compares actual fresh maps.
    maps=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-05-theta0-cuda-dev128-a/Q8_0-b128/server-maps-69345.txt')
    native={}
    for line in maps.read_text().splitlines():
        v=line.split(None,5)
        if 'x' in v[1] and len(v)==6 and v[5].startswith('/'):
            p=Path(v[5]).resolve();native[str(p)]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    (HERE/'native-mapped-code-allowlist.json').write_text(json.dumps({'historical_maps_path':str(maps),'historical_maps_sha256':sha(maps),'files':list(native.values()),'scope':'historical CUDA job executable file bytes only; fresh process must match exact set after synthetic warmup'},indent=2)+'\n')
    print(json.dumps({'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'modules':len(policy['modules']),'groups':groups,'unresolved':unresolved,'native_code_files':len(native)}))
if __name__=='__main__':main()
