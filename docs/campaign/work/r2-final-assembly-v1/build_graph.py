"""Pin two explicit observed software bundles; no model/data scan or imports."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'r2-final-entrypoint-v1/constructor-source-closure.admitted.json'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def build(route):
    old=json.loads(OLD.read_text());observed=json.loads((HERE/(route+'-origins.json')).read_text())
    files={};keys={};absent=set();unresolved=[]
    def add(path,why):
        path=Path(path)
        if not path.is_absolute():return None
        if path.suffix in ('.safetensors','.gguf','.jsonl','.pt'):raise ValueError('source-only path required')
        if not path.is_file():
            if path.suffix=='.pyc':absent.add(str(path))
            return None
        path=path.resolve();key='file_'+hashlib.sha256(str(path).encode()).hexdigest()[:20];keys[str(path)]=key
        if key not in files:files[key]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'dependencies':[],'modules':[],'external_imports':[],'evidence':[why]}
        return key
    interpreter=add(Path(observed['interpreter']).resolve(),'trusted interpreter/frozen/builtin boundary')
    for m in observed['modules']:
        for field in ('file','cached'):
            if m[field]:add(m[field],'actual observed '+field)
    for path in observed['executable_maps']:add(path,'actual observed executable mapping')
    if route=='assembly':
        inventory=json.loads((HERE.parent/'final-binding-integration-v2/source_inventory.json').read_text())
        needed=('production_integration_v3','admission_guard_v3','admission_guard','raw_source_case_builder','raw_source_families_v3','finish_constructor_v4','evaluator_adapter_v3','final_constructor','canonical_scenarios','finish_block','suffix_scenarios','campaign_admission_completion','campaign_protocol','sepalith_init')
        for name in needed:add(inventory['files'][name]['path'],'reviewed constructor file/factory boundary')
        parser=Path('/home/m0hawk/Documents/Sepalith/.venv/lib/python3.10/site-packages')
        for name in ('tree_sitter-0.26.0.dist-info/METADATA','tree_sitter_r-1.3.0.dist-info/METADATA'):add(parser/name,'parser distribution metadata used by accepted builder')
    else:add(HERE.parent/'r2-final-evaluator-v1/profile.json','selected native profile')
    policy={'schema':'dat08.runtime-origin-policy.v1','python_version':observed['version'],'interpreter_realpath':str(Path(observed['interpreter']).resolve()),'sys_path':observed['sys_path'],'modules':{},'must_remain_absent':[],'executable_paths':observed['executable_maps']}
    for m in observed['modules']:
        name=m['name'];bindings=set();kind='file'
        for field in ('file','cached'):
            if m[field] and Path(m[field]).is_file():bindings.add(keys[str(Path(m[field]).resolve())])
        if m['spec_origin'] in ('built-in','frozen'):kind=m['spec_origin'];bindings.add(interpreter)
        elif not m['file'] or not Path(m['file']).is_file():
            previous=old['runtime_policy']['modules'].get(name)
            if previous:
                kind=previous['kind']
                for key in previous['bindings']:
                    bound=add(old['files'][key]['path'],'reviewed '+kind+' factory')
                    if bound:bindings.add(bound)
            else:unresolved.append('unreviewed factory:'+name)
        if not bindings:unresolved.append('unbound module:'+name)
        if m['file'] and Path(m['file']).is_file():files[keys[str(Path(m['file']).resolve())]]['modules'].append(name)
        policy['modules'][name]={'kind':kind,'bindings':sorted(bindings),'observation':m}
    files[interpreter]['dependencies']=[key for key in files if key!=interpreter]
    policy['must_remain_absent']=sorted(absent)
    graph={'schema':'dat08.source-closure.v1','status':'preparation_only','roots':[interpreter],'files':files,'runtime_policy':policy,'unresolved':unresolved,'scope':route+' process only; paired constructor/evaluator graphs are separately frozen','graph_semantics':'Explicit observed process bundle and reviewed factories; membership edges do not claim static import analysis','trust_boundaries':['Trusted pinned fresh CPython -I -S -B and exact explicit import roots','Pinned extension/factory implementation and observed origins; no arbitrary monkeypatch sandbox claim','OS kernel/loader/vdso/vsyscall and device/driver execution remain explicit trust; no direct tensor-memory proof']}
    (HERE/(route+'-source-closure.prepared.json')).write_text(json.dumps(graph,indent=2)+'\n')
    return {'route':route,'files':len(files),'bytes':sum(f['bytes'] for f in files.values()),'modules':len(policy['modules']),'unresolved':unresolved}
if __name__=='__main__':
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});print(json.dumps([build('assembly')]))
