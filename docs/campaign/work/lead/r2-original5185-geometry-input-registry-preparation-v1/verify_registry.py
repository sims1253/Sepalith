#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

SCHEMA='sepalith.dat10.original5185.geometry_input_registry.v1'
CTX_KEYS={'schema_version','path','prefix','region_old','suffix_lines','history','cursor','replacement_range'}

def req(ok,msg):
    if not ok: raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def resolve(path, root):
    p=Path(path); return p if p.is_absolute() else root/p
def first_json(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): return json.loads(line)
    raise ValueError(f'empty jsonl:{path}')
def json_rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def selected(row, prov):
    if 'selected_context' in row: return row['selected_context']
    mode=prov['mode']
    req(mode in ('full_document','complete_span'),'unsupported provenance mode')
    return row['full_context'] if mode=='full_document' else row['bounded_context']
def validate_pair(inp, out, prov):
    rid=prov['row_id']; req(inp['row_id']==rid and out['row_id']==rid,'row join')
    req(inp['preedit_sha256']==prov['preedit_sha256'],'preedit provenance join')
    ctx=selected(out,prov); req(CTX_KEYS <= set(ctx),'selected context fields')
    rr=ctx['replacement_range']; req(rr['content_sha256']==inp['preedit_sha256'],'replacement content identity')
    req(rr['start']==inp['cursor'],'authoritative cursor/replacement start')
    req(isinstance(ctx['history'],list),'history type')
    si=prov['source_identity']; req(si['row_id']==rid,'source identity row')
    req(si['source_sha256']!= '' and inp['preedit_sha256']!='','typed identities')
    return {'row_id':rid,'original_source_sha256':si['source_sha256'],'current_preedit_sha256':inp['preedit_sha256'],'locator':{'source_path':si['source_path'],'prediction_path':inp['path'],'replacement_uri':rr['uri']},'selected_window':{'prefix_lines':len(ctx['prefix']),'region_old_lines':len(ctx['region_old']),'suffix_lines':len(ctx['suffix_lines']),'history_items':len(ctx['history']),'replacement_range':rr}}
def check_manifest(pin, root):
    p=resolve(pin['path'],root); req(p.is_file(),f'missing:{p}'); req(sha(p)==pin['sha256'],f'hash:{p}')
    x=json.loads(p.read_text()); req(x.get('schema')==pin['schema'],f'schema:{p}'); return x
def metadata(reg, root):
    req(reg['schema']==SCHEMA,'registry schema'); req(reg['status']=='preparation_only_root_admission_required','status'); req(reg['training_admission'] is False,'admission')
    cohorts=reg['cohorts']; req([x['name'] for x in cohorts]==['candidate616','candidate133','candidate4435','candidate1'],'cohort order')
    req(sum(x['candidate_rows'] for x in cohorts)==5185,'5185 accounting'); req(reg['accounting']['review_pool_rows']-reg['accounting']['original15006_rows']==5185,'pool accounting')
    census=reg['evidence']['family_census']; p=resolve(census['path'],root); req(sha(p)==census['sha256'],'census hash'); cx=json.loads(p.read_text())
    by={x['cohort']:x for x in cx['cohorts']}; req(cx['total_review_pool_rows']==20191,'census total')
    samples=[]
    for c in cohorts:
        req(by[c['name']]['rows']==c['candidate_rows'],'census cohort count')
        cm=check_manifest(c['candidate_manifest'],root); req(cm['candidate_rows']==c['candidate_rows'],'candidate manifest rows')
        mout=cm['outputs']; req(mout['candidate-provenance.jsonl']['sha256']==c['candidate_provenance']['sha256'] and mout['candidate-provenance.jsonl']['rows']==c['candidate_rows'],'candidate provenance manifest binding')
        req(mout['candidate-tokenrows.jsonl']['sha256']==c['token_rows']['sha256'] and mout['candidate-tokenrows.jsonl']['rows']==c['candidate_rows'],'candidate token manifest binding')
        req(by[c['name']]['sha256']==c['token_rows']['sha256'],'census token binding')
        cp=resolve(c['candidate_provenance']['path'],root); req(cp.is_file(),'candidate provenance missing'); req(cp.stat().st_size==mout['candidate-provenance.jsonl']['bytes'],'candidate provenance bytes')
        # Hashing all provenance is 8MB total and binds original source identities without reading model/token payloads.
        req(sha(cp)==c['candidate_provenance']['sha256'],'candidate provenance hash')
        pr=first_json(cp); req(pr['source_identity']['source_sha256'] and pr['preedit_sha256'],'identity sample')
        inp=first_json(resolve(c['prediction_inputs'][0]['path'],root)); out=first_json(resolve(c['selected_contexts'][0]['path'],root))
        if inp['row_id']==pr['row_id'] and out['row_id']==pr['row_id']:
            samples.append(validate_pair(inp,out,pr))
        else:
            samples.append({'cohort':c['name'],'sample_status':'domain-first-rows-differ; exact row join deferred'})
        for mkey in ('prediction_manifest','selection_manifest'):
            if mkey in c: check_manifest(c[mkey],root)
        for group in ('prediction_inputs','selected_contexts'):
            for pin in c[group]:
                q=resolve(pin['path'],root); req(q.is_file(),f'missing:{q}')
                if 'bytes' in pin: req(q.stat().st_size==pin['bytes'],f'size:{q}')
    return {'schema':'sepalith.dat10.original5185.geometry_input_registry_check.v1','status':'pass','mode':'metadata_and_small_samples','candidate_rows':5185,'cohorts':{x['name']:x['candidate_rows'] for x in cohorts},'sample_evidence':samples,'large_full_join_executed':False}
def full_join(reg,root):
    report=metadata(reg,root); joined={}
    for c in reg['cohorts']:
        prov={x['row_id']:x for x in json_rows(resolve(c['candidate_provenance']['path'],root))}
        inputs={}
        for p in c['prediction_inputs']:
            for x in json_rows(resolve(p['path'],root)): req(x['row_id'] not in inputs,'duplicate input id'); inputs[x['row_id']]=x
        outputs={}
        for p in c['selected_contexts']:
            for x in json_rows(resolve(p['path'],root)): req(x['row_id'] not in outputs,'duplicate selected id'); outputs[x['row_id']]=x
        req(len(prov)==c['candidate_rows'],'provenance count')
        req(set(prov)<=set(inputs),'missing candidate input'); req(set(prov)<=set(outputs),'missing candidate selected context')
        for rid,pv in prov.items(): validate_pair(inputs[rid],outputs[rid],pv)
        joined[c['name']]={'candidate_rows':len(prov),'provider_input_rows':len(inputs),'selected_rows':len(outputs),'exact_candidate_join':True}
    report['mode']='full_join'; report['large_full_join_executed']=True; report['joins']=joined
    return report
def main():
    a=argparse.ArgumentParser(); a.add_argument('registry'); a.add_argument('--mode',choices=('metadata','full-join'),default='metadata'); a.add_argument('--report'); ns=a.parse_args()
    rp=Path(ns.registry).resolve(); root=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
    reg=json.loads(rp.read_text()); result=metadata(reg,root) if ns.mode=='metadata' else full_join(reg,root)
    text=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if ns.report:
        out=Path(ns.report); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(text)
    print(text,end='')
if __name__=='__main__': main()
