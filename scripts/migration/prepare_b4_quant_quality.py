#!/usr/bin/env python3
"""Freeze one shared case cohort before comparing three existing b4 exports."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from prepare_v1c import freeze
from v1c_artifacts import digest
from b4_quant_quality import scenario,noop,ARMS

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/sepalith/src'))
from sepalith.runner import Runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','assets','staging','models','runtime_recipe','archive','output','finish_source'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    if any(not x.is_absolute() for x in vars(a).values()):p.error('Use absolute paths')
    runner=Runner(a.state)
    if not runner.plan()['paused']:p.error('Build/freeze cases between experiments')
    a.staging.mkdir(exist_ok=False)
    # Copy source data before selection. This is CPU preparation, no inference.
    source_files={'finish_block_sample.jsonl':a.finish_source,'holdout.jsonl':scenario.HOLDOUT_REF}
    source_files.update({f+'.jsonl':scenario.SCEN_DIR/(f+'.jsonl') for f in scenario.FAMILIES})
    data,source_inputs=freeze(a.assets/'selection-sources',source_files)
    scenario.SCEN_DIR=data;scenario.HOLDOUT_REF=data/'holdout.jsonl';noop.FINISH_SRC=data/'finish_block_sample.jsonl'
    selected,report=scenario.load_heldout(150);rows=[]
    for family in scenario.FAMILIES:
        for r in selected[family]:rows.append(dict(kind='scenario',id=hashlib.sha1(r['_prompt'].encode()).hexdigest()[:12],prompt=r['_prompt'],scenario=r))
    if len(rows)!=255:raise ValueError('Expected255scenario rows; inspect selection drift')
    # Capture bytes read by the one corpus scan, not a new scan for each model.
    raw_files={};original_read=Path.read_bytes;original_corpus=noop.corpus_functions;functions=[]
    def recorded_read(path):
        value=original_read(path)
        if path.is_relative_to(noop.NORMALIZED):raw_files[str(path)]=value
        return value
    def corpus(n):
        result=original_corpus(n)
        if len(result)!=30:raise ValueError('No-op scan did not return all30corpus functions')
        functions.extend(result);return result
    Path.read_bytes=recorded_read;noop.corpus_functions=corpus
    try:made=noop.build_cases(30,7)
    finally:Path.read_bytes=original_read;noop.corpus_functions=original_corpus
    if len(made)!=258:raise ValueError('Expected258no-op cases; inspect cohort drift')
    for c in made:
        prompt,truncated=noop.build_prompt(c.lines,c.cursor_line,c.cursor_char,c.rel_path)
        rows.append(dict(kind='noop',case_kind=c.kind,id=c.id,prompt=prompt,truncated_prefix=truncated,
            **{key:getattr(c,key) for key in c.__slots__ if key not in ('id','kind')}))
    if len({(r['kind'],r['id']) for r in rows})!=513:raise ValueError('Duplicate case identity')
    (a.staging/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    source_record=[]
    for i,(path,value) in enumerate(sorted(raw_files.items())):
        name=f'normalized-{i:04d}.R';(a.staging/name).write_bytes(value);source_record.append(dict(source=path,file=name,sha256=hashlib.sha256(value).hexdigest()))
    record=dict(scenarios=report,noop_cases=len(made),noop_seed=7,corpus_seed=20260820,corpus_functions=functions,source_reads=source_record,
        versions={name:importlib.metadata.version(name) for name in ('tree-sitter','tree-sitter-r')},
        boundary='Fresh paired cohort, frozen once; historical prediction files are not resumed or overwritten')
    (a.staging/'selection.json').write_text(json.dumps(record,indent=2))
    names=['packaging_b4-Q8_0.gguf','packaging_b4_control-Q4_K_M.gguf','packaging_b4_imatrix-Q4_K_M.gguf']
    files={x.name:x for x in a.staging.iterdir() if x.is_file()};receipts=[]
    for arm,name in zip(ARMS,names):
        model=a.models/name;receipt=model.with_suffix(model.suffix+'.json');meta=json.loads(receipt.read_text())
        if digest(model)!=meta['sha256'] or model.stat().st_size!=meta['bytes']:raise ValueError('Export differs from receipt')
        receipts.append(meta);files[arm+'.gguf']=model;files[arm+'.receipt.json']=receipt
    if len({m['source'] for m in receipts})!=1:raise ValueError('Export parent-path receipts differ')
    if any((m['output_type'],m['embedding_type'])!=('Q8_0','Q8_0') for m in receipts[1:]):raise ValueError('Q4policies differ')
    for name in ('tree_sitter','tree_sitter_r'):
        module=__import__(name);directory=Path(module.__file__).parent
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in ('.py','.so'):
                files[str(Path('python')/name/path.relative_to(directory))]=path
    assets,inputs=freeze(a.assets/'experiment',files);inputs+=source_inputs
    previous=json.loads(a.runtime_recipe.read_text());runtime=Path(previous['env']['S1_RUNTIME'])
    for item in previous['inputs']:
        path=Path(item['path'])
        if path.is_relative_to(runtime) or str(path).startswith(('/usr/','/lib/')):
            if digest(path)!=item['sha256']:raise ValueError('Runtime changed')
            inputs.append(item)
    includes=['packages/sepalith/src','experiments/eval/spec_bench.py','experiments/eval/eval_scenarios.py',
        'experiments/eval/eval_noop_fp.py','experiments/eval/run_eval.py','experiments/post-processing/assemble_sft_v2.py',
        'experiments/synthetic-data/scenarios.py','experiments/synthetic-data/build_astfim.py',
        'experiments/synthetic-data/cases/__init__.py','experiments/synthetic-data/cases/corpus.py','experiments/synthetic-data/cases/validators.py',
        'scripts/migration/prepare_b4_quant_quality.py','scripts/migration/prepare_v1c.py','scripts/migration/s1_gpu.py','scripts/migration/b4_quant_quality.py','scripts/migration/v1c_artifacts.py']
    recipe=dict(schema_version=1,id='b4-export-paired-quality-20260908',snapshot=runner.snapshot(ROOT,includes),resource='gpu',
        python=previous['python'],depends_on=[],inputs=inputs,env=dict(previous['env'],PYTHONPATH=previous['env'].get('PYTHONPATH','')+':'+str(assets/'python')),provenance=dict(queue_item='S2/production quant quality',
            scope='Fresh255scenario+258noop cases x3existing exported models; intent gate separate',source_includes=includes,
            pi_review='Muse46.69s and GLM230.18s, max,300s caps; private pi-b4-quant-quality receipts',runtime_bound_seconds=5400,
            parent_boundary='Receipts identify a common parent path, not a parent content hash; comparison of current exported artifacts only'),steps=[])
    for action,artifacts in [('prepare',['prepared.json']),('measure',['requests.jsonl','token-audit.jsonl']),('evaluate',['evaluation.json','verdict.json'])]:
        recipe['steps'].append(dict(id=action,argv=['{python}','{source}/scripts/migration/b4_quant_quality.py',action,'--run','{run}','--assets',str(assets)],artifacts=artifacts))
    recipe['steps'].append(dict(id='archive',argv=['{python}','{source}/scripts/migration/v1c_artifacts.py','archive','--run','{run}','--archive',str(a.archive)],artifacts=['archive.json']))
    with a.output.open('x') as f:json.dump(recipe,f,indent=2)
    print(a.output)


if __name__=='__main__':main()
