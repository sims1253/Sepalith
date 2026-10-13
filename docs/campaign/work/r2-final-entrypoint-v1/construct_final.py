"""Root-only final constructor wrapper. No public time override."""
import argparse,datetime,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SITE='/home/m0hawk/Documents/Sepalith/.venv/lib/python3.10/site-packages'
sys.path[:0]=[str(HERE),SITE]
import entry_gate as g
import constructor_runtime as runtime
import client_source_policy as policy
import origin_snapshot

def manifest_for_rows(path,freeze):
    raw=Path(path).read_bytes();artifact=json.loads(raw)
    manifest={'schema':g.gate.MANIFEST_SCHEMA,'rows_artifact_sha256':g.gate.sha256_bytes(raw),
        'row_count':artifact['row_count'],'case_ids':[r['id'] for r in artifact['rows']],
        'source_artifact':artifact['source_artifact']}
    manifest['manifest_sha256']=g.gate._manifest_digest(manifest)
    g.gate._validate_manifest(manifest,expected_manifest_sha256=manifest['manifest_sha256'],freeze_sha256=g.gate.sha256_json(freeze))
    rows,audit,_=g.gate._validate_rows_file(g.gate._safe_metadata_path(Path(path)),manifest,g.gate._load_protocol())
    packages=len({row['package_id'] for row in rows})
    return manifest,{'rows':len(rows),'packages':packages,'target_rows':[600,1000],'target_packages':30,
        'row_shortfall':max(0,600-len(rows)),'package_shortfall':max(0,30-packages),'padding_allowed':False}

def execute(args):
    freeze=g.release(args.freeze,args.harness_sha256)
    # Both gates above precede graph/runtime/input/output path inspection.
    graph=g.source_graph(args.source_manifest,args.source_sha256,freeze,'constructor_source_closure_sha256')
    epoch=g.source_epoch(graph)
    integration,_=runtime.preload();g.gate._load_protocol();policy.verify(graph,deep=True)
    inputs={name:json.loads(Path(getattr(args,name)).read_text()) for name in ('requests','case_specs','source_authorization','train_identities','dev_identities')}
    requests=inputs['requests'];requests=requests.get('requests') if isinstance(requests,dict) else requests
    output=Path(args.output)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():raise ValueError('fresh absolute constructor output required')
    output.mkdir(mode=0o700)
    g.check_epoch(graph,epoch);g.release(args.freeze,args.harness_sha256)
    result=integration.integrate_and_publish(requests=requests,case_specs=inputs['case_specs'],
        output_path=output/'integration.json',guard_output_path=output/'source-guard.json',
        evaluator_output_path=output/'constructed-cases.json',rows_output_path=output/'canonical-rows.json',
        allowed_source_roots=args.allowed_root,train_identities=inputs['train_identities'],dev_identities=inputs['dev_identities'],
        freeze_receipt=freeze,source_authorization=inputs['source_authorization'],weights_sha256=g.Q8,
        harness_sha256=args.harness_sha256,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),now_utc=None)
    g.check_epoch(graph,epoch);policy.verify(graph,deep=True)
    g.binding.verify_source_closure(graph,args.source_sha256);g.release(args.freeze,args.harness_sha256)
    manifest,coverage=manifest_for_rows(output/'canonical-rows.json',freeze)
    g.check_epoch(graph,epoch);policy.verify(graph,deep=True)
    g.binding.verify_source_closure(graph,args.source_sha256)
    g.immutable_json(output/'rows-manifest.json',manifest)
    g.immutable_json(output/'constructor-complete.json',{'status':'constructed_and_validated','integration':result,'coverage':coverage,
        'constructor_source_sha256':args.source_sha256,'rows_manifest_sha256':manifest['manifest_sha256'],'final_admission':False})
    return 0

def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if sys.argv[1:]==['--observe-synthetic-only']:
        _,cases=runtime.preload();g.gate._load_protocol()
        observed=origin_snapshot.snapshot();observed['executable_maps']=policy.executable_maps();observed['synthetic_cases']=cases
        with (HERE/'constructor-origins.json').open('x') as f:json.dump(observed,f,indent=2)
        print(json.dumps({'synthetic_families':len(cases),'modules':len(observed['modules'])}));return 0
    parser=argparse.ArgumentParser()
    for name in ('freeze','source-manifest','requests','case-specs','source-authorization','train-identities','dev-identities','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--allowed-root',type=Path,action='append',required=True)
    parser.add_argument('--harness-sha256',required=True);parser.add_argument('--source-sha256',required=True)
    return execute(parser.parse_args())
if __name__=='__main__':raise SystemExit(main())
