"""Root CLI for the frozen assembly component; actual-time freeze and origin gates."""
import argparse, datetime, hashlib, json, os, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ENTRY=HERE.parent/'r2-final-entrypoint-v1'
SITE='/home/m0hawk/Documents/Sepalith/.venv/lib/python3.10/site-packages'
sys.path[:0]=[str(HERE),str(ENTRY),SITE]
import assemble_inputs as assembly
import entry_gate as g
import constructor_runtime as runtime
import client_source_policy as policy
import origin_snapshot


def preload():
    integration,cases=runtime.preload();g.gate._load_protocol()
    modules={k:sys.modules[n] for k,n in [('builder','dat08_integration_builder'),
        ('families','dat08_integration_raw_families_v3'),('finish','dat08_integration_finish_v4')]}
    return integration,modules,sys.modules['dat08_integration_guard'],cases


def construct_candidates(requests,specs,read,modules,integration,guard_module,before_row):
    supported,census=[],[]
    typed={r.row_id:r for r in read.requests}
    for request in requests:
        before_row()
        case=assembly.probe(request,specs[request['row_id']],read.source_bytes[request['row_id']],modules)
        census.append(dict(row_id=request['row_id'],family=request['family'],group_id=request['group_id'],
            status='supported' if case is not None else assembly.ABSENCE[request['family']]))
        if case is not None:
            integration._case_binding(case,typed[request['row_id']],guard_module)
            supported.append(request)
    return supported,census


def guard_for(integration,guard_module,requests,identities,freeze,roots,**clock_fixture):
    return guard_module.AdmissionGuardV3(allowed_source_roots=roots,
        train_identities=identities['train'],dev_identities=identities['dev'],freeze_receipt=freeze,
        source_authorization=assembly.authorization(integration,requests),expected_weights_sha256=g.Q8,
        expected_harness_sha256=freeze['harness_sha256'],expected_source_lock_sha256=assembly.LOCK,
        expected_constructor_sha256=integration.CONSTRUCTOR_SHA256,
        expected_raw_builder_sha256=integration.RAW_BUILDER_CLOSURE_SHA256,
        observed_at=clock_fixture.pop('observed_at',datetime.datetime.now(datetime.timezone.utc).isoformat()),**clock_fixture)


def coverage_for(selected,counts,groups,selection,census,supported):
    return dict(attempts=len(census),supported=len(supported),selected=len(selected),groups=len(groups),
        minimum_groups=selection['min_independent_groups_target'],minimum_rows=600,
        max_cases_per_group=selection['max_cases_per_group'],core_total_ceiling=selection['core_total_ceiling'],
        family_ceilings=selection['planned_family_ceilings'],selected_by_family=counts,
        shortfall_by_family={f:cap-counts.get(f,0) for f,cap in selection['planned_family_ceilings'].items()},
        unsupported_strata=assembly.GAPS,candidate_reference_rows=0,semantic_admission=False,
        metrics_admission=False,coverage_target_met=len(selected)>=600 and len(groups)>=selection['min_independent_groups_target'])


def execute(args):
    freeze=g.release(args.freeze,args.harness_sha256)
    graph=g.source_graph(args.source_manifest,args.source_sha256,freeze,'assembly_source_closure_sha256')
    paths={v['path'] for v in graph['files'].values()}
    if not {str(Path(__file__).resolve()),str(Path(assembly.__file__).resolve())}.issubset(paths):
        raise ValueError('wrapper/component missing from frozen closure')
    epoch=g.source_epoch(graph)
    integration,modules,guard_module,_=preload()
    policy.verify(graph,deep=True)
    selection=assembly.checked_json(args.selection,assembly.LOCK)
    registry=assembly.checked_json(args.registry,assembly.REGISTRY)
    requests,specs,identities=assembly.metadata_plan(selection,registry)
    out=Path(args.output)
    if not out.is_absolute() or out.exists() or not out.parent.is_dir():raise ValueError('fresh absolute output required')
    out.mkdir(mode=0o700)
    admitted=guard_for(integration,guard_module,requests,identities,freeze,['/mnt/h/sepalith/normalized'])
    def still_released():g.release(args.freeze,args.harness_sha256)
    g.check_epoch(graph,epoch);policy.verify(graph,deep=True);still_released()
    read=admitted.authorize_and_read(requests,output_path=out/'probe-source-guard.json',
        before_source_read=lambda request:still_released())
    g.check_epoch(graph,epoch);policy.verify(graph,deep=True);still_released()
    supported,census=construct_candidates(requests,specs,read,modules,integration,guard_module,still_released)
    g.check_epoch(graph,epoch);policy.verify(graph,deep=True);still_released()
    selected,counts,groups=assembly.select(supported,selection)
    coverage=coverage_for(selected,counts,groups,selection,census,supported)
    outputs={'requests':selected,'case_specs':{r['row_id']:specs[r['row_id']] for r in selected},
        'source_authorization':assembly.authorization(integration,selected),'train_identities':identities['train'],
        'dev_identities':identities['dev'],'coverage':coverage,'site-census':census}
    outputs['assembly-binding']=dict(selection_sha256=assembly.LOCK,registry_sha256=assembly.REGISTRY,
        assembly_source_closure_sha256=args.source_sha256,harness_sha256=args.harness_sha256,
        source_read_authorization_id=read.authorization_id,
        output_canonical_hashes={k:assembly.digest(v) for k,v in outputs.items()})
    g.check_epoch(graph,epoch);policy.verify(graph,deep=True)
    integration._closure();g.binding.verify_source_closure(graph,args.source_sha256);still_released()
    for name,value in outputs.items():g.immutable_json(out/(name+'.json'),value)
    # No complete marker can exist unless post-write identity checks also pass.
    g.check_epoch(graph,epoch);policy.verify(graph,deep=True)
    g.binding.verify_source_closure(graph,args.source_sha256);still_released()
    g.immutable_json(out/'assembly-complete.json',dict(status='assembled_not_admitted',coverage=coverage,
        assembly_source_closure_sha256=args.source_sha256,final_admission=False))
    return 0


def synthetic_work():
    """Only fixed authored bytes below; accepts no corpus, freeze or path arguments."""
    integration,modules,guard_module,cases=preload()
    with tempfile.TemporaryDirectory(prefix='synthetic-',dir=HERE) as temporary:
        root=Path(temporary);source=root/'R';source.mkdir();path=source/'example.R'
        raw=b'f <- function(x) {\n  x + 1\n}\nvalue <- 1\n';path.write_bytes(raw)
        selection=dict(split_id='synthetic',global_split_sha256=assembly.REGISTRY,
            requires_weight_and_harness_freeze_receipt=True,
            planned_family_ceilings=dict.fromkeys(set(assembly.FAMILIES)|set(assembly.GAPS),2),
            min_independent_groups_target=30,max_cases_per_group=20,core_total_ceiling=1000,
            source_files=[dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())],
            parents=[dict(group_id='synthetic-final',split='final_candidate_group',identity='pkg:fixture',
                package='Fixture',version=str(root),files=[str(path)])])
        registry=dict(split_id='synthetic',groups=[dict(group_id='synthetic-'+role,split=split,flags=[],
            identity_forms=['pkg:'+package]) for role,split,package in [('final','final_candidate_group','fixture'),
            ('train','train_group','training'),('dev','dev_group','development')]])
        requests,specs,identities=assembly.metadata_plan(selection,registry)
        freeze=dict(status='frozen',weights_frozen=True,harness_frozen=True,final_access_unlocked=True,
            weights_sha256=g.Q8,harness_sha256='a'*64,weights_frozen_at='2026-09-14T10:00:00Z',
            harness_frozen_at='2026-09-14T10:00:00Z')
        # Private accepted guard seam is confined to fixed synthetic temporary files.
        admitted=guard_for(integration,guard_module,requests,identities,freeze,[source],observed_at='2026-09-14T10:01:00Z',now_utc='2026-09-14T10:01:00Z')
        read=admitted.authorize_and_read(requests,output_path=root/'source-guard.json')
        supported,census=construct_candidates(requests,specs,read,modules,integration,guard_module,lambda:None)
        chosen,counts,groups=assembly.select(supported,selection)
        coverage=coverage_for(chosen,counts,groups,selection,census,supported)
        g.immutable_json(root/'coverage.json',coverage)
    return dict(families_preloaded=cases,guarded_probe_count=len(requests),supported=len(supported),
        source_origin='fixed authored R fixture only',final_source_access=False,model_load=False)


def make_parser():
    parser=argparse.ArgumentParser()
    for name in ('freeze','source-manifest','selection','registry','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('harness-sha256','source-sha256'):parser.add_argument('--'+name,required=True)
    return parser


def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    parser=make_parser()
    if sys.argv[1:]==['--observe-synthetic-only']:
        result=synthetic_work()
        observed=origin_snapshot.snapshot();observed['executable_maps']=policy.executable_maps();observed['synthetic']=result
        with (HERE/'assembly-origins.json').open('x') as f:json.dump(observed,f,indent=2)
        print(json.dumps({'synthetic':result,'modules':len(observed['modules'])}));return 0
    return execute(parser.parse_args())
if __name__=='__main__':raise SystemExit(main())
