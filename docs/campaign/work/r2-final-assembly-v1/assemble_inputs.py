"""Private DAT-08 candidate. No CLI clock override; root must admit a NEW closure."""
import collections, hashlib, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
CAM = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
ENTRY = CAM / 'work/r2-final-entrypoint-v1'
LOCK = 'f8037b71373ccb3b0d71241430c40b4113208b76eed67a9937f86c4f3a8b1e1e'
REGISTRY = 'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
FAMILIES = ('finish_block', 'rename_propagation', 'pipe_rewrite', 'na_rm_propagation', 'no_op')
ABSENCE = {'finish_block': 'no_production_finish_signature_candidate', 'no_op': 'no_production_no_op_close_candidate',
           'rename_propagation': 'no_valid_scenario_candidate', 'pipe_rewrite': 'no_valid_scenario_candidate',
           'na_rm_propagation': 'no_valid_scenario_candidate'}
GAPS = {'format_propagation': 'locked raw/normalized pair not supplied', 'roxygen_drafting': 'no admitted constructor'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module

def checked_json(path, sha):
    with Path(path).open('rb') as f:
        raw = f.read(64*1024*1024 + 1)
    if len(raw) > 64*1024*1024: raise ValueError('metadata size bound exceeded')
    if hashlib.sha256(raw).hexdigest() != sha: raise ValueError('metadata hash mismatch')
    return json.loads(raw)

def metadata_plan(selection, registry):
    if selection['split_id'] != registry['split_id'] or selection['global_split_sha256'] != REGISTRY:
        raise ValueError('registry binding mismatch')
    if selection['requires_weight_and_harness_freeze_receipt'] is not True:
        raise ValueError('freeze required')
    if set(selection['planned_family_ceilings']) != set(FAMILIES) | set(GAPS):
        raise ValueError('family policy changed')
    groups = {g['group_id']: g for g in registry['groups']}
    if len(groups) != len(registry['groups']): raise ValueError('duplicate registry group')
    identities = {}
    for role, split in [('train', 'train_group'), ('dev', 'dev_group')]:
        chosen = [g for g in groups.values() if g['split'] == split]
        identities[role] = {'group_ids': sorted(g['group_id'] for g in chosen), 'package_ids': sorted({
            t[4:] for g in chosen for t in g['identity_forms'] if t.startswith('pkg:')}),
            'split_identities': sorted({t for g in chosen for t in g['identity_forms']})}
    sources = {s['path']: s for s in selection['source_files']}
    if len(sources) != len(selection['source_files']): raise ValueError('duplicate source metadata')
    requests, specs, seen = [], {}, set()
    for p in sorted(selection['parents'], key=lambda x: x['group_id']):
        g = groups[p['group_id']]
        if g['split'] != 'final_candidate_group' or p['split'] != g['split'] or g['flags']:
            raise ValueError('ineligible final group')
        if p['identity'] != 'pkg:' + p['package'].lower() or p['identity'] not in g['identity_forms']:
            raise ValueError('package alias binding mismatch')
        for path in sorted(p['files']):
            if path in seen: raise ValueError('duplicate source path')
            seen.add(path)
            rel = Path(path).relative_to(Path(p['version'])).as_posix()
            if 'R' not in Path(rel).parts or not rel.endswith(('.R', '.r')): raise ValueError('not selected R source')
            for family in FAMILIES:
                row = 'dat08-' + digest([p['group_id'], path, sources[path]['sha256'], family, 3407])[:32]
                # repository_id is an explicit package identity, never an invented upstream URL.
                requests.append(dict(row_id=row, split_identity=row, package_id=p['identity'][4:],
                    repository_id=p['identity'], group_id=p['group_id'], family=family,
                    source_path=path, source_sha256=sources[path]['sha256']))
                specs[row] = dict(package_id=p['identity'][4:], group_id=p['group_id'], path=rel, seed=3407)
    return requests, specs, identities

def probe(request, spec, raw, modules):
    """Only exact, family-specific absence is skippable. Source/parse/replay errors abort."""
    family = request['family']
    kw = dict(package_id=request['package_id'], group_id=request['group_id'], path=spec['path'],
              family=family, source_sha256=request['source_sha256'])
    try:
        if family == 'finish_block':
            return modules['finish'].build_raw_family_case(raw, **kw, variant='signature')
        if family == 'no_op': return modules['families'].build_raw_family_case(raw, **kw)
        return modules['builder'].build_raw_source_case(raw, **kw, seed=spec['seed'])
    except ValueError as exc:
        if str(exc) == ABSENCE[family]: return None
        raise

def select(supported, selection):
    queues = collections.defaultdict(list)
    for r in supported: queues[r['group_id']].append(r)
    order = sorted(queues, key=lambda g: digest([3407, g]))
    for group in queues: queues[group].sort(key=lambda r: digest([3407, r['row_id']]))
    counts, groups, chosen = collections.Counter(), collections.Counter(), []
    while any(queues.values()):
        for group in order:
            while queues[group]:
                r = queues[group].pop(0)
                if counts[r['family']] >= selection['planned_family_ceilings'][r['family']]: continue
                if groups[group] >= selection['max_cases_per_group']: continue
                if len(chosen) >= selection['core_total_ceiling']: break
                chosen.append(r); counts[r['family']] += 1; groups[group] += 1; break
            if len(chosen) >= selection['core_total_ceiling']: break
        if len(chosen) >= selection['core_total_ceiling']: break
    return chosen, dict(counts), dict(groups)

def authorization(integration, requests):
    return dict(schema='dat08.source-read-authorization.v2', status='source_read_authorized',
        split='final_locked_v1', source_lock_sha256=LOCK, constructor_sha256=integration.CONSTRUCTOR_SHA256,
        raw_builder_sha256=integration.RAW_BUILDER_CLOSURE_SHA256,
        authorized_case_ids=sorted(r['row_id'] for r in requests))

def execute(*, freeze_path, harness_sha256, source_graph_path, source_graph_sha256,
            selection_path, registry_path, output_dir):
    """Root-only POST-FREEZE API. Full new graph and explicit freeze field are mandatory."""
    sys.path.insert(0, str(ENTRY))
    import entry_gate as gate
    freeze = gate.release(freeze_path, harness_sha256)  # actual time, before any other input
    graph = gate.source_graph(source_graph_path, source_graph_sha256, freeze, 'assembly_source_closure_sha256')
    if str(Path(__file__).resolve()) not in {v['path'] for v in graph['files'].values()}:
        raise ValueError('assembler missing from frozen closure')
    epoch = gate.source_epoch(graph)
    runtime = load('dat08_assembly_runtime', ENTRY / 'constructor_runtime.py')
    integration, _ = runtime.preload()
    modules = {k: sys.modules[n] for k,n in [('builder','dat08_integration_builder'),
        ('families','dat08_integration_raw_families_v3'), ('finish','dat08_integration_finish_v4')]}
    guard_module = sys.modules['dat08_integration_guard']
    selection = checked_json(selection_path, LOCK); registry = checked_json(registry_path, REGISTRY)
    requests, specs, identities = metadata_plan(selection, registry)
    out = Path(output_dir)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output directory required')
    out.mkdir(parents=False)
    admitted = guard_module.AdmissionGuardV3(allowed_source_roots=['/mnt/h/sepalith/normalized'],
        train_identities=identities['train'], dev_identities=identities['dev'], freeze_receipt=freeze,
        source_authorization=authorization(integration, requests), expected_weights_sha256=gate.Q8,
        expected_harness_sha256=harness_sha256, expected_source_lock_sha256=LOCK,
        expected_constructor_sha256=integration.CONSTRUCTOR_SHA256,
        expected_raw_builder_sha256=integration.RAW_BUILDER_CLOSURE_SHA256,
        observed_at=gate.datetime.datetime.now(gate.datetime.timezone.utc).isoformat())
    gate.check_epoch(graph, epoch); gate.release(freeze_path, harness_sha256)
    read = admitted.authorize_and_read(requests, output_path=out/'probe-source-guard.json')
    supported, census = [], []
    typed_requests = {r.row_id: r for r in read.requests}
    for request in requests:
        gate.release(freeze_path, harness_sha256)
        case = probe(request, specs[request['row_id']], read.source_bytes[request['row_id']], modules)
        census.append(dict(row_id=request['row_id'], family=request['family'], group_id=request['group_id'],
                           status='supported' if case is not None else ABSENCE[request['family']]))
        if case is not None:
            integration._case_binding(case, typed_requests[request['row_id']], guard_module)
            supported.append(request)
    selected, counts, group_counts = select(supported, selection)
    coverage = dict(attempts=len(census), supported=len(supported), selected=len(selected),
        groups=len(group_counts), minimum_groups=selection['min_independent_groups_target'],
        minimum_rows=600, core_total_ceiling=selection['core_total_ceiling'],
        family_ceilings=selection['planned_family_ceilings'], selected_by_family=counts,
        shortfall_by_family={f: cap-counts.get(f,0) for f,cap in selection['planned_family_ceilings'].items()},
        unsupported_strata=GAPS, candidate_reference_rows=0,
        selection_rule='seed3407 group round-robin; one exact canonical site per source/family',
        semantic_admission=False, metrics_admission=False,
        coverage_target_met=len(selected)>=600 and len(group_counts)>=selection['min_independent_groups_target'])
    gate.check_epoch(graph, epoch); integration._closure(); gate.release(freeze_path, harness_sha256)
    gate.source_graph(source_graph_path, source_graph_sha256, freeze, 'assembly_source_closure_sha256')
    outputs = {'requests': selected, 'case_specs': {r['row_id']: specs[r['row_id']] for r in selected},
        'source_authorization': authorization(integration, selected), 'train_identities': identities['train'],
        'dev_identities': identities['dev'], 'coverage': coverage, 'site-census': census,
        'assembly-binding': dict(selection_sha256=LOCK, registry_sha256=REGISTRY,
             assembly_source_closure_sha256=source_graph_sha256, harness_sha256=harness_sha256,
             output_canonical_hashes={})}
    outputs['assembly-binding']['output_canonical_hashes'] = {k:digest(v) for k,v in outputs.items() if k!='assembly-binding'}
    for name, value in outputs.items(): gate.immutable_json(out/(name+'.json'), value)
    return coverage
