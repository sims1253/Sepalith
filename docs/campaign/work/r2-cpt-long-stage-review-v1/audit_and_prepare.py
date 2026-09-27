#!/usr/bin/env python3
"""One-core frozen-row audit and preparation only; never imports a model stack."""
import collections, copy, hashlib, json, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAM = HERE.parent.parent
COR = HERE.parent / 'r2-corpus-preparation-v1'
NATIVE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
REG = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
SOURCE = '5149a4065285c681c2230352e5847b861e2ad2e416c8c263131a557e48bd23ca'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def write(name, data):
    p = HERE / name
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    return {'path': str(p), 'sha256': sha(p)}

def check(condition, message):
    if not condition: raise ValueError(message)

def check_row(r, doc, state):
    ids, labels, mask = r['input_ids'], r['labels'], r['attention_mask']
    n, end, overlap = len(ids), r['is_document_end'], r['overlap_context_tokens']
    check(3 <= n <= 2048 and len(labels) == n == len(mask), 'length')
    check(ids[0] == 0 and ids[-1] == 1 and all(type(x) is int and 1 < x < 130560 for x in ids[1:-1]), 'token boundaries')
    check(mask == [1] * n, 'stored attention')
    check(overlap == (1 if r['chunk_index'] else 0), 'overlap count')
    expected = [-100] + ([-100] if overlap else []) + ids[1 + overlap:-1] + ([1] if end else [-100])
    check(labels == expected, 'loss mask')
    check(r['supervised_tokens'] == sum(x != -100 for x in labels[1:]), 'denominator')
    check(r['document_id'] == doc['document_id'] == r['source_sha256'] == doc['sha256'], 'document identity')
    check(r['row_id'] == r['document_id'] + ':' + str(r['chunk_index']), 'row identity')
    check(r['source_path'] == doc['path'] and r['package'] == doc['package'] and r['group_id'] == doc['group_id'], 'provenance')
    check(r['cpt_partition'] == doc['cpt_partition'], 'partition')
    check(r['token_start'] == r['source_token_start'] == state['end'], 'contiguous offset')
    check(r['token_end'] == r['source_token_end'] == r['token_start'] + n - 2 - overlap, 'offset width')
    check(r['document_token_count'] == doc['source_code_tokens'], 'document length')
    check(end == (r['token_end'] == r['document_token_count']), 'document EOS')
    check(r['chunk_index'] == state['chunks'] and not state['finished'], 'chunk order')
    if overlap: check(ids[1] == state['last'], 'overlap token identity')
    state.update(end=r['token_end'], chunks=state['chunks'] + 1, last=ids[-2], finished=end)
    return dict(input_tokens=n, loss_tokens=r['supervised_tokens'], code_tokens=n - 2 - overlap, terminal_document_eos=int(end))

def main():
    started = time.monotonic()
    if hasattr(os, 'sched_setaffinity'): os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    pins = {}
    def pin(p, expected=None):
        got = sha(p)
        if expected: check(got == expected, 'pin mismatch ' + str(p))
        pins[str(p)] = {'sha256': got, 'bytes': Path(p).stat().st_size}
        return got
    pin(REG, 'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09')
    reg = read(REG)
    mapping = {}
    for g in reg['groups']:
        for identity in g['identity_forms']:
            if identity.startswith('pkg:'):
                key = identity[4:].lower()
                check(key not in mapping or mapping[key] == (g['group_id'], g['split']), 'ambiguous registry package')
                mapping[key] = (g['group_id'], g['split'])
    del reg
    partition = read(COR / 'cpt-train-group-partition.json')
    pin(COR / 'cpt-train-group-partition.json', '6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06')
    for g, part in partition['groups'].items():
        want = 'cpt_validation' if int(hashlib.sha256(('DAT10-CPT-validation-v1\0' + g).encode()).hexdigest(), 16) % 20 == 0 else 'cpt_train'
        check(part == want, 'partition hash rule')
    manifest = read(COR / 'broader-shard-v1-2k/manifest.json')
    pin(COR / 'broader-shard-v1-2k/manifest.json', 'f5b615daead1438b8c85aa36c9c672296d90670c87b340441154875f22e72b10')
    pin(COR / 'broader-stage-manifest.json', 'b58dede64ffb180750744a7fcf75cde177ba0d08b643645f9ee51332090dddbb')
    replay = read(COR / 'broader-independent-validation.json')
    check(replay['status'] == 'PASS' and replay['manifest_sha256'] == pins[str(COR / 'broader-shard-v1-2k/manifest.json')]['sha256'], 'prior reconstruction binding')
    pin(COR / 'broader-independent-validation.json')
    pin(COR / 'validate_cpt_shard.py', replay['validator_sha256'])
    pin(COR / 'raw_cpt_broader.py', manifest['source_sha256'])
    pin(COR / 'broader-raw-file-snapshot.jsonl', manifest['metadata_sha256'])
    snapshot = {}
    with (COR / 'broader-raw-file-snapshot.jsonl').open() as stream:
        for line in stream:
            d = json.loads(line); snapshot[d['path']] = d
    for name, artifact in manifest['artifacts'].items():
        if name not in ('cpt_train.jsonl', 'documents.jsonl'):
            pin(COR / 'broader-shard-v1-2k' / name, artifact['sha256'])
    pin(COR / 'known-nontrain-parent-hashes.json', manifest['parent_hash_guard_sha256'])
    parent_hashes = set(read(COR / 'known-nontrain-parent-hashes.json'))
    docs = {}; partitions = collections.defaultdict(set); packages = collections.defaultdict(set)
    for f, part in [('broader-shard-v1-2k/documents.jsonl', 'cpt_train'), ('profile-shard-v1/documents.jsonl', 'cpt_validation')]:
        expected = manifest['artifacts']['documents.jsonl']['sha256'] if part == 'cpt_train' else manifest['reserved_validation_document_manifest_sha256']
        pin(COR / f, expected)
        with (COR / f).open() as stream:
            for line in stream:
                d = json.loads(line)
                if d['cpt_partition'] != part: continue
                check(d['document_id'] not in docs, 'cross-partition/exact duplicate')
                check(mapping[d['package'].lower()] == (d['group_id'], 'train_group'), 'registry binding')
                check(partition['groups'][d['group_id']] == part, 'heldout group exclusion')
                check(not parent_hashes.intersection({d['sha256'], d['sha1'], d['git_blob_sha1']}), 'known heldout hash overlap')
                expected_prefix = '/mnt/h/sepalith/normalized/' + d['package'] + '/' + d['version'] + '/' + d['package'] + '/R/'
                check(d['path'].startswith(expected_prefix) and '..' not in Path(d['path']).parts, 'raw path binding')
                if part == 'cpt_train':
                    check(d['path'] in snapshot, 'frozen snapshot membership')
                    for key, value in snapshot[d['path']].items():
                        check(d.get(key) == value, 'frozen source metadata field ' + key)
                docs[d['document_id']] = d; partitions[part].add(d['group_id']); packages[part].add(d['package'])
    rows = {}; totals = {}; group_code = collections.Counter(); first_row = None
    for f, part, expected in [('broader-shard-v1-2k/cpt_train.jsonl', 'cpt_train', manifest['artifacts']['cpt_train.jsonl']['sha256']), ('profile-shard-v2-2k/cpt_validation.jsonl', 'cpt_validation', 'efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8')]:
        digest = hashlib.sha256(); count = collections.Counter(); states = {}
        with (COR / f).open('rb') as stream:
            for line in stream:
                digest.update(line); r = json.loads(line)
                if first_row is None: first_row = copy.deepcopy(r)
                check(r['row_id'] not in rows, 'duplicate row')
                doc = docs[r['document_id']]
                check(doc['cpt_partition'] == part, 'row partition')
                state = states.setdefault(r['document_id'], dict(end=0, chunks=0, last=None, finished=False))
                stats = check_row(r, doc, state)
                count.update(stats); count['rows'] += 1
                if part == 'cpt_train': group_code[r['group_id']] += stats['code_tokens']
                rows[r['row_id']] = dict(stats, package=r['package'], group_id=r['group_id'], document_id=r['document_id'], partition=part)
        check(digest.hexdigest() == expected, 'row file hash')
        pins[str(COR / f)] = {'sha256': expected, 'bytes': (COR / f).stat().st_size}
        for docid, state in states.items():
            expected_chunks = 1 + (max(0, docs[docid]['source_code_tokens'] - 2046) + 2044) // 2045
            check(state['finished'] and state['end'] == docs[docid]['source_code_tokens'] and state['chunks'] == expected_chunks, 'complete 2K source coverage')
        count.update(documents=len(states), packages=len(packages[part]), groups=len(partitions[part]))
        totals[part] = dict(count)
    check(not partitions['cpt_train'] & partitions['cpt_validation'], 'package holdout')
    schedule = read(COR / 'broader-draws-one-pass.json')
    pin(COR / 'broader-draws-one-pass.json', 'deb29779212a7886f572e955bf82bffa5c8d4be6c1598c07389e1e723321108a')
    ids = schedule['row_ids']; train_ids = [k for k,v in rows.items() if v['partition'] == 'cpt_train']
    check(len(ids) == len(set(ids)) == 27430 and set(ids) == set(train_ids), 'exact permutation')
    shuffled = list(train_ids); random.Random(3407).shuffle(shuffled)
    check(shuffled == ids, 'seeded draw order')
    kept, tail = ids[:1714 * 16], ids[1714 * 16:]
    aggregate = lambda seq: dict(sum((collections.Counter({k: rows[r][k] for k in ('input_tokens','loss_tokens','code_tokens','terminal_document_eos')}) for r in seq), collections.Counter()))
    exposure = {'full_file': totals, 'steps': 1714, 'effective_batch': 16, 'selected_rows': len(kept), 'selected_totals': aggregate(kept), 'omitted_rows': [dict(row_id=r, **rows[r]) for r in tail], 'omitted_totals': aggregate(tail), 'policy': '1714 complete batches; six named rows omitted, no partial tail or replay; incomplete pass'}
    check(exposure['omitted_totals']['loss_tokens'] == 8710 and len(tail) == 6, 'tail accounting')
    write('exposure.json', exposure)
    draw = write('draws-1714.json', dict(schema='sepalith.cpt.draws.v1', split_id=partition['split_id'], method='without_replacement', seed=3407, max_steps=1714, effective_batch=16, token_rows_sha256=schedule['training_rows_sha256'], row_ids=kept))
    all_packages = read(COR / 'raw-train-package-candidates.json')['packages']
    lexical = sorted(all_packages, key=lambda x:x['name'])
    rank = {p['name']:i for i,p in enumerate(lexical)}
    names = packages['cpt_train']; positions = sorted(rank[n] for n in names)
    distribution = dict(collections.Counter(n[0].upper() for n in names))
    next_groups = sorted((p for p in all_packages if partition['groups'][p['group_id']] == 'cpt_train' and p['group_id'] not in partitions['cpt_train']), key=lambda p: hashlib.sha256(('DAT10-next-global-v1\0' + p['group_id']).encode()).digest())
    write('next-global-package-order.json', {'status':'metadata-only future source inventory plan; no content read or token admission', 'seed':'DAT10-next-global-v1', 'source':str(COR/'raw-train-package-candidates.json'), 'source_sha256':pin(COR/'raw-train-package-candidates.json'), 'reserved_validation_groups_excluded':556, 'already_broader_train_groups_excluded':1296, 'eligible_next_count':len(next_groups), 'packages':next_groups})
    coverage = {'eligible_raw_train_groups':len(all_packages), 'eligible_cpt_train_groups':10163, 'broader_groups':len(names), 'fraction_of_cpt_train_groups':len(names)/10163, 'package_initials':distribution, 'case_sensitive_lexicographic_rank_zero_based_min':min(positions), 'case_sensitive_lexicographic_rank_zero_based_max':max(positions), 'first_name':lexical[min(positions)]['name'], 'last_name':lexical[max(positions)]['name'], 'group_code_tokens_max':max(group_code.values()), 'group_code_fraction_max':max(group_code.values())/sum(group_code.values()), 'selection':'Inventory traversed case-sensitive sorted package names; materializer hash-orders only available groups and round-robins files. Not a global representative sample or whole corpus.', 'next_shard':'Cheap global metadata order is frozen here. Traversal, source hashing/tokenization, TRAIN-wide document dedup against current/validation docs, licensing, budget and fresh dataset admission remain required; do not alter a live dataset.'}
    write('coverage.json', coverage)
    recipe = read(HERE.parent / 'lead/r2-cpt-smoke-c/recipe.json')
    old = copy.deepcopy(recipe)
    recipe.update(id='SFT-11-cpt-broad-a-20260913', output_dir=str(NATIVE/'training/SFT11-CPT-broad-a'), archive_dir=str(NATIVE/'checkpoints/SFT11-CPT-broad-a'), resume_from=None, launch_authorized=False, deadline='2026-09-14T06:15:00Z', max_attempt_seconds=2400, checkpoint={'light_every':250,'full_every':250,'evaluation_steps':[250,750,1250,1714]}, decision_steps=[250])
    recipe['parameters']['max_steps'] = 1714
    recipe['identity']['schedule'] = copy.deepcopy(recipe['parameters'])
    recipe['identity']['policy']['status'] = 'prepared_broader_CPT_root_admission_pending'
    recipe['train_rows'] = {'path':str(COR/'broader-shard-v1-2k/cpt_train.jsonl'),'sha256':schedule['training_rows_sha256']}
    recipe['draw_schedule'] = draw
    recipe['identity']['data'].update(train_rows_sha256=recipe['train_rows']['sha256'], draw_schedule_sha256=draw['sha256'], train_package_ids=sorted(names))
    check(recipe['identity']['data']['validation_package_ids'] == sorted(packages['cpt_validation']), 'validation package identity')
    recipe['inputs'] = [r for r in old['inputs'] if r != old['train_rows'] and r != old['draw_schedule']]
    for r in [recipe['train_rows'], draw, {'path':str(COR/'broader-shard-v1-2k/manifest.json'),'sha256':sha(COR/'broader-shard-v1-2k/manifest.json')}, {'path':str(REG),'sha256':pins[str(REG)]['sha256']}]:
        if r not in recipe['inputs']: recipe['inputs'].append(r)
    check(recipe['identity']['source'] == SOURCE and recipe['resume_from'] is None, 'fresh source binding')
    write('recipe.json', recipe)
    # Production metadata validators only: no full preflight, model imports or weights.
    src = NATIVE/'runner-r2-cpt-v1/snapshots'/SOURCE/'source'
    sys.path[:0] = [str(src/'experiments/training'), str(src/'packages/sepalith/src')]
    import campaign_cpt, campaign_cpt_data
    campaign_cpt._check_identity_contract(recipe, recipe['parameters'])
    campaign_cpt._check_checkpoint_contract(recipe, recipe['parameters'])
    campaign_cpt_data.validate_draw_schedule(read(HERE/'draws-1714.json'), [{'id':r} for r in train_ids], token_rows_sha256=recipe['train_rows']['sha256'], max_steps=1714, effective_batch=16)
    negative_pass = []
    for name, index, value in [('BOS supervised',0,0), ('code masked',1,-100), ('internal EOS supervised',-1,1)]:
        r = copy.deepcopy(first_row); r['labels'][index] = value
        try: check_row(r, docs[r['document_id']], dict(end=0,chunks=0,last=None,finished=False))
        except ValueError: negative_pass.append(name)
        else: raise ValueError('negative accepted '+name)
    bad = read(HERE/'draws-1714.json'); bad['row_ids'][0] = 'not-a-row'
    try: campaign_cpt_data.validate_draw_schedule(bad, [{'id':r} for r in train_ids], token_rows_sha256=recipe['train_rows']['sha256'], max_steps=1714, effective_batch=16)
    except ValueError: negative_pass.append('production rejects unknown draw')
    else: raise ValueError('unknown draw accepted')
    for f in ('campaign_cpt.py','campaign_cpt_data.py','campaign_checkpoint.py','campaign_control.py'):
        pin(src/'experiments/training'/f)
    write('stream-audit.json', {'status':'PASS_CPU_candidate_review_not_training_admission', 'seconds':time.monotonic()-started, 'pins':pins, 'counts':totals, 'production_metadata_checks':['identity','checkpoint milestones','draw schedule'], 'negative_checks':negative_pass, 'mask_checks':'all stored rows, BOS/overlap/internalEOS -100, actual document EOS supervised; offsets and previous token exact', 'prior_source_reconstruction':replay, 'limitations':['No raw NAS reread or retokenization in this independent pass; pinned earlier validator reconstructed all documents to byte SHA256.', 'Model weights not read or hashed; root inherited model pins and owns runtime admission.', 'No complete heldout-file hash inventory or near-duplicate proof.', 'Production full preflight and actual collator/fused-loss gates remain root runtime obligations.']})
    print(json.dumps({'status':'PASS','seconds':time.monotonic()-started,'selected':exposure['selected_totals'],'omitted':exposure['omitted_totals'],'coverage':coverage,'recipe_sha256':sha(HERE/'recipe.json'),'draws_sha256':draw['sha256']}))

if __name__ == '__main__': main()
