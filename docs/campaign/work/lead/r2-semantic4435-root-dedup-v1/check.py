import collections
import hashlib
import json
from pathlib import Path

BASE = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
RECEIPT = json.loads((BASE / 'receipts/DAT-10-semantic763-dedup-integration.json').read_text())

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def stream(spec):
    h = hashlib.sha256()
    count = 0
    with Path(spec['path']).open('rb') as file:
        for line in file:
            h.update(line)
            count += 1
            yield json.loads(line)
    assert h.hexdigest() == spec['sha256'], spec['path']
    assert count == spec['rows'], (count, spec['rows'])

def source_keys(value):
    hashes, paths = set(), set()
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                if key in {'source_sha256', 'source_snapshot_sha256', 'before_snapshot_sha256', 'after_snapshot_sha256', 'post_edit_snapshot_sha256'}:
                    hashes.add(item)
                if key in {'source_path', 'source_snapshot_path', 'path', 'event_path'}:
                    path = item.replace('\\', '/')
                    paths.add('R/' + path.split('/R/', 1)[1] if '/R/' in path else path)
            elif isinstance(item, dict):
                nested_hashes, nested_paths = source_keys(item)
                hashes.update(nested_hashes)
                paths.update(nested_paths)
    return hashes, paths

def compact(row):
    return {'id': row['id'], 'pair': digest(row['prompt_text'] + '\0' + row['target_text']), 'prompt': digest(row['prompt_text']), 'target': digest(row['target_body_text']), 'package': row['package_id']}


def keys(row):
    yield 'id', row['id']
    yield 'pair', row['pair']
    yield 'prompt', row['prompt']
    for sha in row['sources'][0]:
        yield 'source_target', (sha, row['target'])
    for path in row['sources'][1]:
        yield 'path_target', (row['package'], path, row['target'])



from datetime import datetime, timezone

def file_record(root,name):
    manifest=json.loads((root/'manifest.json').read_text())
    r=dict(manifest['outputs'][name]);r['path']=str(root/r['path']);return r

specs=[('original15006',RECEIPT['inputs']['authoritative_15006'],RECEIPT['inputs']['authoritative_context']),('candidate616',RECEIPT['outputs']['candidate_tokenrows'],RECEIPT['outputs']['candidate_provenance'])]
for label,root in [('candidate133',Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic-provider-integration-v1/final-01')),('candidate4435',Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4551-provider-materialization-v1/final-01'))]:
    specs.append((label,file_record(root,'candidate-tokenrows.jsonl'),file_record(root,'candidate-provenance.jsonl')))
indexes={name:collections.defaultdict(set) for name in ['id','pair','prompt','source_target','path_target']}
findings=[];counts={};target_by_id={}
for label,rs,ps in specs:
    provenance={r['row_id']:r for r in stream(ps)};seen=set()
    for original in stream(rs):
        row=compact(original);rid=row['id'];assert rid not in seen;seen.add(rid);row['sources']=source_keys(provenance[rid]['source_identity'])
        if label=='candidate4435':
            for kind,key in keys(row):
                matches=indexes[kind].get(key,set())
                if kind=='prompt':matches={m for m in matches if target_by_id[m]!=row['target']}
                if matches:findings.append({'id':rid,'kind':kind,'matches':sorted(matches)})
        for kind,key in keys(row):indexes[kind][key].add(rid)
        target_by_id[rid]=row['target']
    assert seen==set(provenance);counts[label]=len(seen)
result={'task':'DAT-10','at':datetime.now(timezone.utc).isoformat(),'rows':counts,'union_rows':sum(counts.values()),'new_candidate_findings':findings,'all8input_hashes_and_counts_verified':True,'internal_new_candidates_included_in_indexes':True,'training_admitted':False}
print(json.dumps(result,sort_keys=True))
