"""Freeze one-pass ordering and independently audit broader-shard provenance."""
import json,os,random
from collections import Counter
from pathlib import Path
import raw_cpt_v2 as c
HERE=Path(__file__).resolve().parent

def main():
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    folder=HERE/'broader-shard-v1-2k'
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,pin in manifest['artifacts'].items():
        if c.sha(folder/name)!=pin['sha256']:raise ValueError('closed artifact identity mismatch')
    docs=list(map(json.loads,(folder/'documents.jsonl').open()))
    profile=list(map(json.loads,(HERE/'profile-shard-v1/documents.jsonl').open()))
    validation={r['sha256'] for r in profile if r['cpt_partition']=='cpt_validation'}
    initial_train={r['sha256'] for r in profile if r['cpt_partition']=='cpt_train'}
    unique={r['sha256'] for r in docs}
    if len(unique)!=len(docs) or unique & validation:raise ValueError('duplicate or reserved validation document')
    groups=Counter();packages=set();overlap_docs=overlap_tokens=0
    for r in docs:
        if r['split']!='train_group' or r['cpt_partition']!='cpt_train' or c.partition(r['group_id'])!='cpt_train':raise ValueError('split mismatch')
        groups[r['group_id']]+=r['source_code_tokens'];packages.add(r['package'])
        if r['sha256'] in initial_train:overlap_docs+=1;overlap_tokens+=r['source_code_tokens']
    if max(groups.values())>manifest['package_code_token_cap']:raise ValueError('per-group token cap exceeded')
    rows={}
    for row in map(json.loads,(folder/'cpt_train.jsonl').open()):
        if row['row_id'] in rows:raise ValueError('duplicate row ID')
        rows[row['row_id']]={'input_tokens':len(row['input_ids']),'supervised_tokens':row['supervised_tokens']}
    order=list(rows);random.Random(3407).shuffle(order)
    schedule={'schema':1,'seed':3407,'policy':'one fixed shuffled pass without replacement',
        'training_rows_sha256':c.sha(folder/'cpt_train.jsonl'),'row_ids':order,'draw_count':len(order),
        'total_input_tokens':sum(r['input_tokens'] for r in rows.values()),
        'total_supervised_tokens':sum(r['supervised_tokens'] for r in rows.values()),
        'note':'New dataset and sampler identity. A profile-checkpoint continuation requires an explicit reviewed dataset transition.'}
    schedule_path=HERE/'broader-draws-one-pass.json';schedule_path.write_text(json.dumps(schedule,separators=(',',':'))+'\n')
    total=sum(groups.values())
    result={'status':'PASS_CPU_frozen_candidate_pending_root_training_admission',
        'materialization_manifest':str(folder/'manifest.json'),'materialization_manifest_sha256':c.sha(folder/'manifest.json'),
        'counts':manifest['counts'],'actual_packages':len(packages),'actual_groups':len(groups),
        'maximum_group_code_tokens':max(groups.values()),'maximum_group_fraction':max(groups.values())/total,
        'group_code_token_counts':dict(groups),'reserved_validation_documents':len(validation),
        'reserved_validation_overlap':0,'exact_duplicate_retained_documents':0,
        'profile_train_document_overlap':overlap_docs,'profile_train_code_token_overlap':overlap_tokens,
        'new_code_tokens_beyond_profile_train_documents':total-overlap_tokens,
        'draw_schedule':str(schedule_path),'draw_schedule_sha256':c.sha(schedule_path),
        'draw_count':len(order),'source_sha256':c.sha(__file__),
        'limitation':'Distinct whole-document tokens within this bounded corpus. Cross-package near duplicates and exact files absent from available held-out hash metadata are not proven absent.'}
    (HERE/'broader-stage-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='group_code_token_counts'}))

if __name__=='__main__':main()
