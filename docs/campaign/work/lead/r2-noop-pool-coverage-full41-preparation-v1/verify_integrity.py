#!/usr/bin/env python3
"""Verify full-census joins and the provenance envelope without rendering inputs."""
import collections, hashlib, json
from pathlib import Path

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PACKET=PLAN/'docs/campaign/work/lead/r2-noop-pool-coverage-full41-preparation-v1'
REPLAY=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01/shards')
MERGE=PLAN/'docs/campaign/work/lead/r2-sourcewalk41-root-merge-v1/root-result.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def req(value,message):
    if not value: raise ValueError(message)

coverage=json.loads((PACKET/'result.json').read_text())
merge=json.loads(MERGE.read_text())
candidates=[json.loads(line) for line in (PACKET/'candidate-ids.jsonl').open()]
req(coverage['completed_shards']==41 and coverage['pending_shards']==[] and coverage['partial'] is False,'coverage is not terminal41')
req(len(candidates)==len({x['row_id'] for x in candidates})==4227,'candidate IDs are incomplete or duplicated')
req(set(x['status'] for x in candidates)=={'provenance_supported_candidate_root_review_required','hold_independent_provenance_failure'},'candidate statuses differ')
req({x['shard'] for x in candidates} <= set(range(41)),'candidate shard differs')
ours={x['shard']:x['sha256'] for x in coverage['receipts']};theirs={x['shard']:x['sha256'] for x in merge['receipt_pins']}
req(ours==theirs and len(ours)==41,'receipt pins differ from completed root merge')

observed={};counts=collections.Counter();hold_causes=collections.Counter()
for shard in range(41):
    receipt=json.loads((REPLAY/f'shard-{shard:04d}/receipt.json').read_text())
    req(receipt['status']=='complete' and receipt['shard']==shard,'receipt differs')
    for raw in (REPLAY/f'shard-{shard:04d}/ledger.jsonl').open('rb'):
        row=json.loads(raw)
        if row['family']!='no_op': continue
        rid=row['row_id'];req(rid not in observed and row['shard']==shard,'ledger no-op ID/shard differs');observed[rid]=row;counts[row['status']]+=1
        common=(row['global_train'] is True and row['global_group_source_membership'] is True and row['protected_disjoint'] is True and row['protected_partition']=='cpt_train' and row['source_line_group_join'] is True and row['source_stat_stable'] is True and row['description_stat_stable'] is True and row['source_parse_ok'] is True and row['strict_protocol_ok'] is True and row['source_or_target_text_written'] is False and row['semantic_analyzer']=='not_applicable_noop')
        if row['status']=='provenance_supported_candidate_root_review_required':
            geometry=row['noop_geometry'];req(common and row['license_decision']['ok'] is True and geometry['ok'] is True and geometry['geometry_supported'] is True and geometry['supported_kind'] is True and geometry['typed_positions'] is True and geometry['unchanged_target'] is True and geometry['window_occurrences']==1,'supported provenance envelope differs')
        else:
            req(row['status']=='hold_independent_provenance_failure','unexpected no-op status')
            for key,value in [('common',common),('license',row['license_decision']['ok']),('geometry',row['noop_geometry']['ok'])]:
                if not value: hold_causes[key]+=1
req(set(observed)=={x['row_id'] for x in candidates},'candidate-to-ledger join differs')
req(counts==collections.Counter({'provenance_supported_candidate_root_review_required':4106,'hold_independent_provenance_failure':121}),'supported/hold denominator differs')
result={'schema':'sepalith.dat10.noop-full41-integrity.v1','status':'pass_review_only','rows':4227,'unique_row_ids':4227,'shards':41,'supported':4106,'holds':121,'hold_failed_predicate_counts':dict(hold_causes),'root_merge_result_sha256':sha(MERGE),'receipt_pins_match_root_merge':True,'supported_envelope_verified':['TRAIN group membership','protected-set disjointness','stable source and DESCRIPTION','source parse and strict protocol','reviewed license','typed unique source-backed no-op geometry','target text absent from replay evidence'],'full_prediction_inputs_rendered':False,'training_admitted':False}
(PACKET/'integrity-result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
