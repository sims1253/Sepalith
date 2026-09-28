#!/usr/bin/env python3
"""Inspect named corrected TRAIN sources and exercise real binding/reward code."""
import hashlib,json,os,sys,time
from collections import Counter,defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
SOURCE=Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source')
sys.path.insert(0,str(SOURCE/'experiments/training'))
from campaign_rl_data import _validate_row,_validate_sidecar_record,_validate_semantics,RLTrainRecord
from applied_r_reward import bind_train_records,AppliedDocumentExactReward,BindingError
from pinned_r_parser import create_r_parser
P=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/corrected-rl-admission-audit-v1/candidate-data')
EXPECTED={'eligible-train-rows.jsonl':'7e9cf35e8ecbf1af151df78bfc42c07d7b1d24f6ab82ec47770463d967fe9465','context-sidecar.jsonl':'265b80762efc9544230f1aba09e492906760e98a34431593fdeb4813d6e169ac'}
pins=[]
def rows(name):
    h=hashlib.sha256();count=0
    with (P/name).open('rb') as f:
        for line in f:h.update(line);count+=1;yield json.loads(line)
    assert h.hexdigest()==EXPECTED[name]
    pins.append({'path':str(P/name),'sha256':h.hexdigest(),'bytes':(P/name).stat().st_size,'rows':count})
started=time.monotonic();table={}
for row in rows('eligible-train-rows.jsonl'):
    _validate_row(row);assert row['id'] not in table;table[row['id']]=row
parse_r=create_r_parser();full=[];missing=[];families=defaultdict(Counter)
for sidecar in rows('context-sidecar.jsonl'):
    row=table[sidecar['row_id']];_,context,capture,identity,geometry=_validate_sidecar_record(sidecar,row);_validate_semantics(row,context)
    record=RLTrainRecord(row=row,context=context,capture=capture,source_identity=identity,selection_geometry=geometry)
    families[row['family']]['validated_rows']+=1
    if isinstance(identity['source_provenance'].get('selection_source',{}).get('document_text'),str):
        full.append(record);families[row['family']]['full_bound_document']+=1
    else:missing.append(record);families[row['family']]['missing_full_document']+=1
assert len(full)==2222 and len(missing)==6024
bindings=bind_train_records(full,parse_r)
try:bind_train_records([missing[0]],parse_r)
except BindingError as error:missing_rejection=str(error)
else:raise AssertionError('missing document incorrectly admitted')
control={'text':''};reward=AppliedDocumentExactReward(bindings=bindings,parse_r=parse_r,decoder=lambda ids,**kw:control['text'])
probes=[]
for record in full[:12]:
 row=record.row
 for mode,body in [('corrected_gold',row['target_body_text']),('omit_corrected_outer_brace',row['target_body_text'][:-1])]:
    assert row['target_body_text'].endswith('}')
    control['text']=body+'\n>>>>>>> UPDATED'
    score,result=reward.score_one(record.context,row['target_operation'],row['target_body_text'],[100,1],row_id=row['id'],family=row['family'],package_id=row['package_id'])
    if mode=='corrected_gold':assert score==1.2 and result['applied_r_parse']
    else:assert score==0 and not result['applied_r_parse']
    probes.append({'row_id':row['id'],'mode':mode,'reward':score,'legacy_reward_diagnostic':result['legacy_reward_diagnostic'],'applied_r_parse':result['applied_r_parse'],'semantic_status':result['semantic_status']})
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
report={'status':'passed','scope':'existing corrected TRAIN only; no training data emitted','rows_protocol_sidecar_semantics_verified':len(table),'families':families,'gold_full_applied_R_parse_valid':len(bindings),'missing_document_rows':len(missing),'missing_document_rejection':missing_rejection,'old_full_pool_admissible_to_candidate':False,'candidate_subset_is_not_an_admitted_training_pool':True,'decoder_control_probes':probes,'decoder_control_scope':'actual reward and parser; explicit synthetic decoder control over TRAIN source targets, not model generation or tokenizer proof','parser_identity':parse_r.identity,'pins':pins,'seconds':time.monotonic()-started}
(Path(__file__).parent/'train-binding-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','rows_protocol_sidecar_semantics_verified','gold_full_applied_R_parse_valid','missing_document_rows','seconds']}))
