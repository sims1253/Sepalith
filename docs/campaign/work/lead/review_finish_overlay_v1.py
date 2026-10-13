"""Independent streaming verification of the shadow TRAIN overlay."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import collections

ROOT = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
DATA = Path('/mnt/e/sepalith/campaign-20260915/data-work')

def sha(data): return hashlib.sha256(data).hexdigest()

def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec); sys.modules[name]=obj; spec.loader.exec_module(obj)
    return obj

def main():
    report=json.load((ROOT/'docs/campaign/receipts/DAT-04-finish-target-overlay-v1-preparation.json').open())
    for pin in report['pinned_inputs'].values():
        assert sha(Path(pin['path']).read_bytes())==pin['sha256']
    protocol=module('overlay_review_protocol',EXEC/'packages/sepalith/src/sepalith/campaign_protocol.py')
    repair=module('overlay_review_repair',ROOT/'docs/campaign/work/finish-boundary-repair-v2/finish_boundary_repair_v2.py')
    scenarios=module('overlay_review_r',Path('/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/scenarios.py'))
    overlays={};digest=hashlib.sha256();statuses=collections.Counter()
    with (ROOT/report['overlay']['path']).open('rb') as f:
        for raw in f:
            digest.update(raw); row=json.loads(raw); row_id=row['key']['original_id']
            assert row_id not in overlays
            assert row['original']['packet_split']=='train_group' and row['original']['family']=='finish_block'
            overlays[row_id]=row; statuses[row['status']]+=1
    assert digest.hexdigest()==report['overlay']['sha256'] and len(overlays)==5000
    good=0;rejected=0;missing=0;digest=hashlib.sha256()
    with (DATA/'DAT-04B-completion-batch.jsonl').open('rb') as f:
        for raw in f:
            digest.update(raw); case=json.loads(raw); row_id=case['row_ref']['row_id']; row=overlays[row_id]
            assert sha(raw)==row['key']['packet_raw_line_sha256']
            if not row['lineage']['matched_train_identity']:
                assert row['lineage']['registry'] is None and row['lineage']['sft'] is None
                assert not row['eligibility']['strict_v2']; missing+=1;continue
            try:
                plan=repair.prepare_repair(case,utf16_to_codepoint_column=protocol.utf16_to_codepoint_column)
            except repair.RepairRejected as error:
                assert error.code=='target_not_lf_terminated'
                assert not row['eligibility']['strict_v2'];rejected+=1;continue
            assert row['eligibility']['strict_v2']
            post=repair.apply_plan(case,plan,utf16_to_codepoint_column=protocol.utf16_to_codepoint_column)
            assert not scenarios.parser.parse(post.encode()).root_node.has_error
            assert plan.repaired_target_text==row['repaired']['target_body_text']
            assert sha(post.encode())==row['repaired']['document_sha256']
            assert sha(plan.repaired_target_text.encode())==row['repaired']['target_body_sha256']
            assert plan.repaired_target_text==row['original']['target_body_text']+'}'
            good+=1
    assert digest.hexdigest()==report['inputs']['converted_packet']['sha256']
    seen=set(); nonfinish=0;digest=hashlib.sha256()
    with (DATA/'SFT-inputs-v1/train-token-rows.jsonl').open('rb') as f:
        for raw in f:
            digest.update(raw); case=json.loads(raw); assert case['split']=='train'
            assert case['id'] not in seen;seen.add(case['id'])
            if case['family']!='finish_block':
                assert case['id'] not in overlays;nonfinish+=1;continue
            row=overlays[case['id']]
            assert sha(raw)==row['lineage']['sft_line_sha256']
            assert case['target_body_text']==row['original']['target_body_text']
    assert digest.hexdigest()==report['inputs']['sft_train_rows']['sha256']
    assert (good,rejected,missing,nonfinish,len(seen))==(4051,238,711,7475,11764)
    result={'status':'independent_overlay_integrity_and_actual_application_accepted','script_sha256':sha(Path(__file__).read_bytes()),'parent_receipt_sha256':sha((ROOT/'docs/campaign/receipts/DAT-04-finish-target-overlay-v1-preparation.json').read_bytes()),'counts':{'repaired_and_parse_verified':good,'strict_rejected_matched_train':rejected,'absent_downstream':missing,'nonfinish_train_untouched':nonfinish,'original_train_total':len(seen)},'overlay_statuses':dict(statuses),'no_data_or_models_replaced':True,'new_stage_admission':'Separate corrected token-row and source/budget review required.'}
    path=ROOT/'docs/campaign/receipts/DAT-04-finish-overlay-lead-review.json'
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
