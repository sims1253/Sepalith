import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import b4_quant_quality as q


def fixture(path):
    p=dict(arms=['Q8_0','Q4_K_M'],case_ids=[['scenario','s'],['noop','n']]);q.write(path/'prepared.json',p)
    rows=[]
    for arm in p['arms']:
        for kind,identity in p['case_ids']:
            rows.append(dict(arm=arm,kind=kind,id=identity,served_prompt_tokens=10,tokenized_prompt_tokens=10,
                exact=True,valid_pass=True,expectation='no_proposal',proposal=False,response=dict(choices=[dict(finish_reason='stop')])))
    (path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));return rows


def test_paired_quality_does_not_promote_without_intent(tmp_path):
    fixture(tmp_path);q.evaluate(tmp_path)
    assert json.loads((tmp_path/'verdict.json').read_text())['adoption']=='NOT-ASSESSED'
    assert json.loads((tmp_path/'evaluation.json').read_text())['paired']['Q4_K_M']['scenario_exact_within_1pp_observed']


def test_exact_and_valid_margins_are_separate(tmp_path):
    rows=fixture(tmp_path);rows[2]['valid_pass']=False
    (tmp_path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    q.evaluate(tmp_path)
    paired=json.loads((tmp_path/'evaluation.json').read_text())['paired']['Q4_K_M']
    assert paired['scenario_exact_within_1pp_observed']
    assert not paired['scenario_valid_within_1pp_observed']


def test_error_cannot_be_a_noop_success(tmp_path):
    rows=fixture(tmp_path);rows[-1]['error']='timeout'
    (tmp_path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Invalid quality'):q.evaluate(tmp_path)


def test_missing_candidate_case_fails(tmp_path):
    rows=fixture(tmp_path);(tmp_path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
    with pytest.raises(ValueError,match='coverage'):q.evaluate(tmp_path)


def test_exact_mcnemar_retains_direction():
    result=q.paired([True]*4,[False]*4)
    assert result['losses']==4 and result['gains']==0
    assert result['delta_pp']==-100 and result['mcnemar_exact_two_sided']==pytest.approx(.125)
