import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s1_cpu_scout as s


def fixture_run(path,mismatch=False):
    ids=['a2','b2','a8','b8'];arms=['baseline','ngram-simple@8']
    s.write(path/'prepared.json',dict(arms=arms,trace_ids=ids))
    rows=[dict(arm=arm,trace_id=tr,ctx_class='2k' if tr.endswith('2') else '8k',gen_tps=100 if arm=='baseline' else 120,
          error=None,stop_hit=True,n_prompt_srv=100,tokenized_prompt_tokens=100,matches_baseline=not mismatch,
          host_load_before=[1,1,1],host_load_after=[1,1,1]) for arm in arms for tr in ids]
    (path/'per_request.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows


def test_scout_can_only_nominate_confirmation(tmp_path):
    fixture_run(tmp_path);s.evaluate(tmp_path)
    v=json.loads((tmp_path/'verdict.json').read_text())
    assert len(v['confirmation_candidates'])==2
    assert v['adoption']=='NOT-ASSESSED'


def test_mismatching_scout_does_not_nominate(tmp_path):
    fixture_run(tmp_path,True);s.evaluate(tmp_path)
    assert not json.loads((tmp_path/'verdict.json').read_text())['confirmation_candidates']


def test_partial_scout_is_not_complete(tmp_path):
    rows=fixture_run(tmp_path);(tmp_path/'per_request.jsonl').write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='coverage'):s.evaluate(tmp_path)
