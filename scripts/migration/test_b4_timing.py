import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import b4_timing as s2


def fixture_run(path):
    s2.write(path/'prepared.json',dict(formats=['Q8_0','Q4_0'],trace_ids=['a'],reps=1,tier='cpu'))
    rows=[dict(format=fmt,rep=0,trace_id='a',prompt_n=2000,tokenized_prompt_tokens=2000,
          host_load=[1,1,1],raw_response=dict(stop=True,stop_type='word',timings=dict(predicted_n=48)),prompt_ms=100,predicted_ms=ms,wall_ms=ms+105)
          for fmt,ms in [('Q8_0',100),('Q4_0',50)]]
    (path/'per_request.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows


def test_timing_win_does_not_promote_quantization(tmp_path):
    fixture_run(tmp_path);s2.evaluate(tmp_path)
    v=json.loads((tmp_path/'verdict.json').read_text())
    assert v['adoption']=='NOT-ASSESSED'
    e=json.loads((tmp_path/'evaluation.json').read_text())
    assert e['formats']['Q4_0']['speedup_vs_Q8_0']==pytest.approx(4/3)


def test_missing_quantization_row_fails(tmp_path):
    rows=fixture_run(tmp_path)
    (tmp_path/'per_request.jsonl').write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='coverage'):s2.evaluate(tmp_path)


def test_truncated_prompt_is_not_a_fast_result(tmp_path):
    rows=fixture_run(tmp_path);rows[-1]['prompt_n']=1800
    (tmp_path/'per_request.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='request'):s2.evaluate(tmp_path)


def test_bookend_uses_same_frozen_model(tmp_path):
    assert s2.model_path(tmp_path,"Q8_0-bookend")==s2.model_path(tmp_path,"Q8_0")
