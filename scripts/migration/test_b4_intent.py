import copy
import io
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_intent as b


def case():
    return dict(id='one', assertion='Do nothing', input=dict(filename='a.R',
        prefix_lines=['x <- 1'], suffix_lines=[], cursor_partial=''))


def test_cache_key_includes_assertion_and_context():
    a=case();other=copy.deepcopy(a);other['assertion']='Rename x'
    assert b.body_bytes(b.judge_body(a, [])) != b.body_bytes(b.judge_body(other, []))
    other=copy.deepcopy(a);other['input']['prefix_lines']=['x <- 2']
    assert b.body_bytes(b.judge_body(a, [])) != b.body_bytes(b.judge_body(other, []))
    other=copy.deepcopy(a);other['arm']='secret-arm'
    assert b.body_bytes(b.judge_body(a, [])) == b.body_bytes(b.judge_body(other, []))


def test_judge_payload_matches_established_protocol(monkeypatch):
    captured=[]
    def respond(request, timeout):
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps(dict(choices=[dict(message=dict(content=json.dumps(dict(score=2,reason='ok'))))])).encode())
    monkeypatch.setenv('ZAI_API_KEY', 'local-test-placeholder')
    monkeypatch.setattr(b.intent.urllib.request, 'urlopen', respond)
    c=case();c['input']['prefix_lines']=['x'*100]*30;c['input']['suffix_lines']=['y'*100]*12
    prediction=['z'*1300]
    b.intent.judge(c,prediction,retries=1)
    assert captured==[b.judge_body(c,prediction)]


@pytest.mark.parametrize('score', [True, 3, None, '2'])
def test_invalid_scores_rejected(score):
    response=dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(dict(score=score,reason='x'))))])
    with pytest.raises(ValueError):b.judge_score(response)


def test_generation_error_cannot_be_abstention():
    row=dict(tokenized_prompt_tokens=10, response=dict(usage=dict(prompt_tokens=10), choices=[dict(finish_reason='error')]))
    with pytest.raises(ValueError):b.validate_generation(row)


def test_failed_calibration_stops_before_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'suite', lambda _: [case()])
    monkeypatch.setattr(b, 'generation_coverage', lambda *_: None)
    (tmp_path/'generation.jsonl').write_text('')
    monkeypatch.setattr(b.intent, 'anchors', lambda _: [('bad', case(), [], 2)])
    calls=[]
    def judge(body, audit):
        calls.append(body)
        return 'hash', dict(score=0, reason='bad')
    monkeypatch.setattr(b, 'call_judge', judge)
    with pytest.raises(ValueError, match='calibration failed'):b.judge(tmp_path,tmp_path)
    assert len(calls)==1


def test_identical_inputs_share_one_score(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'suite', lambda _: [case()])
    monkeypatch.setattr(b, 'generation_coverage', lambda *_: None)
    monkeypatch.setattr(b.intent, 'anchors', lambda _: [])
    rows=[dict(arm=arm,id='one',prediction=[]) for arm in b.ARMS]
    (tmp_path/'generation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    calls=[]
    def judge(body, audit):
        calls.append(body)
        return 'unused', dict(score=2, reason='good')
    monkeypatch.setattr(b, 'call_judge', judge)
    b.judge(tmp_path,tmp_path)
    scored=b.read_rows(tmp_path/'scores.jsonl')
    assert len(calls)==1
    assert [r['cached'] for r in scored]==[False,True,True]
    assert len({r['request_sha256'] for r in scored})==1
