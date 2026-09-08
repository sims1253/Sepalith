import ast
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import v1a_b4 as v
import prediction_parser


def test_strict_request_failure_cannot_be_silently_skipped(monkeypatch):
    def fail(*args,**kwargs):raise v.RequestFailure('HTTP failure')
    monkeypatch.setattr(v.judge_loop,'complete',fail)
    with pytest.raises(v.RequestFailure):
        v.judge_loop.run_trajectory(dict(key='a',points=[dict(prompt='x')]),18473)


def test_parser_is_unchanged_from_original_source():
    root=Path(__file__).resolve().parents[2]
    def function(path):
        tree=ast.parse(path.read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_prediction'),include_attributes=False)
    assert function(root/'experiments/eval/eval_noop_fp.py')==function(root/'experiments/eval/prediction_parser.py')


def test_missing_episode_is_not_a_valid_baseline(tmp_path):
    v.write(tmp_path/'prepared.json',dict(episodes=[dict(key='a',variant=1,points=1)]))
    (tmp_path/'episodes.jsonl').write_text('');(tmp_path/'requests.jsonl').write_text('')
    with pytest.raises(ValueError,match='episode coverage'):v.evaluate(tmp_path)


def test_complete_episode_uses_existing_simulator_scores(tmp_path,monkeypatch):
    monkeypatch.setattr(v.judge_loop,'complete',lambda *args,**kwargs:'x\n>>>>>>> UPDATED')
    ep=dict(key='a',variant=1,points=[dict(prompt='<[fim-prefix]>f.R\n',label='typing',gt='x',ctx='typing',t_ms=100)])
    result=v.judge_loop.run_trajectory(ep,18473)
    v.write(tmp_path/'prepared.json',dict(episodes=[dict(key='a',variant=1,points=1)]))
    (tmp_path/'episodes.jsonl').write_text(json.dumps(result)+'\n')
    req=dict(episode=0,key='a',variant=1,point=0,served_prompt_tokens=10,tokenized_prompt_tokens=10,response=dict(choices=[dict(finish_reason='stop')]))
    (tmp_path/'requests.jsonl').write_text(json.dumps(req)+'\n')
    v.evaluate(tmp_path)
    assert json.loads((tmp_path/'evaluation.json').read_text())['accepted']==1
    assert json.loads((tmp_path/'verdict.json').read_text())['adoption']=='NOT-ASSESSED'


def test_context_filter_excludes_whole_episode_and_preserves_order():
    import hashlib
    chosen=[dict(key=str(i),points=[dict(prompt='x'),dict(prompt='y')]) for i in range(3)]
    audit=dict(candidate_count=3,ctx=32768,history_reserve=4096,max_tokens=160,margin=16,points=[
        dict(episode=i,point=j,key=str(i),variant=1,tokens=40000 if (i,j)==(1,1) else 10,
             prompt_sha256=hashlib.sha256(p['prompt'].encode()).hexdigest())
        for i,e in enumerate(chosen) for j,p in enumerate(e['points'])])
    kept,excluded=v.filter_context(chosen,audit)
    assert [e['key'] for e in kept]==['0','2']
    assert excluded[0]['points']==2 and excluded[0]['oversized_points']==[dict(point=1,tokens=40000)]
    audit['points'][0]['prompt_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):v.filter_context(chosen,audit)


def test_incomplete_context_audit_cannot_select_cohort():
    chosen=[dict(key='x',points=[dict(prompt='x')])]
    audit=dict(candidate_count=1,ctx=32768,history_reserve=4096,max_tokens=160,margin=16,points=[])
    with pytest.raises(ValueError,match='Incomplete'):v.filter_context(chosen,audit)
