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
