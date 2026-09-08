import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_gpu as s1


def test_gpu_gate_requires_target_and_draft_offload():
    s1.check_offload('offloaded 25/25 layers')
    with pytest.raises(ValueError):
        s1.check_offload('offloaded 25/25 layers', draft=True)
    with pytest.raises(ValueError):
        s1.check_offload('offloaded 25/25 layers\noffloaded 0/25 layers', draft=True)
    s1.check_offload('offloaded 25/25 layers\noffloaded 25/25 layers', draft=True)


def test_warm_error_cannot_disappear_from_summary():
    response = SimpleNamespace(error='timeout', stop_hit=False, ttft_ms=None)
    with pytest.raises(ValueError):
        s1.validate_response(response)


def fixture_run(tmp_path, mismatch=False):
    s1.write(tmp_path / 'prepared.json', dict(arms=['baseline', 'ngram-simple@2'], trace_ids=['a', 'b'], reps=1))
    rows=[]
    for arm in ['baseline', 'ngram-simple@2']:
        for trace in ['a', 'b']:
            rows.append(dict(arm=arm, trace_id=trace, rep=0, ctx_class='2k', error=None,
                warm_error=None, stop_hit=True, warm_stop_hit=True, n_prompt_srv=10,
                tokenized_prompt_tokens=10, gen_tps=100 if arm=='baseline' else 150,
                draft_n=10, gen_text='x', warm_gen_text='x', matches_baseline=not mismatch,
                warm_matches_baseline=not mismatch))
    (tmp_path / 'per_request.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows


def test_fast_mismatching_arm_is_not_winner(tmp_path):
    fixture_run(tmp_path, mismatch=True)
    s1.evaluate(tmp_path)
    assert json.loads((tmp_path / 'verdict.json').read_text())['candidate_winners']==[]


def test_complete_lossless_arm_can_be_candidate(tmp_path):
    fixture_run(tmp_path)
    s1.evaluate(tmp_path)
    assert json.loads((tmp_path / 'verdict.json').read_text())['candidate_winners']==['ngram-simple@2|2k']


def test_missing_measurement_fails_coverage(tmp_path):
    rows=fixture_run(tmp_path)
    (tmp_path / 'per_request.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
    with pytest.raises(ValueError, match='coverage'):
        s1.evaluate(tmp_path)


def test_deadline_is_not_swallowed_by_legacy_exception_handler():
    assert not issubclass(s1.DeadlineExceeded, Exception)


def test_gpu_server_has_one_unambiguous_offload_flag():
    cmd = s1.GpuServer('/model', 18471, [], server='/server').cmd()
    assert cmd.count('-ngl') == 1
    assert cmd[cmd.index('-ngl')+1] == '99'


def test_serving_recount_can_differ_from_hf_metadata_without_truncation():
    # Actual frozen trace: HF metadata 2053; serving tokenizer and no-escape CLI 2057.
    assert s1.check_prompt_count(2057, 10240, baseline_count=2057) is None
    with pytest.raises(ValueError, match='Cross-arm'):
        s1.check_prompt_count(2057, 10240, baseline_count=2053)
    with pytest.raises(ValueError, match='context'):
        s1.check_prompt_count(10200, 10240)


def test_ngram_diagnostic_rejects_ineffective_control(tmp_path, monkeypatch):
    monkeypatch.setenv('S1_RUNTIME', str(tmp_path))
    monkeypatch.setattr(s1, 'selected', lambda _: [dict(trace_id='2038817292-98c44c61-2k', prompt='x')])
    stopped = []
    class Server:
        port = 1
        def __init__(self, *args, **kwargs):
            self.log_path = kwargs['log_path']
        def start(self, **kwargs):
            self.log_path.write_text('offloaded 25/25 layers')
        def stop(self):
            stopped.append(self.log_path)
    monkeypatch.setattr(s1, 'GpuServer', Server)
    monkeypatch.setattr(s1, 'validate_response', lambda _: None)
    monkeypatch.setattr(s1.bench, 'stream_completion', lambda *a, **k: SimpleNamespace(timings={'draft_n': 95}, text='x'))
    with pytest.raises(ValueError, match='control diagnostic failed'):
        s1.check_ngram_control(tmp_path, tmp_path)
    assert len(stopped) == 3
    assert not (tmp_path / 'ngram-control.json').exists()
    assert (tmp_path / 'control-M48.json').exists()
