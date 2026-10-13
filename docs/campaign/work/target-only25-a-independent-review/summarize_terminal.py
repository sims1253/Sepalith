"""Summarize saved, completed host evidence without contacting the host."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
NATIVE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
LEAD = CAMPAIGN / 'work/lead/target-only25-a'
HOST = NATIVE / 'training/SFT-target-only25-a-host-supervision'


def read(path):
    return json.loads(path.read_text())


def pin(path):
    content = path.read_bytes()
    return {'path': str(path), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


def main():
    supervisor = read(LEAD / 'supervisor-terminal.json')
    host = read(HOST / 'terminal.json')
    assert supervisor['result'] == '64e57f7d0b8840a7ba19a03006e00799'
    assert host['status'] == 'completed' and host['child_exit_code'] == 0
    assert host['reason'] is None
    samples = [json.loads(line) for line in (HOST / 'host-memory.jsonl').read_text().splitlines()]
    assert len(samples) == 43
    recipe = read(LEAD / 'recipe.json')
    tokenizer = pin(Path(recipe['model_path']) / 'tokenizer.json')
    assert tokenizer['sha256'] == recipe['identity']['tokenizer']['tokenizer_json_sha256']
    review = read(HERE / 'review.json')
    ceiling = read(CAMPAIGN / 'receipts/SFT-06-target-only25-a-lead-decision.json')['budget']['existing_unused_seconds']
    result = {
        'status': 'completed_evidence_reviewed',
        'at': datetime.now(timezone.utc).isoformat(),
        'host_terminal': host,
        'supervisor_terminal': supervisor,
        'supervisor_result_is_attempt_id_not_status': True,
        'timing': {
            'units': 'seconds',
            'optimizer_updates_sum': review['telemetry']['update_seconds_sum'],
            'trainer_runtime_including_development_evaluation': 576.036,
            'host_guard_elapsed': host['seconds'],
            'outer_supervisor_elapsed': supervisor['seconds'],
            'previous_unallocated_ceiling': ceiling,
            'arithmetic_remaining_using_outer_supervisor': ceiling - supervisor['seconds'],
            'authoritative_budget_charge': 'root decision; distinct scopes must not be added together',
            'not_editor_or_native_inference_latency': True,
        },
        'saved_host_samples': {
            'count': len(samples),
            'first_utc': samples[0]['At'],
            'last_utc': samples[-1]['At'],
            'minimum_available_mbytes': min(r['AvailableMBytes'] for r in samples),
            'maximum_committed_bytes': max(r['CommittedBytes'] for r in samples),
            'commit_limits_bytes': sorted(set(r['CommitLimit'] for r in samples)),
            'maximum_page_reads_per_second': max(r['PageReadsPersec'] for r in samples),
            'maximum_pages_input_per_second': max(r['PagesInputPersec'] for r in samples),
            'maximum_pages_output_per_second': max(r['PagesOutputPersec'] for r in samples),
            'driver_events_count': sum(len(r['DriverEvents']) for r in samples),
            'scope': 'saved polling samples only; no continuous host guarantee or worker PID release proof',
        },
        'tokenizer_json': tokenizer,
        'pins': [pin(p) for p in [LEAD / 'supervisor-terminal.json', HOST / 'terminal.json',
            HOST / 'launch.json', HOST / 'preflight.json', HOST / 'host-memory.jsonl',
            HOST / 'post-load-cache-release.json', HOST / 'process.log',
            LEAD / 'root-input-verification.json', LEAD / 'root-preflight.json',
            CAMPAIGN / 'receipts/SFT-06-target-only25-a-lead-decision.json']],
        'independent_process_release_check': False,
        'root_owns_process_release_and_acceptance': True,
    }
    (HERE / 'host-terminal-review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'timing': result['timing'], 'host': result['saved_host_samples']}, indent=2))


if __name__ == '__main__':
    main()
