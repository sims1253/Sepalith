"""Verify terminal checkpoint450 and prepare a matched evaluation; never launch CUDA."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

LEAD = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead')
PACKET = Path(__file__).resolve().parent
ACTIVE = LEAD / 'r2-selected330-ordinary-root-v1'
ARCHIVE = Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-production-from-selected-packed330-v1')
MODEL = ARCHIVE / 'full/checkpoint-450'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

def prepare():
    # The live process is never interrupted by this preparation command.
    for pid in (2473526, 2474029):
        assert not Path(f'/proc/{pid}').exists(), f'current training handle still exists: {pid}'
    assert sha(Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-selected330-ordinary-to450-root-v1-host-supervision-b/launch.json')) == 'b535342af582062685ebd4e812bd99f3c5609d68713fa789e8e23c29de458a59'
    assert sha(Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-selected330-ordinary-retry-root-v1/launch.json')) == 'b374dfd4300a99a6527f8f18c1a94d8f549d178e68754d71607b194497e31616'
    terminal = read(Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-selected330-ordinary-to450-root-v1-host-supervision-b/terminal.json'))
    assert terminal['status'] == 'completed' and terminal['child_exit_code'] == 0
    controller = read(LEAD / 'r2-selected330-ordinary-retry-root-v1/terminal.json')
    assert controller['exit_code'] == 0 and controller['status'] == 'commands_complete_requires_root_review'
    recipe_path = ACTIVE / 'runtime-recipe.json'
    assert sha(recipe_path) == 'eff266cdb287e8b099990f265760539e024bc20c14dd0707b6e225abce1a6981'
    recipe = read(recipe_path)
    result = read(ARCHIVE / 'run-result.json')
    assert result['status'] == 'root_admitted_execution_stopped'
    assert result['global_step'] == 450 and result['global_optimizer_step_offset'] == 66
    assert result['initial_cursor'] == 4224 and result['observed_draws'] == 1920
    assert result['last_draw_position'] == 6143
    assert Path(result['terminal_checkpoint']) == MODEL
    assert math.isfinite(result['train_loss'])
    lineage = result['ordinary_canary_resume_lineage']
    assert lineage['source_checkpoint_manifest_sha256'] == '2b35854537588499fb703af2127029ff649e410cbab65374948723273b520951'
    assert lineage['source_arm'] == 'varlen_candidate' and lineage['destination_execution'] == 'ordinary_sdpa_unpacked'
    transition = ACTIVE / 'transition.json'
    assert lineage['transition_admission_sha256'] == sha(transition)
    telemetry = [json.loads(line) for line in (ARCHIVE / 'telemetry.jsonl').open()]
    gradients = [row for row in telemetry if row['event'] == 'pre_optimizer']
    logs = [row for row in telemetry if row['event'] == 'log' and 'loss' in row.get('logs', {})]
    assert [row['next_step'] for row in gradients] == list(range(331, 451))
    assert [row['step'] for row in logs] == list(range(331, 451))
    assert all(row['finite_gradient_tensors'] == row['nonzero_gradient_tensors'] == 381 for row in gradients)
    assert all(math.isfinite(row['logs']['loss']) and math.isfinite(row['logs']['grad_norm']) for row in logs)
    manifest = read(MODEL / 'campaign-manifest.json')
    assert manifest['full'] is True and manifest['checkpoint_kind'] == 'full_weights' and manifest['step'] == 450
    source = LEAD / 'r2-selected330-ordinary-continuation-preparation-v2/source/experiments/training/full_weight_cpt_trainer.py'
    spec = importlib.util.spec_from_file_location('reviewed_cpt_trainer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_source(recipe)
    assert manifest['identity'] == module.identity(recipe)
    assert set(manifest['files']) == {'campaign-state.json','chat_template.jinja','config.json','generation_config.json','model.safetensors','optimizer.pt','rng_state.pth','scheduler.pt','tokenizer.json','tokenizer_config.json','trainer_state.json','training_args.bin'}
    actual = {str(p.relative_to(MODEL)) for p in MODEL.rglob('*') if p.is_file()}
    assert actual == set(manifest['files']) | {'campaign-manifest.json'}
    for name, expected in manifest['files'].items():
        path = MODEL / name
        assert not path.is_symlink() and path.stat().st_size == expected['bytes']
        assert sha(path) == expected['sha256'], name
    state = read(MODEL / 'campaign-state.json')
    assert state['identity'] == manifest['identity'] and state['step'] == 450
    assert state['full'] is True and state['checkpoint_kind'] == 'full_weights'
    sampler = state['sampler']
    assert sampler['cursor'] == sampler['stage_cursor'] == 6144
    assert sampler['global_optimizer_step_offset'] == 66 and sampler['global_step'] == 450
    assert sampler['draw_schedule_sha256'] == recipe['cohort']['draw_schedule']['sha256']
    assert read(MODEL / 'trainer_state.json')['global_step'] == 450
    assert sampler['resume_lineage'] == lineage
    # Checkpoint450 requires this separate matched development evaluation.
    # Verify both native hot and durable payloads before evaluating the hot copy.
    native = Path(recipe['outputs']['trainer']) / 'checkpoint-450'
    assert read(native / 'campaign-manifest.json') == manifest
    assert {str(p.relative_to(native)) for p in native.rglob('*') if p.is_file()} == actual
    for name, expected in manifest['files'].items():
        path = native / name
        assert not path.is_symlink() and path.stat().st_size == expected['bytes']
        assert sha(path) == expected['sha256'], name
    assert sha(native / 'campaign-manifest.json') == sha(MODEL / 'campaign-manifest.json')
    binding = read(LEAD / 'r2-cpt90-long-eval-root-v1/binding.json')
    binding.update(model_path=str(native), checkpoint_step=450,
                   resource_condition='Training terminal and root full checkpoint verification passed; fresh exclusive CUDA guard still required')
    binding.pop('at', None)
    binding['model_files'] = {name: manifest['files'][name]['sha256'] for name in binding['model_files']}
    assert sha(LEAD / 'r2-cpt-long-eval-root-v2/evaluate.py') == binding['runner_sha256']
    command = read(LEAD / 'r2-cpt90-long-eval-root-v1/command.json')
    command[command.index('--binding') + 1] = str(PACKET / 'binding.json')
    command[command.index('--output') + 1] = '/mnt/e/sepalith/campaign-20260915/evaluations/SFT11-cpt450-long-root-v1'
    review = {'checkpoint_manifest_sha256': sha(MODEL / 'campaign-manifest.json'), 'files_verified': len(manifest['files']), 'step': 450, 'cursor': 6144, 'native_and_durable_payloads_verified': True, 'evaluation_pending': True, 'cuda_launched': False}
    for name, value in [('binding.json', binding), ('command.json', command), ('checkpoint-review.json', review)]:
        with (PACKET / name).open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    print(json.dumps(review))

if __name__ == '__main__':
    prepare()
