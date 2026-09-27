"""Verify terminal checkpoint194 and prepare a matched evaluation; never launch CUDA."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

LEAD = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead')
PACKET = Path(__file__).resolve().parent
ACTIVE = LEAD / 'r2-cpt90-prefix-extension-root-v2'
ARCHIVE = Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-prefix-extension-v1')
MODEL = ARCHIVE / 'full/checkpoint-194'

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
    for pid in (1831562, 1831597):
        assert not Path(f'/proc/{pid}').exists(), f'current training handle still exists: {pid}'
    recipe_path = ACTIVE / 'bound-recipe.json'
    assert sha(recipe_path) == '579269f0f203ae1eed9cdb96af0f535f19c5953e2cf3e19c52615b7545e10f70'
    recipe = read(recipe_path)
    result = read(ARCHIVE / 'run-result.json')
    assert result['status'] == 'mandatory_milestone_stopped'
    assert result['global_step'] == 194 and result['global_optimizer_step_offset'] == 66
    assert result['initial_cursor'] == 384 and result['observed_draws'] == 1664
    assert result['last_draw_position'] == 2047
    assert Path(result['terminal_checkpoint']) == MODEL
    manifest = read(MODEL / 'campaign-manifest.json')
    assert manifest['full'] is True and manifest['checkpoint_kind'] == 'full_weights' and manifest['step'] == 194
    source = ACTIVE / 'source/experiments/training/full_weight_cpt_trainer.py'
    spec = importlib.util.spec_from_file_location('reviewed_cpt_trainer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_source(recipe)
    assert manifest['identity'] == module.identity(recipe)
    assert {'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'model.safetensors', 'campaign-state.json', 'trainer_state.json'} <= set(manifest['files'])
    actual = {str(p.relative_to(MODEL)) for p in MODEL.rglob('*') if p.is_file()}
    assert actual == set(manifest['files']) | {'campaign-manifest.json'}
    for name, expected in manifest['files'].items():
        path = MODEL / name
        assert not path.is_symlink() and path.stat().st_size == expected['bytes']
        assert sha(path) == expected['sha256'], name
    state = read(MODEL / 'campaign-state.json')
    assert state['identity'] == manifest['identity'] and state['step'] == 194
    sampler = state['sampler']
    assert sampler['cursor'] == sampler['stage_cursor'] == 2048
    assert sampler['global_optimizer_step_offset'] == 66 and sampler['global_step'] == 194
    assert sampler['draw_schedule_sha256'] == recipe['cohort']['draw_schedule']['sha256']
    assert read(MODEL / 'trainer_state.json')['global_step'] == 194
    evaluation = read(ARCHIVE / 'evaluations/step-194.json')
    rows = evaluation['row_metrics']
    assert len(rows) == len({r['row_id'] for r in rows}) == 499
    previous = read(Path('/mnt/e/sepalith/campaign-20260915/evaluations/SFT11-cpt90-long-root-v1/anchor2k.json'))
    assert {r['row_id']: r['loss_tokens'] for r in rows} == {r['row_id']: r['loss_tokens'] for r in previous['row_metrics']}
    assert all(math.isfinite(r['loss_sum']) and r['loss_tokens'] > 0 for r in rows)
    tokens = sum(r['loss_tokens'] for r in rows)
    nll = sum(r['loss_sum'] for r in rows) / tokens
    assert tokens == evaluation['denominators']['validation_loss_tokens'] == 661360
    assert abs(nll - evaluation['metrics']['mean_causal_nll']) < 1e-12
    binding = read(LEAD / 'r2-cpt90-long-eval-root-v1/binding.json')
    binding.update(model_path=str(MODEL), checkpoint_step=194,
                   resource_condition='Training terminal and root full checkpoint verification passed; fresh exclusive CUDA guard still required')
    binding.pop('at', None)
    binding['model_files'] = {name: manifest['files'][name]['sha256'] for name in binding['model_files']}
    assert sha(LEAD / 'r2-cpt-long-eval-root-v2/evaluate.py') == binding['runner_sha256']
    command = read(LEAD / 'r2-cpt90-long-eval-root-v1/command.json')
    command[command.index('--binding') + 1] = str(PACKET / 'binding.json')
    command[command.index('--output') + 1] = '/mnt/e/sepalith/campaign-20260915/evaluations/SFT11-cpt194-long-root-v1'
    review = {'checkpoint_manifest_sha256': sha(MODEL / 'campaign-manifest.json'), 'files_verified': len(manifest['files']), 'step': 194, 'cursor': 2048, 'in_process_anchor_nll': nll, 'anchor_ids_and_denominators_match_checkpoint90': True, 'cuda_launched': False}
    for name, value in [('binding.json', binding), ('command.json', command), ('checkpoint-review.json', review)]:
        with (PACKET / name).open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    print(json.dumps(review))

if __name__ == '__main__':
    prepare()
