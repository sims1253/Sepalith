"""Verify terminal checkpoint322 and prepare a matched evaluation; never launch CUDA."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

LEAD = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead')
PACKET = Path(__file__).resolve().parent
ACTIVE = LEAD / 'r2-native-cpt194-root-admission-v1'
ARCHIVE = Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-prefix-extension-native-v3-from194')
MODEL = ARCHIVE / 'full/checkpoint-322'

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
    for pid in (2064033, 2068268):
        assert not Path(f'/proc/{pid}').exists(), f'current training handle still exists: {pid}'
    recipe_path = ACTIVE / 'runtime-recipe.json'
    assert sha(recipe_path) == '076b9105b88ce125c406d52db37865e782be217824e79f3e33b9ff0fec27fef1'
    recipe = read(recipe_path)
    result = read(ARCHIVE / 'run-result.json')
    assert result['status'] == 'root_admitted_execution_stopped'
    assert result['global_step'] == 322 and result['global_optimizer_step_offset'] == 66
    assert result['initial_cursor'] == 2048 and result['observed_draws'] == 2048
    assert result['last_draw_position'] == 4095
    assert Path(result['terminal_checkpoint']) == MODEL
    manifest = read(MODEL / 'campaign-manifest.json')
    assert manifest['full'] is True and manifest['checkpoint_kind'] == 'full_weights' and manifest['step'] == 322
    source = LEAD / 'r2-native-cpt194-continuation-preparation-v1/source/experiments/training/full_weight_cpt_trainer.py'
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
    assert state['identity'] == manifest['identity'] and state['step'] == 322
    sampler = state['sampler']
    assert sampler['cursor'] == sampler['stage_cursor'] == 4096
    assert sampler['global_optimizer_step_offset'] == 66 and sampler['global_step'] == 322
    assert sampler['draw_schedule_sha256'] == recipe['cohort']['draw_schedule']['sha256']
    assert read(MODEL / 'trainer_state.json')['global_step'] == 322
    # Checkpoint322 has no in-process evaluation under the unchanged recipe.
    # Verify both native hot and durable payloads before evaluating the hot copy.
    native = Path(recipe['outputs']['trainer']) / 'checkpoint-322'
    assert read(native / 'campaign-manifest.json') == manifest
    assert {str(p.relative_to(native)) for p in native.rglob('*') if p.is_file()} == actual
    for name, expected in manifest['files'].items():
        path = native / name
        assert not path.is_symlink() and path.stat().st_size == expected['bytes']
        assert sha(path) == expected['sha256'], name
    assert sha(native / 'campaign-manifest.json') == sha(MODEL / 'campaign-manifest.json')
    binding = read(LEAD / 'r2-cpt90-long-eval-root-v1/binding.json')
    binding.update(model_path=str(native), checkpoint_step=322,
                   resource_condition='Training terminal and root full checkpoint verification passed; fresh exclusive CUDA guard still required')
    binding.pop('at', None)
    binding['model_files'] = {name: manifest['files'][name]['sha256'] for name in binding['model_files']}
    assert sha(LEAD / 'r2-cpt-long-eval-root-v2/evaluate.py') == binding['runner_sha256']
    command = read(LEAD / 'r2-cpt90-long-eval-root-v1/command.json')
    command[command.index('--binding') + 1] = str(PACKET / 'binding.json')
    command[command.index('--output') + 1] = '/mnt/e/sepalith/campaign-20260915/evaluations/SFT11-cpt322-long-root-v1'
    review = {'checkpoint_manifest_sha256': sha(MODEL / 'campaign-manifest.json'), 'files_verified': len(manifest['files']), 'step': 322, 'cursor': 4096, 'native_and_durable_payloads_verified': True, 'evaluation_pending': True, 'cuda_launched': False}
    for name, value in [('binding.json', binding), ('command.json', command), ('checkpoint-review.json', review)]:
        with (PACKET / name).open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    print(json.dumps(review))

if __name__ == '__main__':
    prepare()
