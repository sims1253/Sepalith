"""Pure identity checks; importing this module never reads a checkpoint."""
import hashlib
from pathlib import Path
import re
import shutil


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_selection(selection, recipe, spec):
    require(selection.get('schema') == 1, 'selection schema')
    require(selection.get('decision') == 'selected_for_conditional_export', 'root selection required')
    require(selection.get('producer_terminal') is True, 'producer must be terminal before checkpoint access')
    require(selection.get('development_gate_passed') is True, 'root development gate required')
    require(selection.get('identity') == recipe['identity'] == spec['expected_identity'], 'exact training identity')
    require(selection.get('checkpoint') == spec['checkpoint'], 'checkpoint path')
    require(selection.get('step') == 25 and type(selection['step']) is int, 'step 25 required')
    require(selection.get('source_cursor') == 400 and type(selection['source_cursor']) is int, 'source cursor 400 required')
    require(selection.get('recipe_sha256') == spec['training_recipe']['sha256'], 'training recipe pin')
    for field in ('checkpoint_manifest_sha256', 'terminal_receipt_sha256', 'development_receipt_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', str(selection.get(field))) is not None, field)
    require(recipe['model_path'] == spec['base'], 'theta0 base path')
    require(recipe.get('resume_from') is None, 'fresh LoRA must not be an older resumed adapter')


def validate_checkpoint_metadata(manifest, state, trainer, identity):
    for item in (manifest, state):
        require(item.get('full') is True and item.get('step') == 25, 'full checkpoint 25 required')
        require(item.get('identity') == identity, 'checkpoint identity')
    require(trainer.get('global_step') == 25, 'trainer step 25 required')
    sampler = state.get('sampler', {})
    require(sampler.get('method') == 'frozen sequential draw schedule', 'sampler method')
    require(type(sampler.get('consumed_draws')) is int and sampler['consumed_draws'] == 400, 'observed source cursor 400')
    require(sampler.get('schedule_sha256') == identity['data']['draw_schedule_sha256'], 'sampler schedule pin')
    require(sampler.get('split_id') == identity['data']['split_id'], 'sampler split pin')


def copy_tokenizer_contract(base, destination, pins):
    """Undo framework JSON normalization with the two original pinned files."""
    base, destination = Path(base), Path(destination)
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        require(sha256(base / name) == pins[name], 'base tokenizer bytes: ' + name)
        shutil.copyfile(base / name, destination / name)
        require(sha256(destination / name) == pins[name], 'export tokenizer bytes: ' + name)
