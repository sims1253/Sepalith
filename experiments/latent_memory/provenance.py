"""Version pins, prelaunch freeze, and append-only compute receipts."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from .data import digest, make_dataset


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def backend_identity():
    import transformers.models.qwen3_5.modeling_qwen3_5 as implementation
    import transformers.cache_utils as cache
    import transformers.masking_utils as masks
    import transformers.conversion_mapping as conversion
    if (implementation.is_fast_path_available or implementation.FusedRMSNormGated is not None
            or implementation.causal_conv1d_fn is not None or implementation.chunk_gated_delta_rule is not None):
        raise ValueError('Frozen reference requires pure PyTorch GDN; optional fused backend found')
    return {'versions': {name: importlib.metadata.version(name)
                         for name in ['torch', 'transformers', 'tokenizers', 'safetensors']},
            'python': platform.python_version(),
            'implementation_sha256': file_hash(implementation.__file__),
            'cache_sha256': file_hash(cache.__file__), 'masks_sha256': file_hash(masks.__file__),
            'conversion_sha256': file_hash(conversion.__file__),
            'gdn_backend': 'torch_chunk_gated_delta_rule', 'recurrent_state_dtype': 'float32',
            'attention': 'eager', 'persistent_cache': False}


def inspect_decoder(checkpoint):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from .model import target_names
    root = Path(checkpoint)
    config = Qwen3_5TextConfig.from_pretrained(root, local_files_only=True)
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(config)
    with safe_open(root / 'model.safetensors', framework='pt', device='cpu') as weights:
        targets = {name: list(model.get_submodule(name).weight.shape) for name in target_names(config)}
        saved_names = {}
        for name, shape in targets.items():
            saved_name = name.replace('model.', 'model.language_model.', 1) + '.weight'
            saved_names[name] = saved_name
            if list(weights.get_slice(saved_name).get_shape()) != shape:
                raise ValueError(f'Checkpoint projection mismatch: {name}')
    return {'config': config.to_dict(), 'config_sha256': file_hash(root / 'config.json'),
            'backend': backend_identity(), 'targets': targets, 'saved_tensor_names': saved_names,
            'loader_mapping': '^model.language_model -> model (Transformers qwen3_5_text)',
            'rank16_trainable_parameters': sum(16 * sum(shape) for shape in targets.values()),
            'base_parameters': sum(p.numel() for p in model.parameters()),
            'checkpoint_path': str(root.resolve())}


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'recipe.json',
             root / 'packages/sepalith/src/sepalith/memory.py', Path(__file__).parent / 'requirements.lock']
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def prepare(recipe_path, checkpoint, output, pin_weights=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads(Path(recipe_path).read_text())
    data = make_dataset(recipe['seed'], recipe['train_packages'], recipe['eval_packages'])
    write_json(output / 'dataset.json', data)
    write_json(output / 'recipe.json', recipe)
    inspection = inspect_decoder(checkpoint)
    from transformers import AutoTokenizer
    from .audit import audit_dataset
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    audit = audit_dataset(data, tokenizer, recipe)
    write_json(output / 'data-audit.json', audit)
    write_json(output / 'inspection.json', inspection)
    files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json']
    if pin_weights:
        files += ['model.safetensors']
    identities = {name: file_hash(Path(checkpoint) / name) for name in files}
    frozen = {'recipe_sha256': file_hash(output / 'recipe.json'),
              'dataset_sha256': file_hash(output / 'dataset.json'),
              'split_sha256': digest(data['partitions']), 'data_audit_sha256': file_hash(output / 'data-audit.json'),
              'source_files': source_identity(),
              'checkpoint_files': identities, 'inspection_sha256': file_hash(output / 'inspection.json'),
              'weights_pinned': pin_weights, 'checkpoint_path': str(Path(checkpoint).resolve()),
              'backend': backend_identity(), 'created_at': time.time(),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip()}
    write_json(output / 'freeze.json', frozen)
    return frozen


def verify_freeze(root):
    root = Path(root)
    frozen = json.loads((root / 'freeze.json').read_text())
    if not frozen['weights_pinned']:
        raise ValueError('Preparation only: freeze checkpoint weights inside a resource window before launch')
    for name in ['recipe', 'dataset', 'inspection']:
        if file_hash(root / f'{name}.json') != frozen[f'{name}_sha256']:
            raise ValueError(f'Changed frozen {name}')
    if file_hash(root / 'data-audit.json') != frozen['data_audit_sha256']:
        raise ValueError('Changed data audit')
    if source_identity() != frozen['source_files'] or backend_identity() != frozen['backend']:
        raise ValueError('Code or backend differs from the frozen run')
    for name, expected in frozen['checkpoint_files'].items():
        if file_hash(Path(frozen['checkpoint_path']) / name) != expected:
            raise ValueError(f'Checkpoint changed: {name}')
    return frozen


class Ledger:
    def __init__(self, output, recipe):
        self.path = Path(output) / 'ledger.jsonl'
        self.start = time.monotonic()
        self.recipe = recipe
        self.totals = dict(scored_tokens=0, training_decoder_tokens=0,
                           training_encoder_bytes=0, estimated_training_flops=0)

    def record(self, kind, **fields):
        with self.path.open('a') as stream:
            stream.write(json.dumps({'kind': kind, 'elapsed_seconds': time.monotonic() - self.start,
                                     **fields}) + '\n')
            stream.flush()

    def reserve(self, **counts):
        if time.monotonic() - self.start >= self.recipe['max_wall_seconds']:
            raise RuntimeError('Wall-clock cap reached')
        for name, count in counts.items():
            if self.totals[name] + count > self.recipe['max_' + name]:
                raise RuntimeError(f'Compute cap reached: {name}')
        for name, count in counts.items():
            self.totals[name] += count
