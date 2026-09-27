"""Validate dense safetensors checkpoint layout without loading tensor payloads."""
import json
from pathlib import Path
from safetensors import safe_open


def validate_dense_weights(directory):
    root = Path(directory)
    if any(root.glob('adapter*')):
        raise ValueError('Dense checkpoint contains adapter artifacts')
    config = root / 'config.json'
    if not config.is_file() or config.is_symlink() or not isinstance(json.loads(config.read_text()), dict):
        raise ValueError('Dense checkpoint requires a regular model config')
    single = root / 'model.safetensors'
    index = root / 'model.safetensors.index.json'
    if single.exists() == index.exists():
        raise ValueError('Dense checkpoint requires exactly one single-file or indexed layout')
    weight_map = None
    if index.exists():
        if index.is_symlink():
            raise ValueError('Checkpoint index is a symlink')
        weight_map = json.loads(index.read_text()).get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('Checkpoint weight map is empty or invalid')
        names = set(weight_map.values())
        if any(not isinstance(n, str) or Path(n).name != n or not n.endswith('.safetensors') for n in names):
            raise ValueError('Checkpoint shard path is not a local safetensors filename')
    else:
        names = {'model.safetensors'}
    actual_files = {p.name for p in root.glob('*.safetensors')}
    if names != actual_files:
        raise ValueError('Checkpoint shard inventory differs from weight map')
    tensors = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Checkpoint shard missing or symlinked')
        with safe_open(path, framework='numpy') as reader:
            keys = list(reader.keys())
            if not keys:
                raise ValueError('Checkpoint shard contains no tensors')
            for key in keys:
                if key in tensors:
                    raise ValueError('Duplicate tensor across checkpoint shards')
                if 'lora_A' in key or 'lora_B' in key:
                    raise ValueError('Dense checkpoint contains LoRA factors')
                if weight_map is not None and weight_map.get(key) != name:
                    raise ValueError('Tensor and shard mapping differ')
                tensors[key] = name
    if weight_map is not None and set(tensors) != set(weight_map):
        raise ValueError('Checkpoint weight map references missing tensors')
    return {'kind': 'full_weights', 'weight_files': sorted(names), 'tensor_count': len(tensors)}
