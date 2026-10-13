"""Tiny proof-only dense checkpoint validator; production uses the 381-tensor validator."""
import json
from pathlib import Path
from safetensors import safe_open

EXPECTED = {
    'lm_head.weight', 'model.embed_tokens.weight',
    'model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.k_proj.weight',
    'model.layers.0.self_attn.v_proj.weight', 'model.layers.0.self_attn.o_proj.weight',
    'model.layers.0.mlp.gate_proj.weight', 'model.layers.0.mlp.up_proj.weight',
    'model.layers.0.mlp.down_proj.weight', 'model.norm.weight',
}
def validate_dense_weights(directory):
    path=Path(directory)/'model.safetensors'
    if not path.is_file(): raise ValueError('tiny proof dense weights missing')
    with safe_open(path,framework='pt',device='cpu') as f:
        keys=set(f.keys())
    if keys != EXPECTED: raise ValueError(f'tiny proof dense inventory differs:{sorted(keys^EXPECTED)}')
    return {'checkpoint_kind':'full_weights','tensor_count':len(keys),'proof_only':True}
