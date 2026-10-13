"""Warm-start the pinned upstream DSpark trainer; preserve its training/resume loop."""
from pathlib import Path
import hashlib
import json
from safetensors.torch import load_file
from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
from deepspec.modeling.dspark.qwen3.config import build_draft_config
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel

PUBLIC_SHA256 = 'ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97'
PUBLIC_BYTES = 647558522
FROZEN_TARGET_KEYS = {'embed_tokens.weight', 'lm_head.weight'}


def verify_public_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != PUBLIC_BYTES:
        raise ValueError('Public draft weights have the wrong file size')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != PUBLIC_SHA256:
        raise ValueError('Public draft weights fail the pinned SHA256 check')


class SepalithWarmstartTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        config = build_draft_config(target_config, model_args)
        config.sliding_window = getattr(target_config, 'sliding_window', None)
        draft = Qwen3DSparkModel(config)
        path = Path(model_args.public_draft_weights)
        verify_public_file(path)
        weights = load_file(str(path), device='cpu')
        expected = draft.state_dict()
        if set(expected) - set(weights) != FROZEN_TARGET_KEYS or set(weights) - set(expected):
            raise ValueError('Public draft tensor keys differ beyond the two frozen target matrices')
        import torch
        for key, value in weights.items():
            if value.shape != expected[key].shape or value.dtype != torch.bfloat16:
                raise ValueError('Public draft tensor shape or dtype differs: ' + key)
        result = draft.load_state_dict(weights, strict=False)
        if set(result.missing_keys) != FROZEN_TARGET_KEYS or result.unexpected_keys:
            raise ValueError('Public draft loading did not preserve the exact missing-key contract')
        self._warmstart_receipt = {
            'status': 'public_draft_loaded_target_matrices_pending',
            'weights_sha256': PUBLIC_SHA256,
            'loaded_tensors': len(weights),
            'missing_target_matrices': sorted(FROZEN_TARGET_KEYS),
            'target_layers': list(config.target_layer_ids),
            'num_anchors': int(config.num_anchors),
        }
        return draft

    def build_models(self):
        # Upstream loads the bound target and initializes its exact embedding
        # and output matrices after _build_draft_model, then freezes both.
        draft, tokenizer = super().build_models()
        if draft.embed_tokens.weight.requires_grad or draft.lm_head.weight.requires_grad:
            raise ValueError('Target embedding and output matrices must remain frozen')
        self._warmstart_receipt['status'] = 'public_draft_loaded_and_target_matrices_frozen'
        destination = Path(self.checkpoint_dir_root) / 'sepalith-warmstart.json'
        destination.write_text(json.dumps(self._warmstart_receipt, indent=2) + '\n')
        return draft, tokenizer
