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
PUBLIC_CONFIG_SHA256 = 'bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b'
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
        if self.resume_checkpoint_dir is not None and not bool(model_args.get('allow_resume', False)):
            raise ValueError('Fresh draft profile output contains a checkpoint; explicit resume admission required')
        config = build_draft_config(target_config=target_config, model_args=model_args)
        config.sliding_window = getattr(target_config, 'sliding_window', None)
        draft = Qwen3DSparkModel(config)
        path = Path(model_args.public_draft_weights)
        verify_public_file(path)
        public_config_bytes = path.with_name('config.json').read_bytes()
        if hashlib.sha256(public_config_bytes).hexdigest() != PUBLIC_CONFIG_SHA256:
            raise ValueError('Public draft configuration fails its pinned SHA256 check')
        public_config = json.loads(public_config_bytes)
        semantic_fields = ('block_size', 'mask_token_id', 'target_layer_ids',
                           'markov_rank', 'markov_head_type', 'enable_confidence_head',
                           'confidence_head_with_markov', 'sliding_window',
                           'rms_norm_eps', 'rope_parameters')
        for field in semantic_fields:
            if getattr(config, field, None) != public_config.get(field):
                raise ValueError('Public draft semantic configuration differs: ' + field)
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
            'resume_checkpoint': str(self.resume_checkpoint_dir) if self.resume_checkpoint_dir else None,
            'resume_applied_after_build_models': self.resume_checkpoint_dir is not None,
            'loaded_tensors': len(weights),
            'public_config_sha256': PUBLIC_CONFIG_SHA256,
            'semantic_fields_matched': list(semantic_fields),
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
        if self.global_rank == 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(json.dumps(self._warmstart_receipt, indent=2) + '\n')
            temporary.replace(destination)
        return draft, tokenizer
