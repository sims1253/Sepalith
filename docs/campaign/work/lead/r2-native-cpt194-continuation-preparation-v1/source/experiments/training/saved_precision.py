"""Restore saved FP32 parameters after a BF16 model loader casts them."""
import hashlib
import json
import struct
from pathlib import Path


def restore_saved_fp32(model, weights):
    import torch
    from safetensors import safe_open

    weights = Path(weights)
    with weights.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        assert 0 < size < 16 * 1024 * 1024
        header = json.loads(stream.read(size))
    names = sorted(k for k, v in header.items() if k != '__metadata__' and v['dtype'] == 'F32')
    parameters = dict(model.named_parameters())
    audits = []
    with safe_open(weights, framework='pt', device='cpu') as source, torch.no_grad():
        for name in names:
            assert name in parameters, f'saved FP32 parameter missing: {name}'
            saved = source.get_tensor(name)
            parameter = parameters[name]
            assert saved.numel() <= 1048576, 'unexpected large FP32 tensor requires separate resource review'
            assert saved.shape == parameter.shape and torch.isfinite(saved).all()
            before = parameter.detach().float().cpu()
            changed = int((before != saved).sum())
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
            parameter.copy_(saved.to(parameter.device))
            actual = parameter.detach().cpu()
            assert actual.dtype == torch.float32 and torch.equal(actual, saved)
            audits.append({'name': name, 'elements': saved.numel(), 'changed_elements': changed,
                           'saved_sha256': hashlib.sha256(saved.contiguous().numpy().tobytes()).hexdigest()})
    return {'fp32_tensors_restored': len(audits), 'exact_saved_values_verified': True, 'tensors': audits}
