"""Bounded FP16 compatibility smoke, without model downloads or credentials.

This checks remote execution and numerical plumbing, not model quality.
The submission tool supplies JOB_ID; Kaggle retains this source as version 1.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import time


def main():
    signal.alarm(120)
    start = time.monotonic()
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(3407)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU requested but unavailable')
    device = torch.device('cuda:0')
    x = torch.linspace(-1, 1, 128, device=device, dtype=torch.float16)
    y = 2 * x + 1
    weight = torch.zeros((), device=device, dtype=torch.float16, requires_grad=True)
    bias = torch.zeros((), device=device, dtype=torch.float16, requires_grad=True)
    optimizer = torch.optim.SGD([weight, bias], lr=0.1)
    losses = []
    for _ in range(64):
        optimizer.zero_grad()
        loss = ((weight * x + bias - y).float() ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    # Round-trip the checkpoint and evaluate the restored parameters.
    torch.save({'weight': weight.detach(), 'bias': bias.detach()}, 'checkpoint.pt')
    restored = torch.load('checkpoint.pt', weights_only=True, map_location=device)
    error = ((restored['weight'] * x + restored['bias'] - y).float() ** 2).mean().item()
    if error >= 0.01 or losses[-1] >= losses[0] / 100:
        raise RuntimeError('FP16 regression failed its numerical acceptance gate')
    Path('metrics.json').write_text(json.dumps({'losses': losses, 'restored_mse': error}) + '\n')
    outputs = {}
    for name in ('metrics.json', 'checkpoint.pt'):
        data = Path(name).read_bytes()
        outputs[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    receipt = {
        'job_id': JOB_ID,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'operational_status': 'succeeded', 'scientific_verdict': 'not_applicable',
        'acceptance': 'finite FP16 training; 100x loss reduction; restored MSE < 0.01',
        'python': platform.python_version(), 'interpreter': os.path.realpath(os.sys.executable),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0),
        'elapsed_seconds': time.monotonic() - start, 'outputs': outputs,
    }
    Path('receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
