"""JAX BF16 embedding-style compute, optimization and host checkpoint probe."""
import time
from pathlib import Path


def probe(report):
    if INSTALL_LIBTPU:
        import os, subprocess, sys
        report['stage'] = 'install JAX 0.7.2 TPU plugin libtpu 0.0.23'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'libtpu==0.0.23'],
                       check=True, timeout=90, capture_output=True)
        os.environ['JAX_PLATFORMS'] = 'tpu'
    report['stage'] = 'discover JAX TPU backend'
    import jax
    import jax.numpy as jnp
    import numpy as np
    devices = jax.devices()
    report['devices'] = [str(device) for device in devices]
    assert devices and all(device.platform == 'tpu' for device in devices), 'JAX did not select TPU; CPU fallback rejected'
    report['checks']['tpu_backend'] = True
    report['stage'] = 'BF16 matrix operation'
    x = jnp.ones((1024, 256), dtype=jnp.bfloat16)
    w = jnp.ones((256, 128), dtype=jnp.bfloat16)
    matmul = jax.jit(lambda a, b: a @ b)
    start = time.monotonic()
    out = matmul(x, w).block_until_ready()
    report['compile_and_first_seconds'] = time.monotonic() - start
    start = time.monotonic()
    for _ in range(10): out = matmul(x, w).block_until_ready()
    report['ten_warm_calls_seconds'] = time.monotonic() - start
    assert float(out[0, 0]) == 256.
    report['checks']['bf16_matmul'] = True
    report['stage'] = 'compiled gradient updates'
    values = jnp.linspace(-1., 1., 128)
    targets = 2 * values + 1
    def loss(params): return jnp.mean((params[0] * values + params[1] - targets) ** 2)
    @jax.jit
    def step(params): return params - 0.1 * jax.grad(loss)(params)
    params = jnp.zeros(2)
    initial = float(loss(params))
    for _ in range(64): params = step(params)
    final = float(loss(params))
    assert np.isfinite(final) and final < initial / 100
    report.update(initial_loss=initial, final_loss=final)
    report['checks']['training'] = True
    np.save('checkpoint.npy', np.asarray(params))
    restored = jnp.asarray(np.load('checkpoint.npy', allow_pickle=False))
    assert float(loss(restored)) == final
    report['checks']['checkpoint_reload'] = True
    report['stage'] = 'complete'
