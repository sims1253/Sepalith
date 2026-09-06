**Sepalith core**

This independent Python package contains the local experiment runner, versioned
edit-context records and latent-memory manifests. It uses only the standard
library. It does not import the training stack or modify the root environment.
The runner currently supports Linux, including WSL2.

From the repository root, run the lightweight checks:

```bash
python3 scripts/check_core.py
```

For use from another working directory, set `PYTHONPATH` to the absolute
`packages/sepalith/src` path. Alternatively, install this package in a separate
development environment; it exposes the `sepalith-run` command. The root
`pyproject.toml` and existing `.venv` remain under the active experiments' control.

The public modules are:

| Module | Purpose |
|---|---|
| `sepalith.runner` | Capture source, enqueue recipes, inspect, drain, execute and reconcile attempts |
| `sepalith.protocol` | Versioned edit/evidence records and baseline `zeta2-v1` rendering |
| `sepalith.memory` | Latent payload identity, compatibility and source freshness checks |

See [runner operations](../../docs/EXPERIMENT-RUNNER.md),
[prompt contract](../../docs/PROMPT-CONTRACT.md), and
[latent workspace memory proposal](../../docs/LATENT-WORKSPACE-MEMORY.md).

No existing experiment entry point has been migrated to this package yet. Its
runner is not a replacement for the active supervisors until the documented
cutover is performed. The memory module describes artifacts; it does not train
a compressor, execute a decoder or load KV tensors.

To validate or migrate a captured input and inspect its exact baseline prompt:

```bash
uv venv /tmp/sepalith-dev
uv pip install --python /tmp/sepalith-dev/bin/python ./packages/sepalith
/tmp/sepalith-dev/bin/sepalith-context normalize --legacy example.json > context.json
/tmp/sepalith-dev/bin/sepalith-context render context.json > prompt.txt
```

`normalize` preserves additional fields and validates the versioned record.
`render` writes UTF-8 prompt bytes without adding a trailing newline. The
`--legacy` flag is required for unversioned evaluation records. Static evidence
stays in the record until a separately versioned renderer has been trained and
evaluated to consume it.
