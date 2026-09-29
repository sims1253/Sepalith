**Sepalith core**

This independent Python package contains the local experiment runner, versioned
edit-context records, latent-memory manifests and the full-weight training
runtime. The core modules use only the standard library. `sepalith.training`
needs the training stack (torch, transformers, trl, unsloth) and runs with the
training interpreter; see [its README](src/sepalith/training/README.md). The
package does not modify the root environment.
The runner currently supports Linux, including WSL2.

From the repository root, run the lightweight checks:

```bash
python3 scripts/check_core.py
```

The training runtime has its own CPU suite, which uses the training interpreter
and the campaign data on `/mnt/e`:

```bash
python3 scripts/check_training.py
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
| `sepalith.evaluation` | Paired binary-outcome tests and deterministic bootstrap intervals |
| `sepalith.memory` | Latent payload identity, compatibility and source freshness checks |
| `sepalith.ops.night_queue`, `sepalith.ops.night_runner` | Nightly GPU window queue, runner and morning report ([setup](../../docs/training/plan-20260927/NIGHT-WINDOW.md)) |
| `sepalith.campaign_protocol` | The `zeta2-prm03-v1` editing prompt and output contract used by the campaign |
| `sepalith.training` | Full-weight SFT, DEV gate, CPT and RL runtime (training interpreter only) |

See [runner operations](../../docs/EXPERIMENT-RUNNER.md),
[prompt contract](../../docs/PROMPT-CONTRACT.md), and
[latent workspace memory proposal](../../docs/LATENT-WORKSPACE-MEMORY.md).

The v5 assembler now uses the protocol renderer for its edit rows. Its direct
Python invocation loads the package source from this checkout, including when
started from another working directory; it needs no package installation.
Other assembler prompt families retain their existing rendering conventions.
The runner is not a replacement for the active supervisors until the documented
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

The historical paired-significance audit now uses `sepalith.evaluation`; see
[scoring conventions and corrections](../../docs/SHARED-EVALUATION.md). The v5
assembler also accepts `--data-root`, `--finish-source` and `--out`, resolves its
checkout from its own location, and reports the resolved input/output paths.
