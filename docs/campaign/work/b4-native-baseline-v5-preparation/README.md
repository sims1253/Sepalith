# RUN-03 b4 v5 baseline preparation

The root operator owns the live server and run directories. This packet contains a loopback-only native client and a v4-derived editor harness. The client performs six sequential requests over the protected legacy `/tokenize` and `/completion` contract. It records full native JSON responses, raw completion text, protected parser output, in-memory application text and timing, and an R parse over stdin.

The first request for the first geometry is labelled `cold-ish-changed`; the first request for the other geometries is `changed`; the second request for every geometry is `repeat`. These labels are evidence metadata only. Request bodies contain only the prompt and fixed contract fields.

Static checks:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 client.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import ast
from pathlib import Path
ast.parse(Path('client.py').read_text())
print('python AST: PASS')
PY
node --check editor-v5/extension.js
node --check editor-v5/run_b4_editor_cycle.mjs
```

Root launch order after starting the exact CPU server on `127.0.0.1:18403`:

```sh
PYTHONDONTWRITEBYTECODE=1 taskset -c 1 /usr/bin/timeout --signal=TERM --kill-after=10s 60s \
  python3 client.py --url http://127.0.0.1:18403 \
  --run-root "$RUN_ROOT/b4-native-baseline-v5"
```

Then run `editor-v5/run_b4_editor_cycle.mjs` through the existing bounded supervisor with `--vsix`, `--model`, `--runtime`, `--target-root`, `--port 18403`, and a fresh `--run-root`. The derivative launcher sets `sepalith.debugMode=true`; the harness invokes `sepalith.dumpStats` before and after replacement, acceptance, no-op, and cancellation. Inspect the real Sepalith output channel for the raw completion and `stats:` lines and correlate them with `dump_stats_invoked` events in `host-result.json`.

The packet does not start an editor or server, use SSH/CUDA/network, read model weights, add CDP automation, or make a quality/no-op claim.
