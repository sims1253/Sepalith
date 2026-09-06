#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
CAL="$ROOT/experiments/models/quant-calibration"
F16="$ROOT/experiments/models/packaging_b4-f16.gguf"
# Run only after the f16 export has completed. This is a smoke calibration.
taskset -c 16-23 experiments/bin/llama/llama-b10453/llama-imatrix \
  -m "$F16" -f "$CAL/sft_v7.txt" -o "$CAL/b4-smoke-imatrix.gguf" \
  --parse-special --no-ppl --chunks "${SEPALITH_IMATRIX_CHUNKS:-8}" -c 512 -t 8 -ngl 0 > "$CAL/imatrix.log" 2>&1
python3 - "$F16" "$CAL" "${SEPALITH_IMATRIX_CHUNKS:-8}" <<'PYMETA'
import hashlib, json, sys
from pathlib import Path
source, cal, chunks = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
matrix = cal / 'b4-smoke-imatrix.gguf'
receipt = dict(model=str(source.resolve()), model_sha256=sha(source),
               corpus_sha256=sha(cal / 'sft_v7.txt'), imatrix_sha256=sha(matrix),
               chunks=chunks, context=512, parse_special=True, build='b10453')
matrix.with_suffix('.gguf.json').write_text(json.dumps(receipt, indent=2) + '\n')
PYMETA
python3 experiments/training/export_gguf.py unused unused packaging_b4_control \
  --f16 "$F16" --uncalibrated --tiers Q4_K_M > "$CAL/control.log" 2>&1
python3 experiments/training/export_gguf.py unused unused packaging_b4_imatrix \
  --f16 "$F16" --imatrix "$CAL/b4-smoke-imatrix.gguf" \
  --tiers Q6_K Q4_K_M IQ4_XS > "$CAL/tiers.log" 2>&1
