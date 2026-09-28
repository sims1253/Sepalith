#!/usr/bin/env python3
"""Bounded CPU quantization launcher with durable launch and terminal records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


QUANTIZER = Path("/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-quantize")
INPUT = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step500-runtime-gguf/model-F16.gguf")
ALLOWED = ("Q6_K", "Q5_K_M", "Q4_K_M")


def write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("quant_type", choices=ALLOWED)
    parser.add_argument("output", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--terminal-record", type=Path, required=True)
    args = parser.parse_args(argv)

    if not QUANTIZER.is_file() or not os.access(QUANTIZER, os.X_OK):
        raise SystemExit(f"quantizer is not executable: {QUANTIZER}")
    if not INPUT.is_file():
        raise SystemExit(f"input does not exist: {INPUT}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/timeout", "--signal=TERM", "--kill-after=15s", "240s",
        str(QUANTIZER), "--output-tensor-type", "q8_0",
        "--token-embedding-type", "q8_0", str(INPUT), str(args.output),
        args.quant_type, "2",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    })
    launch = {
        "quant_type": args.quant_type,
        "command": command,
        "input": str(INPUT),
        "input_sha256_expected": "50f523af997f2f77dcbd187adde8a36dbc41529932703fd016e506b172761725",
        "output": str(args.output),
        "quantizer": str(QUANTIZER),
        "quantizer_sha256_expected": "28cd0f04614c2ca5a6de32feb4b2196625fc792052682749da64cc5879035365",
        "output_tensor_type": "q8_0",
        "token_embedding_type": "q8_0",
        "threads": 2,
        "timeout_seconds": 240,
        "cuda_visible_devices": "",
        "started_at_unix": time.time(),
    }
    write_json(args.launch_record, launch)
    start = time.monotonic()
    with args.log.open("wb") as stream:
        stream.write((json.dumps(launch, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=environment,
        )
        return_code = process.wait()
        elapsed = time.monotonic() - start
        stream.flush()
        os.fsync(stream.fileno())
    output_sha256 = None
    output_bytes = None
    if args.output.is_file():
        output_bytes = args.output.stat().st_size
        digest = hashlib.sha256()
        with args.output.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        output_sha256 = digest.hexdigest()
    terminal = {
        "quant_type": args.quant_type,
        "command": command,
        "return_code": return_code,
        "timed_out": return_code == 124,
        "elapsed_seconds": elapsed,
        "output": str(args.output),
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "terminated_at_unix": time.time(),
        "log": str(args.log),
    }
    write_json(args.terminal_record, terminal)
    print(json.dumps(terminal, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
