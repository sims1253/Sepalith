#!/usr/bin/env python3
"""Create the unadmitted six-thread Q8/Q4/IQ3 confirmation packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(source: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"refusing existing output: {output}")
    packet = json.loads(source.read_text(encoding="utf-8"))
    if packet.get("status") != "root_hashes_bound_no_model_load":
        raise ValueError("source packet is not hash-bound")
    packet["schema"] = "sepalith.r2.notebook-quant-eval-packet.six-thread-confirmation.v1"
    packet["status"] = "root_hashes_bound_no_model_load"
    packet["authorized_arms"] = ["q8", "q4_calibrated", "iq3"]
    packet["resource"] = {
        "backend": "cpu",
        "cpu_list": [0, 2, 4, 6, 8, 10],
        "maximum_threads": 6,
        "per_arm_max_seconds": 330,
        "suite_max_seconds": 990,
    }
    packet["remote"]["packet_root"] = "/home/m0hawk/.local/share/sepalith-quant-eval-v1/work/r2-notebook-quant-six-thread-v1"
    packet["launch"]["environment"] = {
        "OMP_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6", "BLIS_NUM_THREADS": "6",
    }
    packet["launch"]["argv_template"] = [
        "/usr/bin/taskset", "--cpu-list", "0,2,4,6,8,10",
        "{binary}", "-m", "{model_path}",
        "--host", "127.0.0.1", "--port", "18407",
        "-t", "6", "-tb", "6", "--threads-http", "2",
        "--parallel", "1", "-c", "4096", "-b", "256", "-ub", "256",
        "-ngl", "0", "-lv", "4", "--metrics",
    ]
    packet["v1"]["purpose"] = "six-thread confirmation after two-thread screen"
    packet["v1"]["requests_per_arm"] = 16
    packet["v1"]["quality_denominator_per_arm"] = 8
    packet["v1"]["root_admission_required"] = True
    packet["v2_later"]["authorized_now"] = False
    packet["v2_later"]["requires_new_root_admission"] = True
    packet["predecessor"] = {
        "packet_sha256": "1056e80572ad5435d841b03e0250e81c5703007c96622e7d1fb48d7aca4057fe",
        "interpretation": "Two-thread screen is terminal and is not the notebook performance ceiling."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packet = prepare(args.source, args.out)
    print(json.dumps({"status": "prepared_unadmitted", "out": str(args.out),
                      "arms": packet["authorized_arms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
