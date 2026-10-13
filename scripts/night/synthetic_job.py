#!/usr/bin/env python3
"""Synthetic night-window job for the C02 dry run.

--mode honour  saves a fake checkpoint and exits 75 once SEPALITH_DEADLINE_EPOCH
               passes, the stop-request file appears or SIGTERM arrives.
--mode ignore  ignores all of that and runs until it is killed.
--device cuda  holds about 2 GiB on the GPU and keeps it busy (needs torch).
--device cpu   no torch; tests the runner mechanics only.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("honour", "ignore"), required=True)
parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
args = parser.parse_args()

stop = []
if args.mode == "ignore":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, lambda *_: stop.append("sigterm"))
deadline = float(os.environ["SEPALITH_DEADLINE_EPOCH"])
stop_file = Path(os.environ["SEPALITH_STOP_REQUEST_FILE"])
attempt = Path(os.environ["SEPALITH_NIGHT_ATTEMPT_DIR"])

step = 0
if args.device == "cuda":
    import torch

    buffer = torch.empty(1024**3, dtype=torch.bfloat16, device="cuda")  # 2 GiB
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    print(f"cuda {torch.cuda.get_device_name()} allocated {torch.cuda.memory_allocated() >> 20} MiB", flush=True)

    def work() -> None:
        global a
        a = torch.tanh(a @ a)
        torch.cuda.synchronize()
else:
    def work() -> None:
        sum(i * i for i in range(20000))

started = time.time()
while True:
    work()
    step += 1
    if step % 200 == 0:
        print(f"step {step} at {time.time() - started:.0f}s", flush=True)
    if args.mode == "ignore":
        continue
    reason = ("sigterm" if stop else "stop_request" if stop_file.exists()
              else "deadline" if time.time() >= deadline else None)
    if reason:
        checkpoint = attempt / f"checkpoint-{step}.json"
        checkpoint.write_text(json.dumps({"step": step, "reason": reason}) + "\n")
        Path(os.environ["SEPALITH_JOB_RESULT"]).write_text(json.dumps({
            "checkpoints": [str(checkpoint)],
            "summary": f"stopped on {reason} after {step} steps, {time.time() - deadline:+.1f}s from the deadline",
        }) + "\n")
        print(f"saved {checkpoint} on {reason}", flush=True)
        sys.exit(75)
