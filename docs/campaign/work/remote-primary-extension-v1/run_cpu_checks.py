#!/usr/bin/env python3
"""Reproduce private CPU checks. Existing dependencies are read-only."""
from pathlib import Path
import json, os, subprocess
root = Path(__file__).resolve().parent
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
commands = [item["argv"] for item in json.loads((root / "cpu-validation.json").read_text())["results"]]
results = []
for command in commands:
    result = subprocess.run(command, cwd=root / "extension", capture_output=True, text=True, timeout=45)
    entry = {"argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    results.append(entry)
    print(json.dumps(entry), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
print(json.dumps({"status": "PASS", "commands": len(results), "cpu_affinity": sorted(os.sched_getaffinity(0))}))
