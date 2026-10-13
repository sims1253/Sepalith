#!/usr/bin/env python3
from pathlib import Path
import json,os,subprocess
root=Path(__file__).resolve().parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for entry in json.loads((root/'cpu-validation.json').read_text())['results']:
    result=subprocess.run(entry['argv'],cwd=root/'extension',env={**os.environ,**entry.get('env',{})},capture_output=True,text=True,timeout=30)
    print(json.dumps({'argv':entry['argv'],'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
