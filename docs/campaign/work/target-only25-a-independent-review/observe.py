"""Read exact run paths at30-second intervals; never hash changing files."""
from pathlib import Path
from datetime import datetime,timezone
import json,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NATIVE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
OUT=NATIVE/'training/SFT-target-only25-a'
CASES=NATIVE/'checkpoints/SFT-target-only25-a/evaluations/cases-step-25.json'
SUP=ROOT/'work/lead/target-only25-a/supervisor-terminal.json'

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def observe():
    gate=read(OUT/'target-only-pre-update-gate.json')
    rows=[]
    try:
        for line in (OUT/'telemetry.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    except FileNotFoundError:pass
    updates=[r for r in rows if r.get('event')=='optimizer_step']
    cases=read(CASES);terminal=read(OUT/'terminal.json');supervisor=read(SUP)
    item={'utc':datetime.now(timezone.utc).isoformat(),'gate_status':None if gate is None else gate.get('status'),
          'update_count':len(updates),'last_step':updates[-1]['step'] if updates else None,
          'last_update_at':updates[-1]['at'] if updates else None,'stop_reason':updates[-1].get('stop_reason') if updates else None,
          'cases_status':None if cases is None else cases.get('status'),'cases_count':len(cases.get('results',[])) if cases else 0,
          'training_terminal':terminal is not None,'supervisor_terminal':supervisor is not None,'live_artifacts_hashed':False}
    with (HERE/'observations.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    print(json.dumps(item),flush=True)
    return supervisor is not None

if __name__=='__main__':
    until=time.monotonic()+1050
    while True:
        if observe() or time.monotonic()>=until:break
        time.sleep(30)
