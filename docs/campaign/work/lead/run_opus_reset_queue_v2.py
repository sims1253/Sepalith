"""Run the admitted source-only subscription queue after its reported reset."""
from pathlib import Path
import datetime as dt
import json
import os
import signal
import subprocess
import time

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
RECEIPT = PLAN/'docs/campaign/receipts/RUN-06-opus-reset-queue-v2.json'
queue = json.loads(RECEIPT.read_text())
OUT = PLAN/'docs/campaign/work/serving-hillclimb/reset-queue-v2-20260913'
OUT.mkdir(exist_ok=False)
start = dt.datetime.fromisoformat(queue['not_before'].replace('Z','+00:00'))
stop = dt.datetime.fromisoformat(queue['hard_stop'].replace('Z','+00:00'))
interrupted = False
def signal_handler(_sig, _frame):
    global interrupted
    interrupted = True
signal.signal(signal.SIGTERM,signal_handler)
signal.signal(signal.SIGINT,signal_handler)
def save(name, value):
    f=OUT/name
    tmp=f.with_suffix('.partial')
    with tmp.open('w') as h:
        json.dump(value,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
    tmp.replace(f)
def now():
    return dt.datetime.now(dt.timezone.utc)
def cancelled():
    return interrupted or (OUT/'cancel').exists() or now()>=stop

save('launch.json',{'at':now().isoformat(),'pid':os.getpid(),'owner':'lead',
    'receipt':str(RECEIPT),'not_before':queue['not_before'],'hard_stop':queue['hard_stop'],
    'scope':'Subscription source-only requests, no tools, edits, GPU, model or cloud jobs.'})
while now()<start and not cancelled():
    save('status.json',{'at':now().isoformat(),'status':'waiting_for_reported_session_reset'})
    time.sleep(min(30,max(.1,(start-now()).total_seconds())))
results=[]
reason='queue_completed'
for job in queue['jobs']:
    if cancelled():
        reason='cancelled_or_hard_stop';break
    if (stop-now()).total_seconds()<job['max_seconds']+60:
        reason='insufficient_time_before_hard_stop';break
    packet=PLAN/job['packet']
    import hashlib
    assert hashlib.sha256(packet.read_bytes()).hexdigest()==job['packet_sha256']
    command=['python3',str(PLAN/'docs/campaign/work/lead/run_opus_patch_stream.py'),
             str(packet),'--task',job['task'],'--seconds',str(job['max_seconds']),
             '--effort',job['effort']]
    with (OUT/(packet.stem+'.stdout.log')).open('wb') as out, (OUT/(packet.stem+'.stderr.log')).open('wb') as err:
        proc=subprocess.Popen(command,cwd=PLAN,stdout=out,stderr=err,start_new_session=True)
        save('status.json',{'at':now().isoformat(),'status':'running','packet':str(packet),'child_pid':proc.pid})
        while proc.poll() is None and not cancelled():
            time.sleep(2)
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    response=packet.parent/(packet.stem+'-run/response.json')
    last=None
    if response.exists():
        for line in response.read_text().splitlines():
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if value.get('type')=='result':last=value
    valid=proc.returncode==0 and last is not None and last.get('is_error') is False
    results.append({'packet':str(packet),'wrapper_exit':proc.returncode,'response':str(response),
                    'has_result':last is not None,'result_is_error':last.get('is_error') if last else None,
                    'status':'artifact_pending_root_review' if valid else 'call_failed'})
    save('results.json',results)
    if not valid:
        reason='first_failed_call_stops_queue';break
save('terminal.json',{'at':now().isoformat(),'reason':reason,'results':results,
                      'acceptance':'Root review and CPU tests pending; no patch was applied.'})
