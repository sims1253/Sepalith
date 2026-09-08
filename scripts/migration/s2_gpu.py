#!/usr/bin/env python3
"""Measure the saved S2 quantization matrix on its missing local GPU tier."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import urllib.request

from s1_gpu import GpuServer, DeadlineExceeded, check_offload, tokenize, write
import spec_bench as bench

FORMATS = ['Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q4_0']


def prepare(run, assets):
    traces = bench.sample_traces(bench.load_traces(assets / 'traces.jsonl')['2k'], 10)
    if len(traces) != 10 or len({r['trace_id'] for r in traces}) != 10:
        raise ValueError('Expected ten unique traces')
    for fmt in FORMATS:
        if not (assets / (fmt + '.gguf')).is_file():
            raise ValueError('Missing quantized model: ' + fmt)
    write(run / 'prepared.json', dict(formats=FORMATS, trace_ids=[r['trace_id'] for r in traces],
        reps=3, expected_rows=150, ctx=4096, max_seconds=1770,
        scope='S2 b1_ref24 GPU timing column; quality gate remains separate'))


def request(port, prompt):
    body = json.dumps(dict(prompt=prompt, n_predict=48, temperature=0.0,
                           stream=False, cache_prompt=False, stop=[bench.STOP])).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=body,
                                 headers={'Content-Type': 'application/json'})
    start=time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as response:
        value=json.load(response)
    return value, (time.monotonic()-start)*1000


def measure(run, assets):
    p=json.loads((run/'prepared.json').read_text())
    traces=bench.sample_traces(bench.load_traces(assets/'traces.jsonl')['2k'],10)
    server=None
    def expire(*_):
        raise DeadlineExceeded('S2 exceeded 30-minute bound')
    old=signal.signal(signal.SIGALRM,expire);deadline=time.monotonic()+p['max_seconds']
    signal.alarm(p['max_seconds'])
    try:
        with (run/'per_request.jsonl').open('x') as output:
            for fmt in FORMATS:
                server=GpuServer(assets/(fmt+'.gguf'),18472,['-lv','4','--seed','20260905'],
                    ctx=p['ctx'], server=Path(os.environ['S1_RUNTIME'])/'llama-server',
                    foreground=True,log_path=run/('server-'+fmt+'.log'))
                try:
                    server.start(ready_timeout=180)
                    check_offload(server.log_path.read_text())
                    with (run/'telemetry.jsonl').open('a') as f:
                        f.write(json.dumps(dict(format=fmt,at=time.time(),load=os.getloadavg(),gpu=subprocess.check_output(
                            ['nvidia-smi','--query-gpu=uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv,noheader'],text=True,timeout=10)))+'\n')
                    token_counts={t['trace_id']:tokenize(server.port,t['prompt']) for t in traces}
                    if max(token_counts.values())+48+16>p['ctx']:
                        raise ValueError('Recounted MiniCPM prompt exceeds context reserve')
                    warmup,_=request(server.port,traces[-1]['prompt'])
                    with (run/'warmups.jsonl').open('a') as f:f.write(json.dumps(dict(format=fmt,response=warmup))+'\n')
                    for rep in range(3):
                        for trace in traces:
                            data,wall=request(server.port,trace['prompt']);timings=data.get('timings',{})
                            row=dict(format=fmt,rep=rep,trace_id=trace['trace_id'],ctx_class='2k',
                                prompt_n=timings.get('prompt_n'),tokenized_prompt_tokens=token_counts[trace['trace_id']],
                                prompt_ms=timings.get('prompt_ms'),predicted_ms=timings.get('predicted_ms'),
                                wall_ms=wall,raw_response=data,recorded_at=time.time(),host_load=os.getloadavg())
                            output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
                            if row['prompt_n']!=row['tokenized_prompt_tokens'] or not data.get('stop'):
                                raise ValueError('Prompt parity or complete-response gate failed')
                            if any(not isinstance(row[k],(int,float)) or not math.isfinite(row[k]) or row[k]<0 for k in ('prompt_ms','predicted_ms')):
                                raise ValueError('Missing or invalid timing')
                    print(json.dumps(dict(format=fmt,rows=30)),flush=True)
                finally:
                    signal.alarm(0);server.stop();server=None
                    signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    finally:
        signal.alarm(0)
        try:
            if server is not None:server.stop()
        finally:signal.signal(signal.SIGALRM,old)


def evaluate(run):
    p=json.loads((run/'prepared.json').read_text())
    rows=[json.loads(l) for l in (run/'per_request.jsonl').read_text().splitlines()]
    expected=[(fmt,rep,tr) for fmt in p['formats'] for rep in range(p['reps']) for tr in p['trace_ids']]
    if [(r['format'],r['rep'],r['trace_id']) for r in rows]!=expected:
        raise ValueError('Incomplete or duplicate S2 coverage')
    if any(r['prompt_n']!=r['tokenized_prompt_tokens'] or not r['raw_response'].get('stop') for r in rows):
        raise ValueError('Invalid S2 request')
    summary={}
    for fmt in p['formats']:
        rs=[r for r in rows if r['format']==fmt]
        summary[fmt]=dict(n=len(rs),cycle_ms_median=statistics.median(r['prompt_ms']+r['predicted_ms'] for r in rs),
            request_wall_ms_median=statistics.median(r['wall_ms'] for r in rs),
            generation_limit_rows=sum(r['raw_response'].get('stop_type')=='limit' for r in rs))
    for value in summary.values():value['speedup_vs_Q8_0']=summary['Q8_0']['cycle_ms_median']/value['cycle_ms_median']
    write(run/'evaluation.json',dict(rows=len(rows),formats=summary))
    write(run/'verdict.json',dict(verdict='GPU-TIMING-MEASURED',adoption='NOT-ASSESSED',
        boundary='Scenario and intent quality within 1pp of Q8 required before quantization promotion; this is b1_ref24, not the production GDN winner.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','measure','evaluate'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--assets',type=Path)
    a=p.parse_args()
    evaluate(a.run) if a.action=='evaluate' else globals()[a.action](a.run,a.assets)
