#!/usr/bin/env python3
"""Measure the three quality-tested b4 exports on GPU or quiet CPU."""
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

FORMATS = ['Q8_0', 'Q6_K_imatrix', 'IQ4_XS_imatrix', 'Q8_0-bookend']


def model_path(assets, fmt):
    return assets / (('Q8_0' if fmt == 'Q8_0-bookend' else fmt) + '.gguf')


def prepare(run, assets):
    traces = bench.sample_traces(bench.load_traces(assets / 'traces.jsonl')['2k'], 10)
    if len(traces) != 10 or len({r['trace_id'] for r in traces}) != 10:
        raise ValueError('Expected ten unique traces')
    for fmt in FORMATS:
        if not model_path(assets, fmt).is_file():
            raise ValueError('Missing quantized model: ' + fmt)
    write(run / 'prepared.json', dict(formats=FORMATS, trace_ids=[r['trace_id'] for r in traces],
        reps=3, expected_rows=120, ctx=4096, max_seconds=1770 if os.environ['B4_TIMING_TIER']=='gpu' else 5370,
        tier=os.environ['B4_TIMING_TIER'],
        scope='S2 paired b4 export cycle timing; fresh Q8 bookend; adoption separate'))


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
    if [r['trace_id'] for r in traces] != p['trace_ids']:
        raise ValueError('Prepared trace identity changed before measurement')
    server=None
    baseline_counts=None
    def expire(*_):
        raise DeadlineExceeded('B4 timing exceeded tier-specific bound')
    old=signal.signal(signal.SIGALRM,expire);deadline=time.monotonic()+p['max_seconds']
    signal.alarm(p['max_seconds'])
    try:
        with (run/'per_request.jsonl').open('x') as output:
            for fmt in FORMATS:
                server_class = GpuServer if p['tier']=='gpu' else bench.SpecServer
                server=server_class(model_path(assets,fmt),18477,['-lv','4','--seed','20260905'],
                    ctx=p['ctx'], threads=8, server=Path(os.environ['S1_RUNTIME'])/'llama-server',
                    foreground=True,log_path=run/('server-'+fmt+'.log'))
                try:
                    server.start(ready_timeout=180)
                    if p['tier']=='gpu':check_offload(server.log_path.read_text())
                    with (run/'telemetry.jsonl').open('a') as f:
                        f.write(json.dumps(dict(format=fmt,at=time.time(),load=os.getloadavg(),gpu=subprocess.check_output(
                            ['nvidia-smi','--query-gpu=uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv,noheader'],text=True,timeout=10) if p['tier']=='gpu' else None))+'\n')
                    token_counts={t['trace_id']:tokenize(server.port,t['prompt']) for t in traces}
                    if baseline_counts is None:baseline_counts=token_counts
                    elif token_counts!=baseline_counts:raise ValueError('Cross-format tokenizer mismatch')
                    if max(token_counts.values())+48+16>p['ctx']:
                        raise ValueError('Recounted b4 prompt exceeds context reserve')
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
            generation_limit_rows=sum(r['raw_response'].get('stop_type')=='limit' for r in rs),
            predicted_tokens_median=statistics.median(r['raw_response']['timings']['predicted_n'] for r in rs),
            stop_types={stop:sum(r['raw_response'].get('stop_type')==stop for r in rs) for stop in sorted({r['raw_response'].get('stop_type','unknown') for r in rs})})
    base={(r['rep'],r['trace_id']):r['prompt_ms']+r['predicted_ms'] for r in rows if r['format']=='Q8_0'}
    for fmt,value in summary.items():
        value['speedup_vs_Q8_0']=summary['Q8_0']['cycle_ms_median']/value['cycle_ms_median']
        rs=[r for r in rows if r['format']==fmt]
        value['paired_cycle_speedup_median']=statistics.median(base[(r['rep'],r['trace_id'])]/(r['prompt_ms']+r['predicted_ms']) for r in rs)
        value['max_load1']=max(r['host_load'][0] for r in rs)
    trace_cycles={fmt:{trace:statistics.median(r['prompt_ms']+r['predicted_ms'] for r in rows if r['format']==fmt and r['trace_id']==trace) for trace in p['trace_ids']} for fmt in summary}
    for fmt,value in summary.items():
        value['per_trace_cycle_ms_median']=trace_cycles[fmt]
        value['trace_median_speedup']=statistics.median(trace_cycles['Q8_0'][t]/trace_cycles[fmt][t] for t in p['trace_ids'])
    write(run/'evaluation.json',dict(rows=len(rows),tier=p['tier'],formats=summary))
    write(run/'verdict.json',dict(verdict=p['tier'].upper()+'-B4-TIMING-MEASURED',adoption='NOT-ASSESSED',
        boundary='Cycle timing of the three frozen b4 exports; varying output lengths, cold cache and ten paired traces. Review bookend drift/load and separate quality results; no automatic promotion.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','measure','evaluate'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--assets',type=Path)
    a=p.parse_args()
    evaluate(a.run) if a.action=='evaluate' else globals()[a.action](a.run,a.assets)
