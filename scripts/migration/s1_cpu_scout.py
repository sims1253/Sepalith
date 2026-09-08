#!/usr/bin/env python3
"""Bounded, explicitly reduced CPU ngram scout with fresh paired controls."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time

from s1_gpu import DeadlineExceeded, check_prompt_count, tokenize, validate_response, write
import spec_bench as bench

ARMS=['baseline']+[f'ngram-simple@{n}' for n in (2,3,4,8,16,48)]+['baseline-bookend']


def select(assets):
    by_class=bench.load_traces(assets/'traces.jsonl')
    return [r for cc in ('2k','8k') for r in bench.sample_traces(by_class[cc],20)]


def prepare(run,assets):
    traces=select(assets)
    if len(traces)!=40 or len({r['trace_id'] for r in traces})!=40:
        raise ValueError('Expected 20 unique traces per class')
    write(run/'prepared.json',dict(arms=ARMS,trace_ids=[r['trace_id'] for r in traces],
        expected_rows=320,ctx=10240,max_seconds=17970,
        deviations='20/class, one cold repetition, no warm pass; ngram only; baseline bookend',
        scope='CPU ngram wall scout; not the complete registered S1 experiment',
        promotion='Exploratory paired speed >=1.15x with exact output parity; requires full confirmation before adoption'))


def measure(run,assets):
    p=json.loads((run/'prepared.json').read_text());traces=select(assets);baseline={};counts={};server=None
    def expire(*_):raise DeadlineExceeded('CPU scout exceeded five-hour bound')
    old=signal.signal(signal.SIGALRM,expire);deadline=time.monotonic()+p['max_seconds'];signal.alarm(p['max_seconds'])
    try:
        with (run/'per_request.jsonl').open('x') as output:
            for arm in ARMS:
                _,flags,_=bench.arm_flags('baseline' if arm=='baseline-bookend' else arm,model=Path(os.environ['SCOUT_MODEL']))
                server=bench.SpecServer(Path(os.environ['SCOUT_MODEL']),18474,flags+['--seed','20260905'],
                    ctx=p['ctx'],threads=8,server=Path(os.environ['SCOUT_RUNTIME'])/'llama-server',
                    foreground=True,log_path=run/('server-'+arm+'.log'))
                try:
                    server.start(ready_timeout=300)
                    for tr in traces:
                        count=tokenize(server.port,tr['prompt']);check_prompt_count(count,p['ctx'],counts.get(tr['trace_id']))
                        start_load=os.getloadavg();res=bench.stream_completion(server.port,tr['prompt'],timeout=300)
                        tps,rate=bench.acceptance_from_response(res.timings)
                        row=dict(arm=arm,trace_id=tr['trace_id'],ctx_class=tr['ctx_class'],rep=0,
                            prompt_tokens=tr['prompt_tokens'],tokenized_prompt_tokens=count,n_prompt_srv=res.timings.get('prompt_n'),
                            gen_text=res.text,gen_tps=bench.gen_tps(res),ttft_ms=res.ttft_ms,wall_ms=res.wall_ms,
                            timings=res.timings,error=res.error,stop_hit=res.stop_hit,accept_tps=tps,accept_rate=rate,
                            draft_n=res.timings.get('draft_n'),draft_n_accepted=res.timings.get('draft_n_accepted'),
                            host_load_before=start_load,host_load_after=os.getloadavg(),at=time.time())
                        if arm=='baseline':baseline[tr['trace_id']]=res.text;counts[tr['trace_id']]=count
                        else:row['matches_baseline']=res.text==baseline[tr['trace_id']]
                        output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
                        validate_response(res)
                        if row['n_prompt_srv']!=count:raise ValueError('Served prompt count differs from tokenizer')
                        print(json.dumps(dict(arm=arm,trace_id=tr['trace_id'],gen_tps=row['gen_tps'])),flush=True)
                finally:
                    signal.alarm(0);server.stop();server=None
                    signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    finally:
        signal.alarm(0)
        try:
            if server is not None:server.stop()
        finally:signal.signal(signal.SIGALRM,old)


def evaluate(run):
    p=json.loads((run/'prepared.json').read_text());rows=[json.loads(l) for l in (run/'per_request.jsonl').read_text().splitlines()]
    if [(r['arm'],r['trace_id']) for r in rows]!=[(a,t) for a in p['arms'] for t in p['trace_ids']]:
        raise ValueError('Incomplete scout coverage')
    if any(r['error'] or not r['stop_hit'] or r['n_prompt_srv']!=r['tokenized_prompt_tokens'] for r in rows):
        raise ValueError('Invalid scout response')
    base={r['trace_id']:r for r in rows if r['arm']=='baseline'};summary={};candidates=[]
    for arm in p['arms']:
        for cc in ('2k','8k'):
            rs=[r for r in rows if r['arm']==arm and r['ctx_class']==cc]
            ratios=[r['gen_tps']/base[r['trace_id']]['gen_tps'] for r in rs]
            key=arm+'|'+cc
            summary[key]=dict(n=len(rs),gen_tps_median=statistics.median(r['gen_tps'] for r in rs),
                paired_speedup_median=statistics.median(ratios),paired_speedup_quartiles=statistics.quantiles(ratios,n=4),
                matches_baseline=sum(r.get('matches_baseline',False) for r in rs),
                max_load1=max(max(r['host_load_before'][0],r['host_load_after'][0]) for r in rs))
            if arm.startswith('ngram') and summary[key]['paired_speedup_median']>=1.15 and all(r['matches_baseline'] for r in rs):candidates.append(key)
    write(run/'evaluation.json',dict(rows=len(rows),arms_summary=summary))
    write(run/'verdict.json',dict(verdict='CPU-NGRAM-SCOUT-MEASURED',confirmation_candidates=candidates,
        adoption='NOT-ASSESSED',boundary='Exploratory small sample, cold-only; review drift/load and confirm candidates at full protocol. MTP/model-draft CPU depths remain unmeasured.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','measure','evaluate']);p.add_argument('--run',type=Path,required=True);p.add_argument('--assets',type=Path)
    a=p.parse_args();evaluate(a.run) if a.action=='evaluate' else globals()[a.action](a.run,a.assets)
