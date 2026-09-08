#!/usr/bin/env python3
"""Summarize paired uncertainty without promoting mismatching speculative arms."""
import argparse
import json
from pathlib import Path
import random
import statistics
import textwrap


def paired_interval(rows, baseline, seed=20260908, draws=1000):
    groups={}
    for row in rows:
        key=(row['trace_id'],row['rep'])
        if key not in baseline:raise ValueError('Missing paired baseline')
        groups.setdefault(row['trace_id'],[]).append((row['gen_tps'],baseline[key]['gen_tps']))
    ids=sorted(groups);rng=random.Random(seed);samples=[]
    for _ in range(draws):
        pairs=[pair for key in rng.choices(ids,k=len(ids)) for pair in groups[key]]
        samples.append(statistics.median(p[0] for p in pairs)/statistics.median(p[1] for p in pairs))
    cuts=statistics.quantiles(samples,n=40,method='inclusive')
    return [cuts[0],cuts[-1]]


def readout(run):
    # A successful evaluator is a prerequisite; this does not summarize partial runs.
    evaluation=json.loads((run/'evaluation.json').read_text())
    rows=[json.loads(l) for l in (run/'per_request.jsonl').read_text().splitlines()]
    baseline={(r['trace_id'],r['rep']):r for r in rows if r['arm']=='baseline'}
    invalid_ngram = run.name == '0ee707385c5648a2883c465ac6cc38a7'
    result={}
    for key,stats in evaluation['arms_summary'].items():
        arm,cc=key.split('|');rs=[r for r in rows if r['arm']==arm and r['ctx_class']==cc]
        if not rs:raise ValueError('Missing summary rows')
        result[key]=dict(n=len(rs),depth_valid=not (invalid_ngram and arm.startswith('ngram-simple')),speedup=stats['speedup_vs_baseline'],
            paired_trace_bootstrap_95=paired_interval(rs,baseline),
            cold_matches=stats.get('matches_baseline'),warm_matches=stats.get('warm_matches_baseline'),
            cold_warm_matches=stats.get('cold_warm_matches'),
            max_load1=max(r['host_load'][0] for r in rs),
            gen_tps=stats['gen_tps_med'],cold_ttft_ms=stats['ttft_ms_med'],warm_ttft_ms=stats['warm_ttft_ms_med'])
    return dict(attempt=run.name,interval_method='1000 paired trace-cluster resamples; ratio of request medians; seed20260908. Intervals are pointwise, not selection-adjusted; they describe this run, not between-day environment uncertainty.',
        ngram_depth_error='Ngram labels changed draft.n_max, which does not control ngram in b10453; all six labels used default size-m=48.' if invalid_ngram else None,
        scientific_boundary='Speed alone does not establish adoption; retain output-parity failures and baseline-bookend drift.',arms=result)


def plot(record,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families=[('ngram-simple',[2,3,4,8,16,48]),('draft-mtp',[1,2,3,4,5]),('model-draft',[1,2,3,4,5])]
    families = [(family, [n for n in depths if f'{family}@{n}|2k' in record['arms']]) for family, depths in families]
    families = [(family, depths) for family, depths in families if depths]
    fig,axes=plt.subplots(1,len(families),figsize=(max(8,4*len(families)),4),sharey=True,squeeze=False)
    axes=axes[0]
    for ax,(family,depths) in zip(axes,families):
        if family == 'ngram-simple' and record.get('ngram_depth_error'):
            ax.text(.5,.5,'Depth control invalid\nSix repeats at default size-m=48',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(family);ax.set_xlabel('Corrected sweep required')
            continue
        for cc,color in [('2k','#2463A0'),('8k','#C46322')]:
            points=[record['arms'][f'{family}@{n}|{cc}'] for n in depths]
            ys=[p['speedup'] for p in points]
            ax.plot(range(len(depths)),ys,color=color,label=cc,linewidth=1.4)
            for i,(p,y) in enumerate(zip(points,ys)):
                low,high=p['paired_trace_bootstrap_95']
                ax.vlines(i,low,high,color=color,linewidth=1)
                valid=p['cold_matches']==p['n'] and p['warm_matches']==p['n']
                ax.scatter([i],[y],edgecolors=color,facecolors=color if valid else 'white',s=40,zorder=3)
        ax.axhline(1,color='#777777',linewidth=1)
        ax.axhline(1.4,color='#777777',linewidth=1,linestyle='--')
        ax.set_xticks(range(len(depths)),[str(n) for n in depths]);ax.set_xlabel('Draft-token limit');ax.set_title(family)
        ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Decode speed / fresh GPU baseline')
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
            break
    fig.suptitle('S1: RTX 5090 speculative-depth sweep',x=.07,ha='left',fontsize=14)
    fig.text(.07,.015,textwrap.fill('Lines: ratio of medians. Bars: paired trace-bootstrap 95% interval. Hollow dots: cold or warm output-parity failure. Dashed: 1.4× gate.',width=110 if len(families)==1 else 200),fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.93));fig.savefig(path);plt.close(fig)
    if path.suffix == '.svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path)
    a=p.parse_args();record=readout(a.run)
    with a.output.open('x') as f:json.dump(record,f,indent=2)
    if a.plot:plot(record,a.plot)
