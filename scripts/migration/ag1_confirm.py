#!/usr/bin/env python3
"""AG1 held-out confirmation: apply the FROZEN entropy gate (mean_entropy_norm
> 0.0702 -> abstain) to the V1a b4 episode cohort (1,208 new points, labels
from the banked lexical-accept decisions). Extract logprobs on b4 Q8 with the
V1a generation settings, compute the feature, report suppression/retention."""
import argparse, json, math, os, signal, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/migration'))
from s1_gpu import GpuServer, DeadlineExceeded, check_offload, write
from ag1_gate1 import features_from_response

THRESHOLD = 0.07021492105215514
MODEL = Path('/home/m0hawk/Documents/Sepalith/experiments/models/b4_qwen35_2b-Q8_0.gguf')
EPISODES = Path('/mnt/h/sepalith/runs/runner-v1a-b4-archive-20260908/8c2ed0c3bbaa46c2b1bdc1e5dceb7d70/episodes.jsonl')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args()
    a.run.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(l) for l in EPISODES.read_text().splitlines()]
    points = [p for e in eps for p in e['points']]
    if len(points) != 1208:
        raise SystemExit('cohort changed: %d' % len(points))
    server = None

    def expire(*_):
        raise DeadlineExceeded('AG1 confirm exceeded bound')
    old = signal.signal(signal.SIGALRM, expire); signal.alarm(5400)
    try:
        server = GpuServer(MODEL, 18478, ['-lv', '4', '--seed', '20260905'], ctx=32768,
                           server=Path(os.environ['S1_RUNTIME']) / 'llama-server', foreground=True,
                           log_path=a.run / 'server.log')
        server.start(ready_timeout=300); check_offload(server.log_path.read_text())
        with (a.run / 'confirm.jsonl').open('x') as out:
            for i, p in enumerate(points):
                body = json.dumps(dict(prompt=p['prompt'], n_predict=160, temperature=0,
                                       top_p=1.0, top_k=0, min_p=0.0, n_probs=8,
                                       stop=['>>>>>>>'], stream=False)).encode()
                req = urllib.request.Request('http://127.0.0.1:%d/completion' % server.port,
                                             data=body, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.load(r)
                feats, fms = features_from_response(data)
                m = (feats or {}).get('mean_entropy_norm')
                out.write(json.dumps(dict(i=i, label=p['label'], decision=p['decision'],
                                          mean_entropy_norm=m,
                                          abstain=(m is not None and m > THRESHOLD))) + chr(10))
                if i % 200 == 0:
                    print(json.dumps(dict(i=i)), flush=True)
    finally:
        signal.alarm(0)
        if server is not None:
            server.stop()
        signal.signal(signal.SIGALRM, old)
    rows = [json.loads(l) for l in (a.run / 'confirm.jsonl').read_text().splitlines()]
    import collections
    stats = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        key = (r['label'], r['decision'])
        stats[key]['n'] += 1
        stats[key]['abstained'] += r['abstain']
        stats[key]['no_feature'] += r['mean_entropy_norm'] is None
    out = {f'{k[0]}/{k[1]}': dict(v) for k, v in stats.items()}
    fs = stats[('noop', 'false_suggestion')]
    acc = stats[('typing', 'accepted')]
    dis = stats[('typing', 'dismissed')]
    report = dict(per_class=out,
                  fp_suppression=round(fs['abstained'] / max(1, fs['n'] - fs['no_feature']), 4),
                  accepted_retention=round(1 - acc['abstained'] / max(1, acc['n']), 4),
                  dismissed_suppression=round(dis['abstained'] / max(1, dis['n'] - dis['no_feature']), 4),
                  threshold=THRESHOLD, note='frozen gate; new-case cohort (V1a episodes); '
                                            'retention n=49 so one loss = 97.96%')
    write(a.run / 'evaluation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
