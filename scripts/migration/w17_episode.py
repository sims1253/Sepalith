#!/usr/bin/env python3
"""W17 episode leg: serve one model over the V1a 1,208-point episode cohort, record
label/decision-equivalent outcomes (proposal parsed, matched the episode's own
decision surface) for the RL-vs-SFT episode column."""
import argparse, json, signal, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
from s1_gpu import GpuServer, DeadlineExceeded
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/migration'))
import eval_noop_fp as noop

EPISODES = Path('/mnt/h/sepalith/runs/runner-v1a-b4-archive-20260908/8c2ed0c3bbaa46c2b1bdc1e5dceb7d70/episodes.jsonl')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(l) for l in EPISODES.read_text().splitlines()]
    points = [p for e in eps for p in e['points']]
    if len(points) != 1208:
        raise SystemExit('cohort changed: %d' % len(points))
    server = None
    def expire(*_):
        raise DeadlineExceeded('W17 episode leg exceeded bound')
    old = signal.signal(signal.SIGALRM, expire); signal.alarm(5400)
    import urllib.request
    try:
        server = GpuServer(a.model, 18478, ['-lv', '4', '--seed', '20260905'], ctx=32768,
                           server=Path(__import__('os').environ['S1_RUNTIME']) / 'llama-server',
                           foreground=True, log_path=a.out.parent / ('server-%s.log' % a.name))
        server.start(ready_timeout=300)
        with a.out.open('x') as f:
            for i, p in enumerate(points):
                body = json.dumps(dict(prompt=p['prompt'], n_predict=160, temperature=0,
                                       top_p=1.0, top_k=0, min_p=0.0, stream=False,
                                       stop=['>>>>>>>'])).encode()
                req = urllib.request.Request('http://127.0.0.1:%d/v1/completions' % server.port,
                                             data=body, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.load(r)
                text = data['choices'][0]['text']
                rec = dict(i=i, label=p['label'], b4_decision=p['decision'],
                           proposed=bool(noop.parse_prediction(text)),
                           stop=data['choices'][0]['finish_reason'], n_tok=len(data.get('tokens') or []),
                          text=text, b4_proposal=p.get('proposal'), b4_pred_head=p.get('pred_head'))
                f.write(json.dumps(rec) + '\n'); f.flush()
                if i % 200 == 0:
                    print(json.dumps(dict(i=i)), flush=True)
    finally:
        signal.alarm(0)
        if server is not None:
            server.stop()
        signal.signal(signal.SIGALRM, old)
    print('DONE', a.name)

if __name__ == '__main__':
    main()
