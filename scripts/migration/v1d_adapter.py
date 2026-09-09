#!/usr/bin/env python3
"""Adapt V1a runner archives (requests.jsonl per-point rows) into the
episode_judged_<arm>.jsonl format pairwise_pref.py's load_points expects."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, '/home/m0hawk/Documents/Sepalith/experiments/eval')
from eval_noop_fp import parse_prediction

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.requests.read_text().splitlines()]
    episodes = {}
    for r in rows:
        key = r.get('key') or r.get('episode')
        variant = r.get('variant', 1)
        ep = episodes.setdefault((key, variant), dict(key=key, variant=variant,
                                                      goal=r.get('goal', ''),
                                                      points=[]))
        ep['points'].append(dict(t_ms=r['point'].get('t_ms'),
                                 ctx=r['point'].get('ctx'),
                                 prompt=r['prompt'],
                                 gt=r['point'].get('gt'),
                                 label=r['point'].get('label'),
                                 proposal=r['response']['choices'][0]['text']))
    n = 0
    with a.out.open('x') as f:
        for ep in episodes.values():
            f.write(json.dumps(ep, allow_nan=False) + chr(10))
            n += len(ep['points'])
    print(json.dumps(dict(episodes=len(episodes), points=n, arm=a.arm)))

if __name__ == '__main__':
    main()
