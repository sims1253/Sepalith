"""Measure serial uncached native llama-server requests on fixed PRM03 contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from urllib.request import Request, urlopen

from sepalith.campaign_protocol import PromptContext, render_prompt


def percentile(values: list[float], p: float) -> float:
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('metrics must be nonempty, finite and nonnegative')
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def measure(lines, started: float, clock=time.perf_counter) -> dict:
    first = None
    final = None
    for line in lines:
        if not line.startswith(b'data: '):
            continue
        payload = line[6:].strip()
        if payload == b'[DONE]':
            break
        event = json.loads(payload)
        if 'error' in event:
            raise ValueError(str(event['error']))
        if first is None and event.get('content'):
            first = clock()
        if event.get('stop'):
            final = event
    if first is None or final is None or final.get('truncated'):
        raise ValueError('missing first content, final event, or truncated prompt')
    timing = final['timings']
    count, rate = timing['predicted_n'], timing.get('predicted_per_second')
    # llama-server excludes the prefill-sampled first token from its decode rate.
    if count < 2 or not rate or rate <= 0:
        raise ValueError('invalid decode timing')
    return {'ttft_ms': (first - started) * 1000, 'tokens_s': rate,
            'predicted_n': count, 'prompt_n': timing['prompt_n'],
            'stop_type': final.get('stop_type'), 'wall_ms': (clock() - started) * 1000}


def request(url: str, endpoint: str, body: dict):
    return urlopen(Request(url + endpoint, data=json.dumps(body).encode(),
                   headers={'Content-Type': 'application/json'}), timeout=300)


def run(url: str, contexts: list[dict], repeats: int) -> dict:
    if repeats < 1 or not contexts:
        raise ValueError('need contexts and positive repeats')
    records = []
    for index, ctx in enumerate(contexts):
        prompt = render_prompt(PromptContext.from_dict(ctx))
        with request(url, '/tokenize', {'content': prompt, 'add_special': False, 'parse_special': False}) as r:
            ids = [0, *json.load(r)['tokens']]
        body = {'prompt': ids, 'n_predict': 1024, 'temperature': 0, 'top_k': 1,
                'samplers': ['top_k'], 'repeat_penalty': 1, 'seed': 3407,
                'stop': [], 'ignore_eos': False, 'cache_prompt': False, 'stream': True}
        for repeat in range(-1, repeats):  # one warm-up per prompt, excluded
            started = time.perf_counter()
            with request(url, '/completion', body) as response:
                row = measure(response, started)
            if repeat >= 0:
                records.append({'prompt': index, 'repeat': repeat, **row})
    summary = {key: {f'p{p}': percentile([r[key] for r in records], p / 100) for p in (50, 95)}
               for key in ('ttft_ms', 'tokens_s', 'wall_ms')}
    return {'samples': records, 'summary': summary}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:18099')
    p.add_argument('--prompts', type=Path, default=Path('docs/training/plan-20260927/c17/prompts.json'))
    p.add_argument('--backend', choices=['cpu', 'vulkan'], required=True)
    p.add_argument('--tier', choices=['Q6_K', 'Q4_K_M-mixB-e4'], required=True)
    p.add_argument('--runtime-commit', required=True)
    p.add_argument('--model-sha256', required=True)
    p.add_argument('--server-command', required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = run(a.url, json.loads(a.prompts.read_text()), a.repeats)
    result.update(schema='sepalith.c17.notebook-latency.v1', backend=a.backend, tier=a.tier,
                  runtime_commit=a.runtime_commit, model_sha256=a.model_sha256,
                  server_command=a.server_command, host=platform.platform(), processor=platform.processor(),
                  prompts_sha256=hashlib.sha256(a.prompts.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
