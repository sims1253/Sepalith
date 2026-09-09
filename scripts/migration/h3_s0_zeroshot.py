#!/usr/bin/env python3
"""H3-S0 zero-shot format-fail bench: render the frozen 513-cohort's scenario
cases in each surviving variant, generate zero-shot on b4 Q8 (temp 0), count
format_fail per variant. The incumbent zeta2 arm is the in-run control.
Expected (pre-registered): unseen orderings/markers fail zero-shot — the
measurement prices the S1 adaptation cost per variant; it does not kill on
quality (S1's job)."""
import argparse, json, os, signal, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/migration'))
import render_variants as rv
from s1_gpu import GpuServer, DeadlineExceeded, check_offload, write

N_CASES = 100
VARIANTS = ['zeta2', 'v10_psmtail_merge', 'v11_psmtail_merge_empty', 'v05_psm_merge',
            'v06_psm_merge_empty', 'v14_psm_merge_nohist', 'v13_zeta1_alpaca']


def cases(assets):
    rows = [json.loads(l) for l in (assets / 'cases.jsonl').read_text().splitlines()]
    sc = [r for r in rows if r['kind'] == 'scenario']
    # deterministic first-N of the frozen cohort (order = frozen file order)
    return sc[:N_CASES]


def render_case(name, row):
    render, _ = rv.VARIANTS[name]
    # the frozen scenario rows carry the ORIGINAL zeta2 prompt; re-render from
    # the scenario example: rebuild the example dict the render functions take
    ex = row['scenario']
    return render(ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    a = ap.parse_args()
    a.run.mkdir(parents=True, exist_ok=True)
    rows = cases(a.assets)
    if len(rows) != N_CASES:
        raise SystemExit('case cohort incomplete')
    server = None

    def expire(*_):
        raise DeadlineExceeded('H3-S0 bench exceeded bound')
    old = signal.signal(signal.SIGALRM, expire); signal.alarm(3600)
    try:
        server = GpuServer(Path('/home/m0hawk/Documents/Sepalith/experiments/models/b4_qwen35_2b-Q8_0.gguf'),
                           18478, ['-lv', '4', '--seed', '20260905'], ctx=8192,
                           server=Path(os.environ['S1_RUNTIME']) / 'llama-server', foreground=True,
                           log_path=a.run / 'server.log')
        server.start(ready_timeout=300); check_offload(server.log_path.read_text())
        results = {}
        with (a.run / 'requests.jsonl').open('x') as out:
            for name in VARIANTS:
                render, parse = rv.VARIANTS[name]
                fails = {}
                for i, row in enumerate(rows):
                    prompt = render(row['scenario'])
                    body = json.dumps(dict(prompt=prompt, n_predict=192, temperature=0,
                                           stream=False, stop=rv.STOP_STRINGS)).encode()
                    req = urllib.request.Request('http://127.0.0.1:%d/completion' % server.port,
                                                 data=body, headers={'Content-Type': 'application/json'})
                    started = time.monotonic()
                    with urllib.request.urlopen(req, timeout=300) as r:
                        data = json.load(r)
                    text = data.get('content', '')
                    stopped = data.get('stop_type') in ('word', 'eos')
                    is_fail, reason = rv.format_fail(text, name, stopped)
                    fails[reason] = fails.get(reason, 0) + 1
                    out.write(json.dumps(dict(variant=name, case=i, fail=is_fail, reason=reason,
                                              prompt_tokens=data.get('tokens_evaluated'),
                                              wall_s=round(time.monotonic() - started, 3))) + chr(10))
                    out.flush()
                n_fail = sum(v for k, v in fails.items() if k)
                results[name] = dict(n=N_CASES, format_fail=n_fail, pass_n=N_CASES - n_fail,
                                     fail_reasons=fails)
                print(json.dumps(results[name]), flush=True)
        write(a.run / 'evaluation.json', dict(
            results=results,
            boundary='Zero-shot compliance tax only; S1 adaptation is where variants compete on '
                     'quality+cache. b4 was trained on zeta2 only; failures here price adaptation, '
                     'they do not kill candidates (the pre-registered kill is on S1 parity+serving).'))
    finally:
        signal.alarm(0)
        if server is not None:
            server.stop()
        signal.signal(signal.SIGALRM, old)


if __name__ == '__main__':
    main()
