#!/usr/bin/env python3
"""H3-S0 K0 control arms: root-cause the llama.cpp prompt-reuse anomaly.

K0a identical-repeat: same prompt twice with cache_prompt=true; the second
request must evaluate ~0 prompt tokens (>=95% cached).
K0b divergence-depth: prompts sharing a d-token prefix then diverging,
alternated on one slot; cached fraction vs d. Hypothesis under test: reuse
granularity ~= n_batch (512) -> step function.
CPU-only serving (b4 Q8, t8); no GPU claim needed. Fail-closed telemetry.
"""
import argparse, json, math, signal, statistics, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
from s1_gpu import DeadlineExceeded, write

MODEL = Path('/home/m0hawk/Documents/Sepalith/experiments/models/b4_qwen35_2b-Q8_0.gguf')
DEPTHS = [64, 128, 256, 384, 512, 768, 1024, 2048]
PORT = 18484


def completion(server_proc, prompt, cache=True, max_tokens=8):
    body = json.dumps(dict(prompt=prompt, n_predict=max_tokens, temperature=0,
                           cache_prompt=cache, stream=False)).encode()
    req = urllib.request.Request('http://127.0.0.1:%d/completion' % PORT, data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def prompt_n(data):
    return (data.get('timings') or {}).get('prompt_n', data.get('tokens_evaluated'))


def measure(run):
    import subprocess, os
    server_bin = Path(os.environ['S1_RUNTIME']) / 'llama-server'
    log = open(run / 'server.log', 'ab')
    argv = [str(server_bin), '-m', str(MODEL), '--port', str(PORT), '--host', '127.0.0.1',
            '-t', '8', '-c', '8192', '-ngl', '0', '--parallel', '1', '-lv', '2']
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        t0 = time.time()
        ready = False
        while time.time() - t0 < 240:
            try:
                completion(proc, 'x <- 1\n', cache=False, max_tokens=2)
                ready = True
                break
            except Exception:
                time.sleep(2)
        if not ready:
            raise SystemExit('server not ready')
        # build a long realistic prompt from repeated R-ish lines
        line = 'fit_%d <- lm(outcome ~ predictor1 + predictor2 + factor(group), data = subset(df, year > 2000))\n'
        full = ''.join(line % i for i in range(180))  # ~6k tokens, fits 8192 ctx
        results = {}
        # K0a
        r1 = completion(proc, full)
        r2 = completion(proc, full)
        n1, n2 = prompt_n(r1), prompt_n(r2)
        results['k0a_identical_repeat'] = dict(first_prompt_n=n1, second_prompt_n=n2,
                                               cached_fraction=round(1 - n2 / max(1, n1), 4),
                                               pass_95=(1 - n2 / max(1, n1)) >= 0.95)
        # K0b: divergence depth sweep
        sweep = []
        for d_lines in [8, 16, 32, 48, 64, 96, 128]:
            head = ''.join(line % i for i in range(d_lines))
            tailA = ''.join('variantA_%d <- mean(x_%d, na.rm = TRUE)\n' % (i, i) for i in range(24))
            tailB = ''.join('variantB_%d <- sum(y_%d, na.rm = TRUE)\n' % (i, i) for i in range(24))
            pA, pB = head + tailA, head + tailB
            a1 = completion(proc, pA)
            b1 = completion(proc, pB)   # diverges at depth d
            a2 = completion(proc, pA)   # back to A: reuses only the shared prefix
            nA1, nB1, nA2 = prompt_n(a1), prompt_n(b1), prompt_n(a2)
            sweep.append(dict(depth_lines=d_lines, first_A=nA1, divergent_B=nB1,
                              return_A=nA2))
            print(json.dumps(sweep[-1]), flush=True)
        results['k0b_divergence_sweep'] = sweep
        write(run / 'telemetry.json', results)
        ok = results['k0a_identical_repeat']['pass_95']
        print(json.dumps(dict(k0a_pass=ok)), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args()
    a.run.mkdir(parents=True, exist_ok=True)
    measure(a.run)
