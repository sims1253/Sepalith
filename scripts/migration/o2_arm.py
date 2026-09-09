#!/usr/bin/env python3
"""O2 GRPO arms: filtered (prescreen-admitted) vs unfiltered control on the
merged b4 base, matched budget (300 steps, seed 3407). The ONLY delta between
arms is the admission filter. Foreground tracked child; readout parses
rl_metrics.jsonl (E1/O1 contract: first50 psg rate + mean reward)."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

PRESCREEN = Path('/mnt/h/sepalith/datasets/rl_o2_prescreen/prescreen_v1.jsonl')
MODEL = '/mnt/h/sepalith/runs/pft1_b4_merged'
DATA = '/mnt/h/sepalith/datasets/sft_v7/train.jsonl'
TRAINER = str(Path(__file__).resolve().parents[2] / 'experiments/training/rl_smoke.py')


def preflight():
    """Fail-closed BEFORE any GPU work: full provenance + ledger validation."""
    import hashlib, collections
    prov_path = PRESCREEN.parent / 'provenance.json'
    prov = json.loads(prov_path.read_text())
    data_sha = hashlib.sha256(open(DATA, 'rb').read()).hexdigest()
    if prov.get('seed') != 3407 or prov.get('k') != 16 or tuple(prov.get('band', ())) != (3, 13):
        raise SystemExit('provenance seed/k/band mismatch')
    if prov.get('data_sha256') != data_sha:
        raise SystemExit('data file content changed vs provenance')
    rows = [json.loads(l) for l in PRESCREEN.read_text().splitlines()]
    if len(rows) != 5578 or any(r['k'] != 16 for r in rows):
        raise SystemExit('ledger row count or k mismatch: %d' % len(rows))
    admitted = sum(1 for r in rows if r['admitted'])
    if admitted != 1592:
        raise SystemExit('admitted count mismatch: %d' % admitted)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/training'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
    import rl_smoke
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/home/m0hawk/Documents/Sepalith/experiments/models/qwen3.5-2b-base-text-hf')
    draw, _ = rl_smoke.build_dataset(tok, quotas=dict(rl_smoke.FAMILY_QUOTA), seed=3407, data_path=DATA)
    adm = {r['prompt_hash'] for r in rows if r['admitted']}
    kept = collections.Counter()
    for r in draw:
        if hashlib.sha1(r['prompt'].encode()).hexdigest() in adm:
            kept[r['family']] += 1
    if dict(kept) != {'rename_propagation': 322, 'format_propagation': 610, 'no_op': 4, 'pipe_rewrite': 6}:
        raise SystemExit('quota-draw keep mismatch: %r' % dict(kept))
    print(json.dumps(dict(preflight='ok', ledger_rows=len(rows), admitted=admitted,
                          quota_draw=len(draw), kept=dict(kept))), flush=True)


def train(run, arm):
    preflight()
    out = run / ('train-' + arm)
    out.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, TRAINER, '--steps', '300', '--model', MODEL,
            '--data', DATA, '--out', str(out)]
    if arm == 'filtered':
        argv += ['--prescreen', str(PRESCREEN)]
    env = dict(os.environ,
               UNSLOTH_COMPILE_DISABLE='1', UNSLOTH_DISABLE_AUTO_PADDING_FREE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    log = (run / ('train-' + arm + '.log')).open('w')
    t0 = time.time()
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env, cwd=str(out))
    rc = proc.wait()
    (run / ('train-' + arm + '.json')).write_text(json.dumps(
        dict(arm=arm, argv=argv, returncode=rc, seconds=round(time.time() - t0, 1))) + '\n')
    if rc != 0:
        raise SystemExit('trainer exited %d for arm %s' % (rc, arm))


def readout(run):
    out = {}
    for arm in ('filtered', 'control'):
        metrics = run / ('train-' + arm) / 'rl_metrics.jsonl'
        rows = [json.loads(l) for l in metrics.read_text().splitlines()] if metrics.exists() else []
        if len(rows) < 300:
            raise SystemExit('%s: expected >=300 metric lines, got %d' % (arm, len(rows)))
        # banked contract: the step-50 summary line carries pooled first50_* fields
        summary = next((r for r in rows if r.get('step') == 50 and 'first50_psg_rate' in r), None)
        if summary is None:
            raise SystemExit('%s: no step-50 summary line with first50 contract' % arm)
        out[arm] = {k: v for k, v in summary.items() if k.startswith('first50')}
    out['note'] = ('Matched-rollout comparison valid on first50 only (both arms repeat-free there); '
                   'full-300 curves are NOT matched (filtered cycles ~2.5 epochs vs control <1); '
                   'family mix differs by design - read per-family psg fields for within-family deltas.')
    (run / 'readout.json').write_text(json.dumps(out, indent=2) + chr(10))
    print(json.dumps(out))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['train', 'readout'])
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--arm', choices=['filtered', 'control'])
    a = ap.parse_args()
    train(a.run, a.arm) if a.action == 'train' else readout(a.run)
