"""Freeze C19's census-weighted subset without changing task bytes or the sampler."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from sepalith.training.grpo_exec import pool


def stratum(family: str, lines: int, passes: int) -> str | None:
    if lines > 16:
        return None
    if passes == 8:
        return 'SOLVED'
    if 1 <= passes <= 7:
        return 'MIXED'
    if passes == 0 and family == 'finish_block' and 3 <= lines <= 16:
        return 'HARD'
    return None


def lines_bin(n: int) -> str:
    return '1' if n < 2 else '2' if n == 2 else '3-4' if n <= 4 else '5-8' if n <= 8 else '9-16' if n <= 16 else '17+'


def select(tasks: list[dict], records: list[dict], excluded_functions: set[str],
           excluded_packages: set[str], seed: int = 3407) -> tuple[list[dict], dict]:
    by_id = {t['id']: t for t in tasks}
    samples: dict = defaultdict(dict)
    non_pool = set()
    for r in records:
        identifier = r['task_id']
        if identifier not in by_id:
            non_pool.add(identifier)
            continue
        sample = r['sample']
        if sample in samples[identifier]:
            pool.require(samples[identifier][sample] == r, f'conflicting duplicate {identifier}:{sample}')
        samples[identifier][sample] = r
    strata: dict = defaultdict(list)
    for identifier, group in sorted(samples.items()):
        if set(group) != set(range(8)):
            continue
        task = by_id[identifier]
        pool.require(task['function_id'] not in excluded_functions and task.get('package') not in excluded_packages,
                     f'hold-out/validation leak: {identifier}')
        first = group[0]
        pool.require(all((r['family'], r['target_lines']) == (first['family'], first['target_lines']) for r in group.values()),
                     f'inconsistent census metadata: {identifier}')
        pool.require(first['family'] == task['family'], f'family differs: {identifier}')
        kind = stratum(first['family'], first['target_lines'], sum(bool(r['all_pass']) for r in group.values()))
        if kind:
            strata[kind].append(identifier)
    n = len(strata['HARD'])
    pool.require(n > 0, 'empty HARD stratum')
    counts = {'HARD': n, 'MIXED': round(n * 25 / 60), 'SOLVED': round(n * 15 / 60)}
    rng = random.Random(seed)
    chosen = []
    for kind, count in counts.items():
        pool.require(len(strata[kind]) >= count, f'short {kind} stratum: {len(strata[kind])} < {count}')
        chosen.extend(strata[kind] if kind == 'HARD' else rng.sample(sorted(strata[kind]), count))
    rows = [by_id[i] for i in sorted(chosen)]
    info = {'strata': counts, 'eligible_strata': {k: len(v) for k, v in strata.items()}, 'non_pool_tasks': len(non_pool),
            'incomplete_tasks': sum(set(g) != set(range(8)) for g in samples.values()), 'seed': seed,
            'functions': len({t['function_id'] for t in rows}), 'family': dict(Counter(t['family'] for t in rows)),
            'bucket': dict(Counter(samples[t['id']][0]['bucket'] for t in rows)),
            'lines_bin': dict(Counter(lines_bin(samples[t['id']][0]['target_lines']) for t in rows))}
    return rows, info


def freeze(root: Path) -> dict:
    from sepalith.training.grpo_exec import c18b

    recipe = json.loads((root / 'docs/training/plan-20260927/c18b/bound/grpo-exec-pilot-1-arm-a-803.json').read_text())
    train, val = (Path(recipe['pool'][k]['path']) for k in ('train', 'val'))
    inputs = [c18b.STATE / 'exec-eval' / name / 'records-c18b-300.jsonl'
              for name in ('C18b-300-train-census', 'C18b-300-pool-census')]
    out = Path('/mnt/e/sepalith/campaign-20260915/data-work/C19-hard-mix-v1/train.jsonl')
    pool.no_sealed(train, val, out, *inputs)
    pool.require(not out.exists() and not out.with_suffix('.receipt.json').exists(), 'freeze output already exists')
    for key, path in (('train', train), ('val', val)):
        pool.require(pool.sha256(path) == recipe['pool'][key]['sha256'], f'{key} differs from C18b pin')
    pins = [{'path': str(p), 'sha256': pool.sha256(p)} for p in inputs]
    records = [r for p in inputs for r in pool.read_jsonl(p)]
    pool.require(all(pool.sha256(p) == pin['sha256'] for p, pin in zip(inputs, pins, strict=True)), 'census changed during freeze')
    held, packages = pool.holdout_identity()
    held |= {str(r['function_id']) for r in pool.read_jsonl(val)}
    rows, info = select(pool.read_jsonl(train), records, held, packages)
    # Preserve each original serialized row, including ExecReward's task metadata.
    selected = {r['id'] for r in rows}
    raw = b''.join(line for line in train.read_bytes().splitlines(keepends=True) if json.loads(line)['id'] in selected)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        stream.write(raw)
    receipt = {**info, 'path': str(out), 'sha256': pool.sha256(out), 'tasks': len(rows), 'inputs': pins,
               'source_train': recipe['pool']['train'], 'validation': recipe['pool']['val'],
               'rule': 'complete k=8; in C18 train only; HARD finish_block 0/8 lines 3..16; MIXED 1..7/8 <=16; SOLVED 8/8 <=16; '
                       'all HARD, round(N*25/60) MIXED, round(N*15/60) SOLVED; seeded draw over sorted ids; legacy on top'}
    pool.write_json_new(out.with_suffix('.receipt.json'), receipt)
    pool.write_json_new(root / 'docs/training/plan-20260927/c19/mix-manifest.json', receipt)
    return receipt
