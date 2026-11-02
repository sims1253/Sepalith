"""C19 worktree jobs and final verdict with checkpoint 300 labels."""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from typing import cast
from pathlib import Path
import shutil

from sepalith.training.grpo_exec import c19, jobs as base

START_LABEL, CANDIDATE_LABEL = 'c18b-300', 'c19-selected'


def command(root: Path, module: str, *args: str) -> list[str]:
    return base.command(root, module, *args, python=c19.PYTHON)[:1] + [
        'PATH=/opt/R/4.6.1/bin:' + os.environ.get('PATH', '')] + base.command(root, module, *args, python=c19.PYTHON)[1:]


def jobs(root: Path) -> dict:
    out = {}
    for kind, identifier, priority, minutes in (('smoke', 'C19-grpo-exec-smoke', 40, 45),
                                               ('pilot', 'C19-grpo-exec-pilot-1', 45, 960)):
        path = root / c19.BOUND / f'grpo-exec-{"pilot-1" if kind == "pilot" else "smoke"}-c18b-300.json'
        out[identifier] = {'id': identifier, 'cwd': str(root), 'env': [], 'priority': priority,
                           'estimated_minutes': minutes, 'resumable': kind == 'pilot',
                           'requires': ['C19-grpo-exec-smoke'] if kind == 'pilot' else [],
                           'command': command(root, 'sepalith.training.grpo_exec.c19_night',
                                              'run' if kind == 'pilot' else 'smoke', '--recipe', str(path)),
                           'note': 'C19 weights-only from C18b 300; hard mix; smoke 1-4/save/gates/resume to 6; '
                                   'pilot R6x and selection unchanged; never auto-enqueues final holdout.'}
    return out


def final_holdout_job(recipe: Mapping, selected: str, work_dir: Path, root: Path) -> dict:
    step = int(Path(selected).name.removeprefix('checkpoint-'))
    work_dir = work_dir.parent / f'final-step-{step}'
    # Reuse the battle-tested final builder in an isolated view of its globals.
    # Do not mutate C18/C18b labels or let a second selection reuse its outputs.
    final = recipe['final_evaluation']
    source = c19.check_file(root, final['start_completions'], '300 completions')
    work_dir.mkdir(parents=True, exist_ok=True)
    seeded = work_dir / f'completions-{START_LABEL}.jsonl'
    if not seeded.exists():
        shutil.copyfile(source, seeded)
    c19.require(c19.sha256(seeded) == final['start_completions']['sha256'], 'seeded completions differ')
    models = work_dir / 'inputs/models.json'
    value = {'models': [{'label': START_LABEL, 'name': 'C18b checkpoint 300', 'path': recipe['parent']['path']},
                        {'label': CANDIDATE_LABEL, 'name': 'C19 selected', 'path': selected}]}
    if models.exists():
        c19.require(json.loads(models.read_text()) == value, 'stale final model inputs')
    else:
        c19.write_new(models, value)
    tasks = final['holdout_tasks']
    return {'id': c19.FINAL_HOLDOUT_JOB_ID, 'cwd': str(root), 'env': [], 'priority': 30, 'resumable': True,
            'estimated_minutes': 120, 'requires': [],
            'command': command(root, 'sepalith.training.grpo_exec.c19_jobs', 'run-final', '--kind', 'holdout',
                               '--job-id', c19.FINAL_HOLDOUT_JOB_ID, '--work-dir', str(work_dir), '--tasks', tasks['path'],
                               '--tasks-sha256', tasks['sha256'], '--models', str(models)),
            'note': 'C19 pre-registered holdout: selected vs pinned C18b 300; verdict via c19_jobs verdict.'}


def verdict(summary_path: Path, metrics_path: Path) -> dict:
    from sepalith.training.grpo_exec import rules

    summary, metrics = json.loads(summary_path.read_text()), json.loads(metrics_path.read_text())
    selected = summary['selected']
    out = rules.final_verdict(metrics[CANDIDATE_LABEL][f'vs_{START_LABEL}'],
        dev250_candidate=selected['dev250']['score250'], dev250_start=summary['start']['metrics']['score250'],
        cpt_rise=selected['cpt_rise_vs_start'])
    cast(dict, out['primary'])['endpoint'] = 'hold-out all-pass difference vs C18b 300 (paired, function-clustered 95% CI)'
    out['protocol_validity'] = {k: metrics[k]['overall']['protocol_valid'] for k in (START_LABEL, CANDIDATE_LABEL)}
    return out


def main() -> int:
    import argparse
    import sys
    from sepalith.training.grpo_exec.c19_night import guard

    if sys.argv[1:2] == ['run-final']:
        from sepalith.exec_eval import jobs as evaluator
        guard()
        return evaluator.main(sys.argv[2:])
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('verdict',))
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verdict(args.summary, args.metrics), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
