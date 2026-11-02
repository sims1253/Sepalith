"""C19 weights-only continuation from C18b 300 with a frozen hard-task mix."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Mapping, Sequence

from sepalith.training.grpo_exec import c18b, recipe as c18
from sepalith.training.grpo_exec.c18b import (  # noqa: F401
    CHECKPOINTS, DEPLOY_ROOT, STATE, RecipeError, check_file, file_record, obj,
    repository_root, require, sha256, write_new,
)

BOUND_SCHEMA = 'sepalith.c19.grpo-exec-bound.v1'
SOURCE_SCHEMA = 'sepalith.c19.grpo-exec-source.v1'
BOUND = 'docs/training/plan-20260927/c19/bound'
SOURCE_MANIFEST = f'{BOUND}/source-manifest.json'
START = CHECKPOINTS / 'C18b-grpo-exec-pilot-1/full/checkpoint-300'
# rig user mode (2026-11-03 user: install/use the version needed; /opt needs sudo). Portable build, same as admin mode's.
R_BIN = '/home/m0hawk/.local/share/rig/r/4.6.1/bin'
R_PATH = f'{R_BIN}/Rscript'
R_VERSION = 'R 4.6.1 (2026-06-24)'
PYTHON = Path('/home/m0hawk/sandbox/Sepalith/.venv-sft/bin/python')
FINAL_HOLDOUT_JOB_ID = 'C19-final-holdout'
PRE_REGISTERED = copy.deepcopy(c18b.PRE_REGISTERED)
PRE_REGISTERED.update(card='C19', start='C18b checkpoint 300, weights only, fresh optimizer; user go-ahead 2026-11-02',
                      prompts='C19-hard-mix-v1; 1-in-5 legacy unchanged')
PRE_REGISTERED['final_evaluation'] = {**obj(c18.PRE_REGISTERED['final_evaluation'], 'final evaluation'),
    'comparison': 'selected C19 checkpoint vs C18b 300, paired by task, function-clustered bootstrap (2,000 resamples)',
    'secondary': ['mean exec reward difference', 'exact-match difference', 'protocol validity',
                  'DEV250 score250 >= checkpoint 300 score250 minus 4', 'CPT 2K NLL rise <= 0.03 nats vs 300']}
SOURCE_FILES = (*c18b.SOURCE_FILES, 'packages/sepalith/src/sepalith/training/grpo_exec/final.py', *(f'packages/sepalith/src/sepalith/training/grpo_exec/{n}.py' for n in
                ('c19', 'c19_mix', 'c19_night', 'c19_trainer', 'c19_val_eval', 'c19_cpt_probe', 'c19_dev_generation', 'c19_jobs', 'c19_final')))
EVALUATOR_FILES = (*c18b.EVALUATOR_FILES, *SOURCE_FILES[len(c18b.SOURCE_FILES):])
ENTRY = {k: f'sepalith.training.grpo_exec.c19_{v}' for k, v in
         {'trainer': 'trainer', 'dev': 'dev_generation', 'val': 'val_eval', 'cpt': 'cpt_probe'}.items()}


def build_source_manifest(root: Path) -> dict:
    return {'schema': SOURCE_SCHEMA, 'root': 'repository',
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'files': [{'path': p, 'bytes': (root / p).stat().st_size, 'sha256': sha256(root / p)} for p in SOURCE_FILES],
            'evaluator_files': list(EVALUATOR_FILES)}


def evaluator_hashes(root: Path) -> dict:
    return {p: sha256(root / p) for p in EVALUATOR_FILES}


def verify_source(root: Path, record: Mapping) -> dict:
    path = check_file(root, {'path': record['manifest_path'], 'sha256': record['manifest_sha256']}, 'C19 source manifest')
    manifest = json.loads(path.read_text())
    require(manifest['schema'] == SOURCE_SCHEMA, 'wrong source schema')
    require([r['path'] for r in manifest['files']] == list(SOURCE_FILES), 'source list differs')
    for r in manifest['files']:
        require(sha256(root / r['path']) == r['sha256'], f"source differs: {r['path']}")
    return manifest


def validate(recipe: Mapping) -> None:
    require(recipe.get('schema') == BOUND_SCHEMA and recipe.get('card') == 'C19', 'not C19')
    require(recipe.get('pre_registered') == PRE_REGISTERED, 'C19 pre-registration differs')
    view = copy.deepcopy(dict(recipe))
    view.update(schema=c18b.BOUND_SCHEMA, card='C18b', pre_registered=c18b.PRE_REGISTERED)
    c18b.validate(view)
    require(recipe['parent']['path'] == str(START) and recipe['parent']['step'] == 300, 'wrong start')
    require(recipe['runtime']['reward']['rscript'] == R_PATH, 'R must use rig')
    require(recipe['runtime']['reward']['rscript_version'] == R_VERSION, 'wrong R version')
    require(recipe['runtime']['prompt_max_tokens'] == 3072 and recipe['runtime']['completion_max_tokens'] == 1024, 'caps differ')
    require(recipe['runtime']['learning_rate'] == 1e-6, 'LR differs')
    require(recipe['runtime']['min_free_disk_bytes'] == 150_000_000_000, 'disk floor differs')
    require(recipe['outputs'].get('trainer') == f"{recipe['outputs']['archive']}/runtime", 'trainer output is not bound')
    require(isinstance(recipe['parent_reference'].get('cpt_recorded_nll_fp32_norms'), float), 'CPT anchor is not bound')
    require(recipe['final_evaluation']['labels'] == ['c18b-300', 'c19-selected'], 'final labels differ')


def bind(kind: str, root: Path) -> dict:
    from sepalith.training.offline.rules import offline_metrics

    root = root.resolve()
    # Copy the immutable C18b binding; never rebind C18 or C18b.
    recipe = json.loads((root / 'docs/training/plan-20260927/c18b/bound/grpo-exec-pilot-1-arm-a-803.json').read_text())
    name = 'C19-grpo-exec-pilot-1' if kind == 'pilot' else 'C19-grpo-exec-smoke'
    recipe.update(schema=BOUND_SCHEMA, card='C19', kind=kind, pre_registered=copy.deepcopy(PRE_REGISTERED),
                  decided_by='2026-11-02 user: sounds good. Prepare that for tonight',
                  arm={'name': name, 'purpose': 'C19 hard-mix continuation from C18b 300, fresh optimizer'})
    recipe.pop('c18_reference', None)
    recipe['parent'] = {'candidate_id': 'C18b-300', 'kind': 'gated_full_weight_checkpoint', 'path': str(START),
                        'step': 300, 'tag': 'c18b-300', 'files': {p: sha256(START / p) for p in c18.START_FILES}}
    result_dir = STATE / 'evaluation-gates/C18b-grpo-exec-pilot-1/results'
    results = {}
    for panel in ('dev75', 'dev250', 'val', 'cpt'):
        matches = [p for p in result_dir.glob(f'{panel}-step-300.*.json') if not p.name.endswith('.admission.json') and '.rows.' not in p.name]
        require(len(matches) == 1, f'ambiguous/missing step-300 {panel}')
        p = matches[0]
        value = json.loads(p.read_text())
        require(value['checkpoint']['campaign_manifest_sha256'] == recipe['parent']['files']['campaign-manifest.json'],
                f'{panel} checkpoint mismatch')
        results[panel] = {'path': str(p), 'sha256': sha256(p)}
    values = {k: json.loads(Path(v['path']).read_text()) for k, v in results.items()}
    recipe['parent_reference'] = {'results': results, 'metrics': offline_metrics(values['dev75']['summary'], values['dev250']['summary']),
                                  'generation_runtime': recipe['parent_reference']['generation_runtime'],
                                  # read by every CPT probe; descriptive anchor only, R6x uses the bound start result
                                  'cpt_recorded_nll_fp32_norms': values['cpt']['mean_causal_nll']}
    receipt = json.loads((root / 'docs/training/plan-20260927/c19/mix-manifest.json').read_text())
    recipe['pool']['train'] = {k: receipt[k] for k in ('path', 'sha256', 'tasks', 'functions')}
    recipe['pool']['c19_mix_receipt'] = file_record(root, 'docs/training/plan-20260927/c19/mix-manifest.json')
    recipe['outputs'] = {**c18.outputs(name)}
    recipe['outputs']['trainer'] = f"{recipe['outputs']['archive']}/runtime"
    gate = recipe['dev_gate']
    gate['evidence_directory'] = recipe['outputs']['evidence_directory']
    gate['override_file_pattern'] = f"{gate['evidence_directory']}/override-step-<step>.json"
    gate['val']['limit'] = 96 if kind == 'smoke' else None
    recipe['smoke'] = {'stops': [4, 6], 'eval_step': 4, 'eval_panels': ['dev75', 'val', 'cpt']} if kind == 'smoke' else None
    recipe['runtime']['reward'].update(rscript=R_PATH, rscript_version=R_VERSION,
        exec_harness_r=f'PATH={R_BIN}:$PATH; c19_night checks exact Rscript --version line before launching; reward parser checks buffer version')
    recipe['source'] = {'repository_root': str(root), 'manifest_path': SOURCE_MANIFEST, 'manifest_sha256': sha256(root / SOURCE_MANIFEST)}
    final = recipe['final_evaluation']
    p = STATE / 'evaluation-gates/C18b-grpo-exec-pilot-1/final-step-300/completions-c18-selected.jsonl'
    final.update(start_completions={'path': str(p), 'sha256': sha256(p)}, labels=['c18b-300', 'c19-selected'],
                 job='sepalith.exec_eval.jobs run --kind holdout --job-id C19-final-holdout; final-step-<selected>')
    validate(recipe)
    return recipe


def load(path: Path, *, root_override: Path | None = None, check_source: bool = True, check_data: bool = True) -> dict:
    recipe = json.loads(path.read_text())
    validate(recipe)
    root = repository_root(recipe, root_override)
    if check_source:
        verify_source(root, recipe['source'])
    check_file(root, recipe['sft_recipe'], 'SFT recipe')
    if check_data:
        for key in ('manifest', 'train', 'val', 'replay', 'cpt_replay', 'c19_mix_receipt'):
            check_file(root, recipe['pool'][key], key)
        for key in ('manifest', 'pool', 'bundle'):
            check_file(root, recipe['pool']['legacy'][key], f'legacy {key}')
        check_file(root, recipe['memory_fit']['receipt'], 'memory fit')
        check_file(root, recipe['dev_gate']['cpt']['panel'], 'CPT panel')
        for panel, record in recipe['parent_reference']['results'].items():
            check_file(root, record, f'start {panel}')
        check_file(root, recipe['final_evaluation']['start_completions'], '300 holdout completions')
        for name, digest in recipe['parent']['files'].items():
            require(sha256(START / name) == digest, f'parent {name} differs')
        for key in ('air', 'rscript'):
            require(os.access(recipe['runtime']['reward'][key], os.X_OK),
                    f"{key} is not executable at {recipe['runtime']['reward'][key]}")
    return recipe


def install() -> None:
    from sepalith.training.grpo_exec import cpt_probe, dev_generation, jobs as job_module, night, trainer, val_eval
    from sepalith.training.grpo_exec.c19_night import C19Arm
    from sepalith.training.grpo_exec.c19_jobs import final_holdout_job

    c18b.install()
    this = sys.modules[__name__]
    for module in (night, trainer, val_eval, cpt_probe, dev_generation):
        module.recipe_module = this  # ty: ignore[invalid-assignment]
    for name, key in [('TRAINER', 'trainer'), ('DEV_EVALUATOR', 'dev'), ('VAL_EVALUATOR', 'val'), ('CPT_PROBE', 'cpt')]:
        setattr(night, name, ENTRY[key])
    night.ExecGRPOArm = C19Arm  # ty: ignore[invalid-assignment]
    job_module.final_holdout_job = final_holdout_job  # ty: ignore[invalid-assignment]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('bind', 'source-manifest', 'verify', 'write-jobs'))
    parser.add_argument('--kind', choices=('pilot', 'smoke'), default='pilot')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--repository-root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()
    if args.command == 'bind':
        write_new(args.output, bind(args.kind, root))
    elif args.command == 'source-manifest':
        write_new(args.output, build_source_manifest(root))
    elif args.command == 'write-jobs':
        from sepalith.training.grpo_exec.c19_jobs import jobs
        for key, job in jobs(root).items():
            write_new(args.output / f'{key}.json', job)
    else:
        recipe = load(args.recipe, root_override=root)
        print(json.dumps({'status': 'verified', 'arm': recipe['arm']['name'], 'sha256': sha256(args.recipe)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
