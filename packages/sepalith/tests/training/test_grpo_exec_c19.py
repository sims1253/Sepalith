import unittest
from sepalith.training.grpo_exec import c19_mix


class C19MixTests(unittest.TestCase):
    def test_strata_boundaries(self):
        for lines in (2, 3, 16, 17):
            for passes in (0, 1, 7, 8):
                expected = ('SOLVED' if passes == 8 and lines <= 16 else
                            'MIXED' if passes in (1, 7) and lines <= 16 else
                            'HARD' if passes == 0 and 3 <= lines <= 16 else None)
                self.assertEqual(c19_mix.stratum('finish_block', lines, passes), expected)
        self.assertIsNone(c19_mix.stratum('rename_propagation', 3, 0))

    def fixture(self):
        tasks, records = [], []
        for i, passes in enumerate([0] * 12 + [4] * 10 + [8] * 10):
            identifier = str(i)
            tasks.append({'id': identifier, 'function_id': identifier, 'family': 'finish_block'})
            records.extend({'task_id': identifier, 'sample': s, 'family': 'finish_block', 'target_lines': 4,
                            'bucket': 'short', 'all_pass': s < passes} for s in range(8))
        return tasks, records

    def test_deterministic_shares_and_incomplete_nonpool_exclusion(self):
        tasks, records = self.fixture()
        rows, info = c19_mix.select(tasks, records, set(), set())
        self.assertEqual(info['strata'], {'HARD': 12, 'MIXED': 5, 'SOLVED': 3})
        self.assertEqual(len(rows), 20)
        self.assertEqual((rows, info), c19_mix.select(tasks[::-1], records[::-1], set(), set()))
        rows, info = c19_mix.select(tasks[1:], records[:-1], set(), set())
        self.assertEqual(info['non_pool_tasks'], 1)
        self.assertEqual(info['incomplete_tasks'], 1)
        self.assertNotIn('31', {r['id'] for r in rows})

    def test_short_stratum_and_leak_fail_closed(self):
        tasks, records = self.fixture()
        with self.assertRaisesRegex(ValueError, 'short MIXED'):
            c19_mix.select(tasks, records[:12 * 8], set(), set())
        with self.assertRaisesRegex(ValueError, 'leak'):
            c19_mix.select(tasks, records, {'0'}, set())

    def test_final_labels_job_and_verdict(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from sepalith.training.grpo_exec import c19, c19_jobs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'completions.jsonl'
            source.write_text('{}\n')
            recipe = {'parent': {'path': str(c19.START)}, 'final_evaluation': {
                'start_completions': {'path': str(source), 'sha256': c19.sha256(source)},
                'holdout_tasks': {'path': '/tasks.jsonl', 'sha256': 'abc'}}}
            job = c19_jobs.final_holdout_job(recipe, '/checkpoint-50', root / 'final', root)
            self.assertEqual(job['id'], 'C19-final-holdout')
            self.assertIn(str(root / 'final-step-50'), job['command'])
            models = json.loads((root / 'final-step-50/inputs/models.json').read_text())
            self.assertEqual([r['label'] for r in models['models']], ['c18b-300', 'c19-selected'])
            self.assertEqual((root / 'final-step-50/completions-c18b-300.jsonl').read_bytes(), source.read_bytes())
            summary = root / 'summary.json'
            summary.write_text(json.dumps({'start': {'metrics': {'score250': 112}},
                                          'selected': {'dev250': {'score250': 108}, 'cpt_rise_vs_start': 0.03}}))
            metrics = root / 'metrics.json'
            metrics.write_text(json.dumps({'c18b-300': {'overall': {'protocol_valid': 1}}, 'c19-selected': {
                'overall': {'protocol_valid': 1}, 'vs_c18b-300': {
                    k: {'mean': 0.02, 'ci95': [0.01, 0.03]} for k in ('all_pass', 'reward', 'exact_match')}}}))
            self.assertTrue(c19_jobs.verdict(summary, metrics)['adopt'])

    def test_recipe_validation_and_install_preserve_parents(self):
        import copy
        import json
        from pathlib import Path
        from sepalith.training.grpo_exec import c18b, c19, recipe, night

        root = Path(__file__).resolve().parents[4]
        original = copy.deepcopy(c18b.PRE_REGISTERED)
        base = copy.deepcopy(recipe.PRE_REGISTERED)
        # Validate a real bound C19 recipe once the CPU binding is available.
        path = root / c19.BOUND / 'grpo-exec-pilot-1-c18b-300.json'
        if path.exists():
            value = json.loads(path.read_text())
            c19.validate(value)
            value['runtime']['reward']['rscript'] = '/usr/bin/Rscript'
            with self.assertRaisesRegex(ValueError, 'rig'):
                c19.validate(value)
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            self.skipTest('install hook requires the optional CPU training dependencies')
        from sepalith.training.grpo_exec import trainer, val_eval, cpt_probe, dev_generation, jobs
        modules = (night, trainer, val_eval, cpt_probe, dev_generation, jobs)
        snapshots = [(m, dict(vars(m))) for m in modules]
        old = night.ExecGRPOArm
        try:
            c19.install()
            self.assertEqual(c18b.PRE_REGISTERED, original)
            self.assertEqual(recipe.PRE_REGISTERED, base)
        finally:
            night.ExecGRPOArm = old
            for module, snapshot in snapshots:
                for key in ('recipe_module', 'ExecGRPOArm', 'ExecGRPOTrainer', 'TRAINER', 'DEV_EVALUATOR', 'VAL_EVALUATOR', 'CPT_PROBE', 'UPDATE_FIELDS', 'final_holdout_job'):
                    if key in snapshot:
                        setattr(module, key, snapshot[key])

    def test_closeout_line_bins_are_reporting_only(self):
        from sepalith.training.grpo_exec.c19_jobs import line_metrics
        report = line_metrics([{'target_lines': n, 'all_pass': n == 2, 'reward': .5} for n in (2, 3, 4, 8, 16, 17)])
        self.assertEqual(report['2'], {'n': 1, 'all_pass': 1, 'mean_reward': .5})
        self.assertEqual(report['3-4']['n'], 2)
        self.assertEqual(set(report), {'2', '3-4', '5-8', '9-16', '17+'})
