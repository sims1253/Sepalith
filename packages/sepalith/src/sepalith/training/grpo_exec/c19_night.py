"""C19 driver hooks; all R6x decisions and checkpoint selection remain C18's."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from collections.abc import Mapping, Sequence
from typing import cast

from sepalith.training.grpo_exec import night


class C19Arm(night.ExecGRPOArm):
    def start_measurements(self) -> dict:
        from sepalith.training.grpo_exec import c19

        results = cast(dict, self.recipe['parent_reference'])['results']
        values = {k: json.loads(c19.check_file(self.repository_root, results[k], f'300 {k}').read_text())
                  for k in ('val', 'cpt')}
        manifest = cast(dict, self.recipe['parent'])['files']['campaign-manifest.json']
        for value in values.values():
            c19.require(value['step'] == 300 and value['checkpoint']['campaign_manifest_sha256'] == manifest,
                        '300 start result identity differs')
        overall = values['val']['metrics']['overall']
        return {'val_all_pass': overall['all_pass'], 'val_mean_reward': overall['mean_reward'],
                'cpt_nll': values['cpt']['mean_causal_nll'], 'val_result': results['val']['path'], 'cpt_result': results['cpt']['path']}

    def current_metrics(self, results: Mapping[str, Path]) -> dict:
        from sepalith.training.grpo_exec import c19
        from sepalith.training.grpo_exec.c19_mix import lines_bin
        from sepalith.training.grpo_exec.pool import read_jsonl

        counts = super().current_metrics(results)
        val = json.loads(results['val'].read_text())
        tasks = {t['id']: t for t in read_jsonl(Path(cast(dict, self.recipe['pool'])['val']['path']))}
        records = read_jsonl(c19.check_file(self.repository_root, val['records'], 'validation records'))
        grouped = {}
        for row in records:
            task = tasks[row['task_id']]
            key = lines_bin(len(str(task['target_body_text']).splitlines()))
            grouped.setdefault(key, []).append(row)
        report = {k: val['metrics'][k] for k in ('by_bucket', 'by_family')}
        report['by_target_lines'] = {k: {'n': len(rows), 'all_pass': sum(r['all_pass'] for r in rows) / len(rows),
                                              'mean_reward': sum(r['reward'] for r in rows) / len(rows)} for k, rows in grouped.items()}
        report_path = results['val'].with_suffix('.c19-breakdown.json')
        if report_path.exists():
            c19.require(json.loads(report_path.read_text()) == report, 'validation breakdown differs')
        else:
            c19.write_new(report_path, report)
        return {**counts, 'validation_breakdown': report}


    def finish_pilot(self, status: str, latest: int) -> int:
        code = super().finish_pilot(status, latest)
        path = self.archive / 'pilot-summary.json'
        if path.exists():
            value = json.loads(path.read_text())
            selected = value.get('selected')
            if selected:
                decision = self.latest_decision(selected['step'])
                if decision:
                    selected['validation_breakdown'] = cast(dict, decision['metrics']).get('validation_breakdown')
                path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        return code


def guard() -> None:
    from sepalith.training.grpo_exec import c19

    version = subprocess.run([c19.R_PATH, '--version'], check=True, capture_output=True, text=True)
    c19.require((version.stdout or version.stderr).splitlines()[0] == 'Rscript (R) version 4.6.1 (2026-06-24)',
                'rig Rscript version line differs')
    os.environ['PATH'] = '/opt/R/4.6.1/bin:' + os.environ.get('PATH', '')


def main(argv: Sequence[str] | None = None) -> int:
    from sepalith.training.grpo_exec import c19

    guard()
    c19.install()
    return night.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
