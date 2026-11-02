"""Optional C19 user-stop closeout, with the same C19 final comparison."""
from collections.abc import Sequence
from pathlib import Path
from typing import cast


def main(argv: Sequence[str] | None = None) -> int:
    from sepalith.training.grpo_exec import c19, final
    from sepalith.training.grpo_exec.c19_jobs import final_holdout_job
    from sepalith.training.grpo_exec.c19_night import C19Arm, guard

    guard()
    c19.install()
    final.ExecGRPOArm = C19Arm  # ty: ignore[invalid-assignment]
    inherited_summary = final.summary
    inherited_job = final.dev250_job

    def summary(arm: C19Arm, step: int) -> dict:
        value = inherited_summary(arm, step)
        job = final_holdout_job(arm.recipe, str(arm.checkpoint(step)), arm.evidence_dir / 'final', arm.repository_root)
        from sepalith.training.sft.night_arm import write_json

        write_json(arm.evidence_dir / 'final-holdout-job.json', job)
        value['final_evaluation'] = {'job': str(arm.evidence_dir / 'final-holdout-job.json'),
            'verdict': 'python -m sepalith.training.grpo_exec.c19_jobs verdict --summary '
                       f"{arm.archive / 'pilot-summary.json'} --metrics {arm.evidence_dir / f'final-step-{step}' / 'metrics.json'}"}
        write_json(arm.archive / 'pilot-summary.json', value)
        return value

    def job(root: Path, recipe: Path, step: int, python: Path) -> dict:
        value = inherited_job(root, recipe, step, python)
        value['id'] = f'C19-final-dev250-step-{step}'
        value['command'] = [str(v).replace('sepalith.training.grpo_exec.final', 'sepalith.training.grpo_exec.c19_final')
                            for v in cast(list, value['command'])]
        value['note'] = 'C19 user-stop DEV250 and closeout, relative to C18b 300.'
        return value

    final.summary = summary  # ty: ignore[invalid-assignment]
    final.dev250_job = job  # ty: ignore[invalid-assignment]
    return final.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
