"""C19 entry for final, chained over C18b."""
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from sepalith.training.grpo_exec import c19, final
    from sepalith.training.grpo_exec.c19_night import guard

    guard()
    c19.install()
    from sepalith.training.grpo_exec.c19_night import C19Arm
    final.ExecGRPOArm = C19Arm  # ty: ignore[invalid-assignment]
    return final.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
