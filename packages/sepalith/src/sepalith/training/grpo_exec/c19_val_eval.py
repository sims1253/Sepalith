"""C19 entry for val_eval, chained over C18b."""
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from sepalith.training.grpo_exec import c19, val_eval
    from sepalith.training.grpo_exec.c19_night import guard

    guard()
    c19.install()
    return val_eval.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
