"""C19 entry for dev_generation, chained over C18b."""
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from sepalith.training.grpo_exec import c19, dev_generation
    from sepalith.training.grpo_exec.c19_night import guard

    guard()
    c19.install()
    return dev_generation.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
