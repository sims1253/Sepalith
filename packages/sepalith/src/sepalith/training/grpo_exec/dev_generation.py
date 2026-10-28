#!/usr/bin/env python3
"""DEV75 or DEV250 generation for one C18 checkpoint (night only, ``cuda:0``).

The C15 GRPO evaluator (``training.grpo.dev_generation``: the same admission file, bf16 load, tokenizer
contract, ``campaign_eval.development_evaluator`` with batch 32, dynamic KV cache, greedy, cap 1,024
and the same result schema, so results compare with arm A 803's bound gate results) with the C18
recipe loader and evaluation steps. This process runs only the evaluator, so it rebinds the two
module-level hooks of the C15 evaluator before calling it instead of copying it.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from sepalith.training.grpo import dev_generation as c15_dev_generation
from sepalith.training.grpo_exec import recipe as recipe_module, rules

Json = dict[str, object]


def allowed_steps(recipe: Json) -> set[int]:
    steps = set(rules.evaluation_steps(int(cast(int, cast(Json, recipe["pre_registered"])["max_updates"]))))
    smoke = recipe.get("smoke")
    if smoke is not None:
        steps.add(int(cast(int, cast(Json, smoke)["eval_step"])))
    return steps


def main(argv: Sequence[str] | None = None) -> int:
    c15_dev_generation.recipe_module = recipe_module  # ty: ignore[invalid-assignment]
    c15_dev_generation.allowed_steps = allowed_steps  # ty: ignore[invalid-assignment]
    return c15_dev_generation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
