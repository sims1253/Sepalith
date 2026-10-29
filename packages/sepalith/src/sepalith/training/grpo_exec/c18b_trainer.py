#!/usr/bin/env python3
"""C18b trainer: the C18 trainer plus the CPT-text replay term (night only, ``cuda:0``).

``CptReplayTrainer.replay_term`` runs the C18 SFT replay term, then backward
``cpt_coefficient x token-mean NLL`` of the update's CPT chunks (``mix.replay_ids`` over the CPT rows with
their own seed, so the schedule is stateless and resumes exactly). Telemetry adds ``cpt_replay_nll``,
``cpt_replay_tokens``, ``cpt_replay_loss`` and ``grad_norm_after_sft`` (the gradient norm between the
two terms; ``grad_norm_grpo`` and ``grad_norm`` bracket them). The entry point is the C18 trainer's,
with ``c18b.install()`` applied first.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import cast

import torch

from sepalith.training.grpo.trainer import Row, require
from sepalith.training.grpo_exec import mix
from sepalith.training.grpo_exec.trainer import ExecGRPOTrainer

Json = dict[str, object]


def load_cpt_rows(recipe: Mapping[str, object]) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    from sepalith.training.grpo_exec import c18b

    record = cast(Json, cast(Json, recipe["pool"])["cpt_replay"])
    path = c18b.check_file(c18b.repository_root(recipe), record, "CPT replay rows")
    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = (tuple(row["prompt_ids"]), tuple(row["target_ids"]))
    require(len(rows) == int(cast(int, record["rows"])), "CPT replay row count differs")
    return rows


class CptReplayTrainer(ExecGRPOTrainer):
    def __init__(self, recipe: Mapping[str, object], recipe_sha: str, *,
                 cpt_replay: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None, **kwargs: object) -> None:
        super().__init__(recipe, recipe_sha, **kwargs)  # ty: ignore[invalid-argument-type]
        self.cpt = dict(cpt_replay) if cpt_replay is not None else load_cpt_rows(recipe)
        self.cpt_ids: Sequence[str] = sorted(self.cpt)
        self.cpt_rows = int(cast(int, self.runtime["cpt_replay_rows_per_update"]))
        self.cpt_coefficient = float(cast(float, self.runtime["cpt_replay_coefficient"]))
        require(self.cpt_ids and self.cpt_rows >= 1 and self.cpt_coefficient >= 0, "CPT replay settings are invalid")

    def replay_term(self, update: int) -> Json:
        out = super().replay_term(update)
        out["grad_norm_after_sft"] = self.grad_norm()
        seed = int(cast(int, self.runtime["seed"])) + 101
        rows = [Row(identifier, *self.cpt[identifier], "eos") for identifier in mix.replay_ids(self.cpt_ids, seed, update, self.cpt_rows)]
        tokens = sum(len(row.completion) for row in rows)
        require(tokens > 0, "CPT replay rows have no target tokens")
        nll = 0.0
        for row in rows:
            logp = self.logprobs(row)
            loss = -logp.sum() * (self.cpt_coefficient / tokens)
            require(bool(torch.isfinite(loss.detach())), "non-finite CPT replay loss")
            loss.backward()
            nll += float(-logp.detach().sum())
        out.update(cpt_replay_rows=len(rows), cpt_replay_tokens=tokens, cpt_replay_nll=nll / tokens,
                   cpt_replay_loss=self.cpt_coefficient * nll / tokens)
        return out


def main(argv: Sequence[str] | None = None) -> int:
    from sepalith.training.grpo_exec import c18b, trainer

    c18b.install()
    return trainer.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
