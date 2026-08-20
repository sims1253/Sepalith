"""cases: the declarative synthetic-data library.

A training-data family is a JSON case spec (specs/*.json) consumed by the
generate.py harness; case-specific logic lives in small registries:

  spec.py        CaseSpec + JSON loading/validation, list_cases/load_case
  corpus.py      corpus_source selectors (dataset file / normalized corpus)
  samplers.py    seeded parameter samplers (template choice + knobs)
  rows.py        target_construction kinds + target normalizers
  validators.py  layer-3 validators + row-structure checks (tree-sitter)
  backends.py    agy / zai / opencode / openrouter / mock adapters with
                 pacing, retry and stats
  generate.py    the CLI harness (gate, dedup, resume, provenance)

Quick start (from experiments/synthetic-data):
  uv run python -m cases.generate --case comment_to_code_styles \
      --n 20 --backend agy --out /mnt/h/sepalith/datasets/cases_v1/x.jsonl
"""
from cases.spec import CaseSpec, SpecError, list_cases, load_case, load_spec_file

__all__ = ["CaseSpec", "SpecError", "list_cases", "load_case", "load_spec_file"]
