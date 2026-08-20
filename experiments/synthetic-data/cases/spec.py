"""Declarative case specs: a training-data family as a JSON description.

A CaseSpec is everything the harness (generate.py) needs to turn a corpus
into validated, provenance-tagged rows:

  name                 unique case identifier (also the --case CLI value)
  version              int, bumped when the spec semantics change
  description          what the family trains
  novelty_note         how it differs from every existing family
  family               value stamped on emitted rows (defaults to name)
  prompt_templates     list of {placeholder}-style templates; the sampler
                       picks one per row
  corpus_source        {kind, path?, selector, params?, provenance?} — a
                       registered selector over the normalized corpus or an
                       existing dataset file
  parameter_sampler    {name, params} — registered seeded sampler (template
                       choice plus any extra knobs)
  target_construction  {kind, params} — registered row constructor turning
                       (corpus item, model target) into a scenario-shaped row
  target_field         the JSON key the backend response must carry
  validator            {name, params} — registered 3rd-layer gate on the
                       target (layer 1 = JSON extraction, layer 2 = schema,
                       layer 3 = this)
  row_check            optional {name, params} extra structural check on the
                       finished row (default: the scenario_block shape check)
  dedup                "target+key" (default) or "target" — content-hash
                       dedup scope
  difficulty           knobs: context_lines, target_lines_min/max, ...
  response_format      "json" (default; backend requests a JSON object) or
                       "text"

Specs load from cases/specs/*.json via load_case(name); --spec <path> on the
CLI loads one directly. Everything that is case-specific logic lives in a
registry (corpus.py, samplers.py, validators.py, rows.py) so a NEW family is
a JSON file plus at most one small registered function — never a new script.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPECS_DIR = HERE / "specs"

REQUIRED_KEYS = ("name", "description", "prompt_templates", "corpus_source",
                 "parameter_sampler", "target_construction", "validator",
                 "novelty_note", "target_field")


class SpecError(ValueError):
    pass


@dataclass
class CaseSpec:
    name: str
    version: int
    description: str
    novelty_note: str
    prompt_templates: list
    corpus_source: dict
    parameter_sampler: dict
    target_construction: dict
    validator: dict
    target_field: str
    family: str = ""
    row_check: dict | None = None
    dedup: str = "target+key"
    difficulty: dict = field(default_factory=dict)
    response_format: str = "json"
    source_path: str = ""          # where the JSON was loaded from
    raw: dict = field(default_factory=dict)

    @property
    def provenance(self) -> dict:
        return dict(self.corpus_source.get("provenance") or {})

    def case_scope(self) -> str:
        """Identity prefix of the content-hash dedup namespace."""
        return f"{self.name}@{self.version}"

    def template_vars(self, item: dict) -> dict:
        """Placeholder values every template may use. Corpus selectors emit
        items with `prefix` (list of lines, ending at the cursor), `block`
        (list of lines), `suffix`, `package`, `path`, `key`."""
        code = "\n".join(str(l) for l in (item.get("block") or []))
        context = "\n".join(str(l) for l in (item.get("prefix") or []))
        return {"code": code, "context": context,
                "package": item.get("package", ""), "path": item.get("path", "")}

    def fill_template(self, index: int, item: dict) -> str:
        return self.prompt_templates[index].format(**self.template_vars(item))


def _check_spec(d: dict, origin: str) -> None:
    for k in REQUIRED_KEYS:
        if k not in d or d[k] in (None, "", []):
            raise SpecError(f"{origin}: missing or empty required key {k!r}")
    if not isinstance(d["prompt_templates"], list):
        raise SpecError(f"{origin}: prompt_templates must be a list")
    for t in d["prompt_templates"]:
        if not isinstance(t, str) or "{" not in t:
            raise SpecError(f"{origin}: template not a string with placeholders: {t!r}")
    cs = d["corpus_source"]
    if not isinstance(cs, dict) or "selector" not in cs:
        raise SpecError(f"{origin}: corpus_source needs a 'selector' name")
    for sec in ("parameter_sampler", "validator"):
        if not isinstance(d[sec], dict) or "name" not in d[sec]:
            raise SpecError(f"{origin}: {sec} must be an object with a 'name'")
    tc = d["target_construction"]
    if not isinstance(tc, dict) or not (tc.get("kind") or tc.get("name")):
        raise SpecError(f"{origin}: target_construction needs a 'kind'")
    if d.get("dedup", "target+key") not in ("target+key", "target"):
        raise SpecError(f"{origin}: dedup must be 'target+key' or 'target'")
    if d.get("response_format", "json") not in ("json", "text"):
        raise SpecError(f"{origin}: response_format must be 'json' or 'text'")


def spec_from_dict(d: dict, origin: str = "<dict>") -> CaseSpec:
    _check_spec(d, origin)
    return CaseSpec(
        name=d["name"],
        version=int(d.get("version", 1)),
        description=d["description"],
        novelty_note=d["novelty_note"],
        prompt_templates=list(d["prompt_templates"]),
        corpus_source=dict(d["corpus_source"]),
        parameter_sampler=dict(d["parameter_sampler"]),
        target_construction=dict(d["target_construction"]),
        validator=dict(d["validator"]),
        target_field=d["target_field"],
        family=d.get("family") or d["name"],
        row_check=dict(d["row_check"]) if d.get("row_check") else None,
        dedup=d.get("dedup", "target+key"),
        difficulty=dict(d.get("difficulty") or {}),
        response_format=d.get("response_format", "json"),
        source_path=origin,
        raw=dict(d),
    )


def load_spec_file(path: str | Path) -> CaseSpec:
    path = Path(path)
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SpecError(f"cannot load spec {path}: {e}") from e
    return spec_from_dict(d, origin=str(path))


def list_cases() -> list[str]:
    """Valid case specs in specs/ (files that fail spec validation — e.g.
    design documents parked in the directory — are skipped, not listed)."""
    out = []
    for p in sorted(SPECS_DIR.glob("*.json")):
        try:
            spec_from_dict(json.loads(p.read_text()), origin=str(p))
        except (SpecError, ValueError, OSError):
            continue
        out.append(p.stem)
    return out


def load_case(name: str) -> CaseSpec:
    if "/" in name or name.endswith(".json"):
        return load_spec_file(name)
    path = SPECS_DIR / f"{name}.json"
    if not path.exists():
        avail = ", ".join(list_cases()) or "<none>"
        raise SpecError(f"unknown case {name!r}; available: {avail}")
    return load_spec_file(path)


if __name__ == "__main__":  # quick inspection helper
    for name in (sys.argv[1:] or list_cases()):
        s = load_case(name)
        print(f"{s.name} v{s.version} family={s.family} "
              f"templates={len(s.prompt_templates)} "
              f"selector={s.corpus_source['selector']} "
              f"validator={s.validator['name']}")
        print(f"  {s.description}")
