"""Seeded parameter samplers (registered by name, referenced from specs).

A sampler receives the spec's parameter_sampler.params plus a seeded
random.Random owned by the harness, and returns a dict of per-row draw
parameters. The default `template_uniform` picks a prompt template index;
`grid_choice` additionally draws named categorical knobs (grid.py-style
coverage cells) for specs that declare them.
"""
from __future__ import annotations

import random

REGISTRY: dict = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def make_sampler(cfg: dict, rng: random.Random):
    name = cfg.get("name", "template_uniform")
    if name not in REGISTRY:
        raise KeyError(f"unknown parameter_sampler {name!r}; "
                       f"registered: {sorted(REGISTRY)}")
    params = dict(cfg.get("params") or {})
    n_templates = int(cfg.get("n_templates", 1))

    def draw() -> dict:
        return REGISTRY[name](rng, n_templates, params)
    return draw


@register("template_uniform")
def template_uniform(rng: random.Random, n_templates: int, params: dict) -> dict:
    """Uniform template choice; optional `weights` list biases the draw."""
    weights = params.get("weights")
    if weights and len(weights) == n_templates:
        i = rng.choices(range(n_templates), weights=weights, k=1)[0]
    else:
        i = rng.randrange(n_templates) if n_templates > 1 else 0
    return {"template_index": i}


@register("grid_choice")
def grid_choice(rng: random.Random, n_templates: int, params: dict) -> dict:
    """template_uniform plus one draw per declared categorical knob:
    params = {"knobs": {"line_target": [8, 12, 16], "style": [...]}}."""
    out = template_uniform(rng, n_templates, params)
    for k, choices in (params.get("knobs") or {}).items():
        out[k] = rng.choice(list(choices))
    return out
