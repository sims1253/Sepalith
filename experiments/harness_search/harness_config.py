"""The H1 harness search space (plan §1.3) as a validated JSON config.

Every knob of the table, expressed machine-checkably. Render markers are
FROZEN (zeta2 canonical); the config only moves the executable knobs.

DEFAULT_CONFIG = the offline rig baseline: the banked eval render (scope
OFF — eval_scenarios' zeta2 render AND eval_noop_fp's scope=null port both
ship without pin/outline sections), the extension's shipped gen/parse
defaults (max_tokens 320, the 7-marker stop set, degenerate-rep cut ON,
marker-line drop ON), the extension's post-heuristics ON, cap 6000 with
protect-pin direction. The extension's LIVE default has scopeContext=true
(pin 4000 + outline 60x1500); that is IN the space as the pin/outline "on"
values, but the offline baseline is the banked eval convention so default-
config runs are comparable to the banked v7 numbers.

The (c) GEPA arm freezes this config and evolves only `texts` (free-text
slots); arms (a)/(b) evolve `cfg` with texts left at defaults.
"""
from __future__ import annotations

import hashlib
import json

# --- frozen markers (never searchable) --------------------------------------
FROZEN_MARKERS = (">>>>>>>", "<<<<<<<", "=======", "<[fim-suffix]>",
                  "<[fim-prefix]>", "<[fim-middle]>", "<|user_cursor|>",
                  "<|outline|>")

STOPS_FULL = [">>>>>>> UPDATED", "<<<<<<< CURRENT", "=======",
              "<[fim-middle]>", "<[fim-suffix]>", "<[fim-prefix]>", "<|outline|>"]
STOPS_MINIMAL = [">>>>>>> UPDATED"]

# --- the space ---------------------------------------------------------------
SPACE = {
    "prefix_suffix_cap": [2000, 4000, 6000, 8000],       # MAX_PREFIX_SUFFIX_CHARS
    "pin": ["off", 2000, 4000],                          # off / on w/ MAX_PIN_CHARS
    "outline": ["off", [20, 800], [20, 1500], [60, 800], [60, 1500]],
    "suffix_truncation": ["protect-pin", "hard-cut"],    # cut direction
    "max_tokens": [128, 320, 640],
    "stops": ["full", "minimal"],
    "parse_rep_cut": [True, False],                      # 3-identical-cut
    "parse_marker_drop": [True, False],                  # marker-line drop
    "comment_heuristic": [True, False],
    "full_region_replace": [True, False],
}

DEFAULT_CONFIG = {
    "prefix_suffix_cap": 6000,
    "pin": "off",              # scope off = the banked eval convention
    "outline": "off",          # (eval_scenarios zeta2 AND eval_noop_fp scope=null)
    "suffix_truncation": "protect-pin",
    "max_tokens": 320,
    "stops": "full",
    "parse_rep_cut": True,
    "parse_marker_drop": True,
    "comment_heuristic": True,
    "full_region_replace": True,
}

# GEPA free-text slots (arm c only; render config frozen at DEFAULT_CONFIG)
DEFAULT_TEXTS = {"instruction_line": "", "checklist_line": "", "outline_header": "<|outline|>"}
TEXT_SLOT_MAX = 220  # chars per slot


def validate_config(cfg: dict) -> dict:
    """Return a normalized copy or raise ValueError with the reason."""
    out = {}
    for k, dom in SPACE.items():
        if k not in cfg:
            raise ValueError(f"missing knob {k}")
        v = cfg[k]
        if v not in dom:
            raise ValueError(f"{k}={v!r} not in {dom!r}")
        out[k] = v
    extra = set(cfg) - set(SPACE)
    if extra:
        raise ValueError(f"unknown knobs {sorted(extra)}")
    return out


def validate_texts(texts: dict) -> dict:
    out = dict(DEFAULT_TEXTS)
    for k, v in texts.items():
        if k not in DEFAULT_TEXTS:
            raise ValueError(f"unknown text slot {k!r}")
        if not isinstance(v, str) or len(v) > TEXT_SLOT_MAX or "\n" in v:
            raise ValueError(f"text slot {k} must be a single line <= {TEXT_SLOT_MAX} chars")
        if k != "outline_header":  # inert under frozen scope-off; default IS the marker
            for m in FROZEN_MARKERS:
                if m in v:
                    raise ValueError(f"text slot {k} contains frozen marker {m!r}")
        out[k] = v
    # outline_header is inert while outline is off (documented; GEPA's render
    # is frozen at DEFAULT_CONFIG) — allowed but flagged by the proposer ctx.
    return out


def fingerprint(cfg: dict, texts: dict | None = None) -> str:
    blob = json.dumps([cfg, texts or DEFAULT_TEXTS], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def stops_for(cfg: dict) -> list[str]:
    return STOPS_FULL if cfg["stops"] == "full" else STOPS_MINIMAL
