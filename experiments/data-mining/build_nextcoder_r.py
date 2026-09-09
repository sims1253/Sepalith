#!/usr/bin/env python3
"""B11: NextCoder-style R edit-seed pack generator (queue row B11; survey Q2).

Task space:  R source seeds (experiments/.cache/repos/r clones + the
scenarios_v1 corpus)  x  transform classes {base_r, tidyverse, data.table,
fix_the_bug, vectorize}  x  3 instruction phrasings, authored by TWO
generators — glm-5.3 (ZaiBackend) + one open model (opencode muse-spark via
the Responses free tier, OpencodeSparkFreeBackend) — with a deterministic
hash split of the task space across backends (per-row model tags make each
backend's rows attributable and purgeable, the rewrite_author_spark
convention).

Subcommands (the wave-driver pattern of rewrite_author_zai/_spark):
  mine      build the seed pool from repos+scenarios, AST-dedupe,
            holdout-exclude, applicability-tag, sample -> seeds.jsonl
  author    generate edit tasks via the two backends; every region_new must
            pass the validator stack (clean tree-sitter parse, statement
            bounds, no-op rejection, class gate) and AST-dedupe (vs the
            corpus signature set + within batch) -> authored.jsonl
  assemble  train/eval split (3% by package, per class family, seed 42),
            zeta2 text rows via the assemble_sft_v2.edit_row() convention
            -> {train,eval}.jsonl + manifest.json (per-class/phrasing
            counts + sha256 hashes)

Resume sidecars per the wave pattern: <out>.done.jsonl (terminal outcomes
only; backend/network errors retried on rerun) + <out>.stats.json. Every
authored row carries the parent-link (source-file sha + rule@v1) and the
backend/model tag.

Fail-closed: missing API key -> refuse before any request; validator
failure -> row dropped + counted; dedupe collisions logged + counted.

--smoke: 5 seeds, 1 phrasing, mock backend (no API, no network) proving the
pipeline end-to-end into /tmp.

Output default: /mnt/h/sepalith/datasets/nextcoder_r_v1/{train,eval}.jsonl
Work dir default: /mnt/h/sepalith/datasets/nextcoder_r_v1_work/
Run resource-polite (CPU only, nice -n 19). Python: system python3 (needs
tree_sitter + tree_sitter_r, the experiments/synthetic-data/cases deps).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # experiments/data-mining
REPO = HERE.parent.parent                              # repo root
sys.path.insert(0, str(REPO / "experiments" / "synthetic-data"))
sys.path.insert(0, str(REPO / "experiments" / "eval"))

import tree_sitter_r                                    # noqa: E402
from tree_sitter import Language, Parser                # noqa: E402

from cases.backends import (Backend, BackendError,      # noqa: E402
                            OpencodeSparkFreeBackend, ZaiBackend,
                            extract_json_object, strip_fences)
from run_eval import render_zeta2                       # noqa: E402  (zeta2 renderer)

NAS = Path("/mnt/h/sepalith")
REPOS_R = REPO / "experiments" / ".cache" / "repos" / "r"
SCENARIOS = NAS / "datasets" / "scenarios_v1"
HOLDOUT_JSON = NAS / "datasets" / "a2" / "holdout_packages.json"
DEFAULT_OUT = NAS / "datasets" / "nextcoder_r_v1"
DEFAULT_WORK = NAS / "datasets" / "nextcoder_r_v1_work"

RULE = "nextcoder-r@v1"
UPDATED = "\n>>>>>>> UPDATED"
MAX_CHARS = 6000          # assemble_sft_v2 v1 convention: prompt+target budget
MAX_BLOCK_LINES = 25      # r_fragment max_lines convention
MAX_STATEMENTS_LO, MAX_STATEMENTS_HI = 1, 10
SEED_RNG = 42

# tree-sitter R loaded exactly as cases/validators.py loads it
_PARSER = Parser(Language(tree_sitter_r.language()))

CLASSES = ("base_r", "tidyverse", "data.table", "fix_the_bug", "vectorize")
PHRASINGS = {
    "base_r": [
        "Rewrite this block in idiomatic base R (no tidyverse, no data.table); keep the behavior identical.",
        "Port this snippet to plain base R functions only.",
        "Rewrite without dplyr/purrr/data.table, using base R idioms.",
    ],
    "tidyverse": [
        "Rewrite this block as a tidyverse pipeline (dplyr/purrr); keep the behavior identical.",
        "Port this snippet to dplyr verbs and the native pipe.",
        "Modernise this code with tidyverse style, preserving semantics.",
    ],
    "data.table": [
        "Rewrite this block using data.table idioms; keep the behavior identical.",
        "Port this snippet to data.table operations.",
        "Convert to data.table syntax, preserving semantics.",
    ],
    "fix_the_bug": [
        "Fix the bug in this block; change as little as possible.",
        "This code has one defect. Repair it with a minimal edit.",
        "Find and fix the single bug, touching nothing else.",
    ],
    "vectorize": [
        "Vectorize this block: remove the explicit loop, keep the behavior identical.",
        "Replace the loop with vectorized or apply-family code.",
        "Rewrite without a for loop, using vectorized R.",
    ],
}

# canonical scenario files whose region_old is real R code (the
# assemble_sft_v2 SCENARIO_FILES convention, minus the comment-only families
# whose regions are roxygen/comment text rather than code)
SCENARIO_SEED_FILES = (
    "rename_propagation.jsonl", "pipe_rewrite.jsonl", "na_rm_propagation.jsonl",
    "format_propagation.jsonl", "doc_sync.jsonl", "comment_to_code_real.jsonl",
    "comment_to_code_synthetic.jsonl", "no_op.jsonl",
)

TIDY_RE = re.compile(r"%>%|\|>|dplyr::|purrr::|tidyr::|stringr::|readr::|forcats::")
DT_RE = re.compile(r"data\.table|\bsetDT\b|\bsetkey\b|:=")
DT_HINT_RE = re.compile(r"\b(data\.frame|subset|merge|aggregate|tapply|sapply|lapply|apply)\s*\(")
FOR_RE = re.compile(r"\bfor\s*\(")
CALL_RE = re.compile(r"\b[a-zA-Z.][\w.]*\s*\(")

# deterministic single-token defect menu (the rewrite_author_zai buinject
# convention: inject -> author the fix; rationale never leaks the exact fix).
# `bad` matches the INJECTED defect token so the gate can demand its absence;
# word/paren aware regexes keep R assignments (x <- y) from matching "<".
DEFECT_MENU = (
    dict(rule="true_false_symbol",
         find=re.compile(r"(?<![\w.])TRUE(?![\w.])"), repl="T",
         bad=re.compile(r"(?<![\w.$])T(?![\w.$])"),
         good=re.compile(r"(?<![\w.])TRUE(?![\w.])"),
         why="T/F can be shadowed; spell the logical constant"),
    dict(rule="seq_safety",
         find=re.compile(r"(?<![\w.])seq_along\s*\("), repl="1:length(",
         bad=re.compile(r"(?<![\w.])1:\s*length\s*\("),
         good=re.compile(r"(?<![\w.])seq_along\s*\("),
         why="1:x yields c(1,0) when x is empty; use the safe idiom"),
    dict(rule="boundary_operator",
         find=re.compile(r"(?<![<>=!])(<=|>=)(?!=)"), repl=None,
         bad=None,   # per-match: the mutated operator, never "<-" / "<=" / "<"
         good=re.compile(r"(?<![<>=!])(<=|>=)(?!=)"),
         why="off-by-one: this comparison boundary uses the wrong operator"),
)


# ---------------------------------------------------------------------------
# AST machinery (validators.py conventions: parse_fragment / fragment_clean,
# plus a structural signature for dedupe)
# ---------------------------------------------------------------------------

def parse_fragment(text: str):
    return _PARSER.parse(text.encode("utf-8", "surrogateescape"))


def _walk(n):
    stack = [n]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def fragment_clean(text: str) -> bool:
    """Fragment parses with no ERROR/missing nodes anywhere (validators.py)."""
    tree = parse_fragment(text)
    if tree.root_node.has_error:
        return False
    return not any(n.type == "ERROR" or n.is_missing
                   for n in _walk(tree.root_node))


def n_statements(text: str) -> int:
    return len([c for c in parse_fragment(text).root_node.children
                if c.is_named])


def _sexp(n) -> str:
    kids = "".join(_sexp(c) for c in n.children if c.is_named)
    return f"({n.type}{kids})"


def ast_hash(text: str) -> str | None:
    """Structural signature: sha1 of the named-node s-expression with
    comments skipped. Structurally identical code (differing only in
    identifiers/literals/comments) collides BY DESIGN — that is the dedupe.
    Comment-only fragments have no signature: caller falls back to text sha."""
    tree = parse_fragment(text)
    if tree.root_node.has_error:
        return None
    parts = [_sexp(c) for c in tree.root_node.children
             if c.is_named and c.type != "comment"]
    if not parts:
        return None
    return hashlib.sha1("".join(parts).encode()).hexdigest()[:16]


def struct_key(text: str) -> tuple:
    """Dedupe key: (ast signature, text sha) — ast first, text fallback."""
    a = ast_hash(text)
    return ("ast", a) if a else ("txt", hashlib.sha1(
        "\n".join(l.strip() for l in text.splitlines())
        .encode("utf-8", "surrogateescape")).hexdigest()[:16])


# ---------------------------------------------------------------------------
# holdout rule (experiments/data-mining/holdout_rule.py; re-derived, the
# audit list is only a cross-check)
# ---------------------------------------------------------------------------

def is_holdout(package_name: str) -> bool:
    return int(hashlib.sha256(package_name.encode("utf-8")).hexdigest(),
               16) % 100 < 2


def load_holdout_audit(path: Path = HOLDOUT_JSON) -> set[str]:
    """Union of the audit artifact's held-out names (belt-and-braces; the
    rule function stays the source of truth)."""
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return set()
    out: set[str] = set()
    for names in (blob.get("held_out") or {}).values():
        out.update(names)
    return out


def holdout_block(packages: set[str], audit: set[str]) -> set[str]:
    """Names blocked for train-side use: rule-derived + audit-listed."""
    return {p for p in packages if is_holdout(p)} | (packages & audit)


# ---------------------------------------------------------------------------
# seed mining
# ---------------------------------------------------------------------------

def tag_applicability(block_text: str) -> list[str]:
    """Cheap AST-free class steering: which transforms make sense here."""
    t = block_text
    tags = []
    tidy = bool(TIDY_RE.search(t))
    dt = bool(DT_RE.search(t))
    if tidy:
        tags.append("base_r")
    elif CALL_RE.search(t):
        tags.append("tidyverse")
    if not dt and (DT_HINT_RE.search(t) or "[" in t):
        tags.append("data.table")
    if FOR_RE.search(t):
        tags.append("vectorize")
    if inject_defect(block_text) is not None:
        tags.append("fix_the_bug")
    return [c for c in CLASSES if c in tags]      # canonical order


def inject_defect(block_text: str):
    """First applicable deterministic defect: dict(rule, mutated_text,
    bad_re, good_re). None when no menu item applies. First match wins,
    comment lines skipped."""
    lines = block_text.split("\n")
    for d in DEFECT_MENU:
        for i, line in enumerate(lines):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = d["find"].search(line)
            if not m:
                continue
            r = d["repl"] if d["repl"] is not None else (
                "<" if m.group(1) == "<=" else ">")
            mutated = line[:m.start()] + r + line[m.end():]
            new_lines = list(lines)
            new_lines[i] = mutated
            bad = d["bad"] if d["bad"] is not None else re.compile(
                re.escape(r) + r"(?![<>=\\-])")
            return dict(rule=d["rule"], bad_re=bad, good_re=d["good"],
                        text="\n".join(new_lines))
    return None


def _fn_blocks(src: bytes, min_lines: int = 4, max_lines: int = MAX_BLOCK_LINES,
               per_file: int = 3) -> list[tuple[int, int]]:
    """(start_row, end_row) of braced top-level function definitions."""
    tree = _PARSER.parse(src)
    out = []
    for n in tree.root_node.children:
        if n.type != "binary_operator" or not n.children:
            continue
        rhs = n.children[-1] if len(n.children) >= 3 else None
        if rhs is None or rhs.type != "function_definition":
            continue
        r0, r1 = n.start_point[0], n.end_point[0]
        if not min_lines <= r1 - r0 + 1 <= max_lines:
            continue
        out.append((r0, r1))
        if len(out) >= per_file:
            break
    return out


def mine_seeds(args) -> dict:
    t0 = time.time()
    audit = load_holdout_audit(Path(args.holdout_json))
    rng = random.Random(args.seed)
    stats = Counter()
    cand: list[dict] = []

    # -- source 1: repo clones ------------------------------------------------
    repos_root = Path(args.repos)
    repo_dirs = sorted(p for p in repos_root.iterdir()
                       if p.is_dir()) if repos_root.is_dir() else []
    blocked = holdout_block({p.name for p in repo_dirs}, audit)
    stats["repos_found"] = len(repo_dirs)
    stats["repos_holdout_blocked"] = len(blocked & {p.name for p in repo_dirs})
    per_repo = max(1, args.per_repo_cap)
    for rd in repo_dirs:
        if rd.name in blocked:
            continue
        got = 0
        files = sorted(set(list(rd.rglob("R/*.R")) + list(rd.rglob("R/*.r"))
                           + list(rd.rglob("*.[Rr]"))))
        for f in files:
            if got >= per_repo:
                break
            try:
                src = f.read_bytes()
            except OSError:
                continue
            if not src or len(src) > 300_000:
                continue
            lines = src.decode("utf-8", "replace").split("\n")
            for r0, r1 in _fn_blocks(src):
                block = lines[r0:r1 + 1]
                text = "\n".join(block)
                if not fragment_clean(text):
                    stats["repo_blocks_unparseable"] += 1
                    continue
                cand.append(dict(
                    source="repo", package=rd.name,
                    path=str(f.relative_to(rd)),
                    prefix=lines[max(0, r0 - 6):r0],
                    block=block, suffix=lines[r1 + 1:r1 + 7],
                    file_sha=hashlib.sha256(src).hexdigest()))
                got += 1
                if got >= per_repo:
                    break
        stats["repo_seeds"] += got

    # -- source 2: scenarios_v1 corpus ---------------------------------------
    scen_cap = args.per_scenario_cap
    for fname in SCENARIO_SEED_FILES:
        path = Path(args.scenarios) / fname
        if not path.exists():
            stats[f"scenario_missing:{fname}"] += 1
            continue
        got = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if got >= scen_cap:
                    break
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                pkg = r.get("package") or "?"
                if is_holdout(pkg) or pkg in audit:
                    stats["scenario_holdout_rows"] += 1
                    continue
                block = [str(l) for l in (r.get("region_old") or [])]
                if not 2 <= len(block) <= MAX_BLOCK_LINES:
                    continue
                text = "\n".join(block)
                if not any(l.strip() for l in text.splitlines()):
                    continue
                if not fragment_clean(text):
                    stats["scenario_blocks_unparseable"] += 1
                    continue
                cand.append(dict(
                    source="scenario", package=pkg,
                    path=r.get("path") or "?",
                    prefix=[str(l) for l in (r.get("prefix") or [])][-6:],
                    block=block, suffix=[],
                    file_sha=hashlib.sha1(line.strip().encode(
                        "utf-8", "surrogateescape")).hexdigest()))
                got += 1
        stats["scenario_seeds"] += got

    # -- dedupe (structural), holdout already applied per-source --------------
    seen: set[tuple] = set()
    uniq: list[dict] = []
    for s in cand:
        k = struct_key("\n".join(s["block"]))
        if k in seen:
            stats["dedupe_collisions"] += 1
            continue
        seen.add(k)
        s["struct_key"] = list(k)
        uniq.append(s)
    stats["seeds_unique"] = len(uniq)

    # -- seeded sample (source-balanced: repos up to 40%) ---------------------
    rng.shuffle(uniq)
    repos_pool = [s for s in uniq if s["source"] == "repo"]
    scen_pool = [s for s in uniq if s["source"] == "scenario"]
    n_target = args.seeds
    n_repos = min(len(repos_pool), max(1, round(n_target * 0.4)))
    picked = repos_pool[:n_repos] + scen_pool[:max(0, n_target - n_repos)]
    rng.shuffle(picked)
    seeds = []
    for i, s in enumerate(picked):
        text = "\n".join(s["block"])
        # stable id: content-derived, immune to reshuffle (review fix)
        seed_id = hashlib.sha1((s["file_sha"] + "|" + s["path"] + "|" +
                                text).encode()).hexdigest()[:12]
        seeds.append(dict(
            key=f"ncr_seed:{seed_id}", source=s["source"], package=s["package"],
            path=s["path"], prefix=s["prefix"], block=s["block"],
            suffix=s["suffix"], file_sha=s["file_sha"],
            applicability=tag_applicability(text),
            struct_key=":".join(s["struct_key"]),
            parent_link=f"{s['file_sha']}@{RULE}"))
    stats["seeds_sampled"] = len(seeds)
    stats["packages_sampled"] = len({s["package"] for s in seeds})
    stats["seeds_no_class"] = sum(1 for s in seeds
                                  if not s["applicability"])
    stats["applicability"] = dict(Counter(
        c for s in seeds for c in s["applicability"]))

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    _write_json(work / "seeds.jsonl.stats.json", dict(
        ts=_now(), rule=RULE, seed_rng=args.seed, stats=dict(stats),
        elapsed_s=round(time.time() - t0, 1)))
    with open(work / "seeds.jsonl", "w") as fh:
        for s in seeds:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_tasks = sum(len(s["applicability"]) * args.phrasings for s in seeds)
    est = dict(tasks=n_tasks, tokens=n_tasks * 600,
               per_backend=round(n_tasks * 600 / max(1, len(args_backends(args)))))
    print(f"[mine] {len(seeds)} seeds ({stats['dedupe_collisions']} dedupe "
          f"collisions dropped, {stats['scenario_holdout_rows']}+"
          f"{stats['repos_holdout_blocked']} holdout-excluded); "
          f"task space = {n_tasks} tasks "
          f"(~{est['tokens']:,} tokens @600/task over "
          f"{len(args_backends(args))} backend(s))", flush=True)
    return dict(stats=dict(stats), estimate=est)


def args_backends(args) -> list[str]:
    names = [b.strip() for b in args.backends.split(",") if b.strip()]
    return names or ["mock"]


# ---------------------------------------------------------------------------
# task space + authoring
# ---------------------------------------------------------------------------

class ZaiAuthorBackend(ZaiBackend):
    """glm-5.3 authoring config (the rewrite_author_zai payload: bigger
    budget, low temperature for minimal diffs)."""
    def _payload(self, prompt: str) -> dict:
        return {
            "model": "glm-5.3", "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "max_tokens": 2000, "temperature": 0.3,
        }


class SparkFreeAuthorBackend(OpencodeSparkFreeBackend):
    """muse-spark-1.3 open model on the free tier via the Responses API (the
    task's 'one open model'; generous output budget — reasoning burns
    first)."""
    model = "muse-spark-1.3-contributor-free"   # pinned, not inherited (review)
    timeout_s = 240.0

    def _payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt}]}],
            "max_output_tokens": 6000,
            "reasoning": {"effort": "low"},
            "text": {"format": {"type": "json_object"}},
        }


class MockAuthorBackend(Backend):
    """Deterministic no-network author (smoke/tests). Derives a valid
    region_new from the machine-readable task hint the driver appends for
    mock runs only. Env knobs: NEXTCODER_MOCK_FAIL_EVERY (unparseable every
    Nth request), NEXTCODER_MOCK_INVALID (schema-ok but gate-failing)."""
    name = "mock"
    model = "mock-0"

    def __init__(self):
        super().__init__()
        self._n = 0

    def _complete_once(self, prompt: str) -> str:
        self._n += 1
        self._bump("ok", 0.0)
        fe = int(os.environ.get("NEXTCODER_MOCK_FAIL_EVERY", "0") or 0)
        if fe and self._n % fe == 0:
            return "not json, sorry"
        hint = json.loads(prompt.rsplit("#MOCK-HINT ", 1)[1])
        lines = list(hint["region_old"])
        if hint["cls"] == "fix_the_bug":
            lines = list(hint["fix_lines"])          # restore the original
        else:
            # three structurally DISTINCT statements, drawn by phrasing, so
            # smoke runs exercise the phrasing axis instead of colliding in
            # within-batch AST dedupe (a real model varies per phrasing)
            h = int(hashlib.sha1(hint["task_key"].encode()).hexdigest()[:6], 16)
            v = int(hint.get("phrasing", 0)) % 3
            base = {0: f"mock_{h % 997} <- {h % 89}",
                    1: f"mock_fn_{h % 997} <- function(x) x + {h % 89}",
                    2: f"if ({h % 89} > 0) mock_{h % 997} <- {h % 89}"}[v]
            # class-idiomatic suffix so the per-class idiom gates pass
            # (mirror what a compliant model answer looks like)
            cls = hint["cls"]
            if cls == "tidyverse":
                idiom = f"mock_tibble_{h % 89} <- dplyr::mutate(x, v = {h % 89})"
            elif cls == "data.table":
                idiom = f"mock_dt_{h % 89} <- data.table::as.data.table(x)[, v := {h % 89}]"
            elif cls == "vectorize":
                idiom = f"mock_vec_{h % 89} <- pmax(x, {h % 89})"
            else:
                idiom = base
            lines = lines + [f"{base}  # {hint['cls']}", idiom]
        if os.environ.get("NEXTCODER_MOCK_INVALID"):
            lines = lines + ["this is << not R"]
        return json.dumps({"region_new": lines,
                           "note": f"mock {hint['cls']} rewrite"})


AUTHOR_BACKENDS = {
    "zai": ZaiAuthorBackend,
    "opencode-spark-free": SparkFreeAuthorBackend,
    "mock": MockAuthorBackend,
}


def preflight_backend(name: str):
    """Fail-closed: refuse before any request when the key is missing."""
    cls = AUTHOR_BACKENDS[name]
    env_key = getattr(cls, "env_key", None)
    if env_key and not os.environ.get(env_key):
        sys.exit(f"refusing: backend {name!r} needs ${env_key}; "
                 f"export it or use --backends mock")
    return cls()


PROMPT_TMPL = """You are editing R code. Apply EXACTLY this task to the CURRENT CODE block and return the edited block.

Task: {task}

Rules:
- return ONLY the lines that replace the block (same indentation style)
- keep the surrounding code untouched; do not add explanations in the code
- respond ONLY with a JSON object: {{"region_new": ["line 1", "line 2"], "note": "one short sentence"}}

CURRENT CODE:
```r
{code}
```"""


def build_task_list(seeds: list[dict], n_phrasings: int) -> list[dict]:
    tasks = []
    for s in seeds:
        for cls in s["applicability"]:
            for p in range(n_phrasings):
                tasks.append(dict(
                    task_key=f"{s['key']}:{cls}:p{p}", seed=s, cls=cls,
                    phrasing=p, note=PHRASINGS[cls][p]))
    return tasks


CANONICAL_BACKENDS = ("zai", "opencode-spark-free")


def assign_backend(task_key: str, backends: list[str]) -> str:
    """Deterministic hash split of the task space across the generators.

    Hashes against the FIXED canonical list, then filters to the wave's
    backends: a task's canonical assignee never changes across waves, so
    single-backend staging waves never orphan the other half (review fix).
    """
    h = int(hashlib.sha1(("::".join([task_key] + list(CANONICAL_BACKENDS))).encode()).hexdigest()[:8], 16)
    canonical = CANONICAL_BACKENDS[h % len(CANONICAL_BACKENDS)]
    if canonical in backends:
        return canonical
    return backends[h % len(backends)]


def corpus_signature_set(seeds: list[dict], scenario_dir: Path | None,
                         per_file_cap: int = 20000) -> set[tuple]:
    """AST dedupe reference: every seed block + the scenario corpus regions
    (region_old AND region_new — authored output must not duplicate existing
    sft_v2 content)."""
    out: set[tuple] = set()
    for s in seeds:
        out.add(struct_key("\n".join(s["block"])))
    if scenario_dir is None:
        return out
    for fname in SCENARIO_SEED_FILES:
        path = scenario_dir / fname
        if not path.exists():
            continue
        n = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if n >= per_file_cap:
                    break
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                for reg in ("region_old", "region_new"):
                    text = "\n".join(str(l) for l in (r.get(reg) or []))
                    if text.strip():
                        out.add(struct_key(text))
                        n += 1
    return out


class Breaker:
    """Global rate-error circuit (rewrite_author_zai.Breaker semantics)."""

    def __init__(self, max_rounds: int = 4):
        self.lock = threading.Lock()
        self.consecutive = 0
        self.rounds = 0
        self.max_rounds = max_rounds
        self.pause_until = 0.0
        self.stop = threading.Event()
        self.rate_errors = 0

    def wait_turn(self) -> bool:
        while True:
            if self.stop.is_set():
                return False
            with self.lock:
                until = self.pause_until
            if time.time() >= until:
                return True
            time.sleep(min(30.0, until - time.time() + 0.5))

    def report(self, ok: bool, rate_error: bool):
        with self.lock:
            if rate_error:
                self.rate_errors += 1
                self.consecutive += 1
                if self.consecutive >= 4:
                    self.rounds += 1
                    if self.rounds > self.max_rounds:
                        self.stop.set()
                    else:
                        pause = min(600.0, 60.0 * (2 ** (self.rounds - 1)))
                        self.pause_until = time.time() + pause
                        print(f"  [breaker] rate errors x4: pausing "
                              f"{pause:.0f}s (round {self.rounds}/"
                              f"{self.max_rounds})", flush=True)
                    self.consecutive = 0
            elif ok:
                self.consecutive = 0


def _normalized_text(text: str) -> str:
    """Whitespace-normalized text for operator-level no-op detection."""
    return " ".join(text.split())


def validate_region_new(region_new: list[str], region_old_text: str,
                        task: dict, defect: dict | None) -> tuple[bool, str]:
    """The validator stack (fail-closed: any failure drops the row)."""
    text = "\n".join(region_new)
    if not any(l.strip() for l in region_new):
        return False, "empty_region_new"
    if len(region_new) > MAX_BLOCK_LINES:
        return False, f"over_{MAX_BLOCK_LINES}_lines"
    if len(text) > 2500:
        return False, "over_2500_chars"
    if not fragment_clean(text):
        return False, "not_clean_r"
    n = n_statements(text)
    if not MAX_STATEMENTS_LO <= n <= MAX_STATEMENTS_HI:
        return False, f"statements_{n}"
    if text == region_old_text:
        return False, "no_op_text"
    if struct_key(text) == struct_key(region_old_text):
        return False, "no_op_ast"
    if task["cls"] == "fix_the_bug" and defect is not None:
        if defect["bad_re"].search(text):
            return False, "defect_still_present"
        if not defect["good_re"].search(text):
            return False, "fix_not_restored"
    # per-class idiom gates (review fix: the sft_v9 poison vector — a
    # 'rewrite as data.table' answer that keeps %>% and appends junk must
    # NOT pass)
    tidy = bool(TIDY_RE.search(text))
    dt = bool(DT_RE.search(text))
    looping = bool(FOR_RE.search(text))
    if task["cls"] == "tidyverse" and not tidy:
        return False, "idiom_tidyverse_missing"
    if task["cls"] == "data.table" and not dt:
        return False, "idiom_data_table_missing"
    if task["cls"] == "vectorize" and looping:
        return False, "idiom_vectorize_still_looping"
    if task["cls"] == "base_r" and (tidy or dt):
        return False, "idiom_base_r_contaminated"
    # operator-aware no-op: struct_key drops operators, so '<=' vs '<'
    # collided; a textual token-level comparison catches those (review fix)
    if _normalized_text(text) == _normalized_text(region_old_text):
        return False, "no_op_normalized_text"
    return True, ""


def author_row(task: dict, region_old: list[str], region_new: list[str],
               defect: dict | None, backend_name: str, model: str,
               prompt: str) -> dict:
    s = task["seed"]
    phrasing = task["note"]
    prefix = list(s["prefix"]) + [f"# task: {phrasing}"]
    fd = next((i for i in range(min(len(region_old), len(region_new)))
               if region_old[i] != region_new[i]), None)
    chash = hashlib.sha1(
        f"{RULE}\x00{task['task_key']}\x00{backend_name}".encode()
    ).hexdigest()
    row = dict(
        family="nextcoder_r", key=task["task_key"], cls=task["cls"],
        phrasing=task["phrasing"], note=phrasing,
        package=s["package"], path=s["path"],
        prefix=prefix, region_old=region_old, region_new=region_new,
        suffix=list(s["suffix"]), cursor_idx=(fd if fd is not None
                                              else len(region_old) - 1),
        event_diff="",
        case="build_nextcoder_r", backend=backend_name, model=model,
        generated_at=_now(), full_prompt=prompt,
        parent_link=f"{s['file_sha']}@{RULE}",
        seed_key=s["key"], seed_source=s["source"],
        transform_id=f"nextcoder_r/{task['cls']}@1",
        derivation=dict(rule_id=f"nextcoder/{task['cls']}", rule_version=1,
                        phrasing=task["phrasing"], seed=s["key"]),
        content_hash=chash, determinism="D3 author-LLM (gated)")
    if defect is not None:
        row["defect_rule"] = defect["rule"]
    return row


def cmd_author(args) -> int:
    work = Path(args.work)
    seeds_path = work / "seeds.jsonl"
    if not seeds_path.exists():
        sys.exit(f"no seed pool at {seeds_path}; run `mine` first")
    seeds = [json.loads(l) for l in seeds_path.read_text().splitlines() if l.strip()]
    backend_names = args_backends(args)
    for b in backend_names:
        if b not in AUTHOR_BACKENDS:
            sys.exit(f"unknown backend {b!r}; known: {sorted(AUTHOR_BACKENDS)}")
    backends = {b: preflight_backend(b) for b in backend_names}

    out_path = work / "authored.jsonl"
    done_path = Path(str(out_path) + ".done.jsonl")
    done = _load_done(done_path)
    hashes = _load_field(out_path, "content_hash")

    tasks = build_task_list(seeds, args.phrasings)
    if args.max > 0:
        tasks = tasks[:args.max]
    pending = []
    for t in tasks:
        if assign_backend(t["task_key"], backend_names) not in backends:
            continue                       # this wave covers other backends
        if t["task_key"] in done:
            continue
        pending.append(t)
    print(f"[author] backends={backend_names} pool={len(tasks)} "
          f"pending={len(pending)} done_keys={len(done)}", flush=True)

    corpus_keys = corpus_signature_set(
        seeds, None if args.no_corpus_scan else Path(args.scenarios),
        per_file_cap=args.corpus_cap)
    print(f"[author] dedupe corpus signatures: {len(corpus_keys)}", flush=True)

    stats = dict(attempted=0, accepted=0, dropped=Counter(),
                 dups_corpus=0, dups_batch=0, backend_error=0,
                 per_class=Counter(), per_backend=Counter(),
                 per_phrasing=Counter())
    stats_lock, batch_lock = threading.Lock(), threading.Lock()
    # warm-start the structural dedupe set from prior waves (review fix:
    # resumed runs must not accept near-duplicate rows authored pre-crash)
    batch_keys: set[tuple] = set()
    authored_path = out_path
    if authored_path.exists():
        for line in authored_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                prior = json.loads(line)
                sk = struct_key(prior.get("region_new", []))
                if sk:
                    batch_keys.add(sk)
            except Exception:
                continue
    breaker = Breaker()
    rows_written = [0]

    def process(task: dict) -> dict:
        s = task["seed"]
        region_old = list(s["block"])
        defect = None
        old_text = "\n".join(region_old)
        if task["cls"] == "fix_the_bug":
            defect = inject_defect(old_text)
            if defect is None:
                return dict(kind="rejected", task=task,
                            reason="no_injectable_defect")
            old_text = defect["text"]
            region_old = old_text.split("\n")
        bname = assign_backend(task["task_key"], backend_names)
        backend = backends[bname]
        prompt = PROMPT_TMPL.format(task=task["note"], code=old_text)
        if isinstance(backend, MockAuthorBackend):
            prompt += "\n#MOCK-HINT " + json.dumps(dict(
                task_key=task["task_key"], cls=task["cls"],
                phrasing=task["phrasing"], region_old=region_old,
                fix_lines=s["block"]))
        with stats_lock:
            stats["attempted"] += 1
        if not breaker.wait_turn():
            return dict(kind="aborted", task=task)
        try:
            raw = backend.complete(prompt)
            breaker.report(ok=True, rate_error=False)
        except BackendError as e:
            breaker.report(ok=False, rate_error=(e.kind == "rate"))
            with stats_lock:
                stats["backend_error"] += 1
            return dict(kind="backend_error", task=task, err=str(e)[:160])
        obj = extract_json_object(strip_fences(raw))          # layer 1
        rn = (obj or {}).get("region_new")
        if not isinstance(rn, list) or not rn or \
                not all(isinstance(l, str) for l in rn):      # layer 2
            return dict(kind="rejected", task=task, reason="layer2_schema")
        ok, why = validate_region_new(rn, old_text, task, defect)  # layer 3
        if not ok:
            return dict(kind="rejected", task=task, reason=why)
        # fix_the_bug is EXEMPT from the vs-corpus check: its region_old is
        # the INJECTED variant (not corpus content) and the tier-1 answer
        # restores the corpus original — the rewrite_author_zai buinject
        # precedent. The no_op gate still demands difference from the
        # injected text, and within-batch dedupe still applies.
        key = struct_key("\n".join(rn))
        with batch_lock:
            if task["cls"] != "fix_the_bug" and key in corpus_keys:
                with stats_lock:
                    stats["dups_corpus"] += 1
                return dict(kind="rejected", task=task,
                            reason="dedupe_corpus")
            if key in batch_keys:
                with stats_lock:
                    stats["dups_batch"] += 1
                return dict(kind="rejected", task=task,
                            reason="dedupe_batch")
            batch_keys.add(key)
        row = author_row(task, region_old, rn, defect, bname,
                         backend.model, prompt)
        return dict(kind="accepted", task=task, row=row)

    def flush_stats(final: bool = False):
        rep = dict(ts=_now(), final=final, rule=RULE,
                   backends={n: backends[n].stats_summary()
                             for n in backends},
                   pending=len(pending), rows_total=rows_written[0],
                   counts={k: (dict(v) if isinstance(v, Counter) else v)
                           for k, v in stats.items()},
                   done_keys=len(done))
        _write_json(Path(str(out_path) + ".stats.json"), rep)

    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=args.workers)
    outstanding: set = set()
    it = iter(pending)
    n_done = 0
    try:
        while True:
            if time.time() - t0 > args.time_budget or breaker.stop.is_set():
                break
            while len(outstanding) < args.workers * 2:
                try:
                    t = next(it)
                except StopIteration:
                    break
                outstanding.add(ex.submit(process, t))
            if not outstanding:
                break
            done_set, _ = wait(outstanding, timeout=30,
                               return_when=FIRST_EXCEPTION)
            for fut in done_set:
                outstanding.discard(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  [worker-exception] {e!r}", flush=True)
                    continue
                n_done += 1
                task = res["task"]
                if res["kind"] == "accepted":
                    row = res["row"]
                    if row["content_hash"] in hashes:
                        with stats_lock:
                            stats["dups_batch"] += 1
                        rec = dict(key=task["task_key"], ok=True,
                                   rows=0, dup=True, ts=_now())
                    else:
                        hashes.add(row["content_hash"])
                        _append_line(out_path, row)
                        rows_written[0] += 1
                        with stats_lock:
                            stats["accepted"] += 1
                            stats["per_class"][task["cls"]] += 1
                            stats["per_phrasing"][task["phrasing"]] += 1
                            stats["per_backend"][row["backend"]] += 1
                        rec = dict(key=task["task_key"], ok=True, rows=1,
                                   ts=_now())
                elif res["kind"] == "rejected":
                    with stats_lock:
                        stats["dropped"][res["reason"]] += 1
                    rec = dict(key=task["task_key"], ok=False,
                               reason=res["reason"][:200], ts=_now())
                else:                       # backend_error / aborted: retry
                    continue
                _append_line(done_path, rec)
                done[task["task_key"]] = rec
            if n_done % 50 == 0 and n_done:
                flush_stats()
    finally:
        for fut in outstanding:
            fut.cancel()
        ex.shutdown(wait=True)
        flush_stats(final=True)
    print(f"[author] FINISHED done={n_done}/{len(pending)} "
          f"accepted={stats['accepted']} drops="
          f"{dict(stats['dropped'])} dups(corpus/batch)="
          f"{stats['dups_corpus']}/{stats['dups_batch']} in "
          f"{time.time()-t0:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# assemble (assemble_sft_v2.edit_row convention + 3%-by-package split)
# ---------------------------------------------------------------------------

def edit_row(ex: dict, family: str, pkg):
    """The assemble_sft_v2.edit_row() convention, verbatim semantics: zeta2
    render + UPDATED-marker target; drops rows with nothing to predict or
    over the 6000-char budget."""
    if not [l for l in ex.get("region_new") or [] if l.strip()]:
        return None, "empty_region_new"
    ex = dict(ex)
    ex.setdefault("suffix", [])
    prompt = render_zeta2(ex)
    target = "\n".join(ex["region_new"]).rstrip() + UPDATED
    if len(prompt) + len(target) > MAX_CHARS:
        return None, f"over_{MAX_CHARS}"
    row = dict(text=prompt + target, prompt=prompt, target=target,
               family=family, package_or_repo=pkg, has_types=False)
    return row, ""


def package_split(rows: list[dict], frac: float = 0.03,
                  seed: int = SEED_RNG) -> dict[str, set[str]]:
    """Per family: hold out max(1, 3%) of that family's packages (the
    scenarios_v1 convention: seeded shuffle, min 1)."""
    rng = random.Random(seed)
    fam_pkgs: dict[str, set[str]] = {}
    for r in rows:
        fam_pkgs.setdefault(r["family"], set()).add(r["package_or_repo"])
    eval_pkgs: dict[str, set[str]] = {}
    for fam in sorted(fam_pkgs):
        ordered = sorted(fam_pkgs[fam])
        rng.shuffle(ordered)
        n_hold = max(1, round(len(ordered) * frac))
        eval_pkgs[fam] = set(ordered[:n_hold])
    return eval_pkgs


def cmd_assemble(args) -> int:
    work = Path(args.work)
    authored = work / "authored.jsonl"
    if not authored.exists():
        sys.exit(f"no authored rows at {authored}; run `author` first")
    audit = load_holdout_audit(Path(args.holdout_json))
    stats = Counter()
    rows = []
    provenance: dict[str, dict] = {}      # content_hash -> cls/phrasing/backend
    seen_hashes: set[str] = set()
    for line in authored.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pkg = r.get("package") or "?"
        if is_holdout(pkg) or pkg in audit:      # belt-and-braces
            stats["holdout_blocked"] += 1
            continue
        fam = f"nextcoder_{r['cls'].replace('.', '_')}"
        if r["content_hash"] in seen_hashes:
            stats["drop:dup_content_hash"] += 1
            continue
        seen_hashes.add(r["content_hash"])
        provenance[r["content_hash"]] = dict(cls=r["cls"],
                                             phrasing=r["phrasing"],
                                             backend=r["backend"],
                                             model=r["model"])
        ex = dict(path=r["path"], prefix=r["prefix"], suffix=r["suffix"],
                  region_old=r["region_old"], region_new=r["region_new"],
                  cursor_idx=r["cursor_idx"], event_diff=r.get("event_diff", ""))
        srow, why = edit_row(ex, fam, pkg)
        if srow is None:
            stats[f"drop:{why}"] += 1
            continue
        srow["_ch"] = r["content_hash"]
        rows.append(srow)
    eval_pkgs = package_split(rows, frac=args.eval_frac)

    train = [r for r in rows if r["package_or_repo"]
             not in eval_pkgs[r["family"]]]
    evals = [r for r in rows if r["package_or_repo"]
             in eval_pkgs[r["family"]]]

    # provenance counters need the hash join BEFORE the schema-conformance
    # strip (the assembled schema is exactly the assemble_sft_v2 row keys)
    allrows = train + evals
    prov = lambda r: provenance.get(r.get("_ch") or "", {})
    n_per_class = Counter(prov(r).get("cls", "?") for r in allrows)
    n_per_cls_phr = Counter((prov(r).get("cls", "?"),
                             prov(r).get("phrasing")) for r in allrows)
    n_models = Counter(prov(r).get("model", "?") for r in allrows)
    n_backends = Counter(prov(r).get("backend", "?") for r in allrows)
    for r in train + evals:
        r.pop("_ch", None)
    random.Random(SEED_RNG).shuffle(train)

    # validation: schema conformance + no train/eval package overlap
    schema = {"text", "prompt", "target", "family", "package_or_repo",
              "has_types"}
    for r in train + evals:
        assert set(r) == schema, f"schema drift: {sorted(r)}"
        assert r["target"].endswith(UPDATED)
        assert r["text"] == r["prompt"] + r["target"]
    tr_pkgs: dict[str, set[str]] = {}
    for r in train:
        tr_pkgs.setdefault(r["family"], set()).add(r["package_or_repo"])
    for fam, ev in eval_pkgs.items():
        overlap = ev & tr_pkgs.get(fam, set())
        assert not overlap, f"eval/train package overlap: {fam}: {overlap}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, data in (("train", train), ("eval", evals)):
        with open(out / f"{split}.jsonl", "w") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = dict(
        rule=RULE, generated_at=_now(),
        sources=dict(workdir=str(work), authored_file=str(authored),
                     authored_sha256=sha(authored) if authored.exists() else None,
                     holdout_json=str(args.holdout_json)),
        split=dict(eval_frac=args.eval_frac, seed=SEED_RNG,
                   eval_packages={f: sorted(v) for f, v in
                                  sorted(eval_pkgs.items())}),
        counts=dict(train=len(train), eval=len(evals),
                    per_class=dict(n_per_class),
                    per_class_train=dict(Counter(r["family"] for r in train)),
                    per_class_eval=dict(Counter(r["family"] for r in evals)),
                    per_class_phasing={f"{k[0]}|p{k[1]}": v
                                       for k, v in sorted(
                                           n_per_cls_phr.items())}),
        backends=dict(models=dict(n_models),
                      per_backend=dict(n_backends)),
        drops=dict(stats),
        hashes=dict(train_sha256=sha(out / "train.jsonl"),
                    eval_sha256=sha(out / "eval.jsonl")),
    )
    _write_json(out / "manifest.json", manifest)
    print(f"[assemble] train={len(train)} eval={len(evals)} -> {out}")
    print(json.dumps({k: manifest[k] for k in
                      ("counts", "drops", "hashes")}, indent=1), flush=True)
    return 0


# ---------------------------------------------------------------------------
# io helpers (drvfs-safe, the wave-driver conventions)
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _append_line(path: Path, obj: dict, tries: int = 20, wait_s: float = 30.0):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    for attempt in range(tries):
        try:
            with open(path, "a") as fh:
                fh.write(line)
                fh.flush()
            return
        except OSError as e:
            if attempt == tries - 1:
                print(f"  [drvfs-write] giving up on {path}: {e}", flush=True)
                return
            print(f"  [drvfs-write] {e}; retry {attempt + 1} in "
                  f"{wait_s:.0f}s", flush=True)
            time.sleep(wait_s)


def _write_json(path: Path, obj, tries: int = 20, wait_s: float = 30.0):
    text = json.dumps(obj, ensure_ascii=False, indent=1, default=str)
    for attempt in range(tries):
        try:
            path.write_text(text)
            return
        except OSError:
            if attempt == tries - 1:
                return
            time.sleep(wait_s)


def _load_done(path: Path) -> dict:
    done: dict = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done[rec["key"]] = rec
            except (ValueError, KeyError):
                pass
    return done


def _load_field(path: Path, field: str) -> set:
    out: set = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                v = json.loads(line).get(field)
                if v:
                    out.add(v)
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("mine", "author", "assemble", "all"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny defaults (5 seeds, 1 phrasing, mock backend, "
                         "/tmp outputs); explicit flags still override")
    ap.add_argument("--repos", default=str(REPOS_R))
    ap.add_argument("--scenarios", default=str(SCENARIOS))
    ap.add_argument("--holdout-json", default=str(HOLDOUT_JSON))
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED_RNG)
    ap.add_argument("--per-repo-cap", type=int, default=None)
    ap.add_argument("--per-scenario-cap", type=int, default=None)
    ap.add_argument("--phrasings", type=int, default=None, choices=(1, 2, 3))
    ap.add_argument("--backends", default=None)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=3600.0)
    ap.add_argument("--no-corpus-scan", action="store_true")
    ap.add_argument("--corpus-cap", type=int, default=20000)
    ap.add_argument("--eval-frac", type=float, default=0.03)
    args = ap.parse_args()

    # smoke = defaults only; an explicit flag always wins
    if args.smoke:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        args.work = args.work or f"/tmp/nextcoder_r_smoke_{stamp}"
        args.out = args.out or args.work + "/out"
        args.seeds = args.seeds if args.seeds is not None else 5
        args.phrasings = args.phrasings if args.phrasings is not None else 1
        args.backends = args.backends or "mock"
        args.workers = args.workers if args.workers is not None else 1
        args.per_repo_cap = args.per_repo_cap if args.per_repo_cap \
            is not None else 2
        args.per_scenario_cap = args.per_scenario_cap \
            if args.per_scenario_cap is not None else 3
        args.no_corpus_scan = True
        print(f"[smoke] work={args.work} out={args.out}")
    else:
        args.work = args.work or str(DEFAULT_WORK)
        args.out = args.out or str(DEFAULT_OUT)
        args.seeds = args.seeds if args.seeds is not None else 2000
        args.phrasings = args.phrasings if args.phrasings is not None else 3
        args.backends = args.backends or "zai,opencode-spark-free"
        args.workers = args.workers if args.workers is not None else 4
        args.per_repo_cap = args.per_repo_cap if args.per_repo_cap \
            is not None else 150
        args.per_scenario_cap = args.per_scenario_cap \
            if args.per_scenario_cap is not None else 800

    if args.command in ("mine", "all"):
        mine_seeds(args)
    if args.command in ("author", "all"):
        cmd_author(args)
    if args.command in ("assemble", "all"):
        cmd_assemble(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
