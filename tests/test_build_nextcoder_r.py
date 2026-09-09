"""Tests for experiments/data-mining/build_nextcoder_r.py (B11 pack).

Pure-CPU, no network, no NAS: fixtures are built under tmp_path; the mock
backend drives the author loop. Covers AST dedupe, the holdout rule, the
defect inject/gate pair (incl. the `<-` assignment trap), validator
no-op rejection, edit_row/zeta2 schema conformance, the 3%-by-package
split, resume sidecars and the fail-closed key preflight.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / \
    "experiments" / "data-mining" / "build_nextcoder_r.py"
_spec = importlib.util.spec_from_file_location("build_nextcoder_r", _SCRIPT)
ncr = importlib.util.module_from_spec(_spec)
sys.modules["build_nextcoder_r"] = ncr
_spec.loader.exec_module(ncr)


# ---------------------------------------------------------------- fixtures

REPO_R = """sum_three <- function(x) {
  y <- x + 1
  z <- mean(y)
  z
}

scale_it <- function(v, k = 2) {
  for (i in seq_along(v)) {
    v[i] <- v[i] * k
  }
  v
}

keep_small <- function(x, hi = 10) {
  ok <- x[x <= hi]
  total <- sum(ok)
  total
}
"""

SEED_TF = "flag <- TRUE\nres <- calc(flag)\n"
SEED_SEQ = "tot <- 0\nfor (i in seq_along(xs)) {\n  tot <- tot + xs[i]\n}\n"
SEED_BOUND = "keep <- x[x <= hi]\n"


def _holdout_pair():
    hold = next(n for n in (f"pkg{i}" for i in range(100000))
                if ncr.is_holdout(n))
    ok = next(n for n in (f"pkg{i}" for i in range(100000))
              if not ncr.is_holdout(n) and n != hold)
    return hold, ok


# ------------------------------------------------------------ AST machinery

def test_ast_hash_structural_collision():
    # identifiers/literals do not matter: structural dedupe BY DESIGN
    assert ncr.struct_key("x <- 1") == ncr.struct_key("y <- 2")
    assert ncr.struct_key("x <- 1") != ncr.struct_key("mean(x)")
    # comment-only fragments fall back to the text key (never collide with
    # the ast namespace)
    k1, k2 = ncr.struct_key("# a"), ncr.struct_key("x <- 1")
    assert k1[0] == "txt" and k2[0] == "ast" and k1 != k2


def test_fragment_clean():
    assert ncr.fragment_clean("x <- 1\ny <- mean(x)")
    assert not ncr.fragment_clean("f <- function( {")
    assert not ncr.fragment_clean("a <- ) b")


def test_dedupe_drops_structural_duplicates():
    seeds = [dict(block=["x <- 1"]), dict(block=["y <- 2"]),   # dup
             dict(block=["mean(z)"])]
    seen, kept, dups = set(), [], 0
    for s in seeds:                       # the mine() dedupe loop, verbatim
        k = ncr.struct_key("\n".join(s["block"]))
        if k in seen:
            dups += 1
            continue
        seen.add(k)
        kept.append(s)
    assert len(kept) == 2 and dups == 1


# ---------------------------------------------------------- holdout rule

def test_holdout_rule_deterministic():
    hold, ok = _holdout_pair()
    assert ncr.is_holdout(hold) and ncr.is_holdout(hold)   # stable
    assert not ncr.is_holdout(ok)
    assert {p for p in (hold, ok) if p in ncr.holdout_block({hold, ok},
                                                            set())} == {hold}


def test_holdout_blocks_audit_listed_names(tmp_path):
    hold, ok = _holdout_pair()
    audit = tmp_path / "holdout_packages.json"
    audit.write_text(json.dumps({"held_out": {"cran": [hold]}}))
    loaded = ncr.load_holdout_audit(audit)
    assert loaded == {hold}
    blocked = ncr.holdout_block({hold, ok}, loaded)
    assert blocked == {hold}          # rule-derived OR audit-listed


# ---------------------------------------------------- defect inject + gate

def _task(cls="fix_the_bug"):
    return dict(cls=cls, note=ncr.PHRASINGS[cls][0])


def test_inject_true_false_and_gate():
    d = ncr.inject_defect(SEED_TF)
    assert d["rule"] == "true_false_symbol"
    assert "T" in d["text"] and "TRUE" not in d["text"]
    ok, why = ncr.validate_region_new(SEED_TF.split("\n")[:-1],
                                      d["text"], _task(), d)
    assert ok, why                                   # restore passes
    # verbatim echo of the injected text is a no-op (dropped earlier)
    ok0, why0 = ncr.validate_region_new(["flag <- T", "res <- calc(flag)"],
                                        d["text"], _task(), d)
    assert not ok0 and why0 == "no_op_ast"
    # structurally different but still defective -> defect gate fires
    bad, why = ncr.validate_region_new(
        ["flag <- T", "res <- calc(flag, na.rm = TRUE)"],
        d["text"], _task(), d)
    assert not bad and why == "defect_still_present"


def test_inject_seq_safety_and_gate():
    d = ncr.inject_defect(SEED_SEQ)
    assert d["rule"] == "seq_safety" and "1:length(" in d["text"]
    ok, _ = ncr.validate_region_new(SEED_SEQ.split("\n")[:-1],
                                    d["text"], _task(), d)
    assert ok
    bad = ["tot <- 0", "for (i in 1:length(xs)) {", "  tot <- tot + xs[i]",
           "}", "extra <- 1"]
    ok2, why = ncr.validate_region_new(bad, d["text"], _task(), d)
    assert not ok2 and why == "defect_still_present"


def test_boundary_defect_never_matches_assignment():
    d = ncr.inject_defect(SEED_BOUND)
    assert d["rule"] == "boundary_operator"
    # a fix containing assignments (`<-`) but restoring `<=` must PASS:
    # the bad regex must not fire on the assignment arrow (regression trap)
    fixed = ["keep <- x[x <= hi]", "other <- v <- w <- 1"]
    ok, why = ncr.validate_region_new(fixed, d["text"], _task(), d)
    assert ok, why
    unfixed = ["keep <- x[x < hi]", "extra <- 1"]
    ok2, why2 = ncr.validate_region_new(unfixed, d["text"], _task(), d)
    assert not ok2 and why2 == "defect_still_present"


def test_no_injectable_defect_returns_none():
    assert ncr.inject_defect("nothing <- to_inject\nhere(x)\n") is None


# ------------------------------------------------------------- validators

def test_noop_edits_rejected():
    t = _task("tidyverse")
    old = ["x <- 1", "y <- 2"]
    ok, why = ncr.validate_region_new(list(old), "\n".join(old), t, None)
    assert not ok and why == "no_op_text"
    # rename-only rewrite: parses clean, structurally identical -> reject
    ok2, why2 = ncr.validate_region_new(["a <- 1", "b <- 2"],
                                        "\n".join(old), t, None)
    assert not ok2 and why2 == "no_op_ast"


def test_validator_drops_broken_r():
    t = _task("tidyverse")
    ok, why = ncr.validate_region_new(["this is << not R"],
                                      "x <- 1", t, None)
    assert not ok and why == "not_clean_r"
    ok2, why2 = ncr.validate_region_new([""], "x <- 1", t, None)
    assert not ok2 and why2 == "empty_region_new"


# --------------------------------------------- edit_row / zeta2 conformance

def test_edit_row_conformance():
    ex = dict(path="R/a.R", prefix=["x <- 1", "# task: Rewrite this block"],
              suffix=["z <- 3"], region_old=["y <- 2"],
              region_new=["y <- mean(x)"], cursor_idx=0, event_diff="")
    row, why = ncr.edit_row(ex, "nextcoder_tidyverse", "pkgA")
    assert row is not None
    assert set(row) == {"text", "prompt", "target", "family",
                        "package_or_repo", "has_types"}
    assert row["text"] == row["prompt"] + row["target"]      # v1 convention
    assert row["target"].endswith("\n>>>>>>> UPDATED")
    assert row["has_types"] is False
    for marker in ("<[fim-suffix]>", "<[fim-prefix]>", "<<<<<<< CURRENT",
                   "=======", "<[fim-middle]>"):
        assert marker in row["prompt"]
    assert "# task: Rewrite this block" in row["prompt"]


def test_edit_row_drops_empty_and_over_budget():
    ex = dict(path="R/a.R", prefix=["x <- 1"], suffix=[],
              region_old=["y <- 2"], region_new=["   "], cursor_idx=0)
    row, why = ncr.edit_row(ex, "f", "p")
    assert row is None and why == "empty_region_new"
    big = ["y <- " + "a" * 200] * 60
    ex2 = dict(path="R/a.R", prefix=[], suffix=[], region_old=["y <- 2"],
               region_new=big, cursor_idx=0)
    row2, why2 = ncr.edit_row(ex2, "f", "p")
    assert row2 is None and why2 == "over_6000"


# ------------------------------------------------------- package split

def test_package_split_three_percent_no_overlap():
    rows = [dict(family=f"nextcoder_c{k}",
                 package_or_repo=f"pkg{ k }-{i % 40}")
            for k in range(3) for i in range(40)]
    eval_pkgs = ncr.package_split(rows, frac=0.03)
    for fam, ev in eval_pkgs.items():
        assert len(ev) >= 1                       # min-1 convention
        assert len(ev) <= 3                       # ~3% of 40
        train_pkgs = {r["package_or_repo"] for r in rows
                      if r["family"] == fam} - ev
        assert not (train_pkgs & ev)


# --------------------------------------------------- backend machinery

def test_assign_backend_deterministic_split():
    keys = [f"ncr_seed:{i}:tidyverse:p0" for i in range(100)]
    bes = ["zai", "opencode-spark-free"]
    a1 = [ncr.assign_backend(k, bes) for k in keys]
    a2 = [ncr.assign_backend(k, bes) for k in keys]
    assert a1 == a2                               # resume-stable
    assert set(a1) == set(bes)                    # both generators used


def test_preflight_fail_closed_without_key(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ZAI_API_KEY"):
        ncr.preflight_backend("zai")


# ------------------------------------------------- end-to-end (mock, tmp)

def _ns(**kw):
    base = dict(work="", out="", holdout_json="/nonexistent/holdout.json",
                repos="", scenarios="", seed=42, seeds=6, phrasings=3,
                backends="mock", per_repo_cap=10, per_scenario_cap=10,
                max=0, workers=2, time_budget=120.0, no_corpus_scan=True,
                scenarios_scan=False, corpus_cap=100, eval_frac=0.03)
    base.update(kw)
    return SimpleNamespace(**base)


def _holdout_free_pair():
    names = (f"pkg{i}" for i in range(100000))
    ok = [n for _, n in zip(range(2), (n for n in names
                                       if not ncr.is_holdout(n)))]
    return ok[0], ok[1]


def _fixtures(tmp_path: Path):
    hold, ok = _holdout_pair()
    ok2, ok3 = _holdout_free_pair()
    repos = tmp_path / "repos" / "r"
    (repos / ok / "R").mkdir(parents=True)
    (repos / ok / "R" / "code.R").write_text(REPO_R)
    (repos / hold / "R").mkdir(parents=True)      # must be excluded
    (repos / hold / "R" / "code.R").write_text(REPO_R)
    scen = tmp_path / "scenarios"
    scen.mkdir()
    rows = [
        dict(family="pipe_rewrite", package=ok2, path="R/a.R",
             prefix=["p <- 1"],
             region_old=["q <- paste(p, 1)", "r <- toupper(q)"],
             region_new=["q <- paste(p, 2)", "r <- toupper(q)"],
             cursor_idx=0, event_diff=""),
        dict(family="pipe_rewrite", package=hold, path="R/b.R",   # excluded
             prefix=["p <- 1"],
             region_old=["q <- paste(p, 1)", "r <- tolower(q)"],
             region_new=["q <- paste(p, 2)", "r <- tolower(q)"],
             cursor_idx=0, event_diff=""),
        dict(family="pipe_rewrite", package=ok3, path="R/c.R",
             prefix=["p <- 2"],
             region_old=["  s <- sum(as.numeric(p))", "  t <- max(s)"],
             region_new=["  s <- sum(p)", "  t <- max(s)"],
             cursor_idx=0, event_diff=""),
    ]
    (scen / "pipe_rewrite.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    return repos, scen, hold, ok


def test_mock_end_to_end_and_resume(tmp_path, capsys):
    repos, scen, hold, ok = _fixtures(tmp_path)
    work = tmp_path / "work"
    args = _ns(repos=str(repos), scenarios=str(scen), work=str(work),
               out=str(work / "out"), seeds=4)
    ncr.mine_seeds(args)

    seeds = [json.loads(l) for l in
             (work / "seeds.jsonl").read_text().splitlines() if l.strip()]
    assert seeds, "no seeds mined"
    assert all(s["package"] != hold for s in seeds)      # holdout excluded
    assert all(s["parent_link"].endswith("@nextcoder-r@v1") for s in seeds)
    assert all(s["applicability"] for s in seeds)

    assert ncr.cmd_author(args) == 0
    authored = [json.loads(l) for l in
                (work / "authored.jsonl").read_text().splitlines() if l.strip()]
    assert authored
    for r in authored:
        assert r["parent_link"].endswith("@nextcoder-r@v1")   # parent-link
        assert r["backend"] == "mock" and r["model"] == "mock-0"  # model tag
        assert r["cls"] in ncr.CLASSES
        assert 0 <= r["phrasing"] < 3
        assert "# task:" in "\n".join(r["prefix"])
        assert r["content_hash"]
    assert (work / "authored.jsonl.done.jsonl").exists()
    assert (work / "authored.jsonl.stats.json").exists()

    # resume: every task key already terminal -> nothing new authored
    n_before = len(authored)
    assert ncr.cmd_author(args) == 0
    out = capsys.readouterr().out
    assert "pending=0" in out
    authored2 = [l for l in (work / "authored.jsonl").read_text().splitlines()
                 if l.strip()]
    assert len(authored2) == n_before

    assert ncr.cmd_assemble(args) == 0
    manifest = json.loads((work / "out" / "manifest.json").read_text())
    train = [json.loads(l) for l in
             (work / "out" / "train.jsonl").read_text().splitlines() if l.strip()]
    evals = [json.loads(l) for l in
             (work / "out" / "eval.jsonl").read_text().splitlines() if l.strip()]
    assert train and evals
    for r in train + evals:
        assert set(r) == {"text", "prompt", "target", "family",
                          "package_or_repo", "has_types"}
        assert r["text"] == r["prompt"] + r["target"]
        assert r["target"].endswith(">>>>>>> UPDATED")
    fams = {r["family"] for r in train}
    tr = {}
    for r in train:
        tr.setdefault(r["family"], set()).add(r["package_or_repo"])
    for r in evals:
        assert r["package_or_repo"] not in tr.get(r["family"], set())
    assert manifest["counts"]["train"] == len(train)
    assert manifest["counts"]["eval"] == len(evals)
    assert manifest["counts"]["per_class_phasing"]
    assert set(manifest["hashes"]) == {"train_sha256", "eval_sha256"}


def test_mock_backend_needs_no_key(monkeypatch):
    for k in ("ZAI_API_KEY", "OPENCODE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    b = ncr.preflight_backend("mock")
    assert b.name == "mock" and b.model == "mock-0"


def test_spec_backends_registered():
    # the B11 spec generators: glm-5.3 via ZaiBackend + muse-spark via the
    # opencode Responses free tier
    assert issubclass(ncr.AUTHOR_BACKENDS["zai"], ncr.ZaiBackend)
    assert issubclass(ncr.AUTHOR_BACKENDS["opencode-spark-free"],
                      ncr.OpencodeSparkFreeBackend)
    assert ncr.AUTHOR_BACKENDS["zai"].model == "glm-5.3"
