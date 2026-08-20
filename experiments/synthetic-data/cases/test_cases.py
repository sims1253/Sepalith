#!/usr/bin/env python3
"""Self-contained tests for the cases library (run:
uv run python experiments/synthetic-data/cases/test_cases.py).

Covers: spec loading + validation (both shipped specs and malformed ones),
the layer-3 validators (comment gate, tidyselect tree-sitter gate, generic
fragment gate, row-structure checks), target normalizers, both corpus
selectors (a temp dataset file; the tidyselect extractor over an in-memory
bundle), the agy CLI contract (never a positional prompt), JSON extraction,
and FULL generate cycles on the mock backend: happy path with provenance,
validator gating, layer-1 JSON-failure regeneration, content-hash dedup,
and done-sidecar resume. No corpus access, no keys, no network.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.backends import MockBackend, extract_json_object
from cases import corpus as C
from cases.corpus import _apply_difficulty, extract_mid_body_edits, \
    extract_tidyselect, select_corpus
from cases.rows import normalize_target
from cases.spec import (SpecError, list_cases, load_case, spec_from_dict)
from cases.validators import (REGISTRY, check_row, get_validator,
                              rc_comment, rc_midline, v_comment,
                              v_fragment, v_mid_body_edit_line, v_tidyselect)
from cases.generate import main as generate_main
from scenarios import Bundle

R_SRC = b'''make_tbl <- function(df) {
  pre <- nrow(df)
  out <- df %>%
    select(USUBJID, starts_with("PARAMCD"), AVAL)
  nums <- summarise(df, across(where(is.numeric), ~ mean(.x)))
  keep <- out[!is.na(out$AVAL), ]
  out
}
'''

MOCK_ENV = ("CASES_MOCK_FAIL_EVERY", "CASES_MOCK_INVALID",
            "CASES_MOCK_CONSTANT")


def _clear_mock_env():
    for k in MOCK_ENV:
        os.environ.pop(k, None)


def toy_spec_dict(path, **over):
    d = {
        "name": "toy_comment",
        "version": 1,
        "description": "toy comment case for tests",
        "novelty_note": "test-only",
        "target_field": "comment",
        "target_normalizer": "comment",
        "prompt_templates": [
            "Write ONE concise R comment (max 80 chars, no code) for:\n\n"
            "{code}\n\nJSON only: {{\"comment\": string}}"],
        "corpus_source": {
            "kind": "dataset_file",
            "path": str(path),
            "selector": "c2c_cache_blocks",
            "params": {"max_items": 5},
            "provenance": {"license": "test-license",
                           "source_url": "https://example.com/src",
                           "note": "test corpus"},
        },
        "parameter_sampler": {"name": "template_uniform", "params": {}},
        "target_construction": {"kind": "comment_prefix",
                                "params": {"comment_indent": "  # "}},
        "validator": {"name": "r_comment_gate", "params": {"max_len": 90}},
        "row_check": {"name": "ends_with_comment_line", "params": {}},
        "dedup": "target+key",
        "difficulty": {"target_chars_min": 4, "target_chars_max": 90},
    }
    d.update(over)
    return d


def write_toy_corpus(tmpdir, n=5):
    cands = [dict(prefix_lines=["f <- function(x) {", "  a <- x + 1"],
                  block=["  b <- a * 2", "  sum(b)"], package="pkgA",
                  path="R/a.R") for _ in range(n)]
    p = Path(tmpdir) / "corpus.json"
    p.write_text(json.dumps({"cands": cands}))
    return p


def write_toy_spec(tmpdir, corpus_path, **over):
    p = Path(tmpdir) / "spec.json"
    p.write_text(json.dumps(toy_spec_dict(corpus_path, **over)))
    return p


class TestSpecLoading(unittest.TestCase):
    def test_shipped_specs_load(self):
        cases = list_cases()
        self.assertIn("comment_to_code_styles", cases)
        self.assertIn("tidyselect_completion", cases)
        s = load_case("comment_to_code_styles")
        self.assertEqual(len(s.prompt_templates), 5)
        self.assertEqual(s.corpus_source["selector"], "c2c_cache_blocks")
        self.assertEqual(s.validator["name"], "r_comment_gate")
        self.assertEqual(s.family, "comment_to_code_styles")
        self.assertTrue(s.novelty_note)
        t = load_case("tidyselect_completion")
        self.assertEqual(len(t.prompt_templates), 2)
        self.assertEqual(t.validator["name"], "tidyselect_call")
        self.assertEqual(t.dedup, "target+key")

    def test_template_fill(self):
        s = load_case("comment_to_code_styles")
        item = dict(block=["  x <- 1", "  y <- x + 1"],
                    prefix=["f <- function() {"], package="p", path="R/p.R")
        prompt = s.fill_template(0, item)
        self.assertIn("  x <- 1\n  y <- x + 1", prompt)   # {code} filled
        self.assertIn('{"comment": string}', prompt)      # {{...}} -> {...}
        self.assertNotIn("{code}", prompt)
        t = load_case("tidyselect_completion")
        prompt = t.fill_template(0, dict(prefix=["  select(USUBJID, "],
                                         block=["starts_with(\"P\")"]))
        self.assertIn("select(USUBJID, ", prompt)
        self.assertNotIn("{context}", prompt)

    def test_missing_and_unknown(self):
        with self.assertRaises(SpecError):
            spec_from_dict({"name": "x"})               # missing required keys
        with self.assertRaises(SpecError):
            spec_from_dict(toy_spec_dict("/dev/null", dedup="nope"))
        with self.assertRaises(SpecError):
            load_case("does_not_exist")

    def test_difficulty_trim(self):
        s = load_case("comment_to_code_styles")
        s.difficulty = {"context_lines": 2}
        item = dict(prefix=["a", "b", "c", "d"], block=["x"])
        self.assertEqual(_apply_difficulty(item, s)["prefix"], ["c", "d"])
        self.assertEqual(item["prefix"], ["a", "b", "c", "d"])  # untouched


class TestValidators(unittest.TestCase):
    def test_comment_gate(self):
        ok, _ = v_comment("keep only rows above the threshold", {})
        self.assertTrue(ok)
        for bad in ("x <- 1", "mean(x)", "a;b", "",
                    "c" * 91, "two\nlines"):
            ok, reason = v_comment(bad, {})
            self.assertFalse(ok, bad)

    def test_tidyselect_gate_accepts(self):
        good = ('starts_with("PARAM")', 'ends_with("DT")',
                'contains("AVIS")', 'matches("^[A-Z]+$")',
                'all_of(c("AGE", "SEX"))', 'any_of(names(panel))',
                'where(is.numeric)', 'tidyselect::last_col(0)',
                'num_range("V", 1:3)')
        for g in good:
            ok, reason = v_tidyselect(g, {})
            self.assertTrue(ok, f"{g}: {reason}")

    def test_tidyselect_gate_rejects(self):
        bad = ('mean(x)',                      # not a helper
               'select(a, b)',                 # verb, not helper
               'starts_with(PARAM)',           # needs a string literal
               'starts_with()',                # no arguments
               'where("numeric")',             # predicate must be name/call
               'all_of(list(a))',              # vector must be c(...) / var
               'starts_with("A") + 1',         # expression, not one call
               'x <- starts_with("A")',        # assignment, not one call
               'foo\nbar',                     # multiline / not a call
               '')
        for b in bad:
            ok, _ = v_tidyselect(b, {})
            self.assertFalse(ok, b)

    def test_fragment_gate(self):
        ok, _ = v_fragment("x <- 1\ny <- x + 1", {})
        self.assertTrue(ok)
        ok, _ = v_fragment("x <- 1", {"min_statements": 2})
        self.assertFalse(ok)

    def test_row_checks(self):
        ok, _ = rc_comment(dict(prefix=["  # fine"]), {})
        self.assertTrue(ok)
        ok, _ = rc_comment(dict(prefix=["  #' roxygen"]), {})
        self.assertFalse(ok)
        ok, _ = rc_comment(dict(prefix=["  x <- 1"]), {})
        self.assertFalse(ok)
        self.assertTrue(rc_midline(dict(prefix=["  select(a, "]), {})[0])
        self.assertTrue(rc_midline(dict(prefix=["  select(-"]), {})[0])
        self.assertFalse(rc_midline(dict(prefix=["  x <- 1"]), {})[0])

    def test_check_row(self):
        row = dict(family="f", package="p", path="R/a.R",
                   prefix=["  # c"], region_old=[""], region_new=["  x <- 1"],
                   cursor_idx=0, event_diff="", note="n", case="c",
                   backend="mock", model="m", full_prompt="p",
                   generated_at="t")
        ok, reason = check_row(row)
        self.assertTrue(ok, reason)
        for mutate in (lambda r: r.pop("full_prompt"),
                       lambda r: r.update(region_old=["x"]),
                       lambda r: r.update(cursor_idx=3),
                       lambda r: r.update(region_new=[]),
                       lambda r: r.update(prefix=["a\nb"])):
            row2 = json.loads(json.dumps(row))
            mutate(row2)
            ok, _ = check_row(row2)
            self.assertFalse(ok)

    def test_registry_names(self):
        for name in ("r_comment_gate", "tidyselect_call", "r_fragment",
                     "validate_py"):
            self.assertIn(name, REGISTRY)
        v = get_validator({"name": "r_comment_gate", "params": {"max_len": 5}})
        ok, _ = v("toolongcomment")
        self.assertFalse(ok)


class TestNormalizers(unittest.TestCase):
    def test_comment(self):
        self.assertEqual(normalize_target("comment", "  # do the thing  "),
                         "do the thing")
        self.assertEqual(normalize_target("comment", '"quoted comment"'),
                         "quoted comment")
        self.assertEqual(normalize_target("comment", "a\n  b"), "a b")

    def test_code_and_raw(self):
        self.assertEqual(normalize_target("code", " x <- 1 \n"), "x <- 1")
        self.assertEqual(normalize_target("raw", " t "), " t ")


class TestBackends(unittest.TestCase):
    def test_agy_never_positional_prompt(self):
        """The agy CLI silently drops positional prompts: the prompt MUST go
        through the --prompt flag (and every call needs --new-project)."""
        import cases.backends as B
        seen = {}

        class R:
            returncode = 0
            stdout = '{"comment": "ok"}'
            stderr = ""

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return R()

        orig = B.subprocess.run
        B.subprocess.run = fake_run
        try:
            be = B.AgyBackend()
            txt = be.complete("MY PROMPT")
        finally:
            B.subprocess.run = orig
        cmd = seen["cmd"]
        self.assertEqual(txt, '{"comment": "ok"}')
        self.assertIn("--prompt", cmd)
        self.assertEqual(cmd[cmd.index("--prompt") + 1], "MY PROMPT")
        self.assertNotIn("MY PROMPT", cmd[:cmd.index("--prompt")])
        self.assertIn("--new-project", cmd)
        self.assertEqual(cmd[0], "agy")

    def test_extract_json_object(self):
        self.assertEqual(extract_json_object('{"comment": "x"}'),
                         {"comment": "x"})
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'),
                         {"a": 1})
        self.assertEqual(extract_json_object('sure! {"a": [1,2]} thanks'),
                         {"a": [1, 2]})
        self.assertIsNone(extract_json_object("no json here"))
        self.assertIsNone(extract_json_object(""))
        self.assertIsNone(extract_json_object("[1,2]"))

    def test_mock_modes(self):
        _clear_mock_env()
        m = MockBackend(target_key="comment")
        obj = extract_json_object(m.complete("prompt-a"))
        self.assertIn("comment", obj)
        m2 = MockBackend(target_key="completion")
        obj = extract_json_object(m2.complete("prompt-b"))
        self.assertIn("completion", obj)
        ok, _ = v_tidyselect(obj["completion"], {})
        self.assertTrue(ok)


class TestCorpusSelectors(unittest.TestCase):
    def test_c2c_cache_selector(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = write_toy_corpus(td, n=4)
            spec = spec_from_dict(toy_spec_dict(corpus),
                                  origin=str(Path(td) / "spec.json"))
            import random
            items = select_corpus(spec, random.Random(7), want=2)
            self.assertEqual(len(items), 4)
            it = items[0]
            self.assertTrue(it["key"].startswith("c2c:"))
            self.assertEqual(it["block"], ["  b <- a * 2", "  sum(b)"])
            self.assertEqual(it["prefix"][-1], "  a <- x + 1")
            self.assertEqual(it["corpus_target"], "  b <- a * 2\n  sum(b)")

    def test_tidyselect_extractor(self):
        b = Bundle("pkgT", "R/t.R", R_SRC)
        items = extract_tidyselect(b, {"select", "relocate", "across"},
                                   dict.fromkeys(
                                       ("starts_with", "ends_with", "contains",
                                        "matches", "num_range", "last_col",
                                        "all_of", "any_of", "where")),
                                   context_lines=8)
        by_target = {it["block"][0]: it for it in items}
        self.assertIn('starts_with("PARAMCD")', by_target)
        self.assertIn("where(is.numeric)", by_target)
        it = by_target['starts_with("PARAMCD")']
        self.assertTrue(it["prefix"][-1].endswith("select(USUBJID,"))
        self.assertEqual(it["suffix"], [", AVAL)"])
        self.assertEqual(it["corpus_target"], 'starts_with("PARAMCD")')
        self.assertEqual(it["package"], "pkgT")
        for it in items:  # every extracted target passes the validator
            ok, reason = v_tidyselect(it["block"][0], {})
            self.assertTrue(ok, reason)


class TestGenerateCycles(unittest.TestCase):
    """Full harness cycles on the mock backend (env knobs drive failures)."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = Path(self._td.name)
        self.corpus = write_toy_corpus(self.td, n=5)
        self.spec_path = write_toy_spec(self.td, self.corpus)
        _clear_mock_env()

    def tearDown(self):
        self._td.cleanup()
        _clear_mock_env()

    def _run(self, n, **over):
        out = self.td / "out.jsonl"
        spec_p = self.spec_path
        if over:
            spec_p = write_toy_spec(self.td, self.corpus, **over)
        rc = generate_main(["--spec", str(spec_p), "--n", str(n),
                            "--backend", "mock", "--out", str(out)])
        self.assertEqual(rc, 0)
        return out

    def _rows(self, out: Path):
        return [json.loads(l) for l in out.read_text().splitlines() if l.strip()]

    def test_full_cycle_and_provenance(self):
        out = self._run(3)
        rows = self._rows(out)
        self.assertEqual(len(rows), 3)
        for row in rows:
            for k in ("case", "case_version", "family", "template_index",
                      "backend", "model", "full_prompt", "generated_at",
                      "seed", "corpus_key", "content_hash", "license",
                      "source_url", "package", "path", "prefix",
                      "region_old", "region_new", "cursor_idx"):
                self.assertIn(k, row, k)
            self.assertEqual(row["case"], "toy_comment")
            self.assertEqual(row["backend"], "mock")
            self.assertEqual(row["license"], "test-license")
            self.assertEqual(row["source_url"], "https://example.com/src")
            self.assertEqual(row["region_old"], [""])
            self.assertEqual(row["cursor_idx"], 0)
            self.assertEqual(row["region_new"], ["  b <- a * 2", "  sum(b)"])
            self.assertTrue(row["prefix"][-1].startswith("  # "))
            self.assertIn("  b <- a * 2\n  sum(b)", row["full_prompt"])
            ok, reason = check_row(row)
            self.assertTrue(ok, reason)
        done = [json.loads(l) for l in
                (Path(str(out) + ".done.jsonl")).read_text().splitlines()]
        self.assertEqual(sum(1 for d in done if d["ok"]), 3)
        stats = json.loads(Path(str(out) + ".stats.json").read_text())
        self.assertEqual(stats["rows_total"], 3)
        self.assertEqual(stats["counts"]["accepted"], 3)
        self.assertEqual(len({r["content_hash"] for r in rows}), 3)

    def test_validator_gating(self):
        os.environ["CASES_MOCK_INVALID"] = "1"   # JSON ok, gate always fails
        out = self._run(3)
        self.assertFalse(out.exists())           # nothing accepted, nothing written
        done = [json.loads(l) for l in
                Path(str(out) + ".done.jsonl").read_text().splitlines()]
        self.assertEqual(len(done), 5)           # every item marked done+rejected
        self.assertTrue(all(not d["ok"] for d in done))
        self.assertTrue(all("layer3" in d["reason"] for d in done))
        stats = json.loads(Path(str(out) + ".stats.json").read_text())
        self.assertEqual(stats["counts"]["accepted"], 0)
        self.assertGreaterEqual(stats["counts"]["rejected_validator"], 10)

    def test_layer1_regeneration(self):
        os.environ["CASES_MOCK_FAIL_EVERY"] = "2"  # every 2nd request: no JSON
        out = self._run(2)
        rows = self._rows(out)
        self.assertEqual(len(rows), 2)             # regen recovers
        stats = json.loads(Path(str(out) + ".stats.json").read_text())
        self.assertGreaterEqual(stats["counts"]["rejected_json"], 1)

    def test_dedup(self):
        os.environ["CASES_MOCK_CONSTANT"] = "1"    # identical target every time
        out = self._run(3, dedup="target")
        rows = self._rows(out)
        self.assertEqual(len(rows), 1)             # one unique target only
        stats = json.loads(Path(str(out) + ".stats.json").read_text())
        self.assertGreaterEqual(stats["counts"]["dups"], 4)
        done = [json.loads(l) for l in
                Path(str(out) + ".done.jsonl").read_text().splitlines()]
        self.assertEqual(sum(1 for d in done if d.get("dup")), 4)

    def test_resume(self):
        out = self._run(2)
        first = self._rows(out)
        self.assertEqual(len(first), 2)
        out2 = self._run(4)                        # same out path: resumes
        rows = self._rows(out2)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[:2], first)          # earlier rows untouched
        keys = [r["corpus_key"] for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        done = [json.loads(l) for l in
                Path(str(out) + ".done.jsonl").read_text().splitlines()]
        self.assertEqual(len(done), 4)             # no item re-burned
        stats = json.loads(Path(str(out) + ".stats.json").read_text())
        self.assertEqual(stats["counts"]["items_skipped_done"], 2)


MID_SRC = b'''midfun <- function(x, df) {
  pre <- head(x, 1)
  a <- round(x, 3)
  b <- mean(df$y)
  c1 <- stats::sd(df$z)
  flag <- grepl("a", x, fixed = TRUE)
  q <- sprintf("%0.4f", 0.05)
  a2 <- a + 1
  out <- c(a, b, c1, pre)
  total <- sum(out)
  total
}
'''


@C.register("mid_body_toy")
def _sel_mid_body_toy(spec, rng, want):
    """Test-only selector: the mid_body_edit extractor over an in-memory
    bundle (the tidyselect test precedent — no corpus access)."""
    items = extract_mid_body_edits(Bundle("pkgM", "R/m.R", MID_SRC), rng,
                                   dict(per_file_cap=12, per_function_cap=12,
                                        window_lines=10))
    for i, it in enumerate(items):
        it["key"] = f"toy_mbe:{i}"
    return items


def mid_body_spec_dict(**over):
    d = {
        "name": "toy_mid_body",
        "version": 1,
        "description": "toy mid_body_edit case for tests",
        "novelty_note": "test-only",
        "target_field": "completion",
        "target_normalizer": "code",
        "prompt_templates": [
            "One line inside this R function changed; the cursor marks where "
            "it goes. The function continues below the cursor.\n\n"
            "Above the cursor:\n\n{context}\n\nBelow the cursor:\n\n"
            "{suffix}\n\nJSON only: {{\"completion\": string}}"],
        "corpus_source": {
            "kind": "normalized_corpus",
            "selector": "mid_body_toy",
            "params": {},
            "provenance": {"license": "test-license",
                           "source_url": "https://example.com/src",
                           "note": "toy bundle"},
        },
        "parameter_sampler": {"name": "template_uniform", "params": {}},
        "target_construction": {
            "kind": "exact_completion",
            "params": {"carry": ["mutation_kind", "corpus_line", "fn_head",
                                 "old_tok", "new_tok", "stat_fn",
                                 "insert_base"]}},
        "validator": {"name": "mid_body_edit_line", "params": {}},
        "row_check": {"name": "mid_body_edit_site", "params": {}},
        "dedup": "target+key",
        "difficulty": {"target_chars_min": 3, "target_chars_max": 240,
                       "target_lines_min": 1, "target_lines_max": 1},
    }
    d.update(over)
    return d


class TestMidBodyEdit(unittest.TestCase):
    """case 6: one deterministic line-level mutation mid-function; the
    suffix pins the post-change function remainder (scope-aware context)."""

    EXTRACT_PARAMS = dict(per_file_cap=12, per_function_cap=12, window_lines=10)

    def test_all_four_kinds_and_invariants(self):
        import random
        seen_kinds = set()
        for seed in range(40):
            b = Bundle("pkgM", "R/m.R", MID_SRC)
            for it in extract_mid_body_edits(b, random.Random(seed),
                                             dict(self.EXTRACT_PARAMS)):
                seen_kinds.add(it["mutation_kind"])
                self._assert_item(it)
        self.assertEqual(seen_kinds, {"arg_edit", "na_rm_insert",
                                      "rename_once", "insert_line"})

    def _assert_item(self, it):
        import scenarios as S
        kind, new = it["mutation_kind"], it["corpus_target"]
        old = it.get("corpus_line") or ""
        self.assertEqual(it["block"], [new])
        if kind == "arg_edit":
            pair = S._single_token_edit(old, new, r"TRUE|FALSE|\d+(?:\.\d+)?L?")
            self.assertIsNotNone(pair, (old, new))
            self.assertNotEqual(pair[0], pair[1])
        elif kind == "na_rm_insert":
            self.assertTrue(S._single_insert_before_close(
                old, new, ", na.rm = TRUE"), (old, new))
        elif kind == "rename_once":
            cands = [S._single_token_edit(old, new, p)
                     for p in S._TOKEN_PATS]
            self.assertTrue(any(c and S._valid_rename_pair(*c)
                                for c in cands), (old, new))
        else:                                   # insert_line
            pair = S._single_token_edit(it["insert_base"], new,
                                        r"[A-Za-z][A-Za-z0-9._]*")
            self.assertIsNotNone(pair, (it["insert_base"], new))
            self.assertTrue(S._valid_rename_pair(*pair))
            self.assertNotIn(new, it["prefix"] + it["suffix"])
        # geometry: typed-partial prefix tail, post-change function remainder
        self.assertTrue(it["prefix"][-1].strip())
        self.assertIn("midfun <- function(x, df) {", it["prefix"])
        self.assertTrue(any(l.strip() == "}" for l in it["suffix"]))
        self.assertTrue(any(l.strip() and l.strip() != "}"
                            for l in it["suffix"]))

    def test_validator_floor(self):
        ok, _ = v_mid_body_edit_line("  b <- mean(df$y, na.rm = TRUE)", {})
        self.assertTrue(ok)
        for bad in ("l1\nl2\nl3\nl4",              # full-function re-emission
                    "a <- 1\nb <- 2",              # two statements
                    "x <- ",                       # broken R
                    "", "   "):
            ok, _ = v_mid_body_edit_line(bad, {})
            self.assertFalse(ok, bad)

    def test_shipped_spec_loads(self):
        s = load_case("mid_body_edit")
        self.assertIn("mid_body_edit", list_cases())
        self.assertEqual(s.corpus_source["selector"], "mid_body_sites")
        self.assertEqual(s.validator["name"], "mid_body_edit_line")
        self.assertEqual(s.row_check["name"], "mid_body_edit_site")
        self.assertEqual(len(s.corpus_source["params"]["kinds"]), 4)
        # templates never leak the ground truth
        for t in s.prompt_templates:
            self.assertNotIn("{code}", t)
            self.assertNotIn("{corpus_target}", t)

    def test_full_cycle_mock_backend(self):
        with tempfile.TemporaryDirectory() as td:
            spec_p = Path(td) / "mbe.json"
            spec_p.write_text(json.dumps(mid_body_spec_dict()))
            out = Path(td) / "mbe.jsonl"
            rc = generate_main(["--spec", str(spec_p), "--n", "6",
                                "--backend", "mock", "--out", str(out)])
            self.assertEqual(rc, 0)
            rows = [json.loads(l) for l in
                    out.read_text().splitlines() if l.strip()]
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(row["case"], "toy_mid_body")
                self.assertEqual(row["region_old"], [""])
                self.assertEqual(row["cursor_idx"], 0)
                self.assertEqual(len(row["region_new"]), 1)
                self.assertTrue(row["suffix"])
                self.assertIn(row["mutation_kind"],
                              ("arg_edit", "na_rm_insert", "rename_once",
                               "insert_line"))
                self.assertIn(row["suffix"][0], row["full_prompt"])
                self.assertIn(row["suffix"][-2] if len(row["suffix"]) > 1
                              else row["suffix"][-1],
                              row["full_prompt"])            # scope pin
                self.assertNotIn(row["region_new"][0],
                                 row["full_prompt"])            # no GT leak
                ok, reason = check_row(row, {"name": "mid_body_edit_site",
                                             "params": {}})
                self.assertTrue(ok, reason)
            stats = json.loads(
                Path(str(out) + ".stats.json").read_text())
            self.assertEqual(stats["counts"]["accepted"], 6)
            self.assertEqual(
                len({r["content_hash"] for r in rows}), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
