"""Synthetic-only tests of the final panel v1 constructor.

No test opens /mnt data, DAT-02, a DEV panel or any final group. A temporary
corpus of invented packages stands in for the normalized CRAN tree. The
end-to-end test labels one invented group `final_candidate_group` so that it
can drive the v2 launcher's own load, semantic-receipt, audit and construct
steps over the emitted artifacts.

    cd docs/campaign/work/final-panel-v1 && python -m unittest -v test_final_panel
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import io
import json
import os
from pathlib import Path
import random
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "final-protocol-v2")]

import final_panel as F  # noqa: E402
import protocol as v2  # noqa: E402
from admission_guard import AdmissionOutputError  # noqa: E402

RENAME_SOURCE = """f <- function(x) {
  old_name <- x + 1
  y <- old_name * 2
  z <- old_name + y
  old_name
}

g <- function(data) {
  total <- sum(data$value, na.rm = TRUE)
  data |>
    head()
}
"""

PIPE_SOURCE = """h <- function(df) {
  a <- df %>% filter(x > 1)
  b <- df %>% select(y)
  c <- df %>% arrange(z)
  list(a, b, c)
}
"""

FINISH_SOURCE = """#' Add values.
#' @param x Numeric values.
#' @return Numeric values.
add_values <- function(x) {
  first <- x + 1
  second <- first * 2
  third <- second - 3
  fourth <- third / 4
  fifth <- fourth + first
  c(first, second, third, fourth, fifth)
}
"""


class TextTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False, split_special_tokens: bool = True) -> list[int]:
        return [ord(char) + 512 for char in text]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = False) -> str:
        return "".join(chr(token - 512) for token in ids)


def recipe(tmp: Path, quotas: dict[str, int] | None = None) -> F.Json:
    base = F.load_recipe()
    base["inputs"] = {**base["inputs"], "corpus_root": str(tmp / "normalized"),
                      "tarball_root": str(tmp / "tarballs")}
    if quotas:
        base["quotas"] = quotas
    return base


def make_package(tmp: Path, name: str, files: dict[str, str], *, raw: dict[str, str] | None = None) -> None:
    root = tmp / "normalized" / name / "1.0.0" / name
    (root / "R").mkdir(parents=True)
    (root / "DESCRIPTION").write_text(f"Package: {name}\nVersion: 1.0.0\nLicense: MIT\n")
    for rel, text in files.items():
        (root / "R" / rel).write_text(text)
    (tmp / "tarballs").mkdir(exist_ok=True)
    with tarfile.open(tmp / "tarballs" / f"{name}_1.0.0.tar.gz", "w:gz") as archive:
        for rel, text in (raw or files).items():
            data = text.encode()
            info = tarfile.TarInfo(f"{name}/R/{rel}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def pin_holdback(rec: F.Json, reg: F.Json, build_groups: int) -> F.Json:
    """Point the recipe's holdback pin at the split of an invented registry."""
    doc = F.holdback_split(reg, F.FINAL_SPLIT, build_groups, rec["inputs"]["dat02"]["sha256"])
    rec["holdback"] = {**rec["holdback"], "build_groups": build_groups,
                       "final_split_sha256": F.sha256_bytes(F.holdback_bytes(doc))}
    return rec


def issue_authorization(tmp: Path, out: Path, marker: Path, name: str = "auth.json",
                        split: str = F.FINAL_SPLIT) -> F.Authorization:
    """A synthetic user build authorization for this exact constructor and recipe."""
    receipt = {"schema": F.SCHEMA + ".build-authorization", "status": "user-approved", "split": split,
               **({"rehearsal": True} if split != F.FINAL_SPLIT else {}),
               "weights_and_harness_frozen": True, "weights_sha256": "a" * 64, "harness_sha256": "b" * 64,
               "memberships_sha256": "c" * 64, "freeze_statement_sha256": "d" * 64, "out_dir": str(out),
               "attempt_marker": str(marker), "constructor_sha256": F.sha256_file(Path(F.__file__)),
               "recipe_sha256": F.sha256_file(F.RECIPE_PATH)}
    path = tmp / name
    path.write_bytes(json.dumps(receipt).encode())
    return F.authorize(path, F.sha256_file(path))


def memberships() -> F.Json:
    empty = {key: [] for key in ("row_ids", "group_ids", "split_identities", "function_ids")}
    return {stage: {**empty, "package_ids": ["unrelated"],
                    "function_ids": ["0" * 16] if stage.startswith("exec_") else []}
            for stage in v2.STAGES}


def registry(groups: list[tuple[str, str, str]]) -> F.Json:
    return {"groups": [{"group_id": g, "split": split, "flags": [], "identity_forms": [f"pkg:{pkg.lower()}"]}
                       for g, split, pkg in groups]}


class NoopRule(unittest.TestCase):
    def test_sites_skip_comments_and_strings(self) -> None:
        lines = [
            "x <- mean(v, na.rm = TRUE)",       # 0 na.rm
            "# y <- sum(v, na.rm = TRUE)",      # comment
            'paste0("sum(", v, ",na.rm=TRUE)")',  # only in a string
            "  df |>",                           # 3 trailing native pipe
            "  df |> head()",                    # pipe not at end
            "z <- max(v, na.rm=TRUE) |>",        # 5 na.rm wins
            "x <- 1 # mention na.rm = TRUE",     # inline comment only
            "x <- 2 # example |>",               # inline comment only
            "y <- mean(v, xna.rm = TRUE)",       # not the na.rm token
        ]
        self.assertEqual(F.noop_sites(lines), [(0, "already_applied_na_rm_true"),
                                               (3, "already_applied_native_pipe_operator"),
                                               (5, "already_applied_na_rm_true")])

    def test_noop_result_is_unchanged_region_with_cursor_at_end(self) -> None:
        lines, eol = F._lines("a <- 1\nb <- sum(x, na.rm = TRUE)\n")
        result = F.noop_result("R/a.R", "file:///sepalith/dev/r/R/a.R", lines, eol, 1)
        context, _ = F.selected_context(result)
        self.assertEqual(result["operation"], "no_op")
        self.assertEqual(result["target_body"], list(context.region_old))
        self.assertEqual(context.cursor.code_point_column, len(lines[1]))
        self.assertEqual(context.history, ())


class Converter(unittest.TestCase):
    def test_rename_example_applies_event_and_finds_region(self) -> None:
        m = F.Modules.load()
        bundle = m.scenarios.Bundle("pkgA", "R/f.R", RENAME_SOURCE.encode())
        examples = m.scenarios.extract_rename(bundle, random.Random(1), cap=4)
        self.assertTrue(examples)
        for example in examples:
            result = F.convert_example(example, RENAME_SOURCE, "file:///sepalith/x/R/f.R", "e" * 32)
            context, _ = F.selected_context(result)
            self.assertEqual(list(context.region_old), example["region_old"])
            (event,) = context.history
            self.assertNotEqual(event.old_text, event.new_text)
            document = result["selection_source"]["document_text"]
            self.assertIn(event.new_text, document)

    def test_event_that_does_not_apply_is_rejected(self) -> None:
        example = {"path": "R/f.R", "prefix": [], "region_old": ["y"], "region_new": ["z"], "cursor_idx": 0,
                   "event_diff": 'User edited "R/f.R":\n\n```diff\n@@ -1 +1 @@\n-nope\n+yes\n```'}
        with self.assertRaises(F.PanelError):
            F.convert_example(example, "x\ny\n", "u", "e")


class Rules(unittest.TestCase):
    def test_parse_gate_rejects_unparseable_post_edit(self) -> None:
        m = F.Modules.load()
        lines, eol = F._lines("f <- function(x) {\n  x + 1\n}\n")
        result = F.noop_result("R/a.R", "u", lines, eol, 1)
        self.assertIsNone(F.parse_gate(m.scenarios.parser, result))
        result["target_body"] = ["  x + ("]
        self.assertEqual(F.parse_gate(m.scenarios.parser, result), "full_post_edit_R_parse_error")

    def test_version_pick_is_highest_and_order_free(self) -> None:
        m = F.Modules.load()
        with TemporaryDirectory() as tmp:
            for name in ("1.10.0", "1.9.2", "1.2-0"):
                (Path(tmp) / name).mkdir()
            self.assertEqual(F.pick_version(Path(tmp), m.astfim).name, "1.10.0")

    def test_closure_refuses_a_module_from_another_checkout(self) -> None:
        F.Modules.load()
        F.verify_closure(F.load_recipe())
        original = sys.modules["scenarios"].__file__
        try:
            sys.modules["scenarios"].__file__ = "/elsewhere/experiments/synthetic-data/scenarios.py"
            with self.assertRaisesRegex(F.PanelError, "constructor_module_from_another_checkout"):
                F.verify_closure(F.load_recipe())
        finally:
            sys.modules["scenarios"].__file__ = original

    def test_mixed_eol_is_detected(self) -> None:
        self.assertTrue(F.mixed_or_lone_cr("a\r\nb\n"))
        self.assertTrue(F.mixed_or_lone_cr("a\rb"))
        self.assertFalse(F.mixed_or_lone_cr("a\r\nb\r\n"))


class Allocation(unittest.TestCase):
    def test_group_cap_and_shortfall_move(self) -> None:
        cands = []
        for g in range(10):
            for f, n in (("finish_block", 3), ("no_op", 1), ("pipe_rewrite", 1)):
                cands += [{"row_id": f"{f}-{g}-{i}", "family": f, "group_id": f"g{g}"} for i in range(n)]
        quotas = {"finish_block": 5, "no_op": 5, "pipe_rewrite": 8, "format_propagation": 2,
                  "rename_propagation": 0, "na_rm_propagation": 0}
        chosen, info = F.allocate(cands, quotas, 3)
        per_group = Counter(c["group_id"] for c in chosen)
        self.assertLessEqual(max(per_group.values()), 3)
        # pipe can give 8 of 8? supply is 10, format has none: 2 slots move to finish.
        self.assertEqual(info["moved_to_finish_block"], 2)
        self.assertEqual(info["selected_by_family"]["format_propagation"], 0)
        self.assertEqual(info["selected_by_family"]["finish_block"], 7)
        for family, count in info["pass1_by_family"].items():
            self.assertGreaterEqual(info["selected_by_family"][family], count)
        self.assertEqual(chosen, F.allocate(list(reversed(cands)), quotas, 3)[0])

    def test_noop_shortfall_is_not_moved(self) -> None:
        cands = [{"row_id": f"f{i}", "family": "finish_block", "group_id": f"g{i}"} for i in range(20)]
        quotas = dict.fromkeys(F.FAMILIES, 0) | {"finish_block": 5, "no_op": 5}
        _, info = F.allocate(cands, quotas, 3)
        self.assertEqual(info["moved_to_finish_block"], 0)
        self.assertEqual(info["shortfall"], {"no_op": 5})


class Gate(unittest.TestCase):
    def test_final_split_requires_authorization_and_fixed_recipe(self) -> None:
        with TemporaryDirectory() as tmp:
            kwargs = {"split": F.FINAL_SPLIT, "registry": {"groups": []}, "recipe": recipe(Path(tmp)),
                      "tokenizer": TextTokenizer(), "workdir": Path(tmp), "cpt": frozenset(),
                      "dev_prompts": frozenset()}
            with self.assertRaisesRegex(F.PanelError, "final_build_requires_build_final"):
                F.build_panel(**kwargs)
            root = Path(tmp)
            inside = issue_authorization(root, root / "out", root / "out.work" / "attempt.json")
            with self.assertRaisesRegex(F.PanelError, "attempt_marker_inside_a_cleanup_tree"):
                F.claim_attempt(inside, root / "out")
            (root / "x").mkdir()
            dotted = issue_authorization(root, root / "out", root / "x" / ".." / "out.work" / "attempt.json",
                                         name="dotted.json")
            with self.assertRaisesRegex(F.PanelError, "attempt_marker_inside_a_cleanup_tree"):
                F.claim_attempt(dotted, root / "out")

    def test_build_final_checks_models_before_anything_else(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            auth = issue_authorization(root, root / "out", root / "attempt.json")
            (root / "models.json").write_text('{"other": 1}')
            with self.assertRaisesRegex(F.PanelError, "models_differ_from_authorization"):
                F.build_final(auth, tokenizer_path=root, memberships_path=root / "m.json",
                              models_path=root / "models.json", freeze_statement_path=root / "f.txt")
            self.assertFalse((root / "attempt.json").exists())
            self.assertFalse((root / "out.work").exists())

    def test_stopped_panel_is_never_published(self) -> None:
        panel = F.Panel(F.FINAL_SPLIT, [], {}, {}, {}, {"status": "stopped_below_floor"})
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(F.PanelError, "stopped_panel_is_not_published"):
            F.write_panel(panel, Path(tmp) / "out", recipe_sha256="r" * 64, inputs={})

    def test_authorize_binds_constructor_and_recipe(self) -> None:
        with TemporaryDirectory() as tmp:
            receipt = {"schema": F.SCHEMA + ".build-authorization", "status": "user-approved",
                       "split": F.FINAL_SPLIT, "weights_and_harness_frozen": True,
                       "weights_sha256": "a" * 64, "harness_sha256": "b" * 64, "memberships_sha256": "c" * 64,
                       "freeze_statement_sha256": "d" * 64, "out_dir": "/x/panel", "attempt_marker": "/x/attempt",
                       "constructor_sha256": F.sha256_file(Path(F.__file__)),
                       "recipe_sha256": F.sha256_file(F.RECIPE_PATH)}
            path = Path(tmp) / "auth.json"
            path.write_bytes(json.dumps(receipt).encode())
            digest = F.sha256_file(path)
            self.assertEqual(F.authorize(path, digest).receipt_sha256, digest)
            with self.assertRaisesRegex(F.PanelError, "authorization_hash_mismatch"):
                F.authorize(path, "1" * 64)
            path.unlink()
            path.write_bytes(json.dumps({**receipt, "recipe_sha256": "2" * 64}).encode())
            with self.assertRaisesRegex(F.PanelError, "authorization_not_for_this_build"):
                F.authorize(path, F.sha256_file(path))

    def test_cli_refuses_final_without_authorization(self) -> None:
        with self.assertRaises(SystemExit):
            F.main(["--split", F.FINAL_SPLIT, "--out", "/nonexistent/x", "--tokenizer", "/nonexistent"])

    def test_symlinked_description_is_refused_before_it_is_read(self) -> None:
        m = F.Modules.load()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_package(root, "pkgA", {"f.R": RENAME_SOURCE})
            (root / "elsewhere").write_text("License: MIT\n")
            desc = root / "normalized/pkgA/1.0.0/pkgA/DESCRIPTION"
            desc.unlink()
            desc.symlink_to(root / "elsewhere")
            opened = []
            original = F._has_license
            F._has_license = lambda path: opened.append(path) or original(path)  # type: ignore[assignment]
            try:
                parents, reasons = F.parents_for(registry([("g-a", "dev_group", "pkgA")])["groups"],
                                                 root / "normalized", m.astfim)
            finally:
                F._has_license = original
            self.assertEqual((parents, reasons["parent_tree_refused"], opened), ([], 1, []))

    def test_symlinked_tree_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_package(root, "pkgA", {"f.R": RENAME_SOURCE})
            (root / "normalized/pkgA/1.0.0/pkgA/R/link.R").symlink_to(root / "normalized/pkgA/1.0.0/pkgA/R/f.R")
            with self.assertRaisesRegex(F.PanelError, "symlink_in_source_tree"):
                F.checked_tree(root / "normalized/pkgA/1.0.0", root / "normalized")


class EndToEnd(unittest.TestCase):
    """Build on an invented corpus, then drive the v2 launcher's own checks over the outputs."""

    def build(self, tmp: Path, split: str) -> tuple[F.Panel, F.Json]:
        make_package(tmp, "pkgA", {"f.R": RENAME_SOURCE, "fin.R": FINISH_SOURCE})
        make_package(tmp, "pkgB", {"h.R": PIPE_SOURCE, "fin.R": FINISH_SOURCE.replace("add_values", "plus")},
                     raw={"h.R": PIPE_SOURCE.replace("df %>% select(y)", "df%>%select(y)")
                          .replace("df %>% arrange(z)", "df%>%arrange(z)"),
                          "fin.R": FINISH_SOURCE.replace("add_values", "plus")})
        reg = registry([("g-a", split, "pkgA"), ("g-b", split, "pkgB"), ("g-other", "train_group", "pkgB")])
        quotas = {"finish_block": 2, "no_op": 2, "pipe_rewrite": 1, "format_propagation": 1,
                  "rename_propagation": 1, "na_rm_propagation": 1}
        rec = pin_holdback(recipe(tmp, quotas), reg, 2) if split == F.FINAL_SPLIT else recipe(tmp, quotas)
        rec["floor"] = {"cases": 1, "families": dict.fromkeys(F.FAMILIES, 0)}
        work = tmp / "work"
        work.mkdir()
        auth = issue_authorization(tmp, tmp / "out", tmp / "attempt.json") if split == F.FINAL_SPLIT else None
        if split == F.FINAL_SPLIT:
            # The fixed-recipe rule is for real final builds; the synthetic corpus needs tiny quotas.
            panel = self._final_build(rec, reg, work, auth)
        else:
            panel = F.build_panel(split=split, registry=reg, recipe=rec, tokenizer=TextTokenizer(), workdir=work,
                                  cpt=frozenset(), dev_prompts=frozenset(), quotas=quotas)
        receipt = F.write_panel(panel, tmp / "out", recipe_sha256="r" * 64, inputs={})
        return panel, receipt

    def _final_build(self, rec: F.Json, reg: F.Json, work: Path, auth: F.Authorization) -> F.Panel:
        """build_final() after its pin checks: claim the attempt, then run the core on the invented corpus."""
        F.claim_attempt(auth, Path(auth.receipt["out_dir"]))
        return F._construct(split=F.FINAL_SPLIT, registry=reg, recipe=rec, tokenizer=TextTokenizer(), workdir=work,
                            cpt=frozenset({"f" * 64}), dev_prompts=frozenset(), group_ids=None, quotas=None,
                            modules=F.Modules.load(), memberships=memberships(), authorization=auth)

    def test_rehearsal_build_is_deterministic(self) -> None:
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            first, r1 = self.build(Path(a), "dev_group")
            second, r2 = self.build(Path(b), "dev_group")
            self.assertTrue(first.cases)
            self.assertEqual(first.payloads, second.payloads)
            self.assertEqual([c["row_id"] for c in first.cases], [c["row_id"] for c in second.cases])
            self.assertEqual(r1["counts"], r2["counts"])
            self.assertEqual({c["group_id"] for c in first.cases} - {"g-a", "g-b"}, set())

    def test_final_build_claims_its_attempt_once(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, F.FINAL_SPLIT)
            marker = json.loads((root / "attempt.json").read_bytes())
            self.assertEqual(marker["out_dir"], str(root / "out"))
            auth = issue_authorization(root, root / "out2", root / "attempt.json", name="auth2.json")
            calls = []
            original = F.parents_for
            F.parents_for = lambda *a, **k: calls.append(1) or original(*a, **k)  # type: ignore[assignment]
            try:
                with self.assertRaises(AdmissionOutputError):
                    reg = registry([("g-a", F.FINAL_SPLIT, "pkgA")])
                    self._final_build(pin_holdback(recipe(root), reg, 1), reg, root, auth)
            finally:
                F.parents_for = original
            self.assertEqual(calls, [])  # refused before any parent or source was touched

    def test_holdback_groups_are_never_opened(self) -> None:
        """An authorized build reads only the build groups; no path of a holdback package is ever opened."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["pkgA", "pkgB", "pkgC", "pkgD", "pkgE", "pkgF"]
            for name in names:
                make_package(root, name, {"f.R": RENAME_SOURCE, "h.R": PIPE_SOURCE,
                                          "fin.R": FINISH_SOURCE.replace("add_values", "fn_" + name.lower())})
            reg = registry([(f"g-{name.lower()}", F.FINAL_SPLIT, name) for name in names])
            rec = pin_holdback(recipe(root), reg, 3)
            rec["floor"] = {"cases": 1, "families": dict.fromkeys(F.FAMILIES, 0)}
            split = F.holdback_split(reg, F.FINAL_SPLIT, 3, rec["inputs"]["dat02"]["sha256"])
            self.assertEqual((len(split["build_group_ids"]), len(split["holdback_group_ids"])), (3, 3))
            held = {name for name in names if f"g-{name.lower()}" in split["holdback_group_ids"]}
            forbidden = [str(root / "normalized" / name) for name in held]
            forbidden += [str(root / "tarballs" / f"{name}_") for name in held]
            touched: list[str] = []
            active = [True]

            def hook(event: str, args: tuple[object, ...]) -> None:
                if active[0] and event in {"open", "os.listdir", "os.scandir", "os.lstat", "os.stat"} and args:
                    path = args[0]
                    if isinstance(path, (str, bytes, Path)):
                        text = os.fsdecode(path)
                        if any(text.startswith(prefix) for prefix in forbidden):
                            touched.append(text)

            sys.addaudithook(hook)
            work = root / "work"
            work.mkdir()
            auth = issue_authorization(root, root / "out", root / "attempt.json")
            try:
                panel = self._final_build(rec, reg, work, auth)
            finally:
                active[0] = False
            self.assertEqual(touched, [])
            self.assertTrue(panel.cases)
            self.assertLessEqual({c["group_id"] for c in panel.cases}, set(split["build_group_ids"]))
            self.assertEqual(panel.receipt["counts"]["groups_in_split"], 3)
            self.assertEqual(panel.receipt["holdback_split_sha256"], rec["holdback"]["final_split_sha256"])

    def test_holdback_pin_is_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = registry([("g-a", F.FINAL_SPLIT, "pkgA"), ("g-b", F.FINAL_SPLIT, "pkgB")])
            rec = pin_holdback(recipe(root), reg, 1)
            rec["holdback"]["final_split_sha256"] = "0" * 64
            auth = issue_authorization(root, root / "out", root / "attempt.json")
            with self.assertRaisesRegex(F.PanelError, "holdback_split_differs_from_pin"):
                F._construct(split=F.FINAL_SPLIT, registry=reg, recipe=rec, tokenizer=TextTokenizer(),
                             workdir=root, cpt=frozenset(), dev_prompts=frozenset(), group_ids=None, quotas=None,
                             modules=F.Modules.load(), memberships=None, authorization=auth)
            with self.assertRaisesRegex(F.PanelError, "final_build_requires_build_final"):
                F._construct(split=F.FINAL_SPLIT, registry=reg, recipe=rec, tokenizer=TextTokenizer(),
                             workdir=root, cpt=frozenset(), dev_prompts=frozenset(), group_ids=None, quotas=None,
                             modules=F.Modules.load(), memberships=None, authorization=None)

    def test_committed_holdback_split_matches_the_recipe_pin(self) -> None:
        rec = F.load_recipe()
        data = (F.ROOT / rec["holdback"]["split_file"]).read_bytes()
        self.assertEqual(F.sha256_bytes(data), rec["holdback"]["final_split_sha256"])
        doc = json.loads(data)
        self.assertEqual(data, F.holdback_bytes(doc))
        self.assertEqual((doc["split"], doc["eligible_groups"], len(doc["build_group_ids"]),
                          len(doc["holdback_group_ids"])), (F.FINAL_SPLIT, 193, 105, 88))
        self.assertEqual(doc["dat02_sha256"], rec["inputs"]["dat02"]["sha256"])
        self.assertFalse(set(doc["build_group_ids"]) & set(doc["holdback_group_ids"]))
        ordered = F.seeded_group_order(doc["build_group_ids"] + doc["holdback_group_ids"])
        self.assertEqual(sorted(ordered[:105]), doc["build_group_ids"])

    def test_dev_rehearsal_authorization_is_accepted_and_final_flag_is_strict(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            auth = issue_authorization(root, root / "out", root / "attempt.json", split="dev_group")
            self.assertEqual(auth.receipt["split"], "dev_group")
            receipt = json.loads((root / "auth.json").read_bytes())
            for bad in ({"rehearsal": False}, {"split": "train_group"}):
                path = root / "bad.json"
                path.unlink(missing_ok=True)
                path.write_bytes(json.dumps({**receipt, **bad}).encode())
                with self.assertRaisesRegex(F.PanelError, "authorization_not_for_this_build"):
                    F.authorize(path, F.sha256_file(path))
            path = root / "count.json"
            path.write_bytes(json.dumps({**receipt, "rehearsal_build_groups": 169}).encode())
            self.assertEqual(F.authorize(path, F.sha256_file(path)).receipt["rehearsal_build_groups"], 169)
            final = {k: v for k, v in receipt.items() if k != "rehearsal"}
            path = root / "final-count.json"
            path.write_bytes(json.dumps({**final, "rehearsal_build_groups": 105}).encode())
            with self.assertRaisesRegex(F.PanelError, "authorization_not_for_this_build"):
                F.authorize(path, F.sha256_file(path))
            path = root / "final-with-rehearsal.json"
            path.write_bytes(json.dumps({**final, "split": F.FINAL_SPLIT, "rehearsal": True}).encode())
            with self.assertRaisesRegex(F.PanelError, "authorization_not_for_this_build"):
                F.authorize(path, F.sha256_file(path))

    def test_finish_function_ids_come_from_the_full_source(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, _ = self.build(root, "dev_group")
            finish = [c for c in panel.cases if c["family"] == "finish_block"]
            self.assertTrue(finish)
            for case in finish:
                self.assertEqual(len(case["function_ids"]), 1)
                self.assertIn(case["function_ids"][0], {F.function_id("pkgA", "R/fin.R", "add_values"),
                                                        F.function_id("pkgB", "R/fin.R", "plus")})

    def test_final_artifacts_pass_the_v2_launcher_checks(self) -> None:
        from admission_guard import AdmissionGuard

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, receipt = self.build(root, F.FINAL_SPLIT)
            out = root / "out"
            cases_doc = json.loads((out / "cases.json").read_bytes())
            self.assertEqual(cases_doc["split"], "final_locked_v1")
            families = Counter(c["family"] for c in cases_doc["cases"])
            self.assertGreaterEqual(len(families), 3, families)
            cases = v2.load_cases(cases_doc, receipt["outputs"]["case_count"])
            for case in cases_doc["cases"]:
                self.assertEqual(set(case), {"row_id", "split_identity", "package_id", "repository_id",
                                             "group_id", "family", "source_path", "source_sha256",
                                             "context_sha256", "target_sha256", "function_ids"})
            semantic = json.loads((out / "semantic-receipt.json").read_bytes())
            self.assertEqual(semantic["source_lock_sha256"], F.sha256_file(out / "cases.json"))
            freeze = {"status": "frozen", "weights_frozen": True, "harness_frozen": True,
                      "final_access_unlocked": True, "weights_frozen_at": "2026-11-23T00:00:00Z",
                      "harness_frozen_at": "2026-11-23T00:00:00Z", "weights_sha256": "a" * 64,
                      "harness_sha256": "b" * 64}
            guard = AdmissionGuard(
                allowed_source_roots=[Path(receipt["outputs"]["source_root"])], train_identities=(),
                dev_identities=(), freeze_receipt=freeze, semantic_receipt=semantic,
                expected_weights_sha256="a" * 64, expected_harness_sha256="b" * 64,
                expected_source_lock_sha256=F.sha256_file(out / "cases.json"),
                observed_at="2026-11-23T01:00:00Z")
            guard._validate_semantic_receipt([c.spec for c in cases])
            overlap, counts = v2.audit(cases, memberships())
            self.assertEqual(panel.receipt["overlap_audit"]["overlap_groups"], 0)
            dry = F.launcher_dry_run(out, TextTokenizer(), weights_sha256="a" * 64, harness_sha256="b" * 64,
                                     rehearsal=False)
            self.assertEqual(dry["constructed"], len(cases))
            self.assertTrue((out / "provenance.json").is_file())
            self.assertEqual(overlap, set())
            prepared, mismatches = v2.construct(cases, out / "payloads.json", guard, overlap, TextTokenizer())
            self.assertEqual(mismatches["cases"], 0)
            self.assertEqual(len(prepared), len(cases))
            # Staged sources are read-only and live under the plan's single source root.
            for case in cases_doc["cases"]:
                path = Path(case["source_path"])
                self.assertTrue(path.is_relative_to(out / "final-sources"))
                self.assertFalse(path.stat().st_mode & 0o222)

    def test_function_ids_follow_exec_pool_scheme(self) -> None:
        m = F.Modules.load()
        ids = F.function_ids("pkgA", "R/f.R", RENAME_SOURCE, [(2, 2)], m.scenarios.parser)
        import hashlib

        self.assertEqual(ids, [hashlib.sha256(b"pkgA\0R/f.R\0f").hexdigest()[:16]])
        self.assertEqual(F.function_ids("pkgA", "R/f.R", RENAME_SOURCE, [(6, 6)], m.scenarios.parser), [])


if __name__ == "__main__":
    unittest.main()
