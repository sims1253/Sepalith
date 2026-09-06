"""Offline correctness checks using invented fixtures, not private datasets."""

import json
import tempfile
import unittest
from pathlib import Path

from b8b import r_tokens, select, split_group
from common import Evidence, keyed, paired
from doc_prompts import constants
from doc_score import aggregate, structural
from inventory import freeze, sha
from loc1 import cache_array, metrics, ranking, rrf
from restore import restore
from uncertainty import cluster_ci


class InventoryTests(unittest.TestCase):
    def test_freeze_restore_and_reject_mutated_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "input"
            source.write_text("original")
            manifest = freeze(root, [source, root / "absent"], root / "snap")
            path = root / "manifest"
            path.write_text(json.dumps(manifest))
            e = Evidence(path, root / "snap")
            self.assertEqual(e.read("/input"), "original")
            self.assertFalse(e.record("/absent")["exists"])
            restore(path, root / "copy")
            source.write_text("changed")
            with self.assertRaises(ValueError):
                restore(path, root / "other")
            self.assertEqual(e.read("/input"), "original")
            (root / "snap" / sha(b"original")).write_text("bad")
            with self.assertRaises(ValueError):
                e.read("/input")

    def test_duplicate_missing_and_mismatched_joins(self):
        a = {"id": "x", "family": "f"}
        with self.assertRaises(ValueError):
            keyed([a, a])
        with self.assertRaises(ValueError):
            list(paired([a], []))
        with self.assertRaises(ValueError):
            list(paired([a], [dict(a, family="g")]))
        self.assertEqual(list(paired([a], [a])), [(a, a)])


class SelectorTests(unittest.TestCase):
    @staticmethod
    def prompt(old="x<-f(a,b)", event_old="y<-f(a,b)", event_new="y <- f(a, b)"):
        return (
            "<filename>edit_history\n-"
            + event_old
            + "\n+"
            + event_new
            + "\n<filename>x.R\n<<<<<<< CURRENT\n"
            + old
            + "\n=======\n<[fim-middle]>"
        )

    def test_preservation_selector_uses_only_prompt_and_outputs(self):
        p = self.prompt()
        self.assertEqual(select(p, "x <- f(a)", "x <- f(a, b)"), "treatment")
        self.assertEqual(select(p, "x <- f(a,b)", "x <- f(a, b)"), "control")
        self.assertEqual(select(p, "x <- f(a)", "x <- f(a)"), "control")

    def test_nonformat_event_keeps_control(self):
        self.assertEqual(
            select(self.prompt(event_new="y <- f(a, c)"), "x<-f(a)", "x<-f(a,b)"),
            "control",
        )

    def test_string_comment_and_identifier_tokens(self):
        self.assertNotEqual(r_tokens('"a b"'), r_tokens('"ab"'))
        self.assertNotEqual(r_tokens("x # a b"), r_tokens("x # ab"))
        self.assertNotEqual(r_tokens("ab"), r_tokens("a b"))
        self.assertNotEqual(r_tokens("a<-b"), r_tokens("a < -b"))
        self.assertEqual(r_tokens('f("a\\"b", x)'), r_tokens('f( "a\\"b" ,x)'))
        self.assertIsNone(r_tokens('"unterminated'))

    def test_malformed_prompt_keeps_control(self):
        self.assertEqual(select("-x<-1\n+x <- 1", "x<-2", "x<-1"), "control")

    def test_split_is_group_stable(self):
        self.assertEqual(split_group("package"), split_group("package"))
        self.assertIn(split_group("package"), ("development", "evaluation"))


class DocumentationTests(unittest.TestCase):
    def test_prompt_constants_are_literal_only(self):
        self.assertEqual(
            constants("PROMPT = 'frozen'", {"PROMPT"}), {"PROMPT": "frozen"}
        )
        with self.assertRaises(ValueError):
            constants("PROMPT = run_model()", {"PROMPT"})
        with self.assertRaises(ValueError):
            constants("OTHER = 'frozen'", {"PROMPT"})

    def packet(self, pred):
        return {
            "case": "example",
            "packet": "p",
            "raw": "raw",
            "old": ["#' @param x Input", "#' @return Output"],
            "prediction": pred,
        }

    def test_parameter_placement_is_not_canonical_adjacency(self):
        p = self.packet(
            [
                "#' @param verbose Prints progress",
                "#' @param x Input",
                "#' @return Output",
            ]
        )
        self.assertTrue(all(structural(p).values()))

    def test_preserve_old_tags_even_with_correct_new_line(self):
        s = structural(
            self.packet(["#' @param verbose Prints progress", "#' @return Output"])
        )
        self.assertTrue(s["coverage"])
        self.assertFalse(s["preservation"])

    def test_duplicate_invented_and_late_tags(self):
        p = self.packet(["#' @param verbose Prints progress"] * 2)
        self.assertFalse(structural(p)["coverage"])
        p = self.packet(
            ["#' @param verbose Prints progress", "#' @param extra Invented"]
        )
        self.assertFalse(structural(p)["coverage"])
        p = self.packet(["#' @return Output", "#' @param verbose Prints progress"])
        self.assertFalse(structural(p)["placement"])

    def test_formatting_rejects_code_and_markers(self):
        for line in ("x <- 1", "<|marker_1|>", "```"):
            self.assertFalse(structural(self.packet([line]))["formatting"])

    def test_unknown_is_not_a_pass_and_history_stays_intact(self):
        p = self.packet(
            [
                "#' @param verbose Prints progress",
                "#' @param x Input",
                "#' @return Output",
            ]
        )
        mapping = [
            {
                "arm": "a",
                "case": "example",
                "packet": "p",
                "exact": 0,
                "valid_pass": 0,
                "fail_kind": "transform",
                "source_sha256": "s",
                "evidence_status": "complete",
            }
        ]
        labels = [{"packet": "p", "target_semantic": "pass", "factual": "unknown"}]
        r = aggregate([p], mapping, labels)["rows"][0]
        self.assertTrue(r["target_usable"])
        self.assertFalse(r["prompt_supported_usable"])
        self.assertEqual(r["exact"], 0)
        mapping[0]["evidence_status"] = "truncated_raw"
        r = aggregate([p], mapping, labels)["rows"][0]
        self.assertIsNone(r["target_usable"])
        self.assertEqual(r["valid_pass"], 0)
        with self.assertRaises(ValueError):
            aggregate([p], mapping, [])


class RetrievalTests(unittest.TestCase):
    def test_hit_differs_from_multigold_coverage(self):
        m = metrics([0, 1, 2, 3], [0, 3], 2)
        self.assertEqual(m["hit@2"], 1)
        self.assertEqual(m["recall@2"], 0.5)
        self.assertEqual(m["all_gold@2"], 0)

    def test_duplicate_gold_chunks_do_not_double_count_function(self):
        m = metrics([0, 1, 2], [0, 1, 2], 1, [{0, 1}, {2}])
        self.assertEqual(m["recall@1"], 0.5)

    def test_invalid_cutoff_and_gold_groups(self):
        with self.assertRaises(ValueError):
            metrics([0, 1], [0], 0)
        with self.assertRaises(ValueError):
            metrics([0, 1], [0], 1, [{1}])
        with self.assertRaises(ValueError):
            rrf([0, 1], [1, 0], -1)

    def test_stable_ties_and_nonfinite(self):
        self.assertEqual(ranking([0, 0, 0]), [0, 1, 2])
        with self.assertRaises(ValueError):
            ranking([float("nan")])
        with self.assertRaises(ValueError):
            metrics([0, 0], [0], 1)
        with self.assertRaises(ValueError):
            metrics([0, 1], [2], 1)

    def test_fusion_keeps_budget_and_same_pool(self):
        r = rrf([0, 1, 2, 3], [3, 2, 1, 0])
        self.assertEqual(len(set(r[:2])), 2)
        self.assertEqual(set(r), {0, 1, 2, 3})
        self.assertEqual(rrf([0, 1], [1, 0]), [0, 1])
        with self.assertRaises(ValueError):
            rrf([0, 1], [0, 2])

    def test_missing_cache_never_invokes_model(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "inventory"
            p.write_text(json.dumps({"inputs": [{"path": "/x.npz", "exists": False}]}))
            with self.assertRaises(FileNotFoundError):
                cache_array(Evidence(p, d), "/x.npz")


class UncertaintyTests(unittest.TestCase):
    def test_invalid_bootstrap_count(self):
        with self.assertRaises(ValueError):
            cluster_ci([1, -1], ["a", "b"], repeats=0)

    def test_clusters_remain_intact(self):
        result = cluster_ci([1, 1, -1, -1], ["a", "a", "b", "b"], repeats=1000)
        self.assertEqual(result["ci"], [-1, 1])
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(
            result, cluster_ci([1, 1, -1, -1], ["a", "a", "b", "b"], repeats=1000)
        )
        self.assertIsNone(cluster_ci([1, 1], ["a", "a"])["ci"])
        with self.assertRaises(ValueError):
            cluster_ci([1], [])


if __name__ == "__main__":
    unittest.main()
