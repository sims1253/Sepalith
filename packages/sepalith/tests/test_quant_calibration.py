"""C09 calibration builder: DEV exclusion, family balance, packing and rendering."""
import unittest
from typing import Any

from sepalith.quant import calibration as cal

CODE = [f"value_{i} <- compute_something(x_{i}, y = {i})" for i in range(12)]


def prompt(lines, region="  x <- old_value_here + another_term_long_enough(1)\n", path="R/a.R"):
    return ("Edit the current region.\n<filename>" + path + "\n" + "\n".join(lines) + "\n<<<<<<< CURRENT\n"
            + region.replace("x <-", "x <-<|user_cursor|>", 1) + "=======\n<[fim-middle]>\n")


def train_row(row_id, family="finish_block", text=None, body="  y <- a_new_value_long_enough_for_a_key(2)",
              operation="replace", ids=None):
    return {"id": row_id, "family": family, "prompt_text": text or prompt(CODE[:3] + [f"unique_{row_id} <- 1"] * 1),
            "target_body_text": body, "target_operation": operation, "input_ids": ids or [0, 5, 6, 1]}


class KeysTest(unittest.TestCase):
    def test_keys(self):
        keys = cal.near_duplicate_keys(prompt(CODE), "  y <- a_new_value_long_enough_for_a_key(2)", "replace")
        self.assertEqual({k[:2] for k in keys}, {"p:", "r:", "t:"})
        noop = cal.near_duplicate_keys(prompt(CODE), "[NO_EDIT]", "no_op")
        self.assertEqual({k[:2] for k in noop}, {"p:", "r:"})
        short = cal.near_duplicate_keys(prompt(CODE, region="}\n"), "}", "replace")
        self.assertEqual({k[:2] for k in short}, {"p:"})

    def test_whitespace_and_cursor_do_not_matter(self):
        a = cal.near_duplicate_keys(prompt(CODE), "x", "replace")
        b = cal.near_duplicate_keys(prompt(CODE).replace("<|user_cursor|>", "").replace(" <- ", "  <-   "), "x", "replace")
        self.assertEqual(a, b)

    def test_current_region(self):
        self.assertEqual(cal.current_region(prompt(CODE, region="a\nb\n")), "a\nb")
        self.assertEqual(cal.current_region("no region"), "")


class ExclusionTest(unittest.TestCase):
    def setUp(self):
        self.dev = cal.build_dev_index([("dev-1", prompt(CODE, path="R/dev.R"), "  z <- dev_target_value_that_is_long_enough(3)", "replace")])
        self.frequency = {}

    def reasons(self, row):
        return cal.exclusion_reasons(row, self.dev, self.frequency)

    def test_clean_row(self):
        row = train_row("t1", text=prompt(["other_thing <- 1", "something_else <- 2"], region="  q <- different_region_content_here_long(5)\n"))
        self.assertEqual(self.reasons(row), [])

    def test_id(self):
        row = train_row("dev-1", text=prompt(["other_thing <- 1"], region="  q <- different_region_content_here_long(5)\n"))
        self.assertEqual(self.reasons(row), ["dev_id"])

    def test_prompt_and_region_hash(self):
        row = train_row("t2", text=prompt(CODE, path="R/dev.R").replace(" <- ", " <-  "))
        self.assertIn("dev_prompt_hash", self.reasons(row))
        self.assertIn("dev_region_hash", self.reasons(row))

    def test_target_hash(self):
        row = train_row("t3", text=prompt(["other_thing <- 1"], region="  q <- different_region_content_here_long(5)\n"),
                        body="  z <- dev_target_value_that_is_long_enough(3)")
        self.assertEqual(self.reasons(row), ["dev_target_hash"])

    def test_shingle_containment(self):
        row = train_row("t4", text=prompt(CODE[:11] + ["extra_line_in_train <- TRUE"], path="R/moved.R",
                                          region="  q <- different_region_content_here_long(5)\n"))
        self.assertEqual(self.reasons(row), ["dev_shingle_containment"])
        # Boilerplate shingles (frequent in TRAIN) do not count.
        self.frequency = {s: cal.SHINGLE_MAX_DOCUMENT_FREQUENCY + 1 for s in cal.shingles(row["prompt_text"])}
        self.assertEqual(self.reasons(row), [])


class SelectionTest(unittest.TestCase):
    def rows(self):
        rows = []
        for family, count in (("a", 30), ("b", 5), ("c", 30)):
            rows += [{"id": f"{family}{i:02d}", "family": family, "input_ids": [0] + [7] * 9 + [1]} for i in range(count)]
        return rows

    def test_family_balanced_and_deterministic(self):
        picked = cal.family_balanced(self.rows(), 150, 3407)
        by_family = {f: sum(1 for r in picked if r["family"] == f) for f in "abc"}
        self.assertEqual(by_family, {"a": 5, "b": 5, "c": 5})
        self.assertEqual([r["id"] for r in picked], [r["id"] for r in cal.family_balanced(self.rows(), 150, 3407)])
        self.assertNotEqual([r["id"] for r in picked], [r["id"] for r in cal.family_balanced(self.rows(), 150, 1)])


class PackingTest(unittest.TestCase):
    def setUp(self):
        self.rows: list[dict[str, Any]] = [{"id": f"r{i}", "input_ids": [0] + [10 + i] * n + [1]} for i, n in enumerate((5, 3, 6, 2))]
        self.fillers: list[dict[str, Any]] = [{"id": "bad", "input_ids": [0, 99, 98, 97, 96, 95, 94, 93, 92, 1]},
                        {"id": "f1", "input_ids": [0] + [50] * 20 + [1]},
                        {"id": "f2", "input_ids": [0] + [60] * 20 + [1]}]
        self.by_id = {r["id"]: r for r in self.rows + self.fillers}

    def test_chunks_are_exact_and_start_at_bos(self):
        chunks, used = cal.pack_chunks(self.rows, 12, iter(self.fillers), lambda ids: 99 not in ids)
        self.assertNotIn("bad", used)
        text, ids = cal.render_calibration(chunks, self.by_id, lambda ids: "".join(f"[{t}]" for t in ids))
        self.assertEqual(len(ids), 12 * len(chunks))
        for start in range(0, len(ids), 12):
            self.assertEqual(ids[start], 0)
            self.assertEqual(ids[start + 11], 1)  # every chunk ends with a whole row
        self.assertFalse(text.startswith("<s>"))
        self.assertTrue(text.endswith("</s>"))
        self.assertEqual(text.count("<s>"), sum(len(c) for c in chunks) - 1)
        whole = sorted(p.row_id for c in chunks for p in c if p.whole)
        self.assertEqual(whole, ["r0", "r1", "r2", "r3"])

    def test_row_longer_than_context(self):
        with self.assertRaises(ValueError):
            cal.pack_chunks(self.rows, 5, iter(self.fillers), lambda ids: True)

    def test_kld_windows(self):
        rows: list[dict[str, Any]] = [{"id": f"k{i}", "family": "ab"[i % 2], "input_ids": [0] + list(range(100 + i, 110 + i)) + [1]} for i in range(6)]
        windows = cal.kld_windows(rows, 8, 4, 3407, lambda ids: True)
        self.assertEqual(len(windows), 4)
        self.assertEqual({w.end - w.start for w in windows}, {7})
        by_id = {r["id"]: r for r in rows}
        text, ids = cal.render_kld(windows, by_id, lambda ids: "".join(f"[{t}]" for t in ids))
        self.assertEqual(len(ids), 4 * 8)
        self.assertEqual([ids[i] for i in range(0, 32, 8)], [0, 8, 8, 8])
        self.assertEqual(text.count("<think>"), 3)
        for window in windows:  # each window ends right before the row's EOS
            self.assertEqual(window.end, len(by_id[window.row_id]["input_ids"]) - 1)
        with self.assertRaises(ValueError):
            cal.kld_windows(rows, 8, 7, 3407, lambda ids: True)

    def test_first_mismatch(self):
        self.assertIsNone(cal.first_mismatch([1, 2], [1, 2]))
        self.assertEqual(cal.first_mismatch([1, 3], [1, 2]), 1)
        self.assertEqual(cal.first_mismatch([1], [1, 2]), 1)


if __name__ == "__main__":
    unittest.main()
