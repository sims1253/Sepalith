import json
import unittest

from sepalith.protocol import (
    EditContext, EvidenceRecord, SourceRange, render_context, serialize_snapshot,
)


class ProtocolTests(unittest.TestCase):
    def context(self, **changes):
        record = dict(path="R/fit.R", prefix=["fit <- function(x) {"],
                      region_old=["  mean(x)", "}"], suffix=["", "fit(values)"], cursor_idx=0)
        record.update(changes)
        return EditContext.from_legacy(record)

    def test_literal_legacy_cursor_on_middle_line(self):
        context = self.context(region_old=["  x <- na.omit(x)", "  mean(x)", "}"], cursor_idx=1)
        expected = (
            "<[fim-suffix]>\n\nfit(values)\n<[fim-prefix]><filename>edit_history\n"
            "<filename>R/fit.R\nfit <- function(x) {\n<<<<<<< CURRENT\n"
            "  x <- na.omit(x)\n  mean(x)<|user_cursor|>\n}\n=======\n<[fim-middle]>"
        )
        self.assertEqual(render_context(context).encode(), expected.encode())

    def test_literal_empty_edit(self):
        context = self.context(path="empty.R", prefix=[], region_old=[], suffix=[], cursor_idx=-1)
        self.assertEqual(render_context(context),
                         "<[fim-suffix]>\n<[fim-prefix]><filename>edit_history\n"
                         "<filename>empty.R\n<<<<<<< CURRENT\n=======\n<[fim-middle]>")

    def test_literal_suffix_history_and_newlines(self):
        context = self.context(prefix=["# café  ", ""], region_old=["x <- 1  ", ""],
                               suffix=["", "# end  ", ""], cursor_idx=1,
                               event_diff="```diff\nUser edited fit.R\n\n-x <- 0\n+x <- 1\n```\n")
        expected = (
            "<[fim-suffix]>\n\n# end  \n\n<[fim-prefix]><filename>edit_history\n"
            "-x <- 0\n+x <- 1\n\n\n<filename>R/fit.R\n# café  \n\n"
            "<<<<<<< CURRENT\nx <- 1  \n<|user_cursor|>\n=======\n<[fim-middle]>"
        )
        self.assertEqual(render_context(context), expected)

    def test_explicit_midline_column(self):
        context = self.context(region_old=["mean(x)"], cursor_idx=0, cursor_column=5)
        self.assertIn("mean(<|user_cursor|>x)", render_context(context))

    def test_unknown_metadata_and_evidence_round_trip_without_prompt_change(self):
        baseline = self.context()
        record = baseline.to_dict()
        record["future_field"] = {"nested": [True, None, "kept"]}
        record["metadata"] = {"project": "science", "tool_result": "DO NOT RENDER"}
        record["evidence"] = [{
            "source_kind": "lsp", "content": "mean(x, na.rm = FALSE)",
            "workspace_revision": "git:abc123+dirty:observed-7",
            "content_identity": "sha256:caller-supplied", "symbol": "mean",
            "source_range": {"start": 0, "end": 27}, "confidence": None,
            "future_evidence_field": {"observed": True},
        }]
        context = EditContext.from_dict(record)
        self.assertEqual(render_context(context), render_context(baseline))
        decoded = json.loads(serialize_snapshot(context))
        self.assertEqual(decoded["future_field"], record["future_field"])
        self.assertEqual(decoded["evidence"][0]["future_evidence_field"], {"observed": True})
        self.assertEqual(serialize_snapshot(EditContext.from_dict(decoded)), serialize_snapshot(context))

    def test_deterministic_key_order_and_significant_evidence_order(self):
        a = self.context(metadata={"z": 1, "a": 2})
        b = self.context(metadata={"a": 2, "z": 1})
        self.assertEqual(serialize_snapshot(a), serialize_snapshot(b))
        records = [EvidenceRecord("file", "a"), EvidenceRecord("file", "b")]
        a = self.context(evidence=[r.to_dict() for r in records])
        b = self.context(evidence=[r.to_dict() for r in reversed(records)])
        self.assertNotEqual(serialize_snapshot(a), serialize_snapshot(b))

    def test_absent_evidence_identity_is_not_fabricated(self):
        record = EvidenceRecord("diagnostic", "object x is missing").to_dict()
        self.assertIsNone(record["workspace_revision"])
        self.assertIsNone(record["content_identity"])
        self.assertIsNone(record["confidence"])

    def test_schema_and_shape_validation(self):
        for changes in ({"schema_version": "v99"}, {"cursor_idx": True},
                        {"prefix": "not an array"}, {"evidence": "not an array"},
                        {"cursor_column": 100}, {"metadata": {"bad": float("nan")}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                EditContext.from_dict({**self.context().to_dict(), **changes})
        with self.assertRaises(ValueError):
            EditContext.from_dict({"path": "R/x.R"})
        with self.assertRaises(ValueError):
            render_context(self.context(), renderer="future-format")
        with self.assertRaises(ValueError):
            SourceRange(4, 3)
        with self.assertRaises(ValueError):
            EvidenceRecord("file", "x", confidence=1.1)
        with self.assertRaises(ValueError):
            EvidenceRecord("file", "x", extra={"content": "shadow"})

    def test_budget_requires_tokenizer_and_uses_its_result(self):
        class SuppliedTokenizer:
            def __init__(self):
                self.seen = None

            def encode(self, text):
                self.seen = text
                return [11, 22, 33]

        context = self.context()
        tokenizer = SuppliedTokenizer()
        with self.assertRaisesRegex(ValueError, "supplied tokenizer"):
            render_context(context, max_tokens=100)
        result = render_context(context, tokenizer=tokenizer, max_tokens=3)
        self.assertEqual(tokenizer.seen, result)
        with self.assertRaisesRegex(ValueError, "3 tokens"):
            render_context(context, tokenizer=tokenizer, max_tokens=2)
        for limit in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                render_context(context, tokenizer=tokenizer, max_tokens=limit)


if __name__ == "__main__":
    unittest.main()
