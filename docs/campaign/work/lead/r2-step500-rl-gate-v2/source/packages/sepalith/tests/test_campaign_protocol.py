import hashlib
import json
from pathlib import Path
import unittest

from sepalith.campaign_protocol import (
    BOS_ID,
    EOS_ID,
    GENERATION_BOUNDARY,
    NO_EDIT,
    ProtocolError,
    PromptContext,
    Cursor,
    Position,
    ReplacementRange,
    StaleContextError,
    TokenBoundaryError,
    build_training_row,
    codepoint_to_utf16_column,
    encode_prompt,
    make_application_plan,
    map_wire_to_document_eol,
    parse_output,
    render_prompt,
    serialize_target,
    utf16_to_codepoint_column,
    validate_training_row,
    valid_generation_tokens,
)


ZERO_SHA = "0" * 64
TOKENIZER_PATH = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain/tokenizer.json")
TOKENIZER_DIR = TOKENIZER_PATH.parent


class PinnedTokenizer:
    def __init__(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_DIR), local_files_only=True, use_fast=True
        )

    def encode(self, text, *, add_special_tokens=False, split_special_tokens=True):
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            split_special_tokens=split_special_tokens,
        )


class AdditiveTokenizer:
    """A boundary-stable test encoder with no protocol special IDs."""

    def encode(self, text, *, add_special_tokens=False, split_special_tokens=True):
        return [ord(char) + 512 for char in text]


class BoundaryChangingTokenizer(AdditiveTokenizer):
    def encode(self, text, *, add_special_tokens=False, split_special_tokens=True):
        ids = super().encode(
            text,
            add_special_tokens=add_special_tokens,
            split_special_tokens=split_special_tokens,
        )
        return [999] + ids if ">>>>>>> UPDATED" in text else ids


def context(**changes):
    value = {
        "schema_version": "sepalith.prompt.prm03.v1",
        "path": "R/café.R",
        "prefix": ["# keep  ", ""],
        "selected_references": [{"path": "R/defs.R", "content": "mean <- function(x) x  "}],
        "history": [],
        "diagnostics": [],
        "retrieval": [],
        "scope_mode": "pin+outline",
        "scope_lines": ["fit <- function(x) {"],
        "suffix_lines": ["}", ""],
        "region_old": ["  😀 <- 1  ", "  literal <- \"<|endoftext|>\"  "],
        "cursor": {"region_line_index": 0, "code_point_column": 4, "utf16_column": 5},
        "replacement_range": {
            "uri": "file:///workspace/R/café.R",
            "document_version": 7,
            "content_sha256": ZERO_SHA,
            "start": {"line": 10, "character": 2},
            "end": {"line": 11, "character": 30},
        },
        "document_eol": "crlf",
    }
    value.update(changes)
    return PromptContext.from_dict(value)


class CampaignProtocolTests(unittest.TestCase):
    def test_python_render_is_versioned_and_has_final_lf(self):
        current = context()
        prompt = render_prompt(current)
        self.assertTrue(prompt.endswith("\n"))
        self.assertEqual(prompt.split("\n")[-2:], [GENERATION_BOUNDARY, ""])
        self.assertIn("  😀 <|user_cursor|><- 1  ", prompt)
        self.assertIn("<filename>selected_references", prompt)
        self.assertIn("<|endoftext|>", prompt)
        self.assertNotIn("region_new", prompt)

    def test_unicode_cursor_round_trip_and_surrogate_rejection(self):
        line = "a😀b"
        self.assertEqual(codepoint_to_utf16_column(line, 2), 3)
        self.assertEqual(utf16_to_codepoint_column(line, 3), 2)
        with self.assertRaises(ProtocolError):
            utf16_to_codepoint_column(line, 2)
        with self.assertRaises(ProtocolError):
            context(cursor={"region_line_index": 0, "code_point_column": 3, "utf16_column": 3})

    def test_context_is_strict_and_has_no_future_target(self):
        record = context().to_dict()
        record["region_new"] = ["future"]
        with self.assertRaises(ProtocolError):
            PromptContext.from_dict(record)
        with self.assertRaises(ProtocolError):
            render_prompt(object())

    def test_context_geometry_and_line_entries_are_strict(self):
        bad_range = context().to_dict()
        bad_range["replacement_range"]["end"]["character"] = 29
        with self.assertRaises(ProtocolError):
            PromptContext.from_dict(bad_range)
        with self.assertRaises(ProtocolError):
            context(region_old=["a\nb"])

    def test_prediction_prompt_ids_are_target_free_and_manually_bos_prefixed(self):
        current = context()
        with_bos = encode_prompt(current, AdditiveTokenizer())
        without_bos = encode_prompt(current, AdditiveTokenizer(), include_bos=False)
        self.assertEqual(with_bos[0], BOS_ID)
        self.assertEqual(with_bos[1:], without_bos)
        self.assertNotIn(EOS_ID, without_bos)

    def test_outputs_cover_noop_full_copy_delete_insert_and_multiline(self):
        current = context()
        no_op = parse_output(f"{NO_EDIT}\n>>>>>>> UPDATED", current)
        self.assertEqual((no_op.status, no_op.operation), ("accepted", "no_op"))
        full_copy = parse_output("\n".join(current.region_old) + "\n>>>>>>> UPDATED", current)
        self.assertEqual(full_copy.operation, "no_op")
        deletion = parse_output(">>>>>>> UPDATED", current)
        self.assertEqual(deletion.operation, "delete")
        insertion = parse_output("a\nb\n>>>>>>> UPDATED", current)
        self.assertEqual((insertion.operation, insertion.body), ("replace", ("a", "b")))
        self.assertEqual(serialize_target("no_op", current.region_old), f"{NO_EDIT}\n>>>>>>> UPDATED")
        self.assertEqual(serialize_target("delete", []), ">>>>>>> UPDATED")
        self.assertEqual(make_application_plan(
            insertion, current, active_uri=current.replacement_range.uri,
            active_document_version=7, active_content_sha256=ZERO_SHA,
        ).mode, "workspace_edit_user_accept")

    def test_same_line_range_allows_multiline_insert_text(self):
        current = context(region_old=[], prefix=["# keep  "], scope_lines=[], suffix_lines=[], document_eol="lf",
                           cursor={"region_line_index": -1, "code_point_column": None, "utf16_column": None},
                           replacement_range={
                               "uri": "file:///workspace/R/café.R", "document_version": 7,
                               "content_sha256": ZERO_SHA,
                               "start": {"line": 10, "character": 2},
                               "end": {"line": 10, "character": 2},
                           })
        insertion = parse_output("a\nb\n>>>>>>> UPDATED", current)
        plan = make_application_plan(insertion, current, active_uri=current.replacement_range.uri,
                                     active_document_version=7, active_content_sha256=ZERO_SHA)
        self.assertEqual(plan.mode, "inline")
        self.assertEqual(plan.replacement_text, "a\nb")

    def test_crlf_mapping_preserves_spaces_and_blank_lines(self):
        payload = "α  \n\n  β  "
        self.assertEqual(map_wire_to_document_eol(payload, "lf"), payload)
        self.assertEqual(map_wire_to_document_eol(payload, "crlf"), "α  \r\n\r\n  β  ")
        with self.assertRaises(ProtocolError):
            map_wire_to_document_eol(payload, "mixed")

    def test_wrong_incomplete_and_ambiguous_outputs_are_distinct(self):
        current = context()
        self.assertEqual(parse_output("x", current).reason, "missing_exact_terminal")
        self.assertEqual(parse_output("x\n>>>>>>> UPDATED\njunk", current).reason, "content_after_terminal")
        self.assertEqual(parse_output("x\n>>>>>>> UPDATED\n>>>>>>> UPDATED", current).reason, "duplicate_exact_terminal")
        self.assertEqual(parse_output("x\n=======\n>>>>>>> UPDATED", current).reason, "reserved_full_body_line")
        quoted = parse_output('x <- ">>>>>>> UPDATED"\n>>>>>>> UPDATED', current)
        self.assertEqual(quoted.operation, "replace")

    def test_empty_range_empty_output_is_noop_without_source_oracle(self):
        current = context(region_old=[], prefix=[], scope_lines=[], suffix_lines=[],
                           cursor={"region_line_index": -1, "code_point_column": None, "utf16_column": None},
                           replacement_range={
                               "uri": "file:///workspace/R/café.R", "document_version": 7,
                               "content_sha256": ZERO_SHA,
                               "start": {"line": 10, "character": 2},
                               "end": {"line": 10, "character": 2},
                           })
        self.assertEqual(parse_output(">>>>>>> UPDATED", current).operation, "no_op")
        self.assertEqual(parse_output("\n>>>>>>> UPDATED", current).operation, "no_op")
        self.assertEqual(parse_output("\n>>>>>>> UPDATED", context()).operation, "delete")
        self.assertEqual(parse_output("\n\n>>>>>>> UPDATED", current).body, ("", ""))
        with self.assertRaises(ProtocolError):
            build_training_row(current, operation="replace", region_new=[""],
                               tokenizer=AdditiveTokenizer(), row_id="PRM04-EMPTY",
                               family="functional", package_id="synthetic-prm04")
        with self.assertRaises(ProtocolError):
            build_training_row(current, operation="replace", region_new=["a\nb"],
                               tokenizer=AdditiveTokenizer(), row_id="PRM04-EMBEDDED-LF",
                               family="functional", package_id="synthetic-prm04")

    def test_reserved_target_lines_are_rejected_before_tokenization(self):
        with self.assertRaises(ProtocolError):
            build_training_row(context(), operation="replace", region_new=["======="],
                               tokenizer=AdditiveTokenizer(), row_id="PRM04-RESERVED",
                               family="functional", package_id="synthetic-prm04")

    def test_stale_context_is_rejected_before_application(self):
        current = context()
        result = parse_output("a\n>>>>>>> UPDATED", current)
        with self.assertRaises(StaleContextError):
            make_application_plan(result, current, active_uri=current.replacement_range.uri,
                                  active_document_version=8, active_content_sha256=ZERO_SHA)

    def test_literal_eos_and_bos_are_admitted_with_split_special_mode(self):
        tokenizer = PinnedTokenizer()
        self.assertEqual(
            tokenizer.encode("literal <- '</s>'", add_special_tokens=False, split_special_tokens=True),
            [124133, 26427, 115170, 104, 42232],
        )
        self.assertEqual(
            tokenizer.encode("literal <- '<s>'", add_special_tokens=False, split_special_tokens=True),
            [124133, 26427, 42467, 104, 42232],
        )
        eos_context = context(region_old=["literal <- '</s>'"],
                              cursor={"region_line_index": 0, "code_point_column": 10, "utf16_column": 10},
                              replacement_range={
                                  "uri": "file:///workspace/R/café.R", "document_version": 7,
                                  "content_sha256": ZERO_SHA,
                                  "start": {"line": 10, "character": 0},
                                  "end": {"line": 10, "character": 17},
                              })
        self.assertEqual(
            parse_output("x <- '</s>'\n>>>>>>> UPDATED", eos_context).operation,
            "replace",
        )
        eos_row = build_training_row(
            eos_context,
            operation="replace",
            region_new=["x <- '</s>'"],
            tokenizer=tokenizer,
            row_id="PRM04-EOS",
            family="literal",
            package_id="synthetic-prm04",
        )
        self.assertEqual(eos_row["split"], "train")
        self.assertEqual(eos_row["input_ids"].count(EOS_ID), 1)
        self.assertEqual(eos_row["input_ids"].count(BOS_ID), 1)
        bos_context = context(region_old=["literal <- '<s>'"],
                              cursor={"region_line_index": 0, "code_point_column": 10, "utf16_column": 10},
                              replacement_range={
                                  "uri": "file:///workspace/R/café.R", "document_version": 7,
                                  "content_sha256": ZERO_SHA,
                                  "start": {"line": 10, "character": 0},
                                  "end": {"line": 10, "character": 16},
                              })
        bos_row = build_training_row(
            bos_context,
            operation="replace",
            region_new=["x <- '<s>'"],
            tokenizer=tokenizer,
            row_id="PRM04-BOS",
            family="literal",
            package_id="synthetic-prm04",
        )
        self.assertEqual(bos_row["input_ids"].count(EOS_ID), 1)
        self.assertEqual(bos_row["input_ids"].count(BOS_ID), 1)

    def test_pinned_rows_have_full_labels_one_bos_one_eos_and_boundary_proof(self):
        current = context()
        tokenizer = PinnedTokenizer()
        rows = []
        for fixture_id, operation, new in (
            ("PRM04-NOOP", "no_op", list(current.region_old)),
            ("PRM04-REPLACE", "replace", ["  😀 <- 2  ", "  literal <- \"<|endoftext|>\"  "]),
            ("PRM04-LEADING-LF", "replace", ["", "  leading whitespace  "]),
            ("PRM04-DELETE", "delete", []),
        ):
            row = build_training_row(current, operation=operation, region_new=new,
                                     tokenizer=tokenizer, row_id=fixture_id,
                                     family="functional", package_id="synthetic-prm04")
            rows.append(row)
            self.assertEqual(row["input_ids"][0], BOS_ID)
            self.assertEqual(row["input_ids"][-1], EOS_ID)
            self.assertEqual(row["input_ids"].count(BOS_ID), 1)
            self.assertEqual(row["input_ids"].count(EOS_ID), 1)
            self.assertEqual(row["target_text"].endswith(">>>>>>> UPDATED"), True)
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["target_body_token_count"], len(row["target_body_tokens"]))
            self.assertEqual(row["target_terminal_token_count"], len(row["target_terminal_tokens"]))
            validate_training_row(row)
        self.assertEqual(len(rows), 4)

    def test_bad_token_boundary_is_not_admitted(self):
        with self.assertRaises(TokenBoundaryError):
            build_training_row(context(), operation="replace", region_new=["x"],
                               tokenizer=BoundaryChangingTokenizer(), row_id="PRM04-BAD",
                               family="functional", package_id="synthetic-prm04")

    def test_full_row_validation_rejects_missing_extra_and_wrong_boundary(self):
        current = context()
        row = build_training_row(current, operation="replace", region_new=["x"],
                                 tokenizer=AdditiveTokenizer(), row_id="PRM04-ROW",
                                 family="functional", package_id="synthetic-prm04")
        for mutation in (
            {key: value for key, value in row.items() if key != "target_text"},
            {**row, "future": True},
            {**row, "target_start": row["target_start"] + 1},
        ):
            with self.assertRaises(ProtocolError):
                validate_training_row(mutation)

    def test_native_control_tokens_are_rejected_but_user_defined_tokens_are_text(self):
        self.assertTrue(valid_generation_tokens([8, 9, 42, 1]))
        for tokens in ([], [42], [42, 130073], [42, 2, 1], [42, 130559, 1],
                       [42, 1, 1], [42, True], [130560, 1], [-1, 1]):
            self.assertFalse(valid_generation_tokens(tokens), tokens)
        row = build_training_row(context(), operation="replace", region_new=["x"],
                                 tokenizer=AdditiveTokenizer(), row_id="PRM04-CONTROL",
                                 family="functional", package_id="synthetic-prm04")
        row['input_ids'][1] = 130082
        with self.assertRaisesRegex(ProtocolError, 'CONTROL'):
            validate_training_row(row)

    def test_reserved_slot_literals_are_explicitly_unsupported(self):
        for text in ('<unused_token_0>', '<unused_token_477>'):
            with self.assertRaisesRegex(ProtocolError, 'reserved_slot_literal'):
                render_prompt(context(prefix=[text]))
            with self.assertRaisesRegex(ProtocolError, 'reserved_slot_literal'):
                serialize_target('replace', [text])
            self.assertEqual(parse_output(text + '\n>>>>>>> UPDATED', context()).status, 'invalid')
        self.assertIn('<unused_token_478>', render_prompt(context(prefix=['<unused_token_478>'])))

    def test_fixture_identity_is_stable(self):
        current = context()
        digest = hashlib.sha256(render_prompt(current).encode("utf-8")).hexdigest()
        self.assertEqual(digest, "d0b6d5f1b57e6606561a141190bbdea7f8cb32501a79a36895bd0c6a1e94def6")
        self.assertEqual(json.loads(json.dumps(current.to_dict())), current.to_dict())


if __name__ == "__main__":
    unittest.main()
