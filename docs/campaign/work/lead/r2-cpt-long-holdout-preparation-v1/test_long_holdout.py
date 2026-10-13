import copy
import importlib.util
import json
import pathlib
import sys
import unittest


PACKET = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prepare_long_holdout", PACKET / "prepare_long_holdout.py")
prep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prep
assert spec.loader is not None
spec.loader.exec_module(prep)
validator = prep.load_module("test_long_holdout_validator", prep.VALIDATOR)


def rows(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line]


class LongHoldoutTests(unittest.TestCase):
    def test_output_passes_exact_frozen_validator(self):
        for label, context, expected_rows, expected_docs in (("8k", 8192, 20, 10), ("16k", 16384, 6, 3)):
            result = validator.validate_materialized_rows(
                rows(PACKET / f"output/cpt_validation_{label}.jsonl"),
                max_sequence_tokens=context,
                require_complete_documents=True,
            )
            self.assertEqual((result["rows"], result["documents"]), (expected_rows, expected_docs))

    def test_strata_are_disjoint_subsets_of_original_holdout_and_train_packages_are_excluded(self):
        original = rows(prep.VALIDATION)
        output = rows(PACKET / "output/cpt_validation_8k.jsonl") + rows(PACKET / "output/cpt_validation_16k.jsonl")
        original_packages = {row["package"] for row in original}
        output_documents = {row["document_id"] for row in output}
        pilot_train = rows(PACKET.parent / "r2-full-weight-cpt-lr-pilot-v1/panel/train384-ctx8192.jsonl")
        self.assertLessEqual({row["package"] for row in output}, original_packages)
        self.assertFalse({row["package"] for row in output} & {row["package"] for row in pilot_train})
        provenance = rows(PACKET / "output/document-provenance.jsonl")
        self.assertEqual(len(output_documents), len(provenance))
        self.assertEqual(len(output_documents), 13)

    def test_every_generated_document_exactly_reassembles_original_tokens(self):
        original = rows(prep.VALIDATION)
        grouped = {}
        for row in original:
            grouped.setdefault(row["document_id"], []).append(row)
        for label in ("8k", "16k"):
            generated = {}
            for row in rows(PACKET / f"output/cpt_validation_{label}.jsonl"):
                generated.setdefault(row["document_id"], []).append(row)
            for ident, new_rows in generated.items():
                _, before, _ = prep.reassemble(grouped[ident])
                _, after, _ = prep.reassemble(new_rows)
                self.assertEqual(before, after)

    def test_terminal_eos_and_prior_carry_contract(self):
        for label in ("8k", "16k"):
            grouped = {}
            for row in rows(PACKET / f"output/cpt_validation_{label}.jsonl"):
                grouped.setdefault(row["document_id"], []).append(row)
            for document in grouped.values():
                document.sort(key=lambda row: row["chunk_index"])
                self.assertEqual(sum(row["is_document_end"] for row in document), 1)
                self.assertEqual(document[-1]["labels"][-1], prep.EOS)
                for previous, current in zip(document, document[1:]):
                    self.assertEqual(current["input_ids"][1], previous["input_ids"][-2])
                    self.assertEqual(current["labels"][1], prep.MASK)

    def test_incomplete_and_cross_document_groups_fail_closed(self):
        original = rows(prep.VALIDATION)
        grouped = {}
        for row in original:
            grouped.setdefault(row["document_id"], []).append(row)
        multi = next(copy.deepcopy(group) for group in grouped.values() if len(group) > 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            prep.reassemble(multi[:-1])
        other = next(group for ident, group in grouped.items() if ident != multi[0]["document_id"])
        mixed = copy.deepcopy(multi)
        mixed[1] = copy.deepcopy(other[0])
        mixed[1]["chunk_index"] = 1
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            prep.reassemble(mixed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
