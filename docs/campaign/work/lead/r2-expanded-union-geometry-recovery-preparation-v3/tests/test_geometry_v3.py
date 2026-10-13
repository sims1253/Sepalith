import copy
import importlib.util
import json
import unittest
from pathlib import Path


PACKET = Path(__file__).resolve().parents[1]
ROOT = PACKET.parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = load("geometry_v3", PACKET / "recover_geometry_v3.py")
C = load("geometry_consumer_v3", PACKET / "validate_geometry_consumer_v3.py")
A = load(
    "audit_v1",
    ROOT / "docs/campaign/work/lead/r2-expanded-union-audit-preparation-v2/audit_expanded_union_v1.py",
)
ORIGINAL = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/context-sidecar.jsonl"
)
SEMANTIC = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/Semantic10948-provider-materialization-v2/selected-01/selected-contexts.jsonl"
)
SEMANTIC_PROVENANCE = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/Semantic10948-provider-materialization-v2/final-01/candidate-provenance.jsonl"
)


def first(path):
    with path.open(encoding="utf-8") as stream:
        return json.loads(next(stream))


class GeometryV3Tests(unittest.TestCase):
    def test_actual_original_cursor_is_absolute_utf16_and_nonzero_range(self):
        row = first(ORIGINAL)
        output = G.recover(row, row, "context", "nested_original15006", row["row_id"])
        geometry = output["source_cursor_geometry"]
        self.assertEqual(row["row_id"], "000a6aaa48aee4291dbcb0cb")
        self.assertEqual(geometry["cursor"], {"line": 42, "character": 25})
        self.assertEqual(geometry["replacement_range"], {
            "start": {"line": 42, "character": 0},
            "end": {"line": 42, "character": 43},
        })
        self.assertNotEqual(geometry["cursor"], geometry["replacement_range"]["start"])
        self.assertEqual(output["cursor_evidence"]["basis"], "explicit_region_cursor")
        self.assertEqual(output["cursor_evidence"]["absolute_position"], geometry["cursor"])
        self.assertEqual(A.geometry_from_provenance(output)[1], None)
        self.assertEqual(C.validate(output), geometry)

    def test_explicit_unicode_cursor_converts_code_points_to_utf16(self):
        lines = ["prefix", "z😀x", "suffix"]
        context = {
            "cursor": {"code_point_column": 2, "region_line_index": 0, "utf16_column": 3},
            "document_eol": "lf",
            "prefix": [lines[0]],
            "region_old": [lines[1]],
            "suffix_lines": [lines[2]],
            "replacement_range": {
                "content_sha256": G.digest_text("\n".join(lines)),
                "document_version": 0,
                "start": {"line": 1, "character": 0},
                "end": {"line": 1, "character": 4},
                "uri": "file:///tmp/fixture.R",
            },
        }
        identity = {
            "source_path": "/tmp/fixture.R",
            "source_sha256": "a" * 64,
        }
        row = {
            "row_id": "unicode",
            "selected_context": context,
            "selection_geometry": {"availability": "full_snapshot"},
        }
        provenance = {
            "row_id": "unicode",
            "preedit_sha256": context["replacement_range"]["content_sha256"],
            "source_identity": identity,
        }
        output = G.recover(row, provenance, "selected_context", "direct", "unicode")
        self.assertEqual(output["source_cursor_geometry"]["cursor"], {"line": 1, "character": 3})
        self.assertEqual(output["cursor_evidence"]["code_point_column"], 2)
        self.assertEqual(A.geometry_from_provenance(output)[1], None)
        self.assertEqual(C.validate(output), output["source_cursor_geometry"])

    def test_semantic_null_cursor_sentinel_requires_zero_width_and_maps_boundary(self):
        row = first(SEMANTIC)
        provenance = first(SEMANTIC_PROVENANCE)
        self.assertEqual(row["row_id"], provenance["row_id"])
        output = G.recover(row, provenance, "selected_context", "direct", row["row_id"])
        geometry = output["source_cursor_geometry"]
        self.assertEqual(geometry["cursor"], {"line": 352, "character": 0})
        self.assertEqual(geometry["replacement_range"]["start"], geometry["replacement_range"]["end"])
        self.assertEqual(output["cursor_evidence"]["basis"], "zero_width_replacement_range_start_sentinel")
        self.assertEqual(A.geometry_from_provenance(output)[1], None)
        self.assertEqual(C.validate(output), geometry)

    def test_nonzero_sentinel_is_unresolved(self):
        row = first(SEMANTIC)
        provenance = first(SEMANTIC_PROVENANCE)
        broken = copy.deepcopy(row)
        broken["selected_context"]["replacement_range"]["end"]["character"] = 1
        with self.assertRaisesRegex(G.RecoveryError, "partial_nonempty_range_unverified|cursor_sentinel_requires_zero_width"):
            G.recover(broken, provenance, "selected_context", "direct", row["row_id"])

    def test_explicit_cursor_must_be_inside_range(self):
        row = first(ORIGINAL)
        broken = copy.deepcopy(row)
        broken["context"]["replacement_range"]["end"]["character"] = 24
        with self.assertRaisesRegex(G.RecoveryError, "cursor_outside_replacement_range"):
            G.recover(broken, broken, "context", "nested_original15006", row["row_id"])

    def test_code_point_and_utf16_columns_must_agree(self):
        row = first(ORIGINAL)
        broken = copy.deepcopy(row)
        broken["context"]["cursor"]["utf16_column"] = 24
        with self.assertRaisesRegex(G.RecoveryError, "cursor_codepoint_utf16_mismatch"):
            G.recover(broken, broken, "context", "nested_original15006", row["row_id"])

    def test_consumer_checks_context_join_without_start_equality(self):
        row = first(ORIGINAL)
        output = G.recover(row, row, "context", "nested_original15006", row["row_id"])
        broken = copy.deepcopy(output)
        broken["source_cursor_geometry"]["cursor"] = {"line": 42, "character": 44}
        broken["source_cursor_geometry_sha256"] = G.digest_text(
            G.canonical(broken["source_cursor_geometry"])
        )
        with self.assertRaisesRegex(C.GeometryConsumerError, "cursor_outside_replacement_range"):
            C.validate(broken)

        broken = copy.deepcopy(output)
        broken["cursor_evidence"]["absolute_position"] = {"line": 42, "character": 0}
        with self.assertRaisesRegex(C.GeometryConsumerError, "cursor_evidence_absolute_join"):
            C.validate(broken)


if __name__ == "__main__":
    unittest.main()
