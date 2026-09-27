import ast
import pathlib
import unittest

ROOT = pathlib.Path("docs/campaign/work/lead")
V3_PROOF = ROOT / "r2-full-weight-resume-proof-v3/source/full_weight_resume_proof.py"
V4_PROOF = ROOT / "r2-full-weight-resume-proof-v4/source/full_weight_resume_proof.py"
CHECKPOINT = ROOT / "r2-full-weight-resume-proof-v4/source/campaign_checkpoint.py"


def functions(path, name):
    tree = ast.parse(path.read_text())
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]


def calls(node):
    names = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call): continue
        target = item.func
        if isinstance(target, ast.Name): names.append(target.id)
        elif isinstance(target, ast.Attribute): names.append(target.attr)
    return names


class CheckpointCallChainTest(unittest.TestCase):
    def test_v4_removes_only_immediate_callback_verify(self):
        v3 = functions(V3_PROOF, "on_save")[0]
        v4 = functions(V4_PROOF, "on_save")[0]
        self.assertIn("seal_checkpoint", calls(v3)); self.assertIn("verify_checkpoint", calls(v3))
        self.assertIn("seal_checkpoint", calls(v4)); self.assertNotIn("verify_checkpoint", calls(v4))
        self.assertGreaterEqual(calls(functions(V4_PROOF, "run_lane")[0]).count("verify_checkpoint"), 2)

    def test_seal_and_verify_each_inventory_full_bytes(self):
        self.assertIn("flush_files", calls(functions(CHECKPOINT, "seal_checkpoint")[0]))
        self.assertIn("inventory", calls(functions(CHECKPOINT, "seal_checkpoint")[0]))
        self.assertIn("inventory", calls(functions(CHECKPOINT, "verify_checkpoint")[0]))

    def test_generic_archive_verifies_copies_and_verifies(self):
        observed = calls(functions(CHECKPOINT, "archive_checkpoint")[0])
        # Source verification, existing-destination verification branch, and
        # copied-temporary verification are all present in the implementation.
        self.assertEqual(observed.count("verify_checkpoint"), 3)
        self.assertIn("copytree", observed)
        self.assertIn("rename", observed)


if __name__ == "__main__": unittest.main()
