import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class RNamespaceEvidenceTests(unittest.TestCase):
    def run_helper(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            namespace = td / "NAMESPACE"
            namespace.write_text(text, encoding="utf-8")
            request = td / "request.json"
            output = td / "output.jsonl"
            request.write_text(json.dumps({"namespaces": [{
                "id": "fixture", "path": str(namespace),
                "expected_sha256": hashlib.sha256(namespace.read_bytes()).hexdigest(),
            }]}))
            completed = subprocess.run(
                ["Rscript", str(HERE / "namespace_evidence.R"), str(request), str(output)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return json.loads(output.read_text())

    def test_comments_strings_and_assignments_do_not_invent_imports(self):
        row = self.run_helper(
            '# importFrom(fake, ghost)\n'
            'label <- "importFrom(fake, phantom)"\n'
            'importFrom(stats, hclust, median)\n'
            'importFrom(magrittr, "%>%")\n'
        )
        self.assertEqual(row["status"], "namespace_evidence_complete")
        pairs = {(d["package"], tuple(d["symbols"])) for d in row["directives"]}
        self.assertEqual(pairs, {("stats", ("hclust", "median")), ("magrittr", ("%>%",))})

    def test_wildcard_is_preserved_but_does_not_invent_symbol_origin(self):
        row = self.run_helper("import(dplyr)\n")
        self.assertEqual(row["directives"], [{"kind": "import", "package": "dplyr", "symbols": []}])

    def test_dynamic_or_malformed_symbol_fails_closed(self):
        row = self.run_helper("importFrom(stats, paste0('med', 'ian'))\n")
        self.assertEqual(row["status"], "hold_namespace_malformed")
        self.assertEqual(row["directives"], [])
        self.assertTrue(row["malformed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
