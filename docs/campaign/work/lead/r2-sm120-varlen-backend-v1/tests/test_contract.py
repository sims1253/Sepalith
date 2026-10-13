import ast
import json
import unittest
from pathlib import Path
import sys

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT))
import backend_contract as contract


class ContractTest(unittest.TestCase):
    def test_static_bindings_and_loss_path(self):
        value=contract.validate()
        self.assertTrue(all(value["checks"].values()))
        self.assertIn("global supervised-token mean",value["loss_denominator"]["effect"])

    def test_gpu_probes_compile_without_importing_gpu_stack(self):
        for name in ("probe_xformers_sm120.py","probe_flex_sm120.py"):
            source=(ROOT/name).read_text()
            ast.parse(source,filename=name)
            self.assertIn("changed_k",source)
            self.assertIn("changed_v",source)

    def test_commands_are_unadmitted_and_overlay_scoped(self):
        value=json.loads((ROOT/"commands.json").read_text())
        self.assertIs(value["execution_authorized"],False)
        self.assertTrue(value["environment"]["PYTHONPATH"].startswith("/mnt/e/"))
        self.assertEqual([x["name"] for x in value["diagnostics_in_order"]],
                         ["official_cu130_xformers_kernel","actual_model_xformers_parity","native_flex_kernel"])

    def test_actual_model_probe_is_bound_to_reviewed_harness(self):
        value=json.loads((ROOT/"commands.json").read_text())
        cmd=value["diagnostics_in_order"][1]["command"]
        self.assertIn("r2-varlen-parity-root-v1/actual_model_varlen_probe.py",cmd[2])
        self.assertIn("--token-cap",cmd)
        self.assertEqual(cmd[cmd.index("--token-cap")+1],"16384")

    def test_flex_is_not_claimed_drop_in(self):
        value=json.loads((ROOT/"environment-audit.json").read_text())
        self.assertIs(value["native_flex"]["packed_cpt_drop_in"],False)
        self.assertEqual(value["isolated_candidate"]["gpu_result"],"not run")


if __name__ == "__main__": unittest.main()
