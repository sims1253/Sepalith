from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from unittest import mock
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "launch_theta0_q8_cuda_dev.py"
ADAPTER = ROOT / "run09_corrected_dev_client.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load_module(LAUNCH, "theta0_q8_cuda_dev_launch_test")
adapter_source = ast.parse(ADAPTER.read_text(encoding="utf-8"))


class PreparationContractTests(unittest.TestCase):
    def test_server_profile_is_selected_cuda_graph0_and_not_cpu_server(self):
        argv = launch.server_argv(18403)
        self.assertIn(str(launch.MODEL), argv)
        self.assertEqual(argv[argv.index("--port") + 1], "18403")
        self.assertEqual(argv[argv.index("-c") + 1], "4096")
        self.assertEqual(argv[argv.index("-b") + 1], "256")
        self.assertEqual(argv[argv.index("-ub") + 1], "256")
        self.assertNotIn("18099", argv)
        env = launch._base_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["GGML_CUDA_GRAPH_OPT"], "0")

    def test_cpu_server_port_is_rejected_without_touching_it(self):
        with self.assertRaises(launch.PreparationError):
            launch._validate_port(18099)
        fake_socket = mock.MagicMock()
        fake_socket.__enter__.return_value = fake_socket
        with mock.patch.object(launch.socket, "socket", return_value=fake_socket):
            launch._validate_port(18403)
        fake_socket.bind.assert_called_once_with(("127.0.0.1", 18403))

    def test_quality_contract_uses_corrected_panel_and_separate_512_cap(self):
        output = Path("/tmp/theta0-q8-cuda-quality.json")
        argv = launch.quality_argv(18403, output)
        self.assertIn(str(launch.ADAPTER), argv)
        self.assertEqual(argv[argv.index("--panel") + 1], str(launch.PANEL))
        self.assertEqual(argv[argv.index("--case-deadline-seconds") + 1], "120")
        self.assertEqual(argv[argv.index("--deadline-seconds") + 1], "1100")
        self.assertEqual(argv[argv.index("--reserve-seconds") + 1], "60")
        self.assertEqual(launch.QUALITY_COMPLETION_CAP, 512)
        self.assertEqual(launch.PANEL_SHA256,
                         "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035")

    def test_diagnostic_contract_is_four_train_rows_and_192_cap(self):
        argv = launch.diagnostic_argv(18403, Path("/tmp/theta0-q8-cuda-diagnostic.json"))
        self.assertEqual(argv[argv.index("--arm") + 1], "baseline")
        self.assertEqual(argv[argv.index("--cap") + 1], "192")
        self.assertEqual(argv[argv.index("--context") + 1], "4096")
        self.assertEqual(argv[argv.index("--user-deadline-ms") + 1], "5000")
        self.assertEqual(argv[argv.index("--diagnostic-timeout-ms") + 1], "60000")
        self.assertEqual(launch.DIAGNOSTIC_COMPLETION_CAP, 192)

    def test_guard_and_cleanup_are_bounded_and_process_group_scoped(self):
        self.assertEqual(launch.SERVER_GUARD_SECONDS, 1200)
        self.assertIn("killpg", launch._terminate_group.__code__.co_names)
        source = LAUNCH.read_text(encoding="utf-8")
        self.assertNotIn("pkill", source)
        self.assertNotIn("killall", source)

    def test_adapter_contains_only_explicit_panel_digest_override(self):
        source = ADAPTER.read_text(encoding="utf-8")
        self.assertIn("REFERENCE_SCORER_SHA256", source)
        self.assertIn("CORRECTED_PANEL_SHA256", source)
        self.assertIn("module.PANEL_SHA256 = CORRECTED_PANEL_SHA256", source)
        self.assertIn("module.PANEL_ROWS = CORRECTED_PANEL_ROWS", source)
        # The adapter is a caller of the reference scorer, not a second scorer.
        names = {node.id for node in ast.walk(adapter_source) if isinstance(node, ast.Name)}
        self.assertIn("module", names)
        self.assertNotIn("requests", names)


if __name__ == "__main__":
    unittest.main()
