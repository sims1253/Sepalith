import json
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve().parent
CHECKPOINT_BYTES = 17_250_917_475
MIN_FREE = 100 * 1024**3


class ProbeResultTest(unittest.TestCase):
    def test_exact_hash_size_cleanup_and_capacity_policy(self):
        result = json.loads((HERE / "probe-result.json").read_text())
        proposal = json.loads((HERE / "proposal.json").read_text())
        self.assertEqual(result["bytes"], 256 * 1024**2)
        self.assertEqual(result["actual_bytes"], result["bytes"])
        self.assertEqual(result["actual_sha256"], result["expected_sha256"])
        self.assertEqual(result["verification"], "pass")
        self.assertTrue(result["cleanup"]["path_absent"])
        self.assertFalse(pathlib.Path(result["path"]).exists())
        policy = proposal["capacity_policy"]
        self.assertEqual(policy["maximum_published_hot_checkpoints"], 2)
        self.assertEqual(policy["two_checkpoint_bytes"], 2 * CHECKPOINT_BYTES)
        self.assertEqual(policy["minimum_c_free_bytes_after_allocation"], MIN_FREE)
        self.assertGreaterEqual(result["free_bytes_after"] - 2 * CHECKPOINT_BYTES, MIN_FREE)


if __name__ == "__main__": unittest.main()
