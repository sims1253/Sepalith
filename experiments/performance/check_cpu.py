"""Run the performance candidates' CPU correctness checks without pytest."""
import inspect
import os
from pathlib import Path
import sys
import unittest

# Set before importing PyTorch or model modules; this check never uses CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import torch
    from experiments.training.poc_diff import test_objective
    from experiments.training.poc_diff.test_objective_chunks import ObjectiveChunksTest
    from experiments.training.poc_twin.test_eval_ce import ChunkedEvalCETest
    from experiments.training.poc_twin.test_ladder_eval_ce import LadderEvalIntegrationTest

    torch.set_num_threads(1)
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(fn)
        for name, fn in inspect.getmembers(test_objective, inspect.isfunction)
        if name.startswith("test_")
    )
    for cls in (ObjectiveChunksTest, ChunkedEvalCETest, LadderEvalIntegrationTest):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
