import unittest
from collections import Counter
from pathlib import Path

from scripts.run_experiments import build_matrix, command_for


class ExperimentMatrixTests(unittest.TestCase):
    def setUp(self):
        self.matrix = build_matrix(Path("data/generated"), seed=0)

    def test_matrix_matches_the_532_cleaned_manuscript_records(self):
        self.assertEqual(len(self.matrix), 532)
        counts = Counter((run.environment, run.method) for run in self.matrix)
        for method in ("ddpg", "gcsl", "wgcsl", "gofar"):
            self.assertEqual(counts[("FetchPush", method)], 54)
            self.assertEqual(counts[("PointMaze", method)], 79)

    def test_ratio_zero_scope_matches_the_completed_logs(self):
        pointmaze_zero = [
            run
            for run in self.matrix
            if run.environment == "PointMaze" and run.ratio == 0.0
        ]
        self.assertEqual(len(pointmaze_zero), 7 * 4)
        self.assertTrue(all(run.dimension != "reachability" for run in pointmaze_zero))

    def test_commands_use_equal_mixture_and_explicit_seed(self):
        command = command_for(self.matrix[0])
        rendered = " ".join(command)
        self.assertIn("--expert_percent 0.5", rendered)
        self.assertIn("--random_percent 0.5", rendered)
        self.assertIn("--seed 0", rendered)
        self.assertIn("--use-mixture", command)
        self.assertNotIn("/home/", rendered)
        self.assertNotIn("\\", rendered)

    def test_method_specific_relabeling_matches_recorded_run_names(self):
        by_method = {}
        for run in self.matrix:
            by_method.setdefault(run.method, command_for(run))
        self.assertIn("0.5", by_method["ddpg"])
        self.assertIn("1.0", by_method["gcsl"])
        self.assertIn("1.0", by_method["wgcsl"])
        self.assertIn("disc", by_method["gofar"])
        self.assertIn("0.0", by_method["gofar"])


if __name__ == "__main__":
    unittest.main()
