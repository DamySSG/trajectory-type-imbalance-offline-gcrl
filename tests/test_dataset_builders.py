import csv
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.data.make_fetchpush_imbalances import (
    DEFAULT_RATIOS as FETCHPUSH_RATIOS,
)
from scripts.data.make_fetchpush_imbalances import (
    build_variants as build_fetchpush,
)
from scripts.data.make_pointmaze_imbalances import (
    DEFAULT_RATIOS as POINTMAZE_RATIOS,
)
from scripts.data.make_pointmaze_imbalances import (
    build_variants as build_pointmaze,
)
from scripts.data.validate_variants import validate_manifest


def fetchpush_buffer():
    goals = np.array(
        [
            [0.5, 0.5],
            [0.5, -0.5],
            [2.0, 0.5],
            [2.0, -0.5],
            [0.2, 0.2],
            [0.2, -0.2],
            [3.0, 0.5],
            [3.0, -0.5],
        ]
    )
    count = len(goals)
    observations = np.zeros((count, 3, 6))
    achieved_goals = np.zeros((count, 3, 3))
    achieved_goals[:, :, 0] = 1.0
    desired_goals = np.repeat(goals[:, None, :], 2, axis=1)
    desired_goals = np.pad(desired_goals, ((0, 0), (0, 0), (0, 1)))
    return {
        "o": observations,
        "ag": achieved_goals,
        "g": desired_goals,
        "u": np.zeros((count, 2, 4)),
    }


def pointmaze_buffer():
    starts = np.array(
        [
            [-3.0, 3.0],
            [3.0, 3.0],
            [-3.0, -3.0],
            [3.0, -3.0],
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )
    horizontal_motion = np.array([-3.0, 3.0, -2.0, 2.0, -1.0, 1.0, -0.5, 0.5, 0.0])
    achieved_goals = np.repeat(starts[:, None, :], 3, axis=1)
    achieved_goals[:, 1, 0] += horizontal_motion * 0.25
    achieved_goals[:, 2, 0] += horizontal_motion
    count = len(starts)
    return {
        "o": np.pad(achieved_goals, ((0, 0), (0, 0), (0, 2))),
        "ag": achieved_goals,
        "g": np.zeros((count, 2, 2)),
        "u": np.zeros((count, 2, 2)),
    }


class DatasetBuilderTests(unittest.TestCase):
    def test_default_ratio_grid_includes_complete_subtype_removal(self):
        self.assertEqual(FETCHPUSH_RATIOS[0], 0.0)
        self.assertEqual(POINTMAZE_RATIOS[0], 0.0)

    def test_fetchpush_builder_writes_six_ratio_one_variants_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_fetchpush(
                fetchpush_buffer(), Path(directory), ratios=[1.0], seed=0
            )

            self.assertEqual(len(manifest), 6)
            files = sorted(Path(directory).rglob("*.pkl"))
            self.assertEqual(len(files), 6)
            self.assertTrue(Path(directory, "manifest.csv").is_file())
            for path in files:
                with path.open("rb") as handle:
                    self.assertEqual(pickle.load(handle)["o"].shape[0], 8)

    def test_pointmaze_builder_writes_nine_ratio_one_variants_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_pointmaze(
                pointmaze_buffer(),
                Path(directory),
                ratios=[1.0],
                seed=0,
                base_episodes=None,
            )

            self.assertEqual(len(manifest), 9)
            self.assertEqual(len(list(Path(directory).rglob("*.pkl"))), 9)
            with Path(directory, "manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertTrue(
                all(row["target_after"] == row["target_before"] for row in rows)
            )

    def test_pointmaze_builder_supports_complete_subtype_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_pointmaze(
                pointmaze_buffer(),
                Path(directory),
                ratios=[0.0],
                seed=0,
                base_episodes=None,
            )

            self.assertEqual(len(manifest), 9)
            self.assertTrue(all(row["target_after"] == 0 for row in manifest))

    def test_manifest_validator_checks_files_and_episode_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_fetchpush(fetchpush_buffer(), root, ratios=[1.0], seed=0)

            report = validate_manifest(root)

            self.assertEqual(report["checked"], 6)
            self.assertEqual(report["errors"], [])

    def test_manifest_validator_reports_a_missing_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_fetchpush(fetchpush_buffer(), root, ratios=[1.0], seed=0)
            root.joinpath(manifest[0]["path"]).unlink()

            report = validate_manifest(root)

            self.assertEqual(report["checked"], 5)
            self.assertEqual(len(report["errors"]), 1)
            self.assertIn("missing", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
