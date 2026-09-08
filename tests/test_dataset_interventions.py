import unittest
from collections import Counter

import numpy as np

from scripts.data.fetchpush import (
    behavior_masks as fetchpush_behavior_masks,
)
from scripts.data.fetchpush import (
    coverage_masks as fetchpush_coverage_masks,
)
from scripts.data.fetchpush import (
    initial_geometry,
)
from scripts.data.fetchpush import (
    reachability_masks as fetchpush_reachability_masks,
)
from scripts.data.interventions import intervene
from scripts.data.pointmaze import (
    behavior_masks as pointmaze_behavior_masks,
)
from scripts.data.pointmaze import (
    coverage_masks as pointmaze_coverage_masks,
)
from scripts.data.pointmaze import (
    path_lengths,
)
from scripts.data.pointmaze import (
    reachability_masks as pointmaze_reachability_masks,
)


def make_buffer(episodes=10, horizon=3):
    episode_ids = np.arange(episodes, dtype=np.int64)
    observations = np.repeat(episode_ids[:, None, None], horizon + 1, axis=1)
    actions = np.repeat(episode_ids[:, None, None], horizon, axis=1)
    goals = np.repeat(episode_ids[:, None, None], horizon, axis=1)
    achieved_goals = np.repeat(episode_ids[:, None, None], horizon + 1, axis=1)
    return {
        "o": observations,
        "u": actions,
        "g": goals,
        "ag": achieved_goals,
        "metadata": np.array([1234]),
    }


def episode_ids(buffer):
    return buffer["o"][:, 0, 0].astype(int).tolist()


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.buffer = make_buffer()
        self.target = np.arange(10) < 4

    def test_ratio_one_preserves_every_episode_and_array_alignment(self):
        result, report = intervene(
            self.buffer,
            self.target,
            ratio=1.0,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(Counter(episode_ids(result)), Counter(range(10)))
        self.assertEqual(report["target_before"], 4)
        self.assertEqual(report["target_after"], 4)
        for row, episode_id in enumerate(episode_ids(result)):
            self.assertTrue(np.all(result["u"][row] == episode_id))
            self.assertTrue(np.all(result["g"][row] == episode_id))
            self.assertTrue(np.all(result["ag"][row] == episode_id))
        np.testing.assert_array_equal(result["metadata"], self.buffer["metadata"])

    def test_downsampling_changes_only_the_target_subset(self):
        result, report = intervene(
            self.buffer,
            self.target,
            ratio=0.5,
            rng=np.random.default_rng(11),
            rounding="floor",
        )

        counts = Counter(episode_ids(result))
        self.assertEqual(report["target_after"], 2)
        self.assertEqual(sum(counts[index] for index in range(4)), 2)
        self.assertEqual(
            {index: counts[index] for index in range(4, 10)},
            {index: 1 for index in range(4, 10)},
        )
        self.assertTrue(all(counts[index] <= 1 for index in range(4)))

    def test_zero_ratio_removes_only_the_target_subset(self):
        result, report = intervene(
            self.buffer,
            self.target,
            ratio=0.0,
            rng=np.random.default_rng(19),
        )

        self.assertEqual(report["target_after"], 0)
        self.assertEqual(Counter(episode_ids(result)), Counter(range(4, 10)))

    def test_append_upsampling_retains_originals_and_reaches_exact_count(self):
        result, report = intervene(
            self.buffer,
            self.target,
            ratio=2.0,
            rng=np.random.default_rng(13),
            rounding="ceil",
            upsampling="append",
        )

        counts = Counter(episode_ids(result))
        self.assertEqual(report["target_after"], 8)
        self.assertEqual(len(result["o"]), 14)
        self.assertTrue(all(counts[index] >= 1 for index in range(4)))
        self.assertEqual(
            {index: counts[index] for index in range(4, 10)},
            {index: 1 for index in range(4, 10)},
        )

    def test_intervention_is_deterministic_for_a_fixed_seed(self):
        first, _ = intervene(
            self.buffer,
            self.target,
            ratio=0.5,
            rng=np.random.default_rng(17),
        )
        second, _ = intervene(
            self.buffer,
            self.target,
            ratio=0.5,
            rng=np.random.default_rng(17),
        )
        np.testing.assert_array_equal(first["o"], second["o"])


class FetchPushClassificationTests(unittest.TestCase):
    def test_initial_geometry_reads_the_reported_buffer_coordinates(self):
        buffer = {
            "o": np.zeros((2, 3, 6)),
            "ag": np.zeros((2, 3, 3)),
            "g": np.zeros((2, 2, 3)),
        }
        buffer["o"][:, 0, :3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        buffer["ag"][:, 0, :2] = [[7.0, 8.0], [9.0, 10.0]]
        buffer["g"][:, 0, :2] = [[11.0, 12.0], [13.0, 14.0]]

        gripper, object_pos, goals = initial_geometry(buffer)

        np.testing.assert_array_equal(gripper, [[1.0, 2.0], [4.0, 5.0]])
        np.testing.assert_array_equal(object_pos, [[7.0, 8.0], [9.0, 10.0]])
        np.testing.assert_array_equal(goals, [[11.0, 12.0], [13.0, 14.0]])

    def test_behavior_uses_signed_planar_cross_product(self):
        gripper = np.zeros((3, 2))
        object_pos = np.array([[1.0, 0.0]] * 3)
        goals = np.array([[1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])

        masks = fetchpush_behavior_masks(gripper, object_pos, goals, eps=1e-8)

        np.testing.assert_array_equal(masks["left"], [True, False, False])
        np.testing.assert_array_equal(masks["right"], [False, True, False])
        np.testing.assert_array_equal(masks["neutral"], [False, False, True])

    def test_reachability_uses_the_configured_weighted_effort_score(self):
        gripper = np.array([[0.0, 0.0], [0.0, 0.0]])
        object_pos = np.array([[1.0, 0.0], [2.0, 0.0]])
        goals = np.array([[2.0, 0.0], [4.0, 0.0]])

        masks, metadata = fetchpush_reachability_masks(
            gripper,
            object_pos,
            goals,
            alpha=1.0,
            beta=1.0,
            gamma=0.5,
        )

        np.testing.assert_allclose(metadata["scores"], [2.0, 4.0])
        self.assertAlmostEqual(metadata["threshold"], 3.0)
        np.testing.assert_array_equal(masks["short"], [True, False])
        np.testing.assert_array_equal(masks["long"], [False, True])

    def test_coverage_uses_mean_gripper_center_and_lower_tail_radius(self):
        gripper = np.zeros((2, 2))
        object_pos = np.array([[1.0, 0.0], [2.0, 0.0]])
        goals = np.array([[0.5, 0.0], [1.5, 0.0]])

        masks, metadata = fetchpush_coverage_masks(
            gripper, object_pos, goals, quantile=0.0
        )

        np.testing.assert_array_equal(metadata["center"], [0.0, 0.0])
        self.assertEqual(metadata["radius"], 1.0)
        np.testing.assert_array_equal(masks["inner"], [True, False])
        np.testing.assert_array_equal(masks["outer"], [False, True])


class PointMazeClassificationTests(unittest.TestCase):
    def test_behavior_uses_net_horizontal_displacement(self):
        achieved_goals = np.array(
            [
                [[0.0, 0.0], [-1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0001, 0.0]],
            ]
        )

        masks = pointmaze_behavior_masks(achieved_goals, eps=1e-3)

        np.testing.assert_array_equal(masks["left"], [True, False, False])
        np.testing.assert_array_equal(masks["right"], [False, True, False])
        np.testing.assert_array_equal(masks["neutral"], [False, False, True])

    def test_path_length_sums_stepwise_euclidean_motion(self):
        achieved_goals = np.array(
            [
                [[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]],
                [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            ]
        )
        np.testing.assert_allclose(path_lengths(achieved_goals), [9.0, 1.0])

    def test_coverage_uses_initial_position_quartiles(self):
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
        achieved_goals = np.repeat(starts[:, None, :], 2, axis=1)

        masks, metadata = pointmaze_coverage_masks(achieved_goals)

        np.testing.assert_array_equal(metadata["quartiles"], [-1.0, 1.0, -1.0, 1.0])
        self.assertEqual(int(masks["top_left"].sum()), 1)
        self.assertEqual(int(masks["top_right"].sum()), 1)
        self.assertEqual(int(masks["bottom_left"].sum()), 1)
        self.assertEqual(int(masks["bottom_right"].sum()), 1)
        self.assertEqual(int(masks["center"].sum()), 5)

    def test_reachability_uses_path_length_tertiles(self):
        achieved_goals = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [3.0, 0.0]],
            ]
        )

        masks, metadata = pointmaze_reachability_masks(achieved_goals)

        np.testing.assert_allclose(metadata["thresholds"], [5.0 / 3.0, 7.0 / 3.0])
        np.testing.assert_array_equal(masks["short"], [True, False, False])
        np.testing.assert_array_equal(masks["middle"], [False, True, False])
        np.testing.assert_array_equal(masks["long"], [False, False, True])


if __name__ == "__main__":
    unittest.main()
