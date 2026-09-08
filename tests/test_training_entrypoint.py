import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT.joinpath("train.py").read_text(encoding="utf-8")


class TrainingEntrypointTests(unittest.TestCase):
    def test_her_ddpg_uses_the_gofar_ddpg_module(self):
        self.assertIn("from rl_modules.ddpg_agent import DDPG", TRAIN_SOURCE)
        self.assertNotIn("ddpg_agent1", TRAIN_SOURCE)

    def test_entrypoint_contains_only_reported_algorithms(self):
        self.assertNotIn("ActionableModel", TRAIN_SOURCE)
        self.assertNotIn("actionablemodel_agent", TRAIN_SOURCE)

    def test_standard_gymnasium_robotics_environments_are_registered(self):
        tree = ast.parse(TRAIN_SOURCE)
        imported_modules = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn("gymnasium_robotics", imported_modules)
        self.assertNotIn("from envs import", TRAIN_SOURCE)
        self.assertNotIn("from envs.", TRAIN_SOURCE)

    def test_wandb_destination_and_mode_are_configurable(self):
        self.assertIn("--wandb-project", TRAIN_SOURCE)
        self.assertIn("--wandb-entity", TRAIN_SOURCE)
        self.assertIn("--wandb-mode", TRAIN_SOURCE)
        self.assertIn("project=args.wandb_project", TRAIN_SOURCE)
        self.assertIn("entity=args.wandb_entity", TRAIN_SOURCE)
        self.assertIn("mode=args.wandb_mode", TRAIN_SOURCE)
        self.assertNotIn('project="gofar-maze-false"', TRAIN_SOURCE)


if __name__ == "__main__":
    unittest.main()
