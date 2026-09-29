"""Regression checks for SACD's automatic entropy target."""

import unittest

import gymnasium as gym
import numpy as np

from stable_baselines3.sacd.sacd import SACD


class SACDTargetEntropyTests(unittest.TestCase):
    def test_auto_target_uses_number_of_discrete_actions(self):
        for env_id, n_actions in (("CartPole-v1", 2), ("FrozenLake-v1", 4)):
            with self.subTest(env_id=env_id):
                env = gym.make(env_id)
                try:
                    model = SACD("MlpPolicy", env, ent_coef="auto", target_entropy="auto")
                    self.assertAlmostEqual(model.target_entropy, 0.98 * np.log(n_actions))
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
