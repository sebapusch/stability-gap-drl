"""Regression checks for CartPole interpolation checkpoint loading."""

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from linear_interpolation_grid import TASK_INDICES, load_policies, make_task_encoding, policy_fn


class LoadPoliciesTests(unittest.TestCase):
    def test_actor_and_dqn_checkpoints_use_the_online_network(self):
        cases = (
            ("actor.latent_pi", 4),
            ("q_net.q_net", 4),
            ("q_net.q_net", 4 + len(TASK_INDICES)),
        )
        for prefix, input_dim in cases:
            with self.subTest(prefix=prefix, input_dim=input_dim), tempfile.TemporaryDirectory() as directory:
                model_path = str(Path(directory) / "model")
                for task_index in TASK_INDICES:
                    value = float(task_index + 1)
                    shapes = ((5, input_dim), (5, 5), (2, 5))
                    state = {}
                    for layer, shape in zip((0, 2, 4), shapes):
                        state[f"{prefix}.{layer}.weight"] = torch.full(shape, value)
                        state[f"{prefix}.{layer}.bias"] = torch.full((shape[0],), value)
                    if prefix == "q_net.q_net":
                        state["q_net_target.q_net.0.weight"] = torch.full((5, input_dim), -99.0)

                    buffer = io.BytesIO()
                    torch.save(state, buffer)
                    with zipfile.ZipFile(f"{model_path}-{task_index}.zip", "w") as archive:
                        archive.writestr("policy.pth", buffer.getvalue())

                first, second, third, fourth = load_policies(model_path)
                for policy, expected in ((first, 1), (second, 2), (third, 3), (fourth, 4)):
                    for weight, bias in policy:
                        np.testing.assert_array_equal(weight, expected)
                        np.testing.assert_array_equal(bias, expected)

                self.assertEqual(make_task_encoding(first).size, input_dim - 4)
                np.testing.assert_array_equal(policy_fn(first, np.zeros((1, input_dim))), [0])


if __name__ == "__main__":
    unittest.main()
