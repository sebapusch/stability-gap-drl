"""Regression checks for InvertedPendulum actor checkpoint loading."""

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from linear_interpolation_grid_ip import (
    TASK_INDICES,
    load_policies,
    make_task_encoding,
    policy_fn,
)


class LoadPoliciesTests(unittest.TestCase):
    def test_sac_and_ddpg_actors(self):
        cases = (
            (("actor.latent_pi.0", "actor.latent_pi.2", "actor.mu"), 4),
            (("actor.mu.0", "actor.mu.2", "actor.mu.4"), 4),
            (("actor.mu.0", "actor.mu.2", "actor.mu.4"), 4 + len(TASK_INDICES)),
        )
        for names, input_dim in cases:
            with self.subTest(names=names, input_dim=input_dim), tempfile.TemporaryDirectory() as directory:
                model_path = str(Path(directory) / "model")
                shapes = ((2, input_dim), (2, 2), (1, 2))
                for task_index in TASK_INDICES:
                    value = (task_index + 1) / 10
                    state = {}
                    for name, shape in zip(names, shapes):
                        state[f"{name}.weight"] = torch.full(shape, value)
                        state[f"{name}.bias"] = torch.full((shape[0],), value)
                    state["actor_target.mu.0.weight"] = torch.full(shapes[0], -99.0)

                    buffer = io.BytesIO()
                    torch.save(state, buffer)
                    with zipfile.ZipFile(f"{model_path}-{task_index}.zip", "w") as archive:
                        archive.writestr("policy.pth", buffer.getvalue())

                first, second, third, fourth = load_policies(model_path)
                for policy, expected in ((first, 0.1), (second, 0.2), (third, 0.3), (fourth, 0.4)):
                    for weight, bias in policy:
                        np.testing.assert_allclose(weight, expected, atol=1e-7)
                        np.testing.assert_allclose(bias, expected, atol=1e-7)

                self.assertEqual(make_task_encoding(first).size, input_dim - 4)
                obs = np.zeros((1, input_dim), dtype=np.float32)
                reference = torch.as_tensor(obs)
                for weight, bias in first[:-1]:
                    reference = torch.relu(
                        torch.nn.functional.linear(
                            reference,
                            torch.as_tensor(np.asarray(weight).copy()),
                            torch.as_tensor(np.asarray(bias).copy()),
                        )
                    )
                weight, bias = first[-1]
                reference = 3 * torch.tanh(
                    torch.nn.functional.linear(
                        reference,
                        torch.as_tensor(np.asarray(weight).copy()),
                        torch.as_tensor(np.asarray(bias).copy()),
                    )
                )
                np.testing.assert_allclose(policy_fn(first, obs), reference.numpy(), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
