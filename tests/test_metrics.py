"""Checks for the thesis definitions of P and min-P."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import (
    compute_final_performance_from_data,
    compute_min_acc_from_data,
    get_env_max_return,
    parse_config,
)


TIMESTEPS = [100, 150, 200, 250, 300]
BENCHMARK = ["0", "1", "2"]


def sample_data():
    """One dead seed and four healthy seeds, all evaluated together."""
    task_returns = {
        "0": [0, 60, 40, 70, 80],
        "1": [0, 0, 0, 90, 50],
        "2": [0, 0, 0, 0, 70],
    }
    data = {}
    for seed in range(5):
        columns = {}
        for task, returns in task_returns.items():
            columns[f"time/{task}/total_timesteps"] = TIMESTEPS
            columns[f"eval/{task}/mean_reward"] = (
                [0] * len(returns) if seed == 0 else returns
            )
        data[seed] = pd.DataFrame(columns)
    return data


class ThesisMetricsTests(unittest.TestCase):
    def test_scalar_seed_count_matches_dispatcher(self):
        config = parse_config(
            str(Path(__file__).resolve().parents[1] / "dispatch/experiments/no_encoding/sac_ip_ji.yaml")
        )
        self.assertEqual(config["seeds"], list(range(50)))
        self.assertEqual(config["hp_values"], [["auto", "1.0"]])

    def test_environment_maxima_are_explicit(self):
        self.assertEqual(get_env_max_return("cartpole"), 500)
        self.assertEqual(get_env_max_return("InvertedPendulum"), 1000)
        with self.assertRaises(ValueError):
            get_env_max_return("unknown_environment")

    def test_iqm_precedes_minimum_and_uses_environment_maximum(self):
        data = sample_data()
        minimum = compute_min_acc_from_data(
            data, BENCHMARK, 2, True, 100, 0.95, 100, 100
        )
        final = compute_final_performance_from_data(
            data, BENCHMARK, 1, True, 100, 0.95, 100, 100
        )

        # The dead seed is removed by timestep-wise IQM. Task 0 reaches 40
        # after its boundary; task 1 reaches 50 after its boundary.
        self.assertAlmostEqual(minimum["min-P_mean"], 45)
        self.assertAlmostEqual(final["P_mean"], (80 + 50 + 70) / 3)
        self.assertEqual(minimum["n_seeds"], 5)

    def test_mean_is_also_aggregated_before_minimum(self):
        minimum = compute_min_acc_from_data(
            sample_data(), BENCHMARK, 2, False, 100, 0.95, 100, 100
        )
        self.assertAlmostEqual(minimum["min-P_mean"], (32 + 40) / 2)

    def test_staggered_seed_dips_do_not_become_the_minimum(self):
        data = {}
        for seed in range(4):
            first_task = [100] * len(TIMESTEPS)
            first_task[seed + 1] = 0
            data[seed] = pd.DataFrame({
                "time/0/total_timesteps": TIMESTEPS,
                "eval/0/mean_reward": first_task,
                "time/1/total_timesteps": TIMESTEPS,
                "eval/1/mean_reward": [0, 0, 0, 100, 100],
            })
        minimum = compute_min_acc_from_data(
            data, BENCHMARK, 2, True, 100, 0.95, 100, 100
        )
        # Every seed individually dips to zero, but at every timestep the IQM
        # of the four returns remains 100.
        self.assertAlmostEqual(minimum["min-P_mean"], 100)

    def test_missing_previous_task_or_post_boundary_values_are_undefined(self):
        data = sample_data()
        for frame in data.values():
            frame.drop(columns=["eval/1/mean_reward", "time/1/total_timesteps"], inplace=True)
        minimum = compute_min_acc_from_data(
            data, BENCHMARK, 2, True, 100, 0.95, 100, 100
        )
        final = compute_final_performance_from_data(
            data, BENCHMARK, 1, True, 100, 0.95, 100, 100
        )
        self.assertTrue(np.isnan(minimum["min-P_mean"]))
        self.assertTrue(np.isnan(final["P_mean"]))

        data = sample_data()
        for frame in data.values():
            frame.loc[frame["time/1/total_timesteps"] > 200, "eval/1/mean_reward"] = np.nan
        minimum = compute_min_acc_from_data(
            data, BENCHMARK, 2, True, 100, 0.95, 100, 100
        )
        self.assertTrue(np.isnan(minimum["min-P_mean"]))

    def test_final_performance_uses_a_shared_evaluation_step(self):
        data = sample_data()
        for frame in data.values():
            frame.loc[frame["time/2/total_timesteps"] == 300, "eval/2/mean_reward"] = np.nan
        final = compute_final_performance_from_data(
            data, BENCHMARK, 1, True, 100, 0.95, 100, 100
        )
        self.assertAlmostEqual(final["P_mean"], (70 + 90 + 0) / 3)


if __name__ == "__main__":
    unittest.main()
