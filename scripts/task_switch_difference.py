import os
from itertools import permutations, combinations

import numpy as np
import gymnasium as gym
from tqdm import tqdm

from transformation.benchmarks.transformed_env_benchmark import TransformedEnvBenchmark


N = 100_000
BENCHMARKS = {
    'easy': [1, 2, 3],
    'hard': [2, 8, 9],
}
ENVS = ['CartPole-v1', 'InvertedPendulum-v5']
PATH = os.path.abspath(os.path.join(__file__, '..', '..', 'output', 'task_switch_difference.md'))


def collect_samples(n: int, env_name: str, seed: int = 42) -> np.ndarray:
    env = gym.make(env_name)
    env.action_space.seed(seed=seed)

    observations = np.zeros((n, 4))

    env = gym.make(env_name)

    obs, info = env.reset(seed=seed)

    for i in tqdm(range(n)):
        action = env.action_space.sample()
        observations[i] = obs

        obs, reward, terminated, truncated, info = env.step(action)

        if truncated or terminated:
            obs, _ = env.reset()

    env.close()

    return observations


def compute_distances(observations: np.ndarray, tasks: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[str, int]:
    distances = {}

    for a, b in combinations(tasks.keys(), 2):
        proj_a, bias_a = tasks[a]
        proj_b, bias_b = tasks[b]

        obs_a = (proj_a @ observations.T).T + bias_a
        obs_b = (proj_b @ observations.T).T + bias_b

        f_dist = (np.linalg.norm(obs_a - obs_b, ord='fro')) / np.sqrt(observations.shape[0])
        distances[f"{a}-{b}"] = f_dist

    return distances


def main() -> None:
    content = ''

    for env_name in ENVS:
        observations = collect_samples(N, env_name)

        for difficulty in BENCHMARKS:
            benchmark = TransformedEnvBenchmark(gym.Env, BENCHMARKS[difficulty], False, size=4)

            distances = compute_distances(observations, benchmark.matrices)

            dist_str = '\n'.join(f'{key}: {val:.2f}' for key, val in distances.items())
            content += f'\n\nbenchmark {env_name} {difficulty}:\n{dist_str}\n'

    with open(PATH, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    main()