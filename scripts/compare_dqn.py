from pathlib import Path
from os.path import join, abspath

import numpy as np
import torch
from gymnasium.envs.classic_control import CartPoleEnv
from highway_env.vehicle.uncertainty import prediction

from common import compute_per_env_final_score
from transformation.benchmarks.transformed_env_benchmark import TransformedEnvBenchmark
from stable_baselines3 import DQN

GAMMA = 0.99
EPISODE_LENGTH = 500
BENCHMARK = ['V1', 'V2', 'V3']
OUTPUT_PATH = abspath(join(__file__, '..', '..', 'output'))
SEEDS = range(10)


def load_scores() -> tuple[dict[int, int], int]:
    scores = {}

    for env in BENCHMARK:
        for seed in SEEDS:
            score = compute_per_env_final_score(
                'dqn_linear_interpolation',
                hp_combo={},
                seed=seed,
                benchmark=['V1', 'V2', 'V3'],
                eval_env=env,
                n_smooth=5,
                data_dir=Path(join(OUTPUT_PATH, 'output')),
                all_ablation_keys=['seed'],
            )
            if seed in scores:
                scores[seed].append(score)
            else:
                scores[seed] = [score]

    best_seed = 0
    for seed in scores:
        scores[seed] = sum(scores[seed]) / len(BENCHMARK)
        print(f'avg seed {seed}: {scores[seed]}')
        if scores[seed] > scores[best_seed]:
            best_seed = seed

    return scores, best_seed

def load_models(seeds: list[int], env: str) -> dict[int, DQN]:
    models: dict[int, DQN] = {}
    for seed in seeds:
        models[seed] = DQN.load(join(OUTPUT_PATH, 'models', 'dqn_linear_interpolation', f'dqn_linear_interpolation-s_{seed}-{env}'))

    return models

def main():
    scores, best_seed = load_scores()
    benchmark = TransformedEnvBenchmark(
        CartPoleEnv,  # type: ignore
        [1, 2, 3],
        True,
        42,
        EPISODE_LENGTH,
    )



    _, env_test = benchmark.make()
    env_test = env_test[1]

    state, _  = env_test.reset()

    models = load_models(seeds=list(SEEDS), env='V2')

    q_values = {s: 0 for s in SEEDS}
    target_q_value = sum(GAMMA ** t for t in range(EPISODE_LENGTH))
    action = np.argmax(models[best_seed].predict(state)[0])

    for seed in scores:
        pred = models[seed].q_net(torch.tensor(state).unsqueeze(0))

        pred = float(pred[0][action])

        q_values[seed] = pred



    print(f'target: {target_q_value}')
    for seed in SEEDS:
        print(f'seed {seed}: {q_values[seed]}')

if __name__ == '__main__':
    main()