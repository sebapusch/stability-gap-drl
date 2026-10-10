import numpy as np
import gymnasium as gym
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from stable_baselines3.common.evaluation import evaluate_policy
from transformation.callbacks import EnvEvalCallback
from dispatch.dispatch_yaml import process_yaml


class OneStepEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)
        self.initial_states = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.np_random.random(dtype=np.float32)
        self.initial_states.append(float(state))
        return np.array([state], dtype=np.float32), {}

    def step(self, action):
        return np.array([0], dtype=np.float32), float(self.initial_states[-1]), True, False, {}


class ZeroPolicy:
    def predict(self, observations, state=None, episode_start=None, deterministic=True):
        return np.zeros(len(observations), dtype=int), state


def test_fixed_episode_seeds_replay_each_initial_state():
    env = OneStepEnv()
    seeds = [11, 22, 33]
    first, _ = evaluate_policy(ZeroPolicy(), env, n_eval_episodes=3,
                               episode_seeds=seeds, return_episode_rewards=True, warn=False)
    second, _ = evaluate_policy(ZeroPolicy(), env, n_eval_episodes=3,
                                episode_seeds=seeds, return_episode_rewards=True, warn=False)

    assert first == second
    assert len(set(first)) == 3


def test_fixed_episode_seeds_require_one_per_episode():
    try:
        evaluate_policy(ZeroPolicy(), OneStepEnv(), n_eval_episodes=2,
                        episode_seeds=[11], warn=False)
    except ValueError as error:
        assert "one seed per evaluation episode" in str(error)
    else:
        raise AssertionError("Expected ValueError for incomplete episode seeds")


def test_callback_logs_both_modes_and_reuses_fixed_starts():
    class Logger:
        def __init__(self):
            self.values = {}

        def record(self, key, value, exclude=None):
            self.values[key] = value

        def dump(self, step):
            pass

    class Model(ZeroPolicy):
        num_timesteps = 0
        logger = Logger()

        def get_vec_normalize_env(self):
            return None

    env = OneStepEnv()
    model = Model()
    callback = EnvEvalCallback([env], n_eval_episodes=3, eval_freq=1,
                               eval_mode=["random", "fixed"], seed=42, verbose=0)
    callback.init_callback(model)
    callback.on_training_start({}, {})
    snapshots = []
    for step in (1, 2):
        model.num_timesteps = step
        assert callback.on_step()
        snapshots.append(dict(model.logger.values))

    assert snapshots[0]["eval/fixed/0/mean_reward"] == snapshots[1]["eval/fixed/0/mean_reward"]
    assert snapshots[0]["eval/random/0/mean_reward"] != snapshots[1]["eval/random/0/mean_reward"]
    assert "eval/fixed/0/mean_ep_length" in snapshots[0]
    assert "eval/random/0/mean_ep_length" in snapshots[0]


def test_yaml_dispatch_passes_both_modes():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "experiment.yaml"
        path.write_text("project: test\neval_mode: [random, fixed]\n")
        output = StringIO()
        with redirect_stdout(output):
            process_yaml(str(path), dry=True)
    assert "--eval_mode random fixed" in output.getvalue()
