import gymnasium as gym
from gymnasium import Env
import numpy as np
from numpy.typing import NDArray


class OneHotWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: Env, task_idx: int, num_tasks: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        env_lb = env.observation_space.low
        env_ub = env.observation_space.high
        one_hot_ub = np.ones(num_tasks)
        one_hot_lb = np.zeros(num_tasks)

        self.one_hot = np.zeros(num_tasks)
        self.one_hot[task_idx] = 1.0

        self._observation_space = gym.spaces.Box(
            np.concatenate([env_lb, one_hot_lb]), np.concatenate([env_ub, one_hot_ub])
        )

    def observation(self, obs: NDArray) -> NDArray:
        return np.concatenate([obs, self.one_hot])


class ObsSpaceInf(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=env.observation_space.shape, dtype=np.float64
        )

    def observation(self, obs: NDArray) -> NDArray:
        return obs


class ObsLinearTransform(gym.ObservationWrapper):
    def __init__(self, env: Env, projection: np.ndarray, bias: np.ndarray | None = None):
        super().__init__(env)

        assert projection.shape == (env.observation_space.shape[-1],
                                    env.observation_space.shape[-1])

        self.projection = np.asarray(projection, dtype=np.float64)
        self.bias = np.zeros(env.observation_space.shape[-1], dtype=np.float64) \
            if bias is None \
            else np.asarray(bias, dtype=np.float64)

    def observation(self, obs: NDArray) -> NDArray:
        obs = np.asarray(obs, dtype=np.float64)
        return self.projection @ obs + self.bias

