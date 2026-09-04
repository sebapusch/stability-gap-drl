import numpy as np
from gymnasium import Env
from gymnasium.wrappers import TimeLimit

from transformation.benchmarks.wrappers import (
    ObsLinearTransform,
    ObsSpaceInf,
    OneHotWrapper,
)
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv


def random_orthogonal(seed: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(size, size))

    q, r = np.linalg.qr(m)

    signs = np.sign(np.diag(r))
    q = q * signs

    return q.astype(np.float64)

def _make_task_seeds(seed: int, num_tasks: int) -> list[int]:
    return [
        num_tasks * seed + i for i in range(num_tasks)
    ]

def _make_transformation_matrices(seed: int, dimension: int, num_tasks: int) -> list[np.ndarray]:
    return [
        random_orthogonal(seed, dimension) for seed in _make_task_seeds(seed, num_tasks)
    ]


class TransformedEnvBenchmark:
    def __init__(
        self,
        env_class: type[Env],
        dimension: int,
        seed: int = 42,
        num_tasks: int = 3,
        encode_task: bool = True,
        time_limit: int | None = None,
    ) -> None:
        self.env_class = env_class
        self.dimension = dimension
        self.seed = seed
        self.transformation_matrices = _make_transformation_matrices(seed, dimension, num_tasks)
        self.num_tasks = num_tasks
        self.encode_task = encode_task
        self.time_limit = time_limit


    def make_single(self, ix: int, test: bool = False, **env_kwargs) -> Env:
        env = self.env_class(**env_kwargs)  # type: ignore

        env = ObsSpaceInf(env)
        env = ObsLinearTransform(env, self.transformation_matrices[ix])
        if self.encode_task:
            env = OneHotWrapper(env, ix, self.num_tasks)
        if self.time_limit is not None:
            env = TimeLimit(env, self.time_limit)

        seed = self.seed + (1 if test else 0)
        env.reset(seed=seed)
        env.action_space.seed(seed)

        return env

    def make_train(self, **env_kwargs) -> list[Env]:
        return [self.make_single(ix, False, **env_kwargs) for ix in range(self.num_tasks)]

    def make_test(self, **env_kwargs) -> list[Env]:
        return [self.make_single(ix, True, **env_kwargs) for ix in range(self.num_tasks)]

    def make(self, **env_kwargs) -> tuple[list[Env], list[Env]]:
        return (
            self.make_train(**env_kwargs),
            self.make_test(**env_kwargs),
        )

    def make_single_vec(self, n_env: int, version: int, test: bool = False, dummy: bool = True, **env_kwargs) -> VecEnv:
        def mk_vec() -> Env:
            return self.make_single(version, test, **env_kwargs)

        return (DummyVecEnv if dummy else SubprocVecEnv)([mk_vec for _ in range(n_env)])

    def make_train_vec(self, n_envs: int, **env_kwargs) -> list[VecEnv]:
        return [self.make_single_vec(n_envs, ix, False, **env_kwargs) for ix in range(self.num_tasks)]

    def make_test_vec(self, n_envs: int, **env_kwargs) -> list[VecEnv]:
        return [self.make_single_vec(n_envs, ix, True, **env_kwargs) for ix in range(self.num_tasks)]

    def make_vec(self, n_envs: int, **env_kwargs) -> tuple[list[VecEnv], list[VecEnv]]:
        return (
            self.make_train_vec(n_envs, **env_kwargs),
            self.make_test_vec(n_envs, **env_kwargs),
        )

    def __len__(self) -> int:
        return self.num_tasks
