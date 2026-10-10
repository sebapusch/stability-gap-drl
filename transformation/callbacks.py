from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from transformation.benchmarks.transformed_env_benchmark import TransformedEnvBenchmark
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (EventCallback,
                                                BaseCallback,
                                                CallbackList,
                                                sync_envs_normalization,
                                                evaluate_policy)
from stable_baselines3.common.type_aliases import GymEnv


def make_callbacks(
        benchmark: TransformedEnvBenchmark,
        envs_test: list[GymEnv],
        eval_freq: int | list[tuple[int, int]],
        n_eval_episodes: int,
        video_freq: int,
        eval_all: bool,
        eval_mode: str | list[str] = "random",
        seed: int = 42,
) -> Callable[[int], CallbackList]:
    wandb_callback = WandbCallback(gradient_save_freq=1000, verbose=2)
    modes = [eval_mode] if isinstance(eval_mode, str) else list(eval_mode)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"random", "fixed"}:
        raise ValueError("eval_mode must be 'random', 'fixed', or both")
    if n_eval_episodes < 1:
        raise ValueError("n_eval_episodes must be positive")
    fixed_episode_seeds = None
    if "fixed" in modes:
        if any(hasattr(env, "num_envs") for env in envs_test):
            raise ValueError("fixed evaluation requires a single Gym environment per task")
        fixed_episode_seeds = [
            np.random.SeedSequence([seed, eval_ix]).generate_state(n_eval_episodes).tolist()
            for eval_ix in range(len(envs_test))
        ]
        for env, seeds in zip(envs_test, fixed_episode_seeds):
            for episode_seed in seeds:
                env.reset(seed=episode_seed)

    def make(env_ix: int) -> CallbackList:
        """
        List of callbacks for each continual learning environment iteration

        evaluate current environment + all previous ones
        """
        callbacks: list[BaseCallback] = [wandb_callback]

        if video_freq > 0:
            video_env = benchmark.make_single(env_ix, test=True, render_mode='rgb_array')
            callbacks.append(
                RegisterVideoCallback(video_freq, video_env),
            )

        n_eval_envs = len(benchmark) if eval_all else env_ix + 1
        callbacks.append(
            EnvEvalCallback(
                eval_envs=envs_test[:n_eval_envs],
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                eval_mode=eval_mode,
                seed=seed,
                fixed_episode_seeds=fixed_episode_seeds[:n_eval_envs] if fixed_episode_seeds is not None else None,
            )
        )

        return CallbackList(callbacks)

    return make


class EnvEvalCallback(EventCallback):
    def __init__(
            self,
            eval_envs: list[GymEnv],
            callback_on_new_best: BaseCallback | None = None,
            n_eval_episodes: int = 5,
            eval_freq: int | list[tuple[int, int]] = 10000,
            deterministic: bool = True,
            verbose: int = 1,
            eval_mode: str | list[str] = "random",
            seed: int = 42,
            fixed_episode_seeds: list[list[int]] | None = None,
    ):
        super().__init__()

        if not eval_envs:
            raise ValueError("EnvEvalCallback requires at least one evaluation environment")

        self.eval_envs = eval_envs
        self.callback_on_new_best = callback_on_new_best
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_modes = [eval_mode] if isinstance(eval_mode, str) else list(eval_mode)
        if not self.eval_modes or len(set(self.eval_modes)) != len(self.eval_modes) or set(self.eval_modes) - {"random", "fixed"}:
            raise ValueError("eval_mode must be 'random', 'fixed', or both")
        if n_eval_episodes < 1:
            raise ValueError("n_eval_episodes must be positive")
        self.fixed_episode_seeds = fixed_episode_seeds or []
        self.random_episode_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xE1A1])) if len(self.eval_modes) > 1 else None
        if "fixed" in self.eval_modes and fixed_episode_seeds is None:
            for eval_ix, env in enumerate(eval_envs):
                if hasattr(env, "num_envs"):
                    raise ValueError("fixed evaluation requires a single Gym environment per task")
                seeds = np.random.SeedSequence([seed, eval_ix]).generate_state(n_eval_episodes).tolist()
                self.fixed_episode_seeds.append(seeds)
                # Sample the initial states once when this run's callback is built.
                for episode_seed in seeds:
                    env.reset(seed=episode_seed)
        self.deterministic = deterministic
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []
        self.best_mean_rewards = {mode: [0.0] * len(eval_envs) for mode in self.eval_modes}
        self.verbose = verbose
        self.task_start_timestep = 0
        self.last_eval_timestep: int | None = None

        if isinstance(self.eval_freq, list):
            assert len(self.eval_freq) > 0

    def _on_training_start(self) -> None:
        """Use a fresh, task-local evaluation schedule for every learn call."""
        self.task_start_timestep = self.num_timesteps
        self.last_eval_timestep = None

        super()._on_training_start()

    def _log_success_callback(self, locals_: dict[str, Any], _: dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_['info']

        if locals_['done']:
            maybe_is_success = info.get('is_success')
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if not self._is_eval_step():
            return continue_training

        for eval_ix, eval_env in enumerate(self.eval_envs):
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            for mode in self.eval_modes:
                self._is_success_buffer = []
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=False,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=False,
                    callback=self._log_success_callback,
                    episode_seeds=(
                        self.fixed_episode_seeds[eval_ix] if mode == "fixed" else
                        self.random_episode_rng.integers(0, 2**32, size=self.n_eval_episodes).tolist()
                        if self.random_episode_rng is not None else None
                    ),
                )

                mean_reward = float(np.mean(episode_rewards))
                mean_ep_length = float(np.mean(episode_lengths))
                self.last_mean_reward = mean_reward

                self.logger.record(f'eval/{mode}/{eval_ix}/mean_reward', mean_reward)
                self.logger.record(f'eval/{mode}/{eval_ix}/mean_ep_length', mean_ep_length)

                if mean_reward > self.best_mean_rewards[mode][eval_ix]:
                    self.best_mean_rewards[mode][eval_ix] = mean_reward
                    if self.verbose >= 1:
                        print(f"New best {mode} mean reward for evaluation environment {eval_ix}!")

                    if self.callback_on_new_best is not None:
                        continue_training = (
                            self.callback_on_new_best.on_step() and continue_training
                        )

            self.logger.record(
                f"time/{eval_ix}/total_timesteps",
                self.num_timesteps,
                exclude="tensorboard",
            )

        # Commit all evaluation metrics in one W&B history row.
        self.logger.dump(self.num_timesteps)

        # Trigger callback after every evaluation, if needed
        if self.callback is not None:
            continue_training = continue_training and self._on_event()

        return continue_training

    def _is_eval_step(self) -> bool:
        # Joint-incremental training invokes callbacks once for every replay
        # environment while temporarily restoring the same model timestep.
        # Only one of those invocations should perform evaluation and logging.
        if self.num_timesteps == self.last_eval_timestep:
            return False

        task_timestep = self.num_timesteps - self.task_start_timestep
        if task_timestep <= 0:
            return False

        if isinstance(self.eval_freq, int):
            freq = self.eval_freq
        else:
            # A schedule entry applies through its max step (inclusive).  Use
            # task-local progress so a newly-created callback starts the same
            # schedule for every task, even though model.num_timesteps is global.
            freq = self.eval_freq[-1][1]
            for max_step, candidate_freq in self.eval_freq:
                if task_timestep <= max_step:
                    freq = candidate_freq
                    break

        should_evaluate = freq > 0 and task_timestep % freq == 0
        if should_evaluate:
            self.last_eval_timestep = self.num_timesteps

        return should_evaluate


class TrackQNet(EventCallback):
    model: DQN

    def __init__(
        self,
        eval_id: str,
        env: GymEnv,
        eval_freq: int | list[tuple[int, int]],
    ) -> None:
        super().__init__()

        self.initial_state = torch.tensor(env.reset()[0]).unsqueeze(0)
        self.eval_freq = eval_freq
        self.cur_eval_freq_ix = 0
        self.eval_id = eval_id

    def _on_step(self) -> bool:
        if not self._is_eval_step():
            return True

        pred = self.model.q_net_target(self.initial_state)

        for action, q_val in enumerate(pred.squeeze()):
            self.logger.record(f'eval/{self.eval_id}/init_q_val/{action}', float(q_val))

        return True


    def _is_eval_step(self) -> bool:
        if isinstance(self.eval_freq, int):
            freq = self.eval_freq
        else:
            max_step, freq = self.eval_freq[self.cur_eval_freq_ix]

            if max_step == self.num_timesteps and self.cur_eval_freq_ix < len(self.eval_freq) - 1:
                self.cur_eval_freq_ix += 1

        return freq > 0 and self.num_timesteps % freq == 0



class RegisterVideoCallback(EventCallback):
    def __init__(self, frequency: int, env: GymEnv, seed: int = 1):
        self.frequency = frequency
        self.env = env
        self.seed = seed

        super().__init__()

    def _on_step(self) -> bool:
        if self.n_calls < 1 or self.n_calls % self.frequency != 0:
            return True

        frames = []

        obs, _ = self.env.reset(seed=self.seed)

        for _ in range(500):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)

            frame = self.env.render()

            frame = np.moveaxis(frame, -1, 0).astype(np.uint8)
            frames.append(frame)

            if terminated or truncated:
                break
        frames_array = np.array(frames)

        wandb.log({'video': wandb.Video(frames_array, fps=30, format='mp4')})

        return True
