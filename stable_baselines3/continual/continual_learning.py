import io
import pathlib
from typing import Protocol

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv


class ContinualLearning(Protocol):
    """
    Interface for continual-learning methods.
    """

    def on_task_change(
        self,
        task_ix: int,
        env: GymEnv,
        logger: Logger,
    ) -> None:
        """Called between tasks to update internal CRL state.

        Implementations typically populate the expert buffer with outputs from
        the current policy evaluated on the replay buffer.
        """
        ...

    def learn(
        self,
        total_timesteps: int,
        callback: CallbackList,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> None: ...


    def save(
        self,
        path: str,
    ) -> None: ...
