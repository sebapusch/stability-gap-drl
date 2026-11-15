from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.off_policy_joint_incremental import (
    OffPolicyJointIncremental,
)


class DQN_JointIncremental(OffPolicyJointIncremental, DQN):
    save = DQN.save

    def __init__(
        self,
        buffer_size: int,
        n_tasks: int,
        env: GymEnv,
        balanced_sampling: bool = False,
        reset_optim_on_task_change: bool = True,
        reset_buffer_on_task_change: bool = True,
        task_learning_starts: int = 0,
        reset_epsilon_on_task_change: bool = False,
        **kwargs,
    ) -> None:
        self.reset_epsilon_on_task_change = reset_epsilon_on_task_change
        self._epsilon_task_start = 0
        self._epsilon_task_horizon = None
        kwargs["env"] = env
        DQN.__init__(self, **kwargs)
        OffPolicyJointIncremental.__init__(
            self,
            buffer_size,
            n_tasks,
            env,
            balanced_sampling=balanced_sampling,
            reset_optim_on_task_change=reset_optim_on_task_change,
            reset_buffer_on_task_change=reset_buffer_on_task_change,
            task_learning_starts=task_learning_starts,
        )

    def reset_optimizer(self) -> None:
        self.policy.optimizer.state.clear()

    def on_task_change(self, task_ix, env, logger) -> None:
        super().on_task_change(task_ix, env, logger)
        self._epsilon_task_start = self.num_timesteps
        self._epsilon_task_horizon = None
        if self.reset_epsilon_on_task_change and task_ix > 0:
            self.exploration_rate = self.exploration_initial_eps

    def learn(self, total_timesteps, *args, **kwargs):
        # Capture the stage budget once so repeated learn calls do not restart decay.
        if self._epsilon_task_horizon is None:
            self._epsilon_task_horizon = max(total_timesteps, 1)
        return super().learn(total_timesteps, *args, **kwargs)

    def _on_step(self) -> None:
        if not self.reset_epsilon_on_task_change or self.task_ix == 0:
            return super()._on_step()
        # Reuse DQN's target update and logging, changing only epsilon's clock.
        # The JI counter advances per task, not once per task in the active set.
        global_progress = self._current_progress_remaining
        self._current_progress_remaining = max(
            0.0,
            1.0 - (self.num_timesteps - self._epsilon_task_start) / self._epsilon_task_horizon,
        )
        try:
            super()._on_step()
        finally:
            self._current_progress_remaining = global_progress
