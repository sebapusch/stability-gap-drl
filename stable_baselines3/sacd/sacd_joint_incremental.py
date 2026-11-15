import torch as th

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.off_policy_joint_incremental import OffPolicyJointIncremental
from stable_baselines3.sacd.sacd import SACD


class SACD_JointIncremental(OffPolicyJointIncremental, SACD):
    save = SACD.save

    def __init__(
            self,
            buffer_size: int,
            n_tasks: int,
            env: GymEnv,
            balanced_sampling: bool = False,
            reset_optim_on_task_change: bool = True,
            reset_buffer_on_task_change: bool = True,
            task_learning_starts: int = 0,
            reset_alpha_on_task_change: bool = False,
            **kwargs
    ) -> None:
        kwargs['env'] = env
        SACD.__init__(self, **kwargs)
        self.reset_alpha_on_task_change = reset_alpha_on_task_change
        # Preserve the actual initialized value, including custom initialization.
        self._initial_log_ent_coef = (
            self.log_ent_coef.detach().item() if self.log_ent_coef is not None else None
        )
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
        self.actor.optimizer.state.clear()
        self.critic.optimizer.state.clear()

        # Clear optimizer state without changing the coefficient value.
        if self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.state.clear()

    def on_task_change(self, task_ix, env, logger) -> None:
        super().on_task_change(task_ix, env, logger)
        if task_ix > 0 and self.reset_alpha_on_task_change and self.log_ent_coef is not None:
            # Reset in place to preserve the entropy optimizer's parameter reference.
            # Optimizer-state resets remain controlled by reset_optim_on_task_change.
            with th.no_grad():
                self.log_ent_coef.fill_(self._initial_log_ent_coef)
            self.ent_coef_tensor = th.exp(self.log_ent_coef.detach())
