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
        **kwargs,
    ) -> None:
        kwargs["env"] = env
        DQN.__init__(self, **kwargs)
        OffPolicyJointIncremental.__init__(
            self,
            buffer_size,
            n_tasks,
            env,
            balanced_sampling=balanced_sampling,
            reset_optim_on_task_change=reset_optim_on_task_change
        )

    def reset_optimizer(self) -> None:
        self.policy.optimizer.state.clear()
