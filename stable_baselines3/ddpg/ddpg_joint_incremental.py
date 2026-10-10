from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.off_policy_joint_incremental import (
    OffPolicyJointIncremental,
)
from stable_baselines3.ddpg.ddpg import DDPG


class DDPG_JointIncremental(OffPolicyJointIncremental, DDPG):
    save = DDPG.save

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
        DDPG.__init__(self, **kwargs)
        OffPolicyJointIncremental.__init__(
            self,
            buffer_size,
            n_tasks,
            env,
            balanced_sampling,
            reset_optim_on_task_change=reset_optim_on_task_change
        )

    def reset_optimizer(self) -> None:
        self.policy.actor.optimizer.state.clear()
        self.policy.critic.optimizer.state.clear()

