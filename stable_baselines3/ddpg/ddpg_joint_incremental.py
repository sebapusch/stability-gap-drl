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
        **kwargs,
    ) -> None:
        kwargs["env"] = env
        DDPG.__init__(self, **kwargs)
        super().__init__(buffer_size, n_tasks, env, balanced_sampling)

    def reset_optimizer(self) -> None:
        self.policy.actor.optimizer.state.clear()
        self.policy.critic.optimizer.state.clear()

