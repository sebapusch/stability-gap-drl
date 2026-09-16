from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.off_policy_joint_incremental import OffPolicyJointIncremental
from stable_baselines3.sacd.sacd import SACD


class SACD_JointIncremental(SACD, OffPolicyJointIncremental):
    """SACD with experience replay across tasks.

    Uses a MultiReplayBuffer that partitions experience by task.
    On task change, resets optimizer state and advances the active
    buffer partition.
    """
    def __init__(self, buffer_size: int, n_tasks: int, env: GymEnv, balanced_sampling: bool = False, **kwargs) -> None:
        kwargs['env'] = env
        SACD.__init__(self, **kwargs)
        OffPolicyJointIncremental.__init__(self, buffer_size, n_tasks, env, balanced_sampling=balanced_sampling)

    def reset_optimizer(self) -> None:
        self.actor.optimizer.state.clear()
        self.critic.optimizer.state.clear()

        # Reset entropy coefficient to its initial value
        if self.ent_coef_optimizer is not None:
            self.log_ent_coef.data.fill_(0.0)
            self.ent_coef_optimizer.state.clear()