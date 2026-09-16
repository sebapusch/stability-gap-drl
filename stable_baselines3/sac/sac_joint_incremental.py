from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.off_policy_joint_incremental import OffPolicyJointIncremental


class SAC_JointIncremental(SAC, OffPolicyJointIncremental):
    def __init__(self, buffer_size: int, n_tasks: int, env: GymEnv, balanced_sampling: bool = False, **kwargs) -> None:
        kwargs['env'] = env
        SAC.__init__(self, **kwargs)
        OffPolicyJointIncremental.__init__(self, buffer_size, n_tasks, env, balanced_sampling=balanced_sampling)
