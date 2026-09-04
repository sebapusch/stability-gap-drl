from typing import Any

from stable_baselines3.ddpg.ddpg_fine_tune import DDPG_FineTune
from transformation.args import get_args, parse_eval_freq
from transformation.callbacks import make_callbacks
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop

import wandb
from transformation.benchmarks.transformed_env_benchmark import TransformedEnvBenchmark, EnvFactory
from transformation.common import make_logger, model_weight_path
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual import ContinualLearning
from stable_baselines3.ddpg.ddpg_joint_incremental import DDPG_JointIncremental
from stable_baselines3.dqn.dqn_fine_tune import DQN_FineTune
from stable_baselines3.dqn.dqn_joint_icremental import DQN_JointIncremental
from stable_baselines3.sac.sac_fine_tune import SAC_FineTune
from stable_baselines3.sac.sac_joint_incremental import SAC_JointIncremental
from stable_baselines3.sacd.sacd_fine_tune import SACD_FineTune
from stable_baselines3.sacd.sacd_joint_incremental import SACD_JointIncremental


OptimizerConfig = tuple[type[Optimizer], dict[str, Any]]


# Env Class, State Vector Dimensions, Max Episode
ENV_REGISTRY: dict[str, tuple[EnvFactory, int, int]] = {
    "cartpole": (CartPoleEnv, 4, 500),
    "inverted_pendulum": (InvertedPendulumEnv, 4, 1000),
}

OPTIMIZERS: dict[str, OptimizerConfig] = {
    "adam": (Adam, {}),
    "sgd": (SGD, {}),
    "sgd_momentum": (SGD, {"momentum": 0.9}),
    "rmsprop": (RMSprop, {}),
    "adamw": (AdamW, {}),
}


def get_benchmark(
    env: str,
    seed: int,
    encode: bool = True,
) -> TransformedEnvBenchmark:
    env_cls, dimension, time_limit = ENV_REGISTRY[env]

    return TransformedEnvBenchmark(
        env_class=env_cls,
        dimension=dimension,
        encode_task=encode,
        seed=seed,
        time_limit=time_limit,
    )



def _build_dqn(
    train_env: GymEnv,
    *,
    lr: float,
    gamma: float,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    target_update: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_frac: float,
    seed: int,
    method: str,
    tau: float,
    network_size: int,
    n_tasks: int,
    balanced_sampling: bool,
    policy_kwargs: dict[str, Any],
    exploration_strategy: str = "eps-greedy",
) -> ContinualLearning:
    policy_kwargs["net_arch"] = [network_size, network_size]

    common_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=lr,
        learning_starts=learning_starts,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_interval=target_update,
        exploration_initial_eps=epsilon_start,
        exploration_final_eps=epsilon_end,
        exploration_fraction=epsilon_decay_frac,
        policy_kwargs=policy_kwargs,
        seed=seed,
        tau=tau,
        exploration_strategy=exploration_strategy,
    )

    match method:
        case "fine_tune":
            return DQN_FineTune(**common_kwargs)
        case "joint_incremental":
            return DQN_JointIncremental(
                n_tasks=n_tasks,
                balanced_sampling=balanced_sampling,
                **common_kwargs,
            )
        case _:
            raise ValueError(f'Unknown method "{method}"')


def _build_sacd(
    train_env: GymEnv,
    *,
    lr: float,
    gamma: float,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    seed: int,
    method: str,
    ent_coef: float | None,
    network_size: int,
    n_tasks: int,
    balanced_sampling: bool,
    policy_kwargs: dict[str, Any],
) -> ContinualLearning:
    policy_kwargs["net_arch"] = [network_size, network_size]

    common_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=lr,
        learning_starts=learning_starts,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        seed=seed,
        ent_coef="auto" if ent_coef is None else ent_coef,
    )

    match method:
        case "fine_tune":
            return SACD_FineTune(**common_kwargs)
        case "joint_incremental":
            return SACD_JointIncremental(
                n_tasks=n_tasks,
                balanced_sampling=balanced_sampling,
                **common_kwargs,
            )
        case _:
            raise ValueError(f'Unknown method "{method}"')


def _build_sac(
    train_env: GymEnv,
    *,
    lr: float,
    gamma: float,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    seed: int,
    method: str,
    ent_coef: float | None,
    network_size: int,
    n_tasks: int,
    balanced_sampling: bool,
    policy_kwargs: dict[str, Any],
) -> ContinualLearning:
    policy_kwargs["net_arch"] = [network_size, network_size]

    common_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=lr,
        learning_starts=learning_starts,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        seed=seed,
        ent_coef="auto" if ent_coef is None else ent_coef,
    )

    match method:
        case "fine_tune":
            return SAC_FineTune(**common_kwargs)
        case "joint_incremental":
            return SAC_JointIncremental(
                n_tasks=n_tasks,
                balanced_sampling=balanced_sampling,
                **common_kwargs,
            )
        case _:
            raise ValueError(f'Unknown method "{method}"')


def _build_ddpg(
    train_env: GymEnv,
    *,
    lr: float,
    gamma: float,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    seed: int,
    method: str,
    network_size: int,
    n_tasks: int,
    balanced_sampling: bool,
    policy_kwargs: dict[str, Any],
) -> ContinualLearning:
    policy_kwargs["net_arch"] = [network_size, network_size]

    common_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=lr,
        learning_starts=learning_starts,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )

    match method:
        case 'fine_tune':
            return DDPG_FineTune(**common_kwargs)
        case 'joint_incremental':
            return DDPG_JointIncremental(
                balanced_sampling=balanced_sampling,
                n_tasks=n_tasks,
                **common_kwargs,
            )
        case _:
            raise ValueError(f"Invalid method {method}")


def train_continual(
    benchmark: TransformedEnvBenchmark,
    envs_train: list[GymEnv],
    envs_test: list[GymEnv],
    model: ContinualLearning,
    tags: list[str],
    name_prefix: str,
    project: str,
    eval_freq: int | list[tuple[int, int]],
    video_freq: int,
    n_eval_episodes: int,
    config: dict[str, Any],
    eval_all: bool,
    total_timesteps: int,
    store_weights: bool,
) -> None:
    run = wandb.init(
        name=f"{name_prefix}",
        project=project,
        config=config,
        tags=tags,
    )

    loggers = make_logger(project, run.name)

    for ix, train_env in enumerate(envs_train):
        model.on_task_change(ix, train_env, loggers)

        callbacks = make_callbacks(
            benchmark=benchmark,
            envs_test=envs_test,
            eval_freq=eval_freq,
            video_freq=video_freq,
            n_eval_episodes=n_eval_episodes,
            eval_all=eval_all,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks(ix),
            reset_num_timesteps=False,
        )

        if store_weights:
            model.save(model_weight_path(project, run.name, ix))

    run.finish()


def train_multitask(
    benchmark: TransformedEnvBenchmark,
    envs_train: list[GymEnv],
    envs_test: list[GymEnv],
    model: ContinualLearning,
    tags: list[str],
    name_prefix: str,
    project: str,
    eval_freq: int | list[tuple[int, int]],
    video_freq: int,
    n_eval_episodes: int,
    config: dict[str, Any],
    total_timesteps: int,
) -> None:
    tags += ["multitask"]

    for ix, train_env in enumerate(envs_train[:-1]):
        model.on_task_change(ix, train_env, make_logger(project, None))

    run = wandb.init(
        name=f"{name_prefix}",
        project=project,
        config=config,
        tags=tags,
    )

    model.on_task_change(
        len(envs_train) - 1, envs_train[-1], make_logger(project, run.name)
    )

    callbacks = make_callbacks(
        benchmark=benchmark,
        envs_test=envs_test,
        eval_freq=eval_freq,
        video_freq=video_freq,
        n_eval_episodes=n_eval_episodes,
        eval_all=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks(len(envs_train) - 1),
        reset_num_timesteps=True,
    )

    run.finish()


def main(
    env: str = "cartpole",
    seed: int = 42,
    name_prefix: str = "",
    project: str = "cartpole",
    method: str = "fine_tune",
    eval_freq: int | list[tuple[int, int]] = 500,
    video_freq: int = 0,
    n_eval_episodes: int = 15,
    lr: float = 3e-4,
    gamma: float = 0.99,
    buffer_size: int = 50_000,
    batch_size: int = 128,
    target_update: int = 1000,
    learning_starts: int = 1000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_frac: float = 0.1,
    total_timesteps: int = 200_000,
    encode_task: bool = False,
    balanced_sampling: bool = False,
    eval_all: bool = True,
    bc_loss_fn: str = "kl",
    algorithm: str = "dqn",
    ent_coef: float | str | None = None,
    dqn_tau: float = 1.0,
    network_size: int | None = None,
    optimizer: str = "adam",
    mode: str = "continual",
    store_weights: bool = False,
    exploration_strategy: str = "eps-greedy",
) -> None:
    bench = get_benchmark(env, seed, encode_task)

    envs_train, envs_test = bench.make()

    common_build_kwargs = dict(
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        seed=seed,
        method=method,
        network_size=network_size,
        n_tasks=len(bench),
        balanced_sampling=balanced_sampling,
        policy_kwargs=dict(
            optimizer_class=OPTIMIZERS[optimizer][0],
            optimizer_kwargs=OPTIMIZERS[optimizer][1],
        ),
    )

    dqn_build_kwargs = dict(
        **common_build_kwargs,
        target_update=target_update,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_frac=epsilon_decay_frac,
        tau=dqn_tau,
        exploration_strategy=exploration_strategy,
    )

    sac_build_kwargs = dict(
        **common_build_kwargs,
        bc_loss_fn=bc_loss_fn,
        ent_coef=ent_coef,
    )

    sacd_build_kwargs = dict(
        **common_build_kwargs,
        ent_coef=ent_coef,
    )

    ddpg_build_kwargs = dict(
        **common_build_kwargs,
    )

    train_env_init = envs_train[0]
    match algorithm:
        case "dqn":
            config = dqn_build_kwargs
            model = _build_dqn(train_env_init, **dqn_build_kwargs)
        case "sac":
            config = sac_build_kwargs
            model = _build_sac(train_env_init, **sac_build_kwargs)
        case "sacd":
            config = sacd_build_kwargs
            model = _build_sacd(train_env_init, **sacd_build_kwargs)
        case "ddpg":
            config = ddpg_build_kwargs
            model = _build_ddpg(train_env_init, **ddpg_build_kwargs)
        case _:
            raise ValueError(f'Unknown algorithm "{algorithm}"')

    match mode:
        case "continual":
            train_continual(
                benchmark=bench,
                envs_train=envs_train,
                envs_test=envs_test,
                model=model,
                tags=[f"s-{str(seed)}", method, optimizer, f"lr-{str(lr)}"],
                name_prefix=name_prefix,
                project=project,
                eval_freq=eval_freq,
                video_freq=video_freq,
                n_eval_episodes=n_eval_episodes,
                config=config,
                eval_all=eval_all,
                total_timesteps=total_timesteps,
                store_weights=store_weights,
            )
        case "multitask":
            train_multitask(
                benchmark=bench,
                envs_train=envs_train,
                envs_test=envs_test,
                model=model,
                tags=[f"s-{str(seed)}", method, optimizer, f"lr-{str(lr)}"],
                name_prefix=name_prefix,
                project=project,
                eval_freq=eval_freq,
                video_freq=video_freq,
                n_eval_episodes=n_eval_episodes,
                config=config,
                total_timesteps=total_timesteps,
            )
        case _:
            raise ValueError(f'Unknown mode "{mode}"')


if __name__ == "__main__":
    args = vars(get_args())
    args["eval_freq"] = parse_eval_freq(args["eval_freq"], args["total_timesteps"])
    if type(args["ent_coef"]) == str:
        if args["ent_coef"] == "auto":
            args["ent_coef"] = None
        else:
            args["ent_coef"] = float(args["ent_coef"])
    main(**args)
