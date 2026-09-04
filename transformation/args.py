import argparse
from argparse import Namespace

METHODS = [
    "fine_tune",
    "joint_incremental",
]
ENVS = ["cartpole", "inverted_pendulum"]
OPTIMIZERS = ["adam", "sgd", "rmsprop", "sgd_momentum", "adamw"]
ALGORITHMS = ["dqn", "sacd", "sac", "ddpg"]
MODES = ["continual", "multitask"]


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", choices=ENVS, default=ENVS[0], type=str, help="Environment to use: cartpole (DQN) or inverted_pendulum (SAC)")
    parser.add_argument("--algorithm", default="dqn", type=str, choices=ALGORITHMS)
    parser.add_argument("--optimizer", default="adam", type=str, choices=OPTIMIZERS)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--name_prefix", default="", type=str)
    parser.add_argument("--project", default="cartpole", type=str)
    parser.add_argument("--method", choices=METHODS, default=METHODS[0], type=str)

    parser.add_argument("--encode_task", action="store_true", help="Append a one-hot encoding of the current task index to the observation")
    parser.add_argument("--balanced_sampling", action="store_true", default=False, help='Whether to maintain original batch size per task on when mode is "continual"')
    parser.add_argument("--eval_all", action="store_false", default=True, help="Whether to only evaluate all environments in the benchmark on every evaluation step")

    # ── Training hyperparameters ────────────────────────────────────
    parser.add_argument("--eval_freq", default=[500], type=int, nargs="+", help="Evaluation frequency schedule. Accepts one or more integers:\n  1 value:  --eval_freq <freq>\n      Evaluate every <freq> steps (constant frequency).\n  ≥3 values (even): --eval_freq <max_step_1> <freq_1> <max_step_2> <freq_2> ...\n      Use <freq_i> until step <max_step_i>, then switch to the next pair.\n  ≥3 values (odd):  --eval_freq <max_step_1> <freq_1> ... <final_freq>\n      Same as even, but the trailing <final_freq> applies from the\n      last max_step up to total_timesteps.\n  2 values: INVALID (ambiguous; use 1 or ≥3).\n")
    parser.add_argument("--video_freq", default=0, type=int)
    parser.add_argument("--n_eval_episodes", default=15, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--buffer_size", default=50_000, type=int)
    parser.add_argument("--target_update", default=1000, type=int)
    parser.add_argument("--learning_starts", default=1000, type=int)
    parser.add_argument("--total_timesteps", default=200_000, type=int)
    parser.add_argument("--network_size", default=None, type=int)
    parser.add_argument("--mode", default=MODES[0], type=str, choices=MODES)
    parser.add_argument("--store_weights", default=False, action="store_true")

    # ── DQN-specific (epsilon-greedy) ───────────────────────────────
    parser.add_argument("--epsilon_start", default=1.0, type=float)
    parser.add_argument("--epsilon_end", default=0.05, type=float)
    parser.add_argument("--epsilon_decay_frac", default=0.1, type=float)
    parser.add_argument("--dqn_tau", default=1.0, type=float)
    parser.add_argument("--exploration_strategy", default="eps-greedy", type=str, choices=["eps-greedy", "boltzmann"])

    # ── SAC-specific ───────────────────────────────
    parser.add_argument("--ent_coef", default=None)

    return parser.parse_args()


def parse_eval_freq(
    raw: list[int] | int,
    total_timesteps: int,
) -> int | list[tuple[int, int]]:
    """Convert a flat list of CLI integers into the eval-frequency format
    expected by :class:`EnvEvalCallback`.

    Rules
    -----
    * **1 value** → plain ``int`` (constant frequency).
    * **2 values** → ``ValueError`` (ambiguous).
    * **≥ 3 values, even count** → ``list[tuple[int, int]]`` of
      ``(max_step, freq)`` pairs.
    * **≥ 3 values, odd count** → same as even, but the last lone value
      becomes ``(total_timesteps, final_freq)``.
    """
    if isinstance(raw, int):
        raw: list[int] = [raw]
    
    n = len(raw)

    if n == 1:
        return raw[0]

    if n == 2:
        raise ValueError(
            f"--eval_freq with exactly 2 values is ambiguous. "
            f"Use 1 value for a constant frequency or ≥3 values for a schedule. "
            f"Got: {raw}"
        )

    pairs: list[tuple[int, int]] = []

    # consume full pairs
    i = 0
    while i + 1 < n:
        pairs.append((raw[i], raw[i + 1]))
        i += 2

    # odd trailing value → pair with total_timesteps
    if n % 2 == 1:
        pairs.append((total_timesteps, raw[-1]))

    return pairs
