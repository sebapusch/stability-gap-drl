import argparse
import zipfile
from functools import reduce
from os import makedirs, path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm

from transformation.benchmarks.transformed_env_benchmark import (
    NUM_TASKS,
    random_orthogonal,
)

type MLP = list[tuple[jax.Array, jax.Array]]

N_STEPS = 200
N_ENV_STEPS = 1000
N_EVAL = 15
OBSERVATION_DIM = 4
EVAL_TASK_INDEX = 0
TASK_INDICES = tuple(range(NUM_TASKS))
MODEL_PATH = path.abspath(path.join(__file__, "..", "..", "output", "__", "models"))
if not path.exists(MODEL_PATH):
    MODEL_PATH = path.abspath(path.join(__file__, "..", "..", "output", "models"))


# Physics constants matching MuJoCo InvertedPendulum-v5
M = 10.47197551
m = 5.01859164
x0 = 0.0005
z0 = 0.3
I = 0.18874977
g = 9.81
J = I + m * (x0**2 + z0**2)


def generate_combinations() -> jax.Array:
    vals = jnp.linspace(-0.5, 1.5, N_STEPS)
    X, Y = jnp.meshgrid(vals, vals)

    grid_matrix = jnp.stack([X, Y], axis=-1)
    grid_matrix = grid_matrix.reshape((-1, 2))

    return grid_matrix


def forward(policy: MLP, x: jax.Array) -> jax.Array:
    Wo, bo = policy[-1]
    latent = reduce(lambda xo, l: jax.nn.relu(l[0] @ xo + l[1]), policy[:-1], x)
    mean = Wo @ latent + bo
    # SAC/DDPG squashes actions to [-1, 1] then scales to [-3, 3] for InvertedPendulum
    return 3.0 * jnp.tanh(mean)


def policy_fn(policy: MLP, batch_obs: jax.Array) -> jax.Array:
    vmap_forward = jax.vmap(forward, in_axes=(None, 0))
    actions = vmap_forward(policy, batch_obs)
    return actions


def dynamics(state: jax.Array, action: float) -> tuple[jax.Array, jax.Array]:
    x, theta, x_dot, theta_dot = state
    F = 100.0 * action

    # Mass matrix and right hand side
    m11 = M + m
    m12 = m * (z0 * jnp.cos(theta) - x0 * jnp.sin(theta))
    m22 = J

    a_prime = -m * (z0 * jnp.sin(theta) + x0 * jnp.cos(theta))
    b1 = F - x_dot - a_prime * (theta_dot**2)
    b2 = -theta_dot - g * a_prime

    # Solve system: Mass * [x_ddot, theta_ddot]^T = [b1, b2]^T
    det = m11 * m22 - m12**2
    x_ddot = (m22 * b1 - m12 * b2) / det
    theta_ddot = (-m12 * b1 + m11 * b2) / det

    return x_ddot, theta_ddot


def rk4_step(state: jax.Array, action: float, dt: float = 0.02) -> jax.Array:
    def f(s):
        x_val, theta_val, x_dot_val, theta_dot_val = s
        x_ddot, theta_ddot = dynamics(s, action)
        return jnp.stack([x_dot_val, theta_dot_val, x_ddot, theta_ddot])

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_2steps(state: jax.Array, action: jax.Array) -> jax.Array:
    action_val = action[0] if action.ndim > 0 else action
    state1 = rk4_step(state, action_val, 0.02)
    state2 = rk4_step(state1, action_val, 0.02)
    return state2


def evaluate(
    policy: MLP,
    task_encoding: jax.Array,
    proj_mat: jax.Array,
    proj_bias: jax.Array,
    angle_limit: float,
    key: jax.Array,
) -> jax.Array:
    # Reset model uniformly matching Gymnasium
    vmap_reset = jax.vmap(lambda k: jax.random.uniform(k, minval=-0.01, maxval=0.01, shape=(4,)))

    batch_task_encoding = jnp.tile(task_encoding, (N_EVAL, 1))

    def process_obs(obs_batch: jax.Array) -> jax.Array:
        proj_obs = obs_batch @ proj_mat.T + proj_bias
        return jnp.concatenate([proj_obs, batch_task_encoding], axis=-1)

    rng, reset_rng = jax.random.split(key)
    reset_keys = jax.random.split(reset_rng, N_EVAL)

    init_state = vmap_reset(reset_keys)
    init_obs_processed = process_obs(init_state)

    def scan_step(carry, _):
        current_state, current_obs_processed, already_done = carry

        actions = policy_fn(policy, current_obs_processed)

        vmap_step_env = jax.vmap(rk4_step_2steps, in_axes=(0, 0))
        next_state = vmap_step_env(current_state, actions)

        # Check termination condition
        angles = next_state[:, 1]
        is_finite = jnp.isfinite(next_state).all(axis=-1)
        dones = jnp.logical_or(~is_finite, jnp.abs(angles) > angle_limit)

        next_obs_processed = process_obs(next_state)

        rewards = 1.0 - already_done
        masked_rewards = jnp.where(already_done, 0.0, rewards)

        next_already_done = jnp.logical_or(already_done, dones)

        next_carry = (next_state, next_obs_processed, next_already_done)

        return next_carry, masked_rewards

    initial_already_done = jnp.zeros(N_EVAL, dtype=bool)
    initial_carry = (init_state, init_obs_processed, initial_already_done)

    final_carry, reward_history = jax.lax.scan(
        scan_step,
        initial_carry,
        jnp.arange(N_ENV_STEPS)
    )

    return reward_history


def combine(a: MLP, b: MLP, c: MLP, d: MLP, alpha: jax.Array, beta: jax.Array) -> MLP:
    comb: MLP = []
    for al, bl, cl, dl in zip(a, b, c, d):
        comb.append((
            (beta * ((1 - alpha) * al[0] + alpha * bl[0]) + (1 - beta) * ((1 - alpha) * cl[0] + alpha * dl[0])),
            (beta * ((1 - alpha) * al[1] + alpha * bl[1]) + (1 - beta) * ((1 - alpha) * cl[1] + alpha * dl[1])),
        ))

    return comb


def evaluate_combination(
    policies: tuple[MLP, MLP, MLP, MLP],
    vals: jax.Array,
    task_encoding: jax.Array,
    proj_mat: jax.Array,
    angle_limit: float,
    key: jax.Array,
) -> jax.Array:
    comb = combine(*policies, alpha=vals[0], beta=vals[1])

    rh = evaluate(
        comb,
        task_encoding,
        proj_mat=proj_mat,
        proj_bias=jnp.zeros(OBSERVATION_DIM, dtype=jnp.float32),
        angle_limit=angle_limit,
        key=key,
    )

    return rh.sum(axis=0)


def load_policies(model_path: str) -> tuple[MLP, MLP, MLP, MLP]:
    models: dict[int, MLP] = {}

    for task_index in TASK_INDICES:
        with zipfile.ZipFile(f"{model_path}-{task_index}.zip") as archive:
            with archive.open("policy.pth", mode="r") as param_file:
                th_object = torch.load(param_file, weights_only=True)

                # Continuous policies (SAC/DDPG) use latent_pi.0, latent_pi.2, and mu layers
                models[task_index] = [
                    (jnp.array(th_object["actor.latent_pi.0.weight"].cpu().numpy()), jnp.array(th_object["actor.latent_pi.0.bias"].cpu().numpy())),
                    (jnp.array(th_object["actor.latent_pi.2.weight"].cpu().numpy()), jnp.array(th_object["actor.latent_pi.2.bias"].cpu().numpy())),
                    (jnp.array(th_object["actor.mu.weight"].cpu().numpy()), jnp.array(th_object["actor.mu.bias"].cpu().numpy())),
                ]

    first, second, third = (models[task_index] for task_index in TASK_INDICES)
    fourth = [
        (third_layer[0] + second_layer[0] - first_layer[0],
         third_layer[1] + second_layer[1] - first_layer[1])
        for first_layer, second_layer, third_layer in zip(first, second, third)
    ]

    return first, second, third, fourth


def make_task_encoding(policy: MLP) -> jax.Array:
    input_dim = policy[0][0].shape[1]
    if input_dim == OBSERVATION_DIM:
        return jnp.empty(0, dtype=jnp.float32)
    if input_dim == OBSERVATION_DIM + NUM_TASKS:
        return jax.nn.one_hot(EVAL_TASK_INDEX, NUM_TASKS, dtype=jnp.float32)

    raise ValueError(
        f"Expected policy input width {OBSERVATION_DIM} or "
        f"{OBSERVATION_DIM + NUM_TASKS}, got {input_dim}"
    )


def task_projection(seed: int) -> jax.Array:
    transformation_seed = NUM_TASKS * seed + EVAL_TASK_INDEX
    return jnp.asarray(
        random_orthogonal(transformation_seed, OBSERVATION_DIM),
        dtype=jnp.float32,
    )


def main(
    seeds: list[int],
    model_path: str,
    output_dir: str,
    hard: bool = False,
    chunk_size: int = 2000,
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    makedirs(output_dir, exist_ok=True)
    angle_limit = 0.1 if hard else 0.2
    print(f"Evaluating task 1 with angle limit {angle_limit} radians")
    combinations = generate_combinations()
    eval_vmap = jax.vmap(
        evaluate_combination,
        in_axes=(None, 0, None, None, None, None),
    )

    for s in tqdm.tqdm(seeds):
        try:
            policies = load_policies(
                path.join(MODEL_PATH, model_path.replace('<s>', str(s)))
            )
        except (FileNotFoundError, zipfile.BadZipFile, KeyError) as e:
            print(f"Warning: Skipping seed {s} due to missing or corrupt model file: {e}")
            continue

        task_encoding = make_task_encoding(policies[0])
        proj_mat = task_projection(s)
        eval_key = jax.random.key(s + 1)

        results = []
        for i in range(0, len(combinations), chunk_size):
            chunk = combinations[i : i + chunk_size]
            res_chunk = eval_vmap(
                policies,
                chunk,
                task_encoding,
                proj_mat,
                angle_limit,
                eval_key,
            )
            res_chunk = res_chunk.mean(axis=-1)
            results.append(res_chunk)

        res = jnp.concatenate(results, axis=0)

        data = jnp.column_stack((combinations, res))

        np.savetxt(path.join(output_dir, f"data_{s}.csv"), np.asarray(data), delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    seed_args = parser.add_mutually_exclusive_group(required=True)
    seed_args.add_argument('--seeds', nargs='+', type=int)
    seed_args.add_argument(
        '--seed_range',
        nargs=2,
        type=int,
        metavar=('START', 'STOP'),
        help='Evaluate seeds in the half-open range [START, STOP)',
    )
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--chunk_size', type=int, default=2000)

    args = parser.parse_args()
    if args.seed_range is not None:
        start, stop = args.seed_range
        if start >= stop:
            parser.error('--seed_range requires START < STOP')
        args.seeds = list(range(start, stop))
    del args.seed_range

    main(**vars(args))
