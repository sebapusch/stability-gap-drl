import argparse
import zipfile
from functools import reduce
from os import makedirs, path

import gymnax
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
N_ENV_STEPS = 500
N_EVAL = 15
OBSERVATION_DIM = 4
EVAL_TASK_INDEX = 0
TASK_INDICES = tuple(range(NUM_TASKS))
MODEL_PATH = path.abspath(path.join(__file__, "..", "..", "output", "models"))


def generate_combinations() -> jax.Array:
    vals = jnp.linspace(-0.5, 1.5, N_STEPS)
    X, Y = jnp.meshgrid(vals, vals)

    grid_matrix = jnp.stack([X, Y], axis=-1)
    grid_matrix = grid_matrix.reshape((-1, 2))

    return grid_matrix


def forward(policy: MLP, x: jax.Array) -> jax.Array:
    Wo, bo = policy[-1]

    return Wo @ reduce(lambda xo, l: jax.nn.relu(l[0] @ xo + l[1]), policy[:-1], x) + bo


def policy_fn(policy: MLP, batch_obs: jax.Array) -> jax.Array:
    vmap_forward = jax.vmap(forward, in_axes=(None, 0))
    q_values = vmap_forward(policy, batch_obs)

    return jnp.argmax(q_values, axis=-1)


def evaluate(
    policy: MLP,
    task_encoding: jax.Array,
    proj_mat: jax.Array,
    key: jax.Array,
) -> jax.Array:
    env, env_params = gymnax.make('CartPole-v1')
    env_params = env_params.replace(max_steps_in_episode=N_ENV_STEPS)

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    batch_task_encoding = jnp.tile(task_encoding, (N_EVAL, 1))

    def process_obs(obs_batch: jax.Array) -> jax.Array:
        proj_obs = obs_batch @ proj_mat.T

        return jnp.concatenate([proj_obs, batch_task_encoding], axis=-1)

    rng, reset_rng = jax.random.split(key)
    reset_keys = jax.random.split(reset_rng, N_EVAL)

    init_obs, init_state = vmap_reset(reset_keys, env_params)
    init_obs = process_obs(init_obs)

    def scan_step(carry, _):
        current_obs, current_state, current_rng, already_done = carry

        current_rng, step_rng = jax.random.split(current_rng)
        step_keys = jax.random.split(step_rng, N_EVAL)

        actions = policy_fn(policy, current_obs)

        next_obs, next_state, rewards, dones, info = vmap_step(
            step_keys, current_state, actions, env_params
        )

        next_obs = process_obs(next_obs)

        masked_rewards = jnp.where(already_done, 0.0, rewards)

        next_already_done = jnp.logical_or(already_done, dones)

        next_carry = (next_obs, next_state, current_rng, next_already_done)

        return next_carry, masked_rewards

    initial_already_done = jnp.zeros(N_EVAL, dtype=bool)
    initial_carry = (init_obs, init_state, rng, initial_already_done)

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
    key: jax.Array,
) -> jax.Array:
    comb = combine(*policies, alpha=vals[0], beta=vals[1])

    rh = evaluate(
        comb,
        task_encoding,
        proj_mat=proj_mat,
        key=key,
    )

    return rh.sum(axis=0)


def load_policies(model_path: str) -> tuple[MLP, MLP, MLP, MLP]:
    models: dict[int, MLP] = {}

    for task_index in TASK_INDICES:
        with zipfile.ZipFile(f"{model_path}-{task_index}.zip") as archive:
            with archive.open("policy.pth", mode="r") as param_file:
                th_object = torch.load(param_file, weights_only=True)

                models[task_index] = [
                    (jnp.array(th_object["actor.latent_pi.0.weight"].cpu().numpy()), jnp.array(th_object["actor.latent_pi.0.bias"].cpu().numpy())),
                    (jnp.array(th_object["actor.latent_pi.2.weight"].cpu().numpy()), jnp.array(th_object["actor.latent_pi.2.bias"].cpu().numpy())),
                    (jnp.array(th_object["actor.latent_pi.4.weight"].cpu().numpy()), jnp.array(th_object["actor.latent_pi.4.bias"].cpu().numpy())),
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
        chunk_size: int = 2000,
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    makedirs(output_dir, exist_ok=True)
    print("Evaluating task 1")
    combinations = generate_combinations()
    eval_vmap = jax.vmap(
        evaluate_combination,
        in_axes=(None, 0, None, None, None),
    )

    for s in tqdm.tqdm(seeds):
        try:
            policies = load_policies(
                path.join(MODEL_PATH, model_path.replace('<s>', str(s)))
            )
        except (FileNotFoundError, zipfile.BadZipFile, KeyError) as error:
            print(f"Warning: skipping seed {s}: {error}")
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
                eval_key,
            )
            results.append(res_chunk.mean(axis=-1))

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
    parser.add_argument('--chunk_size', type=int, default=2000)

    args = parser.parse_args()
    if args.seed_range is not None:
        start, stop = args.seed_range
        if start >= stop:
            parser.error('--seed_range requires START < STOP')
        args.seeds = list(range(start, stop))
    del args.seed_range

    main(**vars(args))
