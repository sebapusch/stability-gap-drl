#!/usr/bin/env python3
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config Parsing & Formatting
# ---------------------------------------------------------------------------

def parse_config(config_path: str) -> dict:
    """Load a YAML experiment config and return a structured dict with parsed keys."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ablations = cfg.get("ablations", {})
    num_tasks = cfg.get("num_tasks", 3)
    benchmark = cfg.get("benchmark")
    if benchmark is None:
        # The current benchmark implementation names evaluation streams by
        # their zero-based task index (eval/0, eval/1, ...).
        benchmark = [str(i) for i in range(num_tasks)]
    else:
        benchmark = [str(task) for task in benchmark]
    name_prefix = cfg.get("name_prefix", "experiment")

    all_ablation_keys = list(ablations.keys())

    def ablation_values(value):
        # Match dispatch_yaml.py: a scalar integer denotes a count, not an ID.
        if isinstance(value, int):
            return list(range(value))
        if isinstance(value, list):
            return value
        return [value]

    seeds = ablation_values(ablations.get("seed", [0]))

    hp_keys = [k for k in all_ablation_keys if k != "seed"]
    hp_values = [ablation_values(ablations[k]) for k in hp_keys]

    project = cfg.get("project", name_prefix)

    cfg.update({
        "name_prefix": name_prefix,
        "project": project,
        "benchmark": benchmark,
        "seeds": seeds,
        "hp_keys": hp_keys,
        "hp_values": hp_values,
        "all_ablation_keys": all_ablation_keys,
        "env": cfg.get("env"),
    })
    return cfg


def enumerate_combinations(cfg: dict):
    """Yield dicts of {hp_name: value, ...} for every hyperparameter combination."""
    import itertools
    if not cfg["hp_keys"]:
        yield {}
        return
    for combo in itertools.product(*cfg["hp_values"]):
        yield dict(zip(cfg["hp_keys"], combo))


def format_value(val) -> str:
    """Format a hyperparameter value the same way dispatch_yaml.py does for filenames."""
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        else:
            return str(val).rstrip("0").rstrip(".").replace(".", "")
    if isinstance(val, str) and "." in val:
        try:
            f_val = float(val)
            if f_val.is_integer():
                return str(int(f_val))
            else:
                return str(f_val).rstrip("0").rstrip(".").replace(".", "")
        except ValueError:
            pass
    return str(val)


def build_suffix(hp_combo: dict, seed: int, all_ablation_keys: list[str]) -> str:
    """Build the filename ablation suffix in the original YAML key order."""
    parts = []
    for k in all_ablation_keys:
        v = seed if k == "seed" else hp_combo[k]
        parts.append(f"{k[0]}_{format_value(v)}")
    return "-".join(parts)


def get_env_max_return(env_name: str) -> float:
    """Return the theoretical maximum return for the environment."""
    normalized_name = (env_name or "").lower().replace("-", "_")
    if "inverted_pendulum" in normalized_name or "invertedpendulum" in normalized_name:
        return 1000.0
    if "cartpole" in normalized_name:
        return 500.0
    raise ValueError(f"Unknown maximum episodic return for environment {env_name!r}")


# ---------------------------------------------------------------------------
# High-Performance Data Loading
# ---------------------------------------------------------------------------

def load_csv_columns(filepath: Path, target_cols: list[str]) -> pd.DataFrame:
    """Efficiently load only selected columns from a CSV file."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return pd.DataFrame()
    try:
        with open(filepath, 'r') as f:
            header = f.readline().strip().split(',')
        cols_to_load = [c for c in target_cols if c in header]
        if not cols_to_load:
            return pd.DataFrame()
        return pd.read_csv(filepath, usecols=cols_to_load, engine='c')
    except Exception:
        return pd.DataFrame()


def load_all_csvs(data_dir: Path, name_prefix: str, benchmark: list[str], hp_combo: dict, seeds: list[int], all_ablation_keys: list[str], project: str = None) -> dict:
    """
    Load the single continual-learning CSV for each seed in an HP combination.

    Since the trainer keeps one logger alive for the entire task loop, the
    filename no longer has a trailing training-task name and its timesteps are
    already cumulative. Returns ``{seed: dataframe}``.
    """
    data = {}
    target_cols = []
    for env in benchmark:
        target_cols.append(f"eval/{env}/mean_reward")
        target_cols.append(f"time/{env}/total_timesteps")
    dir_name = project if project is not None else name_prefix
    for seed in seeds:
        suffix = build_suffix(hp_combo, seed, all_ablation_keys)
        run_name = name_prefix + (f"-{suffix}" if suffix else "")
        filename = f"{run_name}.csv"
        filepath = data_dir / dir_name / filename
        df = load_csv_columns(filepath, target_cols)
        if not df.empty:
            data[seed] = df
    return data


def load_final_reward(filepath: Path, eval_env: str, n_smooth: int) -> float | None:
    """Load smoothed final reward for one evaluation environment."""
    target_col = f"eval/{eval_env}/mean_reward"
    df = load_csv_columns(filepath, [target_col])
    if df.empty or target_col not in df.columns:
        return None
    rewards = df[target_col].dropna().values
    if len(rewards) == 0:
        return None
    n = min(n_smooth, len(rewards))

    return float(np.mean(rewards[-n:]))


def compute_per_env_final_score(
        name_prefix: str,
        hp_combo: dict,
        seed: int,
        benchmark: list[str],
        eval_env: str,
        n_smooth: int,
        data_dir: Path,
        all_ablation_keys: list[str],
        project: str = None,
) -> float | None:
    """Extract a final reward from the single CSV for a seed's full run."""
    suffix = build_suffix(hp_combo, seed, all_ablation_keys)
    run_name = name_prefix + (f"-{suffix}" if suffix else "")
    filename = f"{run_name}.csv"
    dir_name = project if project is not None else name_prefix
    filepath = data_dir / dir_name / filename
    return load_final_reward(filepath, eval_env, n_smooth)


# ---------------------------------------------------------------------------
# IQM & Bootstrap Calculations
# ---------------------------------------------------------------------------

def interquartile_mean(values: np.ndarray) -> float:
    """Compute the interquartile mean (IQM) of a 1-D array."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    q1_idx = int(np.floor(n * 0.25))
    q3_idx = int(np.ceil(n * 0.75))
    if q3_idx <= q1_idx:
        return np.mean(sorted_vals)
    return np.mean(sorted_vals[q1_idx:q3_idx])


def interquartile_mean_batch(values: np.ndarray) -> np.ndarray:
    """Compute IQM along axis=1 of a 2-D array (vectorised)."""
    sorted_vals = np.sort(values, axis=1)
    n = sorted_vals.shape[1]
    q1_idx = int(np.floor(n * 0.25))
    q3_idx = int(np.ceil(n * 0.75))
    if q3_idx <= q1_idx:
        return np.mean(sorted_vals, axis=1)
    return np.mean(sorted_vals[:, q1_idx:q3_idx], axis=1)


def bootstrap_iqm(seed_values: np.ndarray, n_bootstrap: int = 10_000, confidence: float = 0.95):
    """
    Compute IQM and confidence interval via bootstrap over seeds.
    seed_values can be (n_seeds,) or (n_timesteps, n_seeds).
    """
    if seed_values.ndim == 1:
        n_seeds = len(seed_values)
        if n_seeds == 0:
            return np.nan, np.nan, np.nan
        rng = np.random.default_rng(42)
        indices = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
        boot_samples = seed_values[indices]
        boot_iqms = interquartile_mean_batch(boot_samples)
        alpha = (1 - confidence) / 2
        ci_low = np.percentile(boot_iqms, 100 * alpha)
        ci_high = np.percentile(boot_iqms, 100 * (1 - alpha))
        iqm = interquartile_mean(seed_values)
        return iqm, ci_low, ci_high

    # Batched path: (n_timesteps, n_seeds)
    n_ts, n_seeds = seed_values.shape
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    q1_idx = int(np.floor(n_seeds * 0.25))
    q3_idx = int(np.ceil(n_seeds * 0.75))
    alpha = (1 - confidence) / 2
    ci_low = np.empty(n_ts)
    ci_high = np.empty(n_ts)

    # Indexing all timesteps at once creates two enormous 3-D arrays (samples
    # and sorted samples). With 2,400 timesteps, 10,000 resamples and 50 seeds,
    # each float64 array alone takes almost 9 GiB. Reuse the same resampling
    # indices for every row, but keep each batch of samples below 64 MiB.
    bytes_per_row = n_bootstrap * n_seeds * seed_values.dtype.itemsize
    batch_size = max(1, (64 * 1024 * 1024) // bytes_per_row)
    for start in range(0, n_ts, batch_size):
        stop = min(start + batch_size, n_ts)
        boot_samples = seed_values[start:stop, indices]
        boot_samples.sort(axis=2)
        middle = (boot_samples if q3_idx <= q1_idx
                  else boot_samples[:, :, q1_idx:q3_idx])
        boot_iqms = np.mean(middle, axis=2)
        ci_low[start:stop] = np.percentile(boot_iqms, 100 * alpha, axis=1)
        ci_high[start:stop] = np.percentile(boot_iqms, 100 * (1 - alpha), axis=1)

    sorted_sv = np.sort(seed_values, axis=1)
    if q3_idx <= q1_idx:
        iqm = np.mean(sorted_sv, axis=1)
    else:
        iqm = np.mean(sorted_sv[:, q1_idx:q3_idx], axis=1)

    return iqm, ci_low, ci_high


def bootstrap_mean(seed_values: np.ndarray, n_bootstrap: int = 10_000, confidence: float = 0.95):
    """Compute the mean and a percentile bootstrap CI over seeds for each timestep."""
    n_ts, n_seeds = seed_values.shape
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    alpha = (1 - confidence) / 2
    ci_low = np.empty(n_ts)
    ci_high = np.empty(n_ts)

    # Bound the temporary (timesteps × resamples × seeds) array as for IQM.
    bytes_per_row = n_bootstrap * n_seeds * seed_values.dtype.itemsize
    batch_size = max(1, (64 * 1024 * 1024) // bytes_per_row)
    for start in range(0, n_ts, batch_size):
        stop = min(start + batch_size, n_ts)
        boot_means = np.mean(seed_values[start:stop, indices], axis=2)
        ci_low[start:stop] = np.percentile(boot_means, 100 * alpha, axis=1)
        ci_high[start:stop] = np.percentile(boot_means, 100 * (1 - alpha), axis=1)

    return np.mean(seed_values, axis=1), ci_low, ci_high


# ---------------------------------------------------------------------------
# Metric Core Computations
# ---------------------------------------------------------------------------

def get_aligned_curves(
    combo_data: dict,
    eval_env: str,
    timestep_start: int | None = None,
    timestep_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Align eval data for a given eval_env across all seeds in combo_data,
    collapsing duplicate rows by taking the first observation at each timestep.
    Returns:
        timesteps: 1D array of aligned timesteps of length T
        stacked_curves: 2D array of shape (n_seeds, T) containing raw rewards
        valid_seeds: list of seed IDs that were successfully loaded
    """
    seed_frames = []
    reward_col = f"eval/{eval_env}/mean_reward"
    timestep_col = f"time/{eval_env}/total_timesteps"

    for seed, df in combo_data.items():
        if reward_col not in df.columns or timestep_col not in df.columns:
            continue
        mask = df[reward_col].notna() & df[timestep_col].notna()
        if timestep_start is not None:
            mask &= df[timestep_col] >= timestep_start
        if timestep_end is not None:
            mask &= df[timestep_col] <= timestep_end
        subset = df.loc[mask, [timestep_col, reward_col]].copy()
        if subset.empty:
            continue
        subset.columns = ["timestep", "reward"]
        subset = subset.groupby("timestep", as_index=False).first()
        subset["seed"] = seed
        seed_frames.append(subset)

    if not seed_frames:
        return np.array([]), np.array([]), []

    combined_all = pd.concat(seed_frames, ignore_index=True)
    pivot = combined_all.pivot_table(
        index="timestep", columns="seed", values="reward", aggfunc="first"
    )
    min_seeds = min(2, len(pivot.columns))
    seed_counts = pivot.notna().sum(axis=1)
    pivot = pivot.loc[seed_counts >= min_seeds]

    if pivot.empty:
        return np.array([]), np.array([]), []

    valid_ts = pivot.index.values
    stacked_curves = pivot.values.T
    valid_seeds = list(pivot.columns)

    return valid_ts, stacked_curves, valid_seeds


def aggregate_curves_vectorized(curves: np.ndarray, use_iqm: bool) -> np.ndarray:
    """
    curves: shape (T, n_seeds), may contain NaNs.
    Returns: shape (T,)
    """
    T, n_seeds = curves.shape
    out = np.empty(T)

    # Sort along seeds axis: NaNs are placed at the end
    sorted_curves = np.sort(curves, axis=1)

    # Count non-NaNs per row
    m = np.sum(~np.isnan(curves), axis=1)

    for unique_m in np.unique(m):
        row_indices = np.where(m == unique_m)[0]
        if unique_m == 0:
            out[row_indices] = np.nan
        elif not use_iqm or unique_m < 2:
            out[row_indices] = np.mean(sorted_curves[row_indices, :unique_m], axis=1)
        else:
            lowercut = int(0.25 * unique_m)
            uppercut = unique_m - lowercut
            if uppercut <= lowercut:
                out[row_indices] = np.mean(sorted_curves[row_indices, :unique_m], axis=1)
            else:
                out[row_indices] = np.mean(sorted_curves[row_indices, lowercut:uppercut], axis=1)

    return out



def compute_final_performance_from_data(
    combo_data: dict,
    benchmark: list[str],
    n_smooth: int,
    use_iqm: bool,
    env_max: float,
    confidence: float,
    n_bootstrap: int,
    timesteps_per_env: int,
) -> dict:
    """Compute P at the final evaluation from environment-normalized returns.

    Aggregate seeds at each evaluation step before averaging over tasks. With
    ``n_smooth=1`` this is P(t_N) as defined in the thesis; larger values
    optionally average the last few evaluation steps for each task.
    """
    if not np.isfinite(env_max) or env_max <= 0:
        raise ValueError("env_max must be positive and finite")
    aligned_curves = {}
    common_seeds = None

    for eval_env in benchmark:
        ts, curves, seeds = get_aligned_curves(
            combo_data, eval_env, timestep_end=len(benchmark) * timesteps_per_env
        )
        if len(seeds) == 0:
            common_seeds = set()
            break
        aligned_curves[eval_env] = (ts, curves, seeds)
        if common_seeds is None:
            common_seeds = set(seeds)
        else:
            common_seeds = common_seeds.intersection(seeds)

    if not common_seeds:
        res = {"n_seeds": 0}
        for env in benchmark:
            res[f"{env}_mean"] = np.nan
            res[f"{env}_ci_low"] = np.nan
            res[f"{env}_ci_high"] = np.nan
        res["P_mean"] = np.nan
        res["P_ci_low"] = np.nan
        res["P_ci_high"] = np.nan
        return res

    valid_seeds = sorted(list(common_seeds))
    n_valid = len(valid_seeds)

    common_timesteps = None
    for ts, _, _ in aligned_curves.values():
        common_timesteps = ts if common_timesteps is None else np.intersect1d(common_timesteps, ts)
    if common_timesteps is None or len(common_timesteps) == 0:
        res = {"n_seeds": 0}
        for env in benchmark:
            res[f"{env}_mean"] = np.nan
            res[f"{env}_ci_low"] = np.nan
            res[f"{env}_ci_high"] = np.nan
        res["P_mean"] = np.nan
        res["P_ci_low"] = np.nan
        res["P_ci_high"] = np.nan
        return res
    final_timesteps = common_timesteps[-n_smooth:]

    stacked_mapped = {}
    for eval_env in benchmark:
        ts, curves, seeds = aligned_curves[eval_env]
        seed_indices = [seeds.index(s) for s in valid_seeds]
        timestep_indices = np.searchsorted(ts, final_timesteps)
        stacked_mapped[eval_env] = (
            curves[np.ix_(seed_indices, timestep_indices)].T / env_max
        ) * 100.0

    def calc_perf_single_bootstrap(boot_col_indices: np.ndarray) -> np.ndarray:
        env_scores = np.empty(len(benchmark))
        for idx, env in enumerate(benchmark):
            sub_curves = stacked_mapped[env][:, boot_col_indices]
            agg_curve = aggregate_curves_vectorized(sub_curves, use_iqm)
            env_scores[idx] = np.mean(agg_curve)
        return env_scores

    obs_env_scores = calc_perf_single_bootstrap(np.arange(n_valid))
    
    # Run bootstrap
    rng = np.random.default_rng(42)
    boot_env_scores = np.empty((n_bootstrap, len(benchmark)))
    for b in range(n_bootstrap):
        boot_col_indices = rng.integers(0, n_valid, size=n_valid)
        boot_env_scores[b, :] = calc_perf_single_bootstrap(boot_col_indices)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        obs_pt = float(np.mean(obs_env_scores))
        boot_pt = np.mean(boot_env_scores, axis=1)

    alpha = (1 - confidence) / 2

    res = {"n_seeds": n_valid}
    for idx, env in enumerate(benchmark):
        res[f"{env}_mean"] = float(obs_env_scores[idx])
        valid_boot = boot_env_scores[:, idx][~np.isnan(boot_env_scores[:, idx])]
        if len(valid_boot) == 0:
            res[f"{env}_ci_low"] = np.nan
            res[f"{env}_ci_high"] = np.nan
        else:
            res[f"{env}_ci_low"] = float(np.percentile(valid_boot, 100 * alpha))
            res[f"{env}_ci_high"] = float(np.percentile(valid_boot, 100 * (1 - alpha)))

    valid_boot_pt = boot_pt[~np.isnan(boot_pt)]
    if len(valid_boot_pt) == 0:
        res["P_mean"] = obs_pt
        res["P_ci_low"] = np.nan
        res["P_ci_high"] = np.nan
    else:
        res["P_mean"] = obs_pt
        res["P_ci_low"] = float(np.percentile(valid_boot_pt, 100 * alpha))
        res["P_ci_high"] = float(np.percentile(valid_boot_pt, 100 * (1 - alpha)))
    return res


def compute_min_acc_from_data(
    combo_data: dict,
    benchmark: list[str],
    k_idx: int,
    use_iqm: bool,
    env_max: float,
    confidence: float,
    n_bootstrap: int,
    timesteps_per_env: int,
) -> dict:
    """Compute min-P from environment-normalized, seed-aggregated curves.

    For each previous task i, take the minimum strictly after its training
    boundary through the end of task k, then average those minima.
    """
    if not np.isfinite(env_max) or env_max <= 0:
        raise ValueError("env_max must be positive and finite")
    aligned_curves = {}
    common_seeds = None

    for i in range(k_idx):
        eval_env = benchmark[i]
        ts, curves, seeds = get_aligned_curves(
            combo_data,
            eval_env,
            timestep_start=(i + 1) * timesteps_per_env,
            timestep_end=(k_idx + 1) * timesteps_per_env,
        )
        if len(seeds) == 0:
            common_seeds = set()
            break
        after_boundary = ts > (i + 1) * timesteps_per_env
        if not after_boundary.any():
            common_seeds = set()
            break
        ts, curves = ts[after_boundary], curves[:, after_boundary]
        aligned_curves[eval_env] = (ts, curves, seeds)
        if common_seeds is None:
            common_seeds = set(seeds)
        else:
            common_seeds = common_seeds.intersection(seeds)

    if not common_seeds:
        return {
            "n_seeds": 0,
            "min-P_mean": np.nan,
            "min-P_ci_low": np.nan,
            "min-P_ci_high": np.nan,
        }

    valid_seeds = sorted(list(common_seeds))
    n_valid = len(valid_seeds)

    stacked_curves_mapped = {}
    for i in range(k_idx):
        eval_env = benchmark[i]
        _, curves, seeds = aligned_curves[eval_env]
        seed_indices = [seeds.index(s) for s in valid_seeds]
        stacked_curves_mapped[eval_env] = (curves[seed_indices, :].T / env_max) * 100.0

    def calc_metric_single_bootstrap(boot_col_indices: np.ndarray) -> float:
        task_mins = np.empty(k_idx)
        for i in range(k_idx):
            eval_env = benchmark[i]
            sub_curves = stacked_curves_mapped[eval_env][:, boot_col_indices]
            agg_curve = aggregate_curves_vectorized(sub_curves, use_iqm)
            valid = agg_curve[~np.isnan(agg_curve)]
            if len(valid) == 0:
                return np.nan
            task_mins[i] = np.min(valid)
        return float(np.mean(task_mins))

    observed_stat = calc_metric_single_bootstrap(np.arange(n_valid))
    
    # Run bootstrap
    rng = np.random.default_rng(42)
    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        boot_col_indices = rng.integers(0, n_valid, size=n_valid)
        boot_stats[b] = calc_metric_single_bootstrap(boot_col_indices)

    alpha = (1 - confidence) / 2
    valid_boot = boot_stats[~np.isnan(boot_stats)]
    if len(valid_boot) == 0:
        ci_low = np.nan
        ci_high = np.nan
    else:
        ci_low = float(np.percentile(valid_boot, 100 * alpha))
        ci_high = float(np.percentile(valid_boot, 100 * (1 - alpha)))

    return {
        "n_seeds": n_valid,
        "min-P_mean": observed_stat,
        "min-P_ci_low": ci_low,
        "min-P_ci_high": ci_high,
    }



# ---------------------------------------------------------------------------
# Plotting Data Processing Utilities
# ---------------------------------------------------------------------------

def load_eval_data(
    method: str,
    seed: int,
    test_env: str,
    train_envs: list[str],
    timesteps_per_env: int,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Load eval data from the single CSV spanning the complete task sequence.
    """
    reward_col = f"eval/{test_env}/mean_reward"
    timestep_col = f"time/{test_env}/total_timesteps"
    target_cols = [reward_col, timestep_col]

    filename = f"{method}.csv".replace('<s>', str(seed))
    filepath = data_dir / filename
    df = load_csv_columns(filepath, target_cols)
    if df.empty or reward_col not in df.columns or timestep_col not in df.columns:
        return pd.DataFrame(columns=["timestep", "reward"])

    mask = df[reward_col].notna() & df[timestep_col].notna()
    result = df.loc[mask, [timestep_col, reward_col]].copy()
    result.columns = ["timestep", "reward"]
    # A shared logger can contain duplicate dumps at a task boundary.
    result = result.groupby("timestep", as_index=False).first()
    result = result.sort_values("timestep").reset_index(drop=True)
    return result


def compute_aggregate_curve(
    method: str,
    test_env: str,
    seeds: list[int],
    train_envs: list[str],
    timesteps_per_env: int,
    data_dir: Path,
    aggregation: str = "iqm",
    n_bootstrap: int = 10_000,
):
    """
    Compute an aggregate reward curve for a method on a test environment.

    ``aggregation="iqm"`` returns the IQM and its 95% bootstrap confidence
    interval. ``aggregation="mean"`` returns the arithmetic mean and a band
    one standard error below and above the mean. ``aggregation="mean_ci"``
    returns the mean and its 95% bootstrap confidence interval.
    """
    if aggregation not in {"iqm", "mean", "mean_ci"}:
        raise ValueError(f"Unknown aggregation {aggregation!r}; expected 'iqm', 'mean', or 'mean_ci'.")

    seed_frames = []
    for seed in seeds:
        df = load_eval_data(
            method, seed, test_env,
            train_envs=train_envs,
            timesteps_per_env=timesteps_per_env,
            data_dir=data_dir,
        )
        if not df.empty:
            df = df.copy()
            df["seed"] = seed
            seed_frames.append(df)

    if not seed_frames:
        return np.array([]), np.array([]), np.array([]), np.array([])

    combined = pd.concat(seed_frames, ignore_index=True)

    # Pivot to (timestep × seed) matrix; NaN where a seed has no data
    pivot = combined.pivot_table(
        index="timestep", columns="seed", values="reward", aggfunc="first"
    )
    # Keep only timesteps with >= 2 seeds present
    seed_counts = pivot.notna().sum(axis=1)
    pivot = pivot.loc[seed_counts >= 2]

    if pivot.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    valid_ts = pivot.index.values

    # Group rows by their set of available seeds so we can aggregate dense
    # matrices in batches even when seed availability varies over time.
    present_mask = pivot.notna().values  # (n_ts, n_seed_cols)
    patterns = [tuple(row) for row in present_mask]
    unique_patterns = list(set(patterns))

    aggregate_values = np.empty(len(valid_ts))
    interval_lows = np.empty(len(valid_ts))
    interval_highs = np.empty(len(valid_ts))

    for pat in unique_patterns:
        row_indices = np.array([i for i, p in enumerate(patterns) if p == pat])
        col_mask = np.array(pat)
        # Extract the dense (n_rows, n_present_seeds) sub-matrix
        seed_matrix = pivot.values[np.ix_(row_indices, col_mask)]
        if aggregation == "iqm":
            center, lower, upper = bootstrap_iqm(
                seed_matrix, n_bootstrap=n_bootstrap, confidence=0.95
            )
        elif aggregation == "mean_ci":
            center, lower, upper = bootstrap_mean(
                seed_matrix, n_bootstrap=n_bootstrap, confidence=0.95
            )
        else:
            center = np.mean(seed_matrix, axis=1)
            stderr = np.std(seed_matrix, axis=1, ddof=1) / np.sqrt(seed_matrix.shape[1])
            lower = center - stderr
            upper = center + stderr

        aggregate_values[row_indices] = center
        interval_lows[row_indices] = lower
        interval_highs[row_indices] = upper

    return valid_ts, aggregate_values, interval_lows, interval_highs


def compute_iqm_curve(
    method: str,
    test_env: str,
    seeds: list[int],
    train_envs: list[str],
    timesteps_per_env: int,
    data_dir: Path,
    n_bootstrap: int = 10_000,
):
    """Compute an IQM + 95% bootstrap CI curve."""
    return compute_aggregate_curve(
        method=method,
        test_env=test_env,
        seeds=seeds,
        train_envs=train_envs,
        timesteps_per_env=timesteps_per_env,
        data_dir=data_dir,
        aggregation="iqm",
        n_bootstrap=n_bootstrap,
    )
