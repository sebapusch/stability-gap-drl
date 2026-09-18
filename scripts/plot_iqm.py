#!/usr/bin/env python3
"""
Generate aggregate reward plots for each test environment.

Supports two modes:

1. CLI mode (original): produces one plot per test env.
   python plot_iqm.py --methods continual sequential fine_tune

2. YAML config mode: produces a grid of subplots from a config file.
   python plot_iqm.py --config path/to/config.yaml

Pass --compute-only to write the line data to CSV without creating figures.
Pass --aggregation mean to plot the mean with a standard-error band instead
of the default IQM with a 95% bootstrap confidence interval.
"""

import argparse
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter

from common import (
    compute_aggregate_curve,
    smooth_peak_aware as _smooth,
)

try:
    import yaml
except ImportError:
    yaml = None

DATA_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "plots"
CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "cache"

DEFAULT_METHODS = ["continual", "sequential", "fine_tune"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TRAIN_ENVS = ["0", "1", "2"]
TEST_ENVS = ["0", "1", "2"]
TIMESTEPS_PER_ENV = 40_000
FS = 1.2
X_LABEL = "Cumulative training timesteps"
Y_LABELS = {
    "iqm": "IQM episodic return (95% CI)",
    "mean": "Mean episodic return (standard error)",
}
CACHE_SCHEMA_VERSION = "single-run-v2"

# Known labels
METHOD_LABELS = {
    "continual": "Continual",
    "sequential": "Sequential",
    "fine_tune": "Fine-tune",
    "continual_encode": "Continual (encode)",
}

COLOR_PALETTE = [
    "#2196F3",  # blue
    "#FF9800",  # orange
    "#4CAF50",  # green
    "#E91E63",  # pink
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#FF5722",  # deep orange
    "#607D8B",  # blue-grey
]


def get_label(method: str) -> str:
    """Return a display label for a method, auto-generating if unknown."""
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    return method.replace("_", " ").title()


def get_color(method: str, index: int) -> str:
    """Return a color for a method, cycling through the palette."""
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]


def make_cache_key(methods: list[str], prefix: str, aggregation: str = "iqm") -> str:
    """Generate a short hash from the sorted methods + prefix combination."""
    canonical = "||".join((
        CACHE_SCHEMA_VERSION,
        aggregation,
        "|".join(sorted(methods)),
        prefix,
    ))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def cache_path_for(cache_key: str, method: str, test_env: str, aggregation: str) -> Path:
    """Return the CSV cache path for a specific method/test_env under a cache key."""
    safe_method = method.replace("/", "__")
    return CACHE_DIR / cache_key / f"{safe_method}_{test_env}_{aggregation}.csv"


def save_to_cache(cache_key: str, method: str, test_env: str, aggregation: str,
                  ts: np.ndarray, center: np.ndarray,
                  lower: np.ndarray, upper: np.ndarray):
    """Save a computed aggregate curve to a CSV cache file."""
    path = cache_path_for(cache_key, method, test_env, aggregation)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "timestep": ts, "center": center, "lower": lower, "upper": upper,
    })
    df.to_csv(path, index=False)


def load_from_cache(cache_key: str, method: str, test_env: str, aggregation: str):
    """Load a cached aggregate curve. Returns (ts, center, lower, upper)."""
    path = cache_path_for(cache_key, method, test_env, aggregation)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return (
        df["timestep"].values, df["center"].values,
        df["lower"].values, df["upper"].values,
    )


def compute_line_data(
    cache_key: str,
    method: str,
    test_env: str,
    *,
    label: str,
    seeds: list[int],
    train_envs: list[str],
    timesteps_per_env: int,
    aggregation: str,
    smooth: int | None,
    use_cache: bool,
) -> pd.DataFrame:
    """Compute the exact (optionally smoothed) data used to draw one line."""
    cached = load_from_cache(cache_key, method, test_env, aggregation) if use_cache else None
    if cached is not None:
        ts, center, lower, upper = cached
        print(f"Loaded cached {aggregation} for {label} on {test_env}")
    else:
        print(f"Computing {aggregation} for {label} on {test_env}...")
        ts, center, lower, upper = compute_aggregate_curve(
            method,
            test_env,
            seeds=seeds,
            train_envs=train_envs,
            timesteps_per_env=timesteps_per_env,
            data_dir=DATA_DIR,
            aggregation=aggregation,
        )
        if len(ts) > 0:
            save_to_cache(
                cache_key, method, test_env, aggregation, ts, center, lower, upper
            )

    if len(ts) == 0:
        return pd.DataFrame(columns=["timestep", "center", "lower", "upper"])

    return pd.DataFrame({
        "timestep": ts,
        "center": _smooth(center, smooth),
        "lower": _smooth(lower, smooth),
        "upper": _smooth(upper, smooth),
    })


def load_config(path: str) -> dict:
    """Load and validate a YAML config file for grid plotting."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required for --config mode. Install with: pip install pyyaml"
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "plots" not in cfg or not isinstance(cfg["plots"], list):
        raise ValueError("YAML config must contain a 'plots' list.")

    for i, plot in enumerate(cfg["plots"]):
        if "lines" not in plot or not isinstance(plot["lines"], list):
            raise ValueError(f"Plot entry {i} must contain a 'lines' list.")
        plot_test_env = plot.get("test_env")
        for j, line in enumerate(plot["lines"]):
            if "method" not in line:
                raise ValueError(f"Line {j} in plot {i} must specify 'method'.")
            if plot_test_env is None and "test_env" not in line:
                raise ValueError(
                    f"Plot entry {i} must specify 'test_env', or each of its lines must specify 'test_env'."
                )
            if "envs" in line and not isinstance(line["envs"], list):
                raise ValueError(
                    f"Line {j} in plot {i} must specify 'envs' as a list of strings."
                )
            if "num_tasks" in line and (
                not isinstance(line["num_tasks"], int)
                or isinstance(line["num_tasks"], bool)
                or line["num_tasks"] <= 0
            ):
                raise ValueError(f"Line {j} in plot {i} key 'num_tasks' must be a positive integer.")
            if "linewidth" in line and not isinstance(line["linewidth"], (int, float)):
                raise ValueError(f"Line {j} in plot {i} key 'linewidth' must be a number.")
        zooms = plot.get("zooms")
        if zooms is not None:
            if not isinstance(zooms, list):
                raise ValueError(f"Plot entry {i} has 'zooms' that is not a list.")
            for z_idx, z in enumerate(zooms):
                if "t_start" not in z or "t_end" not in z:
                    raise ValueError(f"Zoom {z_idx} in plot {i} must specify both 't_start' and 't_end'.")
        for key in ["show_timesteps", "show_y_label", "show_x_label"]:
            if key in plot and not isinstance(plot[key], bool):
                raise ValueError(f"Plot entry {i} key '{key}' must be a boolean.")
        if "output_file" in plot and not isinstance(plot["output_file"], str):
            raise ValueError(f"Plot entry {i} key 'output_file' must be a string.")
        if "linewidth" in plot and not isinstance(plot["linewidth"], (int, float)):
            raise ValueError(f"Plot entry {i} key 'linewidth' must be a number.")
        if plot.get("aggregation", "iqm") not in Y_LABELS:
            raise ValueError(
                f"Plot entry {i} key 'aggregation' must be 'iqm' or 'mean'."
            )
    if "defaults" in cfg and isinstance(cfg["defaults"], dict):
        for key in ["format", "ext", "output_file", "output_dir"]:
            if key in cfg["defaults"] and not isinstance(cfg["defaults"][key], str):
                raise ValueError(f"YAML config 'defaults' key '{key}' must be a string.")
        if "linewidth" in cfg["defaults"] and not isinstance(cfg["defaults"]["linewidth"], (int, float)):
            raise ValueError("YAML config 'defaults' key 'linewidth' must be a number.")
        if cfg["defaults"].get("aggregation", "iqm") not in Y_LABELS:
            raise ValueError("YAML config 'defaults' key 'aggregation' must be 'iqm' or 'mean'.")
    return cfg


def _line_envs(line_cfg: dict, fallback: list[str]) -> list[str]:
    """Resolve task IDs used for plot boundaries and cache identity."""
    if "envs" in line_cfg:
        return [str(env) for env in line_cfg["envs"]]
    if "num_tasks" in line_cfg:
        return [str(i) for i in range(line_cfg["num_tasks"])]
    return [str(env) for env in fallback]


def _nice_floor(value: float) -> float:
    """Round *value* down to the nearest 'nice' number in {1, 2, 5} * 10^n."""
    if value <= 0:
        return 0.0
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    mantissa = value / base
    for nice in (5, 2, 1):
        if mantissa >= nice:
            return nice * base
    return base


def _format_timestep(value: float, _position: int) -> str:
    """Format timestep ticks compactly without hiding their scale."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def _decorate_ax(ax, train_envs, timesteps_per_env, title=None, test_env=None, zoomed=False, show_task_labels=True,
                 show_timesteps=True, show_y_label=True, show_x_label=True,
                 y_label=Y_LABELS["iqm"]):
    """Add environment boundary lines, labels, and grid to an axis."""
    fs = FS
    x_lo, x_hi = ax.get_xlim()

    test_env_idx = None
    if test_env and test_env in train_envs:
        test_env_idx = train_envs.index(test_env)

    # Boundary lines
    for i in range(1, len(train_envs)):
        boundary = i * timesteps_per_env
        if x_lo < boundary < x_hi:
            ax.axvline(
                x=boundary,
                color="black",
                linestyle="-",
                alpha=1,
                linewidth=0.5,
                zorder=5,
            )

    # Task labels
    if show_task_labels:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for i, env in enumerate(train_envs):
            env_start = i * timesteps_per_env
            env_end = (i + 1) * timesteps_per_env
            if env_end <= x_lo or env_start >= x_hi:
                continue
            if test_env_idx is not None and test_env_idx > i:
                continue
            center = (i + 0.5) * timesteps_per_env
            ax.text(
                center,
                -0.08,
                f"Task {i + 1}",
                ha="center",
                va="top",
                fontsize=9 * fs,
                color="gray",
                transform=trans,
            )

    if zoomed:
        ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend:
            legend.remove()
        if show_y_label:
            ax.set_ylabel(y_label, fontsize=10 * fs)
        ax.tick_params(axis="both", labelsize=10 * fs)
        ax.tick_params(axis="x", labelbottom=show_timesteps)
        transition_ticks = [
            i * timesteps_per_env
            for i in range(1, len(train_envs))
            if x_lo <= i * timesteps_per_env <= x_hi
        ]
        ax.set_xticks(transition_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(_format_timestep))
        # Connector lines occupy the corridor below the inset. Keep the lone
        # transition label readable even when a connector passes behind it.
        for tick_label in ax.get_xticklabels():
            tick_label.set_bbox({
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 1.5,
            })
            tick_label.set_zorder(10)
        y_lo, y_hi = ax.get_ylim()
        y_max = _nice_floor(y_hi)
        yticks = [0, y_max / 2, y_max]
        ax.set_yticks(yticks)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(False)
    else:
        if show_x_label:
            task_label_pad = 34 if show_task_labels else 6
            ax.set_xlabel(X_LABEL, fontsize=10 * fs, labelpad=task_label_pad)
        if show_y_label:
            ax.set_ylabel(y_label, fontsize=10 * fs)
        if title:
            ax.set_title(title, pad=14, fontsize=11 * fs)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=9 * fs)
        ax.tick_params(axis='both', labelsize=10 * fs)
        if not show_timesteps:
            ax.tick_params(axis='x', labelbottom=False)
        # Set x-ticks only at task boundaries and 0
        xticks = [0] + [(idx + 1) * timesteps_per_env for idx in range(len(train_envs))]
        xticks = [x for x in xticks if x_lo <= x <= x_hi]
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(_format_timestep))
        y_lo, y_hi = ax.get_ylim()
        y_max = _nice_floor(y_hi)
        yticks = [0, y_max / 2, y_max]
        ax.set_yticks(yticks)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(False)


def plot_zoom_figure(plot_cfg, cache_key, use_cache, seeds, envs, timesteps, env_name, plot_output_dir, output_file, dpi, defaults, index, total_plots):
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import ConnectionPatch

    zooms = plot_cfg["zooms"]
    n_zooms = len(zooms)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, n_zooms, height_ratios=[1, 1.2])

    ax_zooms = [fig.add_subplot(gs[0, i]) for i in range(n_zooms)]
    ax_main = fig.add_subplot(gs[1, :])

    plot_timesteps = plot_cfg.get("timesteps", timesteps)
    plot_envs = plot_cfg.get("envs", envs)
    if "envs" not in plot_cfg and plot_cfg.get("lines"):
        plot_envs = _line_envs(plot_cfg["lines"][0], envs)

    test_env = plot_cfg.get("test_env")
    aggregation = plot_cfg.get("aggregation", defaults.get("aggregation", "iqm"))
    y_label = Y_LABELS[aggregation]
    title = plot_cfg.get("title", None)
    if title is None and env_name:
        if test_env:
            title = f"Evaluation on {env_name}-{test_env}"
        else:
            title = f"Evaluation on {env_name}"

    n_lines_plotted = 0
    for line_idx, line_cfg in enumerate(plot_cfg["lines"]):
        method = line_cfg["method"]
        label = line_cfg.get("label", get_label(method))
        color = line_cfg.get("color", get_color(method, line_idx))
        line_test_env = line_cfg.get("test_env", test_env)
        line_envs = _line_envs(line_cfg, plot_envs)
        line_timesteps = line_cfg.get("timesteps", plot_timesteps)

        smooth = plot_cfg.get("smooth", defaults.get("smooth"))
        line_data = compute_line_data(
            cache_key,
            method,
            line_test_env,
            label=label,
            seeds=seeds,
            train_envs=line_envs,
            timesteps_per_env=line_timesteps,
            aggregation=aggregation,
            smooth=smooth,
            use_cache=use_cache,
        )
        if line_data.empty:
            print(f"  No data for {method}/{line_test_env}")
            continue

        ts = line_data["timestep"].to_numpy()
        center_smoothed = line_data["center"].to_numpy()
        lower_smoothed = line_data["lower"].to_numpy()
        upper_smoothed = line_data["upper"].to_numpy()

        linewidth = line_cfg.get("linewidth", plot_cfg.get("linewidth", defaults.get("linewidth", 0.7)))

        # Plot on main
        ax_main.plot(ts, center_smoothed, label=label, color=color, linewidth=linewidth)
        ax_main.fill_between(ts, lower_smoothed, upper_smoothed, alpha=0.2, color=color)
        n_lines_plotted += 1

        # Plot on each zoom axis
        for i, z_cfg in enumerate(zooms):
            ax_zooms[i].plot(ts, center_smoothed, label=label, color=color, linewidth=linewidth)
            ax_zooms[i].fill_between(ts, lower_smoothed, upper_smoothed, alpha=0.2, color=color)

    if n_lines_plotted == 0:
        plt.close(fig)
        raise RuntimeError(f"No data found for zoom plot: {title or output_file}")

    show_timesteps = plot_cfg.get("show_timesteps", defaults.get("show_timesteps", True))
    show_y_label = plot_cfg.get("show_y_label", defaults.get("show_y_label", True))
    show_x_label = plot_cfg.get("show_x_label", defaults.get("show_x_label", True))

    # Decorate main axis
    _decorate_ax(ax_main, plot_envs, plot_timesteps, title=None, test_env=test_env, zoomed=False, show_task_labels=True,
                 show_timesteps=show_timesteps, show_y_label=False, show_x_label=show_x_label,
                 y_label=y_label)
    if title:
        fig.suptitle(title, fontsize=11 * FS, y=0.965)
    if show_y_label:
        fig.supylabel(y_label, fontsize=10 * FS, x=0.015)

    # Sync y limits and mark the ranges represented by the zoom panels.
    y_min, y_max = ax_main.get_ylim()
    for i, z_cfg in enumerate(zooms):
        z_start = z_cfg["t_start"]
        z_end = z_cfg["t_end"]

        ax_zooms[i].set_xlim(left=z_start, right=z_end)
        ax_zooms[i].set_ylim(bottom=y_min, top=y_max)

        # Decorate zoom axis (hide task labels)
        _decorate_ax(ax_zooms[i], plot_envs, plot_timesteps, title=None, test_env=test_env, zoomed=True, show_task_labels=False,
                     show_timesteps=show_timesteps, show_y_label=False, show_x_label=show_x_label,
                     y_label=y_label)

        # Draw a lightly shaded range on the main plot.  The visible outline is
        # also where the connector lines will terminate.
        rect = plt.Rectangle((z_start, y_min), z_end - z_start, y_max - y_min,
                             facecolor=(0, 0, 0, 0.035), edgecolor="0.3",
                             linewidth=1.0, zorder=6)
        ax_main.add_patch(rect)

    # Leave a deliberate corridor between the zooms and the overview.  The
    # previous default tight layout made the connectors so short that the top
    # panels looked like independent plots rather than magnified ranges.
    fig.tight_layout(
        rect=(0.035 if show_y_label else 0, 0, 1, 0.965 if title else 1),
        h_pad=3.0,
        w_pad=2.0,
    )

    # Draw the connectors only after layout has fixed the axes positions, so
    # each line precisely joins a zoom-panel corner to its overview range.
    for i, z_cfg in enumerate(zooms):
        z_start = z_cfg["t_start"]
        z_end = z_cfg["t_end"]

        con_left = ConnectionPatch(xyA=(0, 0), xyB=(z_start, y_max), coordsA="axes fraction", coordsB="data",
                                   axesA=ax_zooms[i], axesB=ax_main, color="0.3", linestyle="--",
                                   linewidth=1.0, clip_on=False, zorder=-1)
        con_right = ConnectionPatch(xyA=(1, 0), xyB=(z_end, y_max), coordsA="axes fraction", coordsB="data",
                                    axesA=ax_zooms[i], axesB=ax_main, color="0.3", linestyle="--",
                                    linewidth=1.0, clip_on=False, zorder=-1)
        fig.add_artist(con_left)
        fig.add_artist(con_right)

    file_format = defaults.get("format", defaults.get("ext", "svg")).lstrip(".")
    # Save figure
    plot_output_file = plot_cfg.get("output_file")
    if plot_output_file:
        out_path = plot_output_dir / f"{plot_output_file}.{file_format}"
    elif total_plots == 1:
        out_path = plot_output_dir / f"{output_file}.{file_format}"
    else:
        # The index prevents plots that share a test environment from silently
        # overwriting one another.
        test_env_suffix = test_env if test_env else "mixed"
        out_path = plot_output_dir / f"{output_file}_{index + 1:02d}_{test_env_suffix}_zoom.{file_format}"

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved zoom plot to {out_path}")


def grid_cache_key(config: dict) -> str:
    """Return the cache key shared by plotting and compute-only config mode."""
    defaults = config.get("defaults", {})
    timesteps = defaults.get("timesteps", TIMESTEPS_PER_ENV)
    envs = defaults.get("envs", TRAIN_ENVS)
    output_file = defaults.get("output_file", "iqm_grid")
    all_methods = []
    for plot_cfg in config["plots"]:
        plot_timesteps = plot_cfg.get("timesteps", timesteps)
        aggregation = plot_cfg.get("aggregation", defaults.get("aggregation", "iqm"))
        for line_cfg in plot_cfg["lines"]:
            line_envs = _line_envs(line_cfg, plot_cfg.get("envs", envs))
            line_timesteps = line_cfg.get("timesteps", plot_timesteps)
            all_methods.append(
                f"{line_cfg['method']}_{'-'.join(line_envs)}_{line_timesteps}_{aggregation}"
            )
    return make_cache_key(all_methods, output_file, aggregation="configured")


def compute_plot_data(config: dict, use_cache: bool = True) -> pd.DataFrame:
    """Compute all YAML-configured lines and return the values used for plotting.

    The returned frame has one row per timestep and includes plot/line metadata
    so lines with the same label or method remain distinguishable.
    """
    defaults = config.get("defaults", {})
    seeds = defaults.get("seeds", SEEDS)
    timesteps = defaults.get("timesteps", TIMESTEPS_PER_ENV)
    envs = defaults.get("envs", TRAIN_ENVS)
    cache_key = grid_cache_key(config)
    frames = []

    if use_cache:
        print(f"Cache key: {cache_key}  (use --no-cache to force recompute)")

    for plot_idx, plot_cfg in enumerate(config["plots"]):
        plot_timesteps = plot_cfg.get("timesteps", timesteps)
        plot_envs = plot_cfg.get("envs", envs)
        if "envs" not in plot_cfg and plot_cfg.get("lines"):
            plot_envs = _line_envs(plot_cfg["lines"][0], envs)
        plot_test_env = plot_cfg.get("test_env")
        aggregation = plot_cfg.get("aggregation", defaults.get("aggregation", "iqm"))

        for line_idx, line_cfg in enumerate(plot_cfg["lines"]):
            method = line_cfg["method"]
            label = line_cfg.get("label", get_label(method))
            test_env = line_cfg.get("test_env", plot_test_env)
            line_envs = _line_envs(line_cfg, plot_envs)
            line_timesteps = line_cfg.get("timesteps", plot_timesteps)
            smooth = plot_cfg.get("smooth", defaults.get("smooth"))
            frame = compute_line_data(
                cache_key,
                method,
                test_env,
                label=label,
                seeds=seeds,
                train_envs=line_envs,
                timesteps_per_env=line_timesteps,
                aggregation=aggregation,
                smooth=smooth,
                use_cache=use_cache,
            )
            if frame.empty:
                print(f"  No data for {method}/{test_env}")
                continue

            frame.insert(0, "test_env", test_env)
            frame.insert(0, "aggregation", aggregation)
            frame.insert(0, "method", method)
            frame.insert(0, "label", label)
            frame.insert(0, "line_index", line_idx)
            frame.insert(0, "plot_title", plot_cfg.get("title", ""))
            frame.insert(0, "plot_index", plot_idx)
            frames.append(frame)

    columns = [
        "plot_index", "plot_title", "line_index", "label", "method",
        "test_env", "aggregation", "timestep", "center", "lower", "upper",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def plot_grid(config: dict, use_cache: bool):
    """Main driver for YAML-config grid plotting."""
    defaults = config.get("defaults", {})
    plots = config["plots"]

    seeds = defaults.get("seeds", SEEDS)
    timesteps = defaults.get("timesteps", TIMESTEPS_PER_ENV)
    envs = defaults.get("envs", TRAIN_ENVS)
    env_name = defaults.get("env_name", "")
    output_subdir = defaults.get("output_dir", "")
    output_file = defaults.get("output_file", "iqm_grid")
    dpi = defaults.get("dpi", 300)

    if output_subdir:
        plot_output_dir = OUTPUT_DIR / output_subdir
    else:
        plot_output_dir = OUTPUT_DIR
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    cache_key = grid_cache_key(config)
    if use_cache:
        print(f"Cache key: {cache_key}  (use --no-cache to force recompute)")

    zoom_plots = [p for p in plots if "zooms" in p]
    grid_plots = [p for p in plots if "zooms" not in p]

    for zoom_idx, plot_cfg in enumerate(zoom_plots):
        plot_zoom_figure(plot_cfg, cache_key, use_cache, seeds, envs, timesteps, env_name, plot_output_dir, output_file, dpi, defaults, zoom_idx, len(plots))

    if not grid_plots:
        return

    n_plots = len(grid_plots)
    grid = defaults.get("grid", None)
    if grid:
        nrows, ncols = grid
    elif n_plots == 1:
        nrows, ncols = 1, 1
    else:
        ncols = math.ceil(math.sqrt(n_plots))
        nrows = math.ceil(n_plots / ncols)

    any_zoomed = any(
        p.get("t_start", defaults.get("t_start")) is not None
        or p.get("t_end", defaults.get("t_end")) is not None
        for p in grid_plots
    )

    if any_zoomed:
        fig_w = 7 * ncols
        fig_h = 5 * nrows
    else:
        fig_w = 10 * ncols
        fig_h = 5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for plot_idx, plot_cfg in enumerate(grid_plots):
        row = plot_idx // ncols
        col = plot_idx % ncols
        ax = axes[row][col]

        plot_timesteps = plot_cfg.get("timesteps", timesteps)
        plot_envs = plot_cfg.get("envs", envs)
        if "envs" not in plot_cfg and plot_cfg.get("lines"):
            plot_envs = _line_envs(plot_cfg["lines"][0], envs)

        test_env = plot_cfg.get("test_env")
        aggregation = plot_cfg.get("aggregation", defaults.get("aggregation", "iqm"))
        y_label = Y_LABELS[aggregation]
        title = plot_cfg.get("title", None)
        if title is None and env_name:
            if test_env:
                title = f"Evaluation on {env_name}-{test_env}"
            else:
                title = f"Evaluation on {env_name}"

        n_lines_plotted = 0
        for line_idx, line_cfg in enumerate(plot_cfg["lines"]):
            method = line_cfg["method"]
            label = line_cfg.get("label", get_label(method))
            color = line_cfg.get("color", get_color(method, line_idx))
            line_test_env = line_cfg.get("test_env", test_env)
            line_envs = _line_envs(line_cfg, plot_envs)
            line_timesteps = line_cfg.get("timesteps", plot_timesteps)

            smooth = plot_cfg.get("smooth", defaults.get("smooth"))
            line_data = compute_line_data(
                cache_key,
                method,
                line_test_env,
                label=label,
                seeds=seeds,
                train_envs=line_envs,
                timesteps_per_env=line_timesteps,
                aggregation=aggregation,
                smooth=smooth,
                use_cache=use_cache,
            )
            if line_data.empty:
                print(f"  No data for {method}/{line_test_env}")
                continue

            ts = line_data["timestep"].to_numpy()
            center = line_data["center"].to_numpy()
            lower = line_data["lower"].to_numpy()
            upper = line_data["upper"].to_numpy()

            linewidth = line_cfg.get("linewidth", plot_cfg.get("linewidth", defaults.get("linewidth", 0.7)))
            ax.plot(ts, center, label=label, color=color, linewidth=linewidth)
            ax.fill_between(ts, lower, upper, alpha=0.2, color=color)
            n_lines_plotted += 1

        if n_lines_plotted == 0:
            plt.close(fig)
            raise RuntimeError(f"No data found for plot: {title or output_file}")

        t_start = plot_cfg.get("t_start", defaults.get("t_start"))
        t_end = plot_cfg.get("t_end", defaults.get("t_end"))
        zoomed = t_start is not None or t_end is not None
        if zoomed:
            ax.set_xlim(left=t_start, right=t_end)

        show_timesteps = plot_cfg.get("show_timesteps", defaults.get("show_timesteps", True))
        show_y_label = plot_cfg.get("show_y_label", defaults.get("show_y_label", True))
        show_x_label = plot_cfg.get("show_x_label", defaults.get("show_x_label", True))
        _decorate_ax(
            ax, plot_envs, plot_timesteps, title=title, test_env=test_env, zoomed=zoomed,
            show_timesteps=show_timesteps, show_y_label=show_y_label, show_x_label=show_x_label,
            y_label=y_label,
        )

    for idx in range(n_plots, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].set_visible(False)

    plt.tight_layout()
    file_format = defaults.get("format", defaults.get("ext", "svg")).lstrip(".")
    out_path = plot_output_dir / f"{output_file}.{file_format}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate aggregate reward plots for each test environment.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file for grid plotting.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        help=f"Method names to plot (default: {DEFAULT_METHODS})",
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Filename prefix for output plots, e.g. 'encode' -> iqm_encode_V1.png",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached results and recompute everything.",
    )
    parser.add_argument(
        "--aggregation", choices=sorted(Y_LABELS), default=None,
        help=("Statistic and uncertainty band to plot: 'iqm' uses a 95%% bootstrap CI; "
              "'mean' uses standard error (default: iqm)."),
    )
    parser.add_argument(
        "--compute-only", action="store_true",
        help="Compute the plotted line data, save it as CSV, and skip figure generation.",
    )
    parser.add_argument(
        "--data-output", type=str, default=None,
        help="CSV path for --compute-only (default: <plot output>/<name>_data.csv).",
    )
    parser.add_argument(
        "--envs", "--env_order", nargs="+", default=TEST_ENVS,
        help=f"Environments to plot and their training order (default: {TEST_ENVS})",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help=f"Seeds to include (default: {SEEDS})",
    )
    parser.add_argument(
        "--timesteps", type=int, default=TIMESTEPS_PER_ENV,
        help=f"Timesteps per environment (default: {TIMESTEPS_PER_ENV})",
    )
    parser.add_argument(
        "--env_name", type=str, help="Base name of the environment",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Subdirectory inside output/plots to save plots",
    )
    parser.add_argument(
        "--t_start", type=int, default=None,
        help="Start of global timestep range to zoom into.",
    )
    parser.add_argument(
        "--t_end", type=int, default=None,
        help="End of global timestep range to zoom into.",
    )
    parser.add_argument(
        "--smooth", type=int, default=None,
        help="Window size for peak-aware moving-average smoothing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_cache = not args.no_cache

    if args.config:
        cfg = load_config(args.config)
        defaults = cfg.setdefault("defaults", {})
        if args.env_name and "env_name" not in defaults:
            defaults["env_name"] = args.env_name
        if args.output_dir and "output_dir" not in defaults:
            defaults["output_dir"] = args.output_dir
        if args.smooth is not None and "smooth" not in defaults:
            defaults["smooth"] = args.smooth
        if args.aggregation is not None:
            defaults["aggregation"] = args.aggregation
        if args.compute_only:
            data = compute_plot_data(cfg, use_cache)
            if data.empty:
                raise RuntimeError("No data found for any configured plot line.")
            if args.data_output:
                data_path = Path(args.data_output)
            else:
                output_dir = OUTPUT_DIR / defaults.get("output_dir", "")
                data_path = output_dir / f"{defaults.get('output_file', 'iqm_grid')}_data.csv"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(data_path, index=False)
            print(f"Saved plot data to {data_path}")
            print("Done!")
            return
        plot_grid(cfg, use_cache)
        print("Done!")
        return

    methods = args.methods
    prefix = args.prefix
    train_envs = args.envs
    test_envs = args.envs
    seeds = args.seeds
    timesteps = args.timesteps
    env_name = args.env_name
    output_subdir = args.output_dir
    t_start = args.t_start
    t_end = args.t_end
    smooth = args.smooth
    aggregation = args.aggregation or "iqm"

    if output_subdir:
        plot_output_dir = OUTPUT_DIR / output_subdir
    else:
        plot_output_dir = OUTPUT_DIR
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    cache_key = make_cache_key(methods, prefix, aggregation=aggregation)
    if use_cache:
        print(f"Cache key: {cache_key}  (use --no-cache to force recompute)")

    if args.compute_only:
        frames = []
        for test_env in test_envs:
            for line_idx, method in enumerate(methods):
                label = get_label(prefix)
                frame = compute_line_data(
                    cache_key,
                    method,
                    test_env,
                    label=label,
                    seeds=seeds,
                    train_envs=train_envs,
                    timesteps_per_env=timesteps,
                    aggregation=aggregation,
                    smooth=smooth,
                    use_cache=use_cache,
                )
                if frame.empty:
                    print(f"  No data for {method}/{test_env}")
                    continue
                frame.insert(0, "test_env", test_env)
                frame.insert(0, "aggregation", aggregation)
                frame.insert(0, "method", method)
                frame.insert(0, "label", label)
                frame.insert(0, "line_index", line_idx)
                frames.append(frame)

        if not frames:
            raise RuntimeError("No data found for any requested plot line.")
        data = pd.concat(frames, ignore_index=True)
        if args.data_output:
            data_path = Path(args.data_output)
        else:
            parts = [aggregation]
            if prefix:
                parts.append(prefix)
            data_path = plot_output_dir / f"{'_'.join(parts)}_data.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_path, index=False)
        print(f"Saved plot data to {data_path}")
        print("Done!")
        return

    for test_env in test_envs:
        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, method in enumerate(methods):
            label = get_label(prefix)
            line_data = compute_line_data(
                cache_key,
                method,
                test_env,
                label=label,
                seeds=seeds,
                train_envs=train_envs,
                timesteps_per_env=timesteps,
                aggregation=aggregation,
                smooth=smooth,
                use_cache=use_cache,
            )
            if line_data.empty:
                print(f"  No data for {method}/{test_env}")
                continue

            color = get_color(method, idx)
            ts = line_data["timestep"].to_numpy()
            center = line_data["center"].to_numpy()
            lower = line_data["lower"].to_numpy()
            upper = line_data["upper"].to_numpy()
            ax.plot(ts, center, label=label, color=color, linewidth=0.7)
            ax.fill_between(ts, lower, upper, alpha=0.2, color=color)

        zoomed = t_start is not None or t_end is not None
        if zoomed:
            ax.set_xlim(left=t_start, right=t_end)

        _decorate_ax(
            ax, train_envs, timesteps,
            title=f"Evaluation on {env_name}-{test_env}",
            test_env=test_env,
            zoomed=zoomed,
            y_label=Y_LABELS[aggregation],
        )

        plt.tight_layout()
        parts = [aggregation]
        if prefix:
            parts.append(prefix)
        parts.append(test_env)
        out_path = plot_output_dir / f"{'_'.join(parts)}.png"

        fig.savefig(out_path, dpi=1000, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
