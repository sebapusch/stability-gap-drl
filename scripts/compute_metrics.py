#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from common import (
    parse_config,
    enumerate_combinations,
    load_all_csvs,
    get_env_max_return,
    compute_final_performance_from_data,
    compute_min_acc_from_data,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_SMOOTH = 5
N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95


def format_makecell(mean: float, ci_low: float, ci_high: float, precision: int = 1) -> str:
    """Format as \\makecell{<mean> \\footnotesize (<ci_low>,<ci_high>)}."""
    if np.isnan(mean):
        return r"\makecell{N/A}"
    return (
        rf"\makecell{{{mean:.{precision}f} "
        rf"\footnotesize ({ci_low:.{precision}f},{ci_high:.{precision}f})}}"
    )


def format_ci(mean: float, ci_low: float, ci_high: float, precision: int = 1) -> str:
    """Format as 'mean [ci_low, ci_high]'."""
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{precision}f} [{ci_low:.{precision}f}, {ci_high:.{precision}f}]"


def main():
    parser = argparse.ArgumentParser(
        description="Compute final performance P(Vi), P(T), and min-ACC for RL ablation experiments.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML experiment config file.",
    )
    parser.add_argument(
        "--smooth", type=int, default=DEFAULT_SMOOTH,
        help=f"Number of last evaluation points to average for final performance smoothing (default: {DEFAULT_SMOOTH}).",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=f"Override the data directory (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save output files (CSV, Markdown, LaTeX). If not set, prints to stdout only.",
    )
    parser.add_argument(
        "--confidence", type=float, default=CONFIDENCE,
        help=f"Confidence level for bootstrap CI (default: {CONFIDENCE}).",
    )
    parser.add_argument(
        "--mean", action="store_true",
        help="Use Mean (instead of IQM) across seeds for timestep aggregation (applies to both P(T) and min-ACC).",
    )
    parser.add_argument(
        "--precision", type=int, default=1,
        help="Number of decimal places for formatted values (default: 1).",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    cfg = parse_config(args.config)

    env_name = cfg["env"]
    benchmark = cfg["benchmark"]
    seeds = cfg["seeds"]
    name_prefix = cfg["name_prefix"]
    project = cfg["project"]
    all_ablation_keys = cfg["all_ablation_keys"]
    hp_keys = cfg["hp_keys"]
    use_iqm = not args.mean
    env_max = get_env_max_return(env_name)

    k_target = 3
    k_idx = k_target - 1

    print(f"Config: {args.config}", file=sys.stderr)
    print(f"  Name prefix:  {name_prefix}", file=sys.stderr)
    print(f"  Project:      {project}", file=sys.stderr)
    print(f"  Benchmark:    {benchmark}", file=sys.stderr)
    print(f"  Seeds:        {seeds}", file=sys.stderr)
    print(f"  Smoothing:    last {args.smooth} eval points", file=sys.stderr)
    print(f"  Confidence:   {args.confidence * 100:.0f}%", file=sys.stderr)
    print(f"  Aggregation:  {'IQM' if use_iqm else 'Mean'} by timestep", file=sys.stderr)
    print(file=sys.stderr)

    if len(benchmark) <= k_idx:
        print(f"Error: Benchmark needs at least {k_target} tasks for min-ACC at k={k_target}.", file=sys.stderr)
        sys.exit(1)

    results = []

    timesteps_per_env = cfg.get("total_timesteps", 40000)

    for hp_combo in enumerate_combinations(cfg):
        # 1. Load data once for this HP combination
        combo_data = load_all_csvs(data_dir, name_prefix, benchmark, hp_combo, seeds, all_ablation_keys, project)

        # 2. Compute final performance
        perf = compute_final_performance_from_data(
            combo_data, benchmark, args.smooth, use_iqm, env_max, args.confidence, N_BOOTSTRAP, timesteps_per_env
        )

        # 3. Compute min-ACC
        minacc = compute_min_acc_from_data(
            combo_data, benchmark, k_idx, use_iqm, args.confidence, N_BOOTSTRAP, timesteps_per_env
        )

        # Merge results into a single row dictionary
        row = dict(hp_combo)
        row["n_seeds"] = perf["n_seeds"]
        
        # Add final performance fields
        for env in benchmark:
            row[f"{env}_mean"] = perf[f"{env}_mean"]
            row[f"{env}_ci_low"] = perf[f"{env}_ci_low"]
            row[f"{env}_ci_high"] = perf[f"{env}_ci_high"]
        row["P(T)_mean"] = perf["P(T)_mean"]
        row["P(T)_ci_low"] = perf["P(T)_ci_low"]
        row["P(T)_ci_high"] = perf["P(T)_ci_high"]

        # Add min-ACC fields
        row["min-ACC_mean"] = minacc["min-ACC_mean"]
        row["min-ACC_ci_low"] = minacc["min-ACC_ci_low"]
        row["min-ACC_ci_high"] = minacc["min-ACC_ci_high"]

        results.append(row)

    df_results = pd.DataFrame(results)

    # ---------------------------------------------------------------------------
    # Generate Output Formats
    # ---------------------------------------------------------------------------
    precision = args.precision

    # 1. LaTeX Rows
    latex_lines = []
    for _, row in df_results.iterrows():
        if hp_keys:
            hp_label = ", ".join(f"{k}={row[k]}" for k in hp_keys)
        else:
            hp_label = name_prefix

        parts = []
        for env in benchmark:
            parts.append(format_makecell(row[f"{env}_mean"], row[f"{env}_ci_low"], row[f"{env}_ci_high"], precision))
        parts.append(format_makecell(row["P(T)_mean"], row["P(T)_ci_low"], row["P(T)_ci_high"], precision))
        parts.append(format_makecell(row["min-ACC_mean"], row["min-ACC_ci_low"], row["min-ACC_ci_high"], precision))

        cells = " & ".join(f" {p}" for p in parts)
        latex_lines.append(f"{hp_label} & {cells} \\\\")
    latex_output = "\n".join(latex_lines)

    # 2. Markdown Table
    md_lines = []
    md_lines.append(f"# Performance Metrics — {name_prefix}")
    md_lines.append("")
    md_lines.append(f"Smoothing: last {args.smooth} eval points | Seeds: {len(seeds)} | Confidence: {args.confidence * 100:.0f}% | Aggregation: {'IQM' if use_iqm else 'Mean'}")
    md_lines.append("")
    
    headers = [k for k in hp_keys] + [env for env in benchmark] + ["P(T)", "min-ACC", "Seeds"]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df_results.iterrows():
        parts = [str(row[k]) for k in hp_keys]
        for env in benchmark:
            parts.append(format_ci(row[f"{env}_mean"], row[f"{env}_ci_low"], row[f"{env}_ci_high"], precision))
        parts.append(format_ci(row["P(T)_mean"], row["P(T)_ci_low"], row["P(T)_ci_high"], precision))
        parts.append(format_ci(row["min-ACC_mean"], row["min-ACC_ci_low"], row["min-ACC_ci_high"], precision))
        parts.append(str(int(row["n_seeds"])))
        md_lines.append("| " + " | ".join(parts) + " |")
    md_output = "\n".join(md_lines)

    # 3. CSV DataFrame
    csv_rows = []
    for _, row in df_results.iterrows():
        out = {k: row[k] for k in hp_keys}
        for env in benchmark:
            out[f"{env}_mean"] = row[f"{env}_mean"]
            out[f"{env}_ci_low"] = row[f"{env}_ci_low"]
            out[f"{env}_ci_high"] = row[f"{env}_ci_high"]
            out[f"{env}_formatted"] = format_ci(row[f"{env}_mean"], row[f"{env}_ci_low"], row[f"{env}_ci_high"], precision)
        out["P(T)_mean"] = row["P(T)_mean"]
        out["P(T)_ci_low"] = row["P(T)_ci_low"]
        out["P(T)_ci_high"] = row["P(T)_ci_high"]
        out["P(T)_formatted"] = format_ci(row["P(T)_mean"], row["P(T)_ci_low"], row["P(T)_ci_high"], precision)
        out["min-ACC_mean"] = row["min-ACC_mean"]
        out["min-ACC_ci_low"] = row["min-ACC_ci_low"]
        out["min-ACC_ci_high"] = row["min-ACC_ci_high"]
        out["min-ACC_formatted"] = format_ci(row["min-ACC_mean"], row["min-ACC_ci_low"], row["min-ACC_ci_high"], precision)
        out["n_seeds"] = int(row["n_seeds"])
        csv_rows.append(out)
    df_csv = pd.DataFrame(csv_rows)

    # Print LaTeX to stdout by default (matches compute_table_row.py)
    print(latex_output)

    # Save outputs if output_dir is requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tex_path = out_dir / f"{name_prefix}_metrics.tex"
        tex_path.write_text(latex_output)
        print(f"Saved LaTeX: {tex_path}", file=sys.stderr)

        md_path = out_dir / f"{name_prefix}_metrics.md"
        md_path.write_text(md_output)
        print(f"Saved Markdown: {md_path}", file=sys.stderr)

        csv_path = out_dir / f"{name_prefix}_metrics.csv"
        df_csv.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
