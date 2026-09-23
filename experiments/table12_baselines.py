# -*- coding: utf-8 -*-
"""Table 12 aggregate performance comparison: RALF vs. classical and
single-algorithm baselines.

Compares RALF (the heterogeneous A2C/DDPG/PPO architecture) against:
  - (s, S) policy              -- classical inventory control, no learning
  - Base-stock policy          -- classical inventory control, no learning
  - Single-agent PPO ("MAPPO") -- same 3-agent structure, but every agent
                                   uses PPO instead of RALF's heterogeneous
                                   A2C/DDPG/PPO split

on: Fill Rate (%), Inv. Turnover (cyc/yr), Profit ($), Overstock (%),
Stockout (%).

Uses the same scaled-down budget as the Table 9 sweep (see
experiments/sweep_table9.py) for consistency: a handful of seeds, a
reduced episode count -- illustrative for a paper, not a
publication-grade convergence study.

Inv. Turnover is annualized from the per-episode (60-day)
'inventory_turnover' metric via a straight (365 / episode_length) scale-up;
it is not re-derived from first principles.

Usage:
    python experiments/table12_baselines.py
"""
import contextlib
import io
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ralf.baselines import BaseStockPolicy, SSPolicy, run_classical_policy, train_homogeneous_ppo
from ralf.config import ExperimentConfig
from ralf.data import load_retail_dataset
from ralf.training import train_heterogeneous_marl_with_timing

DATA_PATH = "data/retail_store_inventory.csv"
OUTPUT_DIR = "outputs"

SEEDS = [42, 1, 2, 3]
EPISODES = 300
TAIL_WINDOW = 20
EPISODE_LENGTH_DAYS = 60

METRIC_KEYS = ['Fill_Rate_pct', 'Inv_Turnover_cyc_yr', 'Profit', 'Overstock_pct', 'Stockout_pct']

SS_POLICY = SSPolicy(s=150.0, S=300.0, max_order_quantity=500)
BASE_STOCK_POLICY = BaseStockPolicy(base_stock_level=300.0, max_order_quantity=500)


def summarize(metrics_history):
    tail = metrics_history[-TAIL_WINDOW:]
    turnover = np.mean([m.get('inventory_turnover', 0.0) for m in tail]) * (365.0 / EPISODE_LENGTH_DAYS)
    return {
        'Fill_Rate_pct': float(np.mean([m.get('service_level', 0.0) for m in tail])),
        'Inv_Turnover_cyc_yr': float(turnover),
        'Profit': float(np.mean([m.get('total_profit', 0.0) for m in tail])),
        'Overstock_pct': float(np.mean([m.get('overstock_rate', 0.0) for m in tail])),
        'Stockout_pct': float(np.mean([m.get('stockout_rate', 0.0) for m in tail])),
    }


def run_ralf(df, seed):
    exp_config = ExperimentConfig(num_episodes=EPISODES, seed=seed, device="cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, _, metrics, _, _ = train_heterogeneous_marl_with_timing(df, exp_config)
    return summarize(metrics)


def run_all_ppo(df, seed):
    exp_config = ExperimentConfig(num_episodes=EPISODES, seed=seed, device="cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, _, metrics = train_homogeneous_ppo(df, exp_config)
    return summarize(metrics)


def run_ss(df, seed):
    exp_config = ExperimentConfig(num_episodes=EPISODES, seed=seed, device="cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = run_classical_policy(df, SS_POLICY, exp_config)
    return summarize(metrics)


def run_base_stock(df, seed):
    exp_config = ExperimentConfig(num_episodes=EPISODES, seed=seed, device="cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = run_classical_policy(df, BASE_STOCK_POLICY, exp_config)
    return summarize(metrics)


METHODS = [
    ("RALF", run_ralf),
    ("(s,S) policy", run_ss),
    ("Base-stock policy", run_base_stock),
    ("Single-agent PPO (MAPPO)", run_all_ppo),
]


def plot_comparison(results_df, output_path):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    metrics_to_plot = [
        ('Fill_Rate_pct', 'Fill Rate (%)', axes[0, 0], 'steelblue'),
        ('Inv_Turnover_cyc_yr', 'Inv. Turnover (cyc/yr)', axes[0, 1], 'darkorange'),
        ('Profit', 'Profit ($)', axes[0, 2], 'seagreen'),
        ('Overstock_pct', 'Overstock (%)', axes[1, 0], 'indianred'),
        ('Stockout_pct', 'Stockout (%)', axes[1, 1], 'purple'),
    ]
    for key, title, ax, color in metrics_to_plot:
        ax.bar(results_df['Method'], results_df[key], yerr=results_df[f'{key}_std'],
               capsize=5, color=color, edgecolor='black', alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=20, labelsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    axes[1, 2].axis('off')
    fig.suptitle(f'RALF vs. Baselines ({len(SEEDS)} seeds, {EPISODES} episodes/run)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def format_markdown(results_df):
    lines = [
        "| Method | Fill Rate (%) | Inv. Turnover (cyc/yr) | Profit ($) | Overstock (%) | Stockout (%) |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['Method']} | {row['Fill_Rate_pct']:.1f} +/- {row['Fill_Rate_pct_std']:.1f} | "
            f"{row['Inv_Turnover_cyc_yr']:.2f} +/- {row['Inv_Turnover_cyc_yr_std']:.2f} | "
            f"{row['Profit']:.0f} +/- {row['Profit_std']:.0f} | "
            f"{row['Overstock_pct']:.1f} +/- {row['Overstock_pct_std']:.1f} | "
            f"{row['Stockout_pct']:.1f} +/- {row['Stockout_pct_std']:.1f} |"
        )
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_retail_dataset(DATA_PATH)

    rows = []
    for name, run_fn in METHODS:
        print(f"\n=== {name} ===")
        seed_results = []
        for seed in SEEDS:
            t0 = time.time()
            result = run_fn(df, seed)
            seed_results.append(result)
            print(f"  seed={seed} -> Fill Rate={result['Fill_Rate_pct']:.1f}%  "
                  f"Turnover={result['Inv_Turnover_cyc_yr']:.2f}  Profit={result['Profit']:.0f}  "
                  f"Overstock={result['Overstock_pct']:.1f}%  Stockout={result['Stockout_pct']:.1f}%  "
                  f"({time.time() - t0:.1f}s)")

        row = {'Method': name}
        for key in METRIC_KEYS:
            values = [r[key] for r in seed_results]
            row[key] = float(np.mean(values))
            row[f'{key}_std'] = float(np.std(values))
        rows.append(row)

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'table12_baseline_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_path = os.path.join(OUTPUT_DIR, 'table12_baseline_comparison.png')
    plot_comparison(results_df, plot_path)

    summary_path = os.path.join(OUTPUT_DIR, 'table12_summary.md')
    with open(summary_path, 'w') as f:
        f.write(format_markdown(results_df))
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 100)
    print("TABLE 12 COMPARISON COMPLETE")
    print("=" * 100)
    print(format_markdown(results_df))

    return results_df


if __name__ == "__main__":
    main()
