# -*- coding: utf-8 -*-
"""Hyperparameter sensitivity sweep for the paper's Table 9.

Sweeps two dimensions independently (one at a time, all else held at
default), per the table's own instruction to vary one dimension while
keeping the others fixed:

  - w_stockout:w_overstock ratio: RetailConfig.stockout_penalty_multiplier,
    with overstock_penalty_multiplier held fixed at 0.2. Ratios are realized
    as stockout_multiplier = ratio * 0.2, so the table's stated "5:1
    (default)" row corresponds to stockout_multiplier=1.0.
  - gamma: the RL discount factor (the table's lambda_forecast dimension),
    shared across all three agents (Forecaster, Stock Manager, Coordinator).

This is a scaled-down sweep meant to produce illustrative plots for a paper,
not a publication-grade convergence study: each setting trains a handful of
seeds for a reduced episode budget rather than the thesis's original 50k-100k
episode runs. For each setting we report the mean +/- std (across seeds) of
the system-wide average reward (AdvancedMetricsCalculator's
'avg_total_reward'), itself averaged over the final TAIL_WINDOW episodes of
each run to smooth out per-episode noise.

The two sweeps share one "default" configuration point (5:1 ratio, gamma
0.99) — it is only trained once and reused in both tables.

Usage:
    python experiments/sweep_table9.py
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

from ralf.config import ExperimentConfig, RetailConfig
from ralf.data import load_retail_dataset
from ralf.training import train_heterogeneous_marl_with_timing

DATA_PATH = "data/retail_store_inventory.csv"
OUTPUT_DIR = "outputs"

SEEDS = [42, 1, 2, 3]
EPISODES = 300
TAIL_WINDOW = 20  # average system reward over the final N episodes of a run

DEFAULT_OVERSTOCK_MULT = 0.2
DEFAULT_STOCKOUT_MULT = 1.0  # realizes the paper's stated 5:1 default
DEFAULT_GAMMA = 0.99

RATIO_SETTINGS = [
    ("1:1", 0.2),
    ("2.5:1", 0.5),
    ("5:1 (default)", DEFAULT_STOCKOUT_MULT),
    ("10:1", 2.0),
]

GAMMA_SETTINGS = [
    ("0.90", 0.90),
    ("0.95", 0.95),
    ("0.97", 0.97),
    ("0.99 (default)", DEFAULT_GAMMA),
    ("0.995", 0.995),
]


def _base_kwargs(param_name, value):
    kwargs = {
        'stockout_mult': DEFAULT_STOCKOUT_MULT,
        'overstock_mult': DEFAULT_OVERSTOCK_MULT,
        'gamma': DEFAULT_GAMMA,
    }
    kwargs[param_name] = value
    return kwargs


def run_one(df, stockout_mult, overstock_mult, gamma, seed):
    """Train one (hyperparameter setting, seed) run and summarize it."""
    retail_config = RetailConfig(
        stockout_penalty_multiplier=stockout_mult,
        overstock_penalty_multiplier=overstock_mult,
    )
    exp_config = ExperimentConfig(
        num_episodes=EPISODES, seed=seed, device="cpu", gamma=gamma,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, _, metrics, _, _ = train_heterogeneous_marl_with_timing(
            df, exp_config, retail_config=retail_config
        )
    tail = metrics[-TAIL_WINDOW:]
    return {
        'mean_reward': float(np.mean([m.get('avg_total_reward', 0.0) for m in tail])),
        'mean_profit': float(np.mean([m.get('total_profit', 0.0) for m in tail])),
    }


def run_sweep(df, settings, param_name, cache):
    """Run every (setting, seed) combination in `settings`, reusing cached
    results for configurations already trained under a different sweep."""
    rows = []
    for label, value in settings:
        kwargs = _base_kwargs(param_name, value)
        config_key = (kwargs['stockout_mult'], kwargs['overstock_mult'], kwargs['gamma'])
        seed_results = cache.setdefault(config_key, {})

        for seed in SEEDS:
            if seed in seed_results:
                print(f"  [{param_name}={label}] seed={seed} -> reused from cache")
                continue
            t0 = time.time()
            result = run_one(df, seed=seed, **kwargs)
            seed_results[seed] = result
            print(f"  [{param_name}={label}] seed={seed} -> "
                  f"reward={result['mean_reward']:.2f} profit={result['mean_profit']:.0f} "
                  f"({time.time() - t0:.1f}s)")

        rewards = [seed_results[s]['mean_reward'] for s in SEEDS]
        profits = [seed_results[s]['mean_profit'] for s in SEEDS]
        rows.append({
            'Setting': label,
            'Value': value,
            'Mean_System_Reward': float(np.mean(rewards)),
            'Std_System_Reward': float(np.std(rewards)),
            'Mean_Profit': float(np.mean(profits)),
            'Std_Profit': float(np.std(profits)),
            'Seeds': str(SEEDS),
        })

    return pd.DataFrame(rows)


def plot_sweep(ratio_df, gamma_df, output_path):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(ratio_df['Setting'], ratio_df['Mean_System_Reward'],
                yerr=ratio_df['Std_System_Reward'], capsize=5,
                color='steelblue', edgecolor='black', alpha=0.85)
    axes[0].set_title(f'Sensitivity to $w_{{stockout}}:w_{{overstock}}$ ratio\n'
                       f'({len(SEEDS)} seeds, {EPISODES} episodes/run)',
                       fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Ratio (stockout : overstock)', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Mean System Reward', fontsize=10, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(gamma_df['Setting'], gamma_df['Mean_System_Reward'],
                yerr=gamma_df['Std_System_Reward'], capsize=5,
                color='darkorange', edgecolor='black', alpha=0.85)
    axes[1].set_title(f'Sensitivity to $\\gamma$ (discount factor)\n'
                       f'({len(SEEDS)} seeds, {EPISODES} episodes/run)',
                       fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Gamma', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Mean System Reward', fontsize=10, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def format_table9_markdown(ratio_df, gamma_df):
    lines = [
        "| Hyperparameter | Value | System Reward |",
        "|---|---|---|",
    ]
    for _, row in ratio_df.iterrows():
        lines.append(f"| $w_{{stockout}}:w_{{overstock}}$ | {row['Setting']} | "
                      f"{row['Mean_System_Reward']:.1f} +/- {row['Std_System_Reward']:.1f} |")
    for _, row in gamma_df.iterrows():
        lines.append(f"| $\\gamma$ | {row['Setting']} | "
                      f"{row['Mean_System_Reward']:.1f} +/- {row['Std_System_Reward']:.1f} |")
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_retail_dataset(DATA_PATH)

    # Cache keyed by the full underlying config (not by which sweep asked for
    # it), so the shared default point trains only once across both sweeps.
    cache = {}

    print("\n=== Sweeping w_stockout:w_overstock ratio ===")
    ratio_df = run_sweep(df, RATIO_SETTINGS, 'stockout_mult', cache)
    ratio_path = os.path.join(OUTPUT_DIR, 'table9_ratio_sweep.csv')
    ratio_df.to_csv(ratio_path, index=False)
    print(f"Saved: {ratio_path}")

    print("\n=== Sweeping gamma (paper's lambda_forecast / discount factor) ===")
    gamma_df = run_sweep(df, GAMMA_SETTINGS, 'gamma', cache)
    gamma_path = os.path.join(OUTPUT_DIR, 'table9_gamma_sweep.csv')
    gamma_df.to_csv(gamma_path, index=False)
    print(f"Saved: {gamma_path}")

    plot_path = os.path.join(OUTPUT_DIR, 'table9_hyperparameter_sensitivity.png')
    plot_sweep(ratio_df, gamma_df, plot_path)

    summary_path = os.path.join(OUTPUT_DIR, 'table9_summary.md')
    with open(summary_path, 'w') as f:
        f.write(format_table9_markdown(ratio_df, gamma_df))
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 80)
    print("TABLE 9 SWEEP COMPLETE")
    print("=" * 80)
    print(format_table9_markdown(ratio_df, gamma_df))

    return ratio_df, gamma_df


if __name__ == "__main__":
    main()
