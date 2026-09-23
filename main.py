#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RALF CLI entrypoint.

Usage:
    python main.py --data data/retail_store_inventory.csv --episodes 1000
"""

import argparse
import os

import pandas as pd
import torch

from ralf.analysis import build_algorithm_comparison_table, run_agent_snapshots
from ralf.config import ExperimentConfig
from ralf.data import fetch_kaggle_dataset, load_retail_dataset
from ralf.training import train_heterogeneous_marl_with_timing
from ralf.visualization import TrainingVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RALF multi-agent retail RL system.")
    parser.add_argument("--data", type=str, default="data/retail_store_inventory.csv",
                         help="Path to the retail inventory CSV dataset.")
    parser.add_argument("--no-auto-download", action="store_true",
                         help="Don't auto-download the dataset from Kaggle via kagglehub "
                              "if it's not found at --data.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None,
                         help="Device to use ('cpu' or 'cuda'). Defaults to cuda if available.")
    parser.add_argument("--output-dir", type=str, default="outputs",
                         help="Directory for plots, metrics CSVs, and snapshots.")
    parser.add_argument("--experiment-name", type=str, default="ralf_run")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating visualizations.")
    parser.add_argument("--skip-snapshots", action="store_true", help="Skip post-training agent snapshots.")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        if args.no_auto_download:
            raise FileNotFoundError(
                f"Dataset not found at '{data_path}'. Place your retail CSV there, pass --data <path>, "
                "or drop --no-auto-download to fetch it from Kaggle automatically."
            )
        print(f"No dataset found at '{data_path}' — fetching from Kaggle instead.")
        data_path = fetch_kaggle_dataset()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = load_retail_dataset(data_path)

    config = ExperimentConfig(
        num_episodes=args.episodes,
        seed=args.seed,
        device=device,
        experiment_name=args.experiment_name,
    )

    forecaster, stock_manager, coordinator, env, metrics, agent_times, total_times = \
        train_heterogeneous_marl_with_timing(df, config)

    if not args.skip_plots:
        visualizer = TrainingVisualizer(metrics, agent_times, total_times, config.num_episodes,
                                         output_dir=args.output_dir)
        visualizer.generate_all_plots()

    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(args.output_dir, "ralf_training_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics to: {metrics_csv_path}")

    final_metrics = metrics[-1]
    first_metrics = metrics[0]

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\n{'METRIC':<30} {'EPISODE 1':>15} {'FINAL EPISODE':>15} {'IMPROVEMENT':>15}")
    print("-" * 75)

    def pct_improve(first, final, higher_is_better=True):
        base = max(0.1, abs(first)) if first != 0 else 0.1
        change = (final - first) / base * 100
        return change if higher_is_better else -change

    print(f"{'Profit ($)':<30} ${first_metrics.get('total_profit', 0):>14.0f} ${final_metrics.get('total_profit', 0):>14.0f} "
          f"{pct_improve(first_metrics.get('total_profit', 0), final_metrics.get('total_profit', 0)):>14.1f}%")
    print(f"{'Revenue ($)':<30} ${first_metrics.get('total_revenue', 0):>14.0f} ${final_metrics.get('total_revenue', 0):>14.0f} "
          f"{pct_improve(first_metrics.get('total_revenue', 0), final_metrics.get('total_revenue', 0)):>14.1f}%")
    print(f"{'MAPE (%)':<30} {first_metrics.get('mape', 0):>14.1f}% {final_metrics.get('mape', 0):>14.1f}% "
          f"{pct_improve(first_metrics.get('mape', 0), final_metrics.get('mape', 0), higher_is_better=False):>14.1f}%")
    print(f"{'MAE (units)':<30} {first_metrics.get('mae', 0):>14.1f} {final_metrics.get('mae', 0):>14.1f} "
          f"{pct_improve(first_metrics.get('mae', 0), final_metrics.get('mae', 0), higher_is_better=False):>14.1f}%")
    print(f"{'RMSE (units)':<30} {first_metrics.get('rmse', 0):>14.1f} {final_metrics.get('rmse', 0):>14.1f} "
          f"{pct_improve(first_metrics.get('rmse', 0), final_metrics.get('rmse', 0), higher_is_better=False):>14.1f}%")
    print(f"{'Stockout Rate (%)':<30} {first_metrics.get('stockout_rate', 0):>14.1f}% {final_metrics.get('stockout_rate', 0):>14.1f}% "
          f"{pct_improve(first_metrics.get('stockout_rate', 0), final_metrics.get('stockout_rate', 0), higher_is_better=False):>14.1f}%")
    print(f"{'Overstock Rate (%)':<30} {first_metrics.get('overstock_rate', 0):>14.1f}% {final_metrics.get('overstock_rate', 0):>14.1f}% "
          f"{pct_improve(first_metrics.get('overstock_rate', 0), final_metrics.get('overstock_rate', 0), higher_is_better=False):>14.1f}%")
    print(f"{'Service Level (%)':<30} {first_metrics.get('service_level', 0):>14.1f}% {final_metrics.get('service_level', 0):>14.1f}% "
          f"{pct_improve(first_metrics.get('service_level', 0), final_metrics.get('service_level', 0)):>14.1f}%")
    print(f"{'Avg Total Reward':<30} {first_metrics.get('avg_total_reward', 0):>14.1f} {final_metrics.get('avg_total_reward', 0):>14.1f} "
          f"{pct_improve(first_metrics.get('avg_total_reward', 0), final_metrics.get('avg_total_reward', 0)):>14.1f}%")
    print("=" * 80)

    if not args.skip_snapshots:
        run_agent_snapshots(env, forecaster, stock_manager, coordinator,
                             output_dir=args.output_dir)

    build_algorithm_comparison_table(metrics, total_times, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
