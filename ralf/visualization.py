# -*- coding: utf-8 -*-
"""Training visualization dashboards."""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


class TrainingVisualizer:
    """Comprehensive visualization of training results"""

    def __init__(self, metrics_history: List[Dict], agent_times: Dict[str, List[float]],
                 total_times: List[float], num_episodes: int, output_dir: str = "outputs"):
        self.metrics_history = metrics_history
        self.agent_times = agent_times
        self.total_times = total_times
        self.num_episodes = num_episodes
        self.episodes = np.arange(1, num_episodes + 1)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def plot_financial_metrics(self):
        """Plot revenue and profit over episodes"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        revenues = [m['total_revenue'] for m in self.metrics_history]
        axes[0].plot(self.episodes, revenues, 'b-o', linewidth=2, markersize=4, label='Revenue')
        axes[0].fill_between(self.episodes, revenues, alpha=0.3, color='blue')
        axes[0].set_xlabel('Episode', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Total Revenue ($)', fontsize=11, fontweight='bold')
        axes[0].set_title('Revenue Progression Over Training', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)

        z = np.polyfit(self.episodes, revenues, 3)
        p = np.poly1d(z)
        axes[0].plot(self.episodes, p(self.episodes), 'r--', linewidth=2, alpha=0.7, label='Trend')

        profits = [m['total_profit'] for m in self.metrics_history]
        axes[1].plot(self.episodes, profits, 'g-o', linewidth=2, markersize=4, label='Profit')
        axes[1].fill_between(self.episodes, profits, alpha=0.3, color='green')
        axes[1].set_xlabel('Episode', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Total Profit ($)', fontsize=11, fontweight='bold')
        axes[1].set_title('Profit Progression Over Training', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)

        z = np.polyfit(self.episodes, profits, 3)
        p = np.poly1d(z)
        axes[1].plot(self.episodes, p(self.episodes), 'r--', linewidth=2, alpha=0.7, label='Trend')

        plt.tight_layout()
        plt.savefig(self._path('01_financial_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {self._path('01_financial_metrics.png')}")

    def plot_inventory_metrics(self):
        """Plot stockouts and overstocks over episodes"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        stockout_rates = [m['stockout_rate'] for m in self.metrics_history]
        axes[0, 0].plot(self.episodes, stockout_rates, 'r-o', linewidth=2, markersize=4)
        axes[0, 0].fill_between(self.episodes, stockout_rates, alpha=0.3, color='red')
        axes[0, 0].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[0, 0].set_ylabel('Stockout Rate (%)', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('Stockout Rate Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=2, color='g', linestyle='--', linewidth=2, label='Target: 2%')
        axes[0, 0].legend(fontsize=9)

        overstock_rates = [m['overstock_rate'] for m in self.metrics_history]
        axes[0, 1].plot(self.episodes, overstock_rates, 'orange', marker='o', linewidth=2, markersize=4)
        axes[0, 1].fill_between(self.episodes, overstock_rates, alpha=0.3, color='orange')
        axes[0, 1].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[0, 1].set_ylabel('Overstock Rate (%)', fontsize=10, fontweight='bold')
        axes[0, 1].set_title('Overstock Rate Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        total_stockouts = [m['total_stockouts'] for m in self.metrics_history]
        axes[1, 0].bar(self.episodes, total_stockouts, color='darkred', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[1, 0].set_ylabel('Total Stockout Units', fontsize=10, fontweight='bold')
        axes[1, 0].set_title('Cumulative Stockouts Per Episode', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        total_overstocks = [m['total_overstocks'] for m in self.metrics_history]
        axes[1, 1].bar(self.episodes, total_overstocks, color='darkorange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[1, 1].set_ylabel('Total Overstock Units', fontsize=10, fontweight='bold')
        axes[1, 1].set_title('Cumulative Overstocks Per Episode', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self._path('02_inventory_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {self._path('02_inventory_metrics.png')}")

    def plot_forecast_metrics(self):
        """Plot MAE, MAPE, and RMSE over episodes"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        maes = [m['mae'] for m in self.metrics_history]
        axes[0, 0].plot(self.episodes, maes, 'purple', marker='o', linewidth=2, markersize=4)
        axes[0, 0].fill_between(self.episodes, maes, alpha=0.3, color='purple')
        axes[0, 0].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[0, 0].set_ylabel('MAE (units)', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        mapes = [m['mape'] for m in self.metrics_history]
        axes[0, 1].plot(self.episodes, mapes, 'brown', marker='o', linewidth=2, markersize=4)
        axes[0, 1].fill_between(self.episodes, mapes, alpha=0.3, color='brown')
        axes[0, 1].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[0, 1].set_ylabel('MAPE (%)', fontsize=10, fontweight='bold')
        axes[0, 1].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        rmses = [m['rmse'] for m in self.metrics_history]
        axes[1, 0].plot(self.episodes, rmses, 'pink', marker='o', linewidth=2, markersize=4)
        axes[1, 0].fill_between(self.episodes, rmses, alpha=0.3, color='pink')
        axes[1, 0].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[1, 0].set_ylabel('RMSE (units)', fontsize=10, fontweight='bold')
        axes[1, 0].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        accuracies = [m['forecast_accuracy'] for m in self.metrics_history]
        axes[1, 1].plot(self.episodes, accuracies, 'teal', marker='o', linewidth=2, markersize=4)
        axes[1, 1].fill_between(self.episodes, accuracies, alpha=0.3, color='teal')
        axes[1, 1].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[1, 1].set_ylabel('Forecast Accuracy (%)', fontsize=10, fontweight='bold')
        axes[1, 1].set_title('Forecast Accuracy (1 - MAPE)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(self._path('03_forecast_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {self._path('03_forecast_metrics.png')}")

    def plot_reward_metrics(self):
        """Plot rewards for all agents over episodes"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        f_rewards = [m['avg_forecaster_reward'] for m in self.metrics_history]
        axes[0, 0].plot(self.episodes, f_rewards, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].fill_between(self.episodes, f_rewards, alpha=0.3, color='blue')
        axes[0, 0].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[0, 0].set_ylabel('Average Reward', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('Forecaster Agent Rewards', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        s_rewards = [m['avg_stock_mgr_reward'] for m in self.metrics_history]
        axes[0, 1].plot(self.episodes, s_rewards, 'g-o', linewidth=2, markersize=4)
        axes[0, 1].fill_between(self.episodes, s_rewards, alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[0, 1].set_ylabel('Average Reward', fontsize=10, fontweight='bold')
        axes[0, 1].set_title('Stock Manager Agent Rewards', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        c_rewards = [m['avg_coordinator_reward'] for m in self.metrics_history]
        axes[1, 0].plot(self.episodes, c_rewards, 'r-o', linewidth=2, markersize=4)
        axes[1, 0].fill_between(self.episodes, c_rewards, alpha=0.3, color='red')
        axes[1, 0].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[1, 0].set_ylabel('Average Reward', fontsize=10, fontweight='bold')
        axes[1, 0].set_title('Coordinator Agent Rewards', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        total_rewards = [m['avg_total_reward'] for m in self.metrics_history]
        axes[1, 1].plot(self.episodes, total_rewards, 'purple', marker='o', linewidth=2, markersize=4)
        axes[1, 1].fill_between(self.episodes, total_rewards, alpha=0.3, color='purple')
        axes[1, 1].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[1, 1].set_ylabel('Average Total Reward', fontsize=10, fontweight='bold')
        axes[1, 1].set_title('System-Wide Average Reward', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(self._path('04_reward_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {self._path('04_reward_metrics.png')}")

    def plot_training_time(self):
        """Plot training time per agent and total system time"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        forecaster_times = self.agent_times.get('forecaster', [])
        stock_mgr_times = self.agent_times.get('stock_manager', [])
        coord_times = self.agent_times.get('coordinator', [])

        if len(forecaster_times) > 0:
            axes[0].plot(self.episodes, forecaster_times, 'b-o', linewidth=2, markersize=4, label='Forecaster')
            axes[0].plot(self.episodes, stock_mgr_times, 'g-o', linewidth=2, markersize=4, label='Stock Manager')
            axes[0].plot(self.episodes, coord_times, 'r-o', linewidth=2, markersize=4, label='Coordinator')
            axes[0].set_xlabel('Episode', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
            axes[0].set_title('Individual Agent Training Time Per Episode', fontsize=12, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

        if len(self.total_times) > 0:
            axes[1].bar(self.episodes, self.total_times, color='steelblue', alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Episode', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
            axes[1].set_title('Total System Training Time Per Episode', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')

            avg_time = np.mean(self.total_times)
            axes[1].axhline(y=avg_time, color='r', linestyle='--', linewidth=2, label=f'Average: {avg_time:.2f}s')
            axes[1].legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(self._path('05_training_time.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {self._path('05_training_time.png')}")

    def plot_comprehensive_summary(self):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :2])
        profits = [m['total_profit'] for m in self.metrics_history]
        ax1.plot(self.episodes, profits, 'g-o', linewidth=2.5, markersize=5)
        ax1.fill_between(self.episodes, profits, alpha=0.3, color='green')
        ax1.set_ylabel('Profit ($)', fontsize=10, fontweight='bold')
        ax1.set_title('Cumulative Profit Progression', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 2])
        mapes = [m['mape'] for m in self.metrics_history]
        ax2.plot(self.episodes, mapes, 'r-o', linewidth=2, markersize=4)
        ax2.set_ylabel('MAPE (%)', fontsize=9, fontweight='bold')
        ax2.set_title('Forecast Accuracy', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        stockout_rates = [m['stockout_rate'] for m in self.metrics_history]
        ax3.plot(self.episodes, stockout_rates, 'darkred', marker='o', linewidth=2, markersize=4)
        ax3.set_ylabel('Stockout Rate (%)', fontsize=9, fontweight='bold')
        ax3.set_title('Stockouts', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=2, color='g', linestyle='--', linewidth=1.5, alpha=0.5)

        ax4 = fig.add_subplot(gs[1, 1])
        overstock_rates = [m['overstock_rate'] for m in self.metrics_history]
        ax4.plot(self.episodes, overstock_rates, 'orange', marker='o', linewidth=2, markersize=4)
        ax4.set_ylabel('Overstock Rate (%)', fontsize=9, fontweight='bold')
        ax4.set_title('Overstocks', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 2])
        revenues = [m['total_revenue'] for m in self.metrics_history]
        ax5.plot(self.episodes, revenues, 'b-o', linewidth=2, markersize=4)
        ax5.set_ylabel('Revenue ($)', fontsize=9, fontweight='bold')
        ax5.set_title('Total Revenue', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 0])
        maes = [m['mae'] for m in self.metrics_history]
        ax6.plot(self.episodes, maes, 'purple', marker='o', linewidth=2, markersize=4)
        ax6.set_xlabel('Episode', fontsize=9, fontweight='bold')
        ax6.set_ylabel('MAE (units)', fontsize=9, fontweight='bold')
        ax6.set_title('Forecast Error (MAE)', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        ax7 = fig.add_subplot(gs[2, 1])
        rmses = [m['rmse'] for m in self.metrics_history]
        ax7.plot(self.episodes, rmses, 'pink', marker='o', linewidth=2, markersize=4)
        ax7.set_xlabel('Episode', fontsize=9, fontweight='bold')
        ax7.set_ylabel('RMSE (units)', fontsize=9, fontweight='bold')
        ax7.set_title('Forecast Error (RMSE)', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[2, 2])
        total_rewards = [m['avg_total_reward'] for m in self.metrics_history]
        ax8.plot(self.episodes, total_rewards, 'teal', marker='o', linewidth=2, markersize=4)
        ax8.set_xlabel('Episode', fontsize=9, fontweight='bold')
        ax8.set_ylabel('Avg Reward', fontsize=9, fontweight='bold')
        ax8.set_title('System Reward', fontsize=11, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        fig.suptitle('RALF System: Comprehensive Training Dashboard', fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(self._path('00_comprehensive_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {self._path('00_comprehensive_dashboard.png')}")

    def generate_all_plots(self):
        """Generate all plots in sequence"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 80 + "\n")

        print("Creating comprehensive dashboard...")
        self.plot_comprehensive_summary()

        print("Creating financial metrics plots...")
        self.plot_financial_metrics()

        print("Creating inventory metrics plots...")
        self.plot_inventory_metrics()

        print("Creating forecast metrics plots...")
        self.plot_forecast_metrics()

        print("Creating reward metrics plots...")
        self.plot_reward_metrics()

        print("Creating training time plots...")
        self.plot_training_time()

        print("\n" + "=" * 80)
        print("ALL VISUALIZATIONS GENERATED!")
        print("=" * 80 + "\n")
        print(f"Files saved to: {self.output_dir}/")
