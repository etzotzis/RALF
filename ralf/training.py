# -*- coding: utf-8 -*-
"""Training loop for the heterogeneous MARL retail system."""

import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch.optim as optim

from ralf.agents import Coordinator, Forecaster, StockManager
from ralf.config import ExperimentConfig, RetailConfig
from ralf.environment import RetailMARLExplainable
from ralf.metrics import AdvancedMetricsCalculator


def train_heterogeneous_marl_with_timing(df: pd.DataFrame, config: ExperimentConfig = None,
                                          retail_config: RetailConfig = None) -> Tuple:
    """Complete training loop with detailed timing and metrics"""

    if config is None:
        config = ExperimentConfig()

    print(f"\n{'='*80}")
    print("STARTING HETEROGENEOUS MARL TRAINING")
    print(f"{'='*80}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Device: {config.device}")
    print(f"Timestamp: {config.timestamp}")
    print(f"{'='*80}\n")

    if retail_config is None:
        retail_config = RetailConfig()
    env = RetailMARLExplainable(df, config=retail_config, seed=config.seed)

    forecaster = Forecaster(obs_dim=10, hidden_dim=config.hidden_dim,
                             lr=config.forecaster_lr, device=config.device,
                             max_demand=retail_config.max_demand, gamma=config.gamma)
    stock_manager = StockManager(obs_dim=11, hidden_dim=config.hidden_dim,
                                  actor_lr=config.stock_manager_actor_lr,
                                  critic_lr=config.stock_manager_critic_lr,
                                  device=config.device,
                                  max_order_quantity=retail_config.max_order_quantity,
                                  gamma=config.gamma)
    coordinator = Coordinator(obs_dim=12, hidden_dim=config.hidden_dim,
                               lr=config.coordinator_lr, device=config.device,
                               ppo_epsilon=config.ppo_epsilon, gamma=config.gamma)

    forecaster_scheduler = optim.lr_scheduler.StepLR(
        forecaster.policy_optimizer, step_size=500, gamma=0.95
    )
    stock_manager_actor_scheduler = optim.lr_scheduler.StepLR(
        stock_manager.actor_optimizer, step_size=500, gamma=0.95
    )
    stock_manager_critic_scheduler = optim.lr_scheduler.StepLR(
        stock_manager.critic_optimizer, step_size=500, gamma=0.95
    )
    coordinator_scheduler = optim.lr_scheduler.StepLR(
        coordinator.policy_optimizer, step_size=500, gamma=0.95
    )

    metrics_calc = AdvancedMetricsCalculator()
    metrics_history = []

    agent_times = {
        'forecaster': [],
        'stock_manager': [],
        'coordinator': []
    }
    total_times = []

    training_start = time.time()

    for episode in range(config.num_episodes):
        episode_start = time.time()

        obs, info = env.reset()
        terminated = False
        step_count = 0

        forecaster_time = 0
        stock_mgr_time = 0
        coord_time = 0

        while not terminated and step_count < env.episode_length:
            f_start = time.time()
            f_action = forecaster.select_action(obs['forecaster'])
            forecaster_time += time.time() - f_start

            s_start = time.time()
            s_action = stock_manager.select_action(obs['stock_manager'])
            stock_mgr_time += time.time() - s_start

            c_start = time.time()
            c_action = coordinator.select_action(obs['coordinator'])
            coord_time += time.time() - c_start

            obs, rewards, terminated, _, info = env.step({
                'forecaster': np.array([f_action]),
                'stock_manager': s_action,
                'coordinator': c_action
            })

            forecaster.store_transition(rewards['forecaster'])
            stock_manager.store_transition(rewards['stock_manager'], obs['stock_manager'], terminated)
            coordinator.store_transition(rewards['coordinator'])

            step_count += 1

        # Update agents ONCE per episode: on-policy A2C/PPO batches for the
        # Forecaster/Coordinator, off-policy replay-buffer sampling for the
        # Stock Manager's DDPG update.
        forecaster.update(next_value=0.0, n_epochs=config.n_epochs)
        stock_manager.update(next_value=0.0, n_epochs=config.n_epochs)
        coordinator.update(next_value=0.0, n_epochs=config.n_epochs)

        forecaster_scheduler.step()
        stock_manager_actor_scheduler.step()
        stock_manager_critic_scheduler.step()
        coordinator_scheduler.step()

        agent_times['forecaster'].append(forecaster_time)
        agent_times['stock_manager'].append(stock_mgr_time)
        agent_times['coordinator'].append(coord_time)

        episode_time = time.time() - episode_start
        total_times.append(episode_time)

        episode_metrics = metrics_calc.calculate_detailed_metrics(env, retail_config)
        metrics_history.append(episode_metrics)

        if (episode + 1) % 100 == 0 or episode == 0:
            print(f"Episode {episode+1:3d}/{config.num_episodes} | "
                  f"Profit: ${episode_metrics.get('total_profit', 0):>8.0f} | "
                  f"Revenue: ${episode_metrics.get('total_revenue', 0):>8.0f} | "
                  f"MAPE: {episode_metrics.get('mape', 0):>5.1f}% | "
                  f"Stockouts: {episode_metrics.get('total_stockouts', 0):>3.0f} | "
                  f"Overstocks: {episode_metrics.get('total_overstocks', 0):>3.0f} | "
                  f"Time: {episode_time:.2f}s")

    total_time = time.time() - training_start

    avg_forecaster_time = np.mean(agent_times['forecaster'])
    avg_stock_mgr_time = np.mean(agent_times['stock_manager'])
    avg_coord_time = np.mean(agent_times['coordinator'])
    avg_total_time = np.mean(total_times)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Avg Time per Episode: {avg_total_time:.2f}s")
    print("\nPer-Agent Average Time per Episode:")
    print(f"  Forecaster: {avg_forecaster_time:.4f}s")
    print(f"  Stock Manager: {avg_stock_mgr_time:.4f}s")
    print(f"  Coordinator: {avg_coord_time:.4f}s")
    print(f"{'='*80}\n")

    return forecaster, stock_manager, coordinator, env, metrics_history, agent_times, total_times
